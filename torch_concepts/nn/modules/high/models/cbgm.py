"""Concept Bottleneck Generative Model (CBGM), the VAE variant.

Where the discriminative models run ``input → latent → concepts → tasks``, this
one runs the generative direction ``z → concepts → input`` (Ismail et al.,
ICLR 2024). A latent ``z ~ N(0, I)`` produces per-concept state embeddings; each
concept is decoded from its own embeddings; the embeddings are mixed by the
predicted concept probabilities (the CEM mixture, reused verbatim); and the
resulting bottleneck — the ``k`` mixed concept contexts plus one *unsupervised*
context — is decoded back into the observation.

The model is assembled by a single builder, :meth:`_build_model`, and shares the
CEM's level factories: concepts and their embeddings are grouped identically by
:meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.build_concept_variables`
and
:meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.build_concept_embedding_variables`,
so the two lists align element-by-element.

Inference is variational: the guide ``q(z | input)`` is registered with the Pyro
:class:`~torch_concepts.nn.VariationalInference` engine, which therefore requires
``pyro-ppl``.

References
----------
Ismail et al. "Concept Bottleneck Generative Models", ICLR 2024.
https://openreview.net/forum?id=L9U5MJJleF
"""
import copy
from typing import Optional, Type

import torch
import torch.nn as nn

from torch.distributions import Bernoulli, Normal, OneHotCategorical

import torch_concepts as pyc
from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from .....distributions import Delta
from ...low.dense_layers import LinearEmbeddingEncoder
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.mix import MixConceptEmbeddingToEmbedding
from ...low.priors import FixedPrior
from ...mid.inference.base import BaseInference
from ...mid.inference.pyro.variational import VariationalInference
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.factors.cpd import ParametricCPD
from ...mid.variable import EmbeddingVariable
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ..base.graph import DirectedGraphModel


class ConceptBottleneckGenerativeModel(DirectedGraphModel):
    """Concept Bottleneck Generative Model (VAE variant).

    Generative process ``z → concepts → input``, trained as a VAE through a
    variational guide ``q(z | input)``. The concept bottleneck layer
    (:class:`~torch_concepts.nn.MixConceptEmbeddingToEmbedding`) sits between the
    two halves of the decoder, so intervening on a concept steers the generated
    output.

    Parameters
    ----------
    input_size : int
        Dimensionality of one observation (the generated variable).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types). Every concept is
        supervised; there are no task variables.
    encoder : nn.Module
        The guide's location head, mapping an observation (``input_size``) to
        ``latent_size`` values: the mean of ``q(z | input)``.
    decoder : nn.Module
        The post-concept-bottleneck network, mapping the flattened bottleneck
        (``embedding_size * (n_concepts + 1)``) to ``input_size`` **raw** values.
        The observation parameter's activation is composed on top by the model
        (a sigmoid for a ``Bernoulli``'s ``probs``), so a decoder that squashes
        its own output would be activated twice.
    latent_size : int, default 64
        Dimensionality of ``z``.
    embedding_size : int, default 16
        Width ``m`` of a single context embedding.
    observation : type, default ``torch.distributions.Bernoulli``
        Distribution family of the generated variable. ``Bernoulli`` for images
        in ``[0, 1]``; ``Normal`` adds a learned scale head.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration. Defaults to
        :class:`~torch_concepts.nn.VariationalInference`, with the guide on
        ``z`` injected into ``inference_kwargs['latents']``.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default None
        Per-level plate preference (see :class:`BaseModel`). ``None``/``True``
        group homogeneous concepts into the minimum number of plates; ``False``
        gives one variable per concept, which is the reference implementation's
        layout.
    **kwargs
        Forwarded to :class:`BaseModel`.

    Notes
    -----
    Deviations from the authors' released code, which implements the VAE-**GAN**
    variant (paper Eq. 2) rather than the pure VAE (Eq. 1) built here:

    * The decoder input is the paper's ``m(k + 1)`` bottleneck; the code also
      prepends the concept probabilities.
    * The reference context generators end in ``BatchNorm1d``;
      :class:`~torch_concepts.nn.LinearEmbeddingEncoder` does not.
    * The reference concept head is a per-concept ``Linear(bins * m, bins)``;
      the reused CEM head is a ``Linear(m, 1)`` shared across states.
    * The reference gives a binary concept ``bins = 2`` — two state embeddings
      and a 2-way one-hot. A cardinality-1 concept here gets *one* embedding,
      which
      :class:`~torch_concepts.nn.MixConceptEmbeddingToEmbedding` splits in two
      with a learned ``Linear(m, 2m) + LeakyReLU``. Declare such concepts as
      2-way ``categorical`` to match the reference exactly.
    * Under Pyro, concepts sample through a straight-through relaxation rather
      than a plain softmax — Appendix A of the paper endorses Gumbel-Softmax
      here, and the engine's temperature schedule controls it. The mixture
      therefore reads a hard one-hot (with gradients) where the reference reads
      the continuous softmax, so an intervention selects a state instead of
      forming the convex combination of Section 3.1.

    Examples
    --------
    >>> import torch
    >>> from torch.distributions import Bernoulli
    >>> from torch_concepts.annotations import Annotations
    >>> from torch_concepts.nn import ConceptBottleneckGenerativeModel, MLP
    >>>
    >>> ann = Annotations(labels=['digit', 'color'], cardinalities=[10, 2],
    ...                   types=['categorical', 'categorical'])
    >>> model = ConceptBottleneckGenerativeModel(
    ...     input_size=784, annotations=ann,
    ...     encoder=MLP(784, 128, 32),
    ...     # Raw: the Bernoulli's `probs` sigmoid is composed on top for you.
    ...     decoder=MLP(3 * 8, 128, 784),
    ...     latent_size=32, embedding_size=8, observation=Bernoulli,
    ... )  # doctest: +SKIP
    >>> out = model(query=list(model.pgm.variables), input=torch.rand(4, 784))  # doctest: +SKIP

    See Also
    --------
    torch_concepts.nn.MixConceptEmbeddingToEmbedding : the concept bottleneck layer
    torch_concepts.nn.functional.concept_orthogonality : the orthogonality penalty
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    # The reference mixes the state embeddings by the concept *probabilities*,
    # so the bottleneck reads a normalised score rather than a raw logit.
    param_for_discrete_var = "probs"

    variable_distributions = {
        'binary': Bernoulli,
        'categorical': OneHotCategorical,
        'continuous': Normal,
    }
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        latent_size: int = 64,
        embedding_size: int = 16,
        observation: Type = Normal,
        inference: Optional[BaseInference] = VariationalInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        plate: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            latent_size=latent_size,
            lightning=lightning,
            plate=plate,
            **kwargs,
        )
        self.embedding_size = embedding_size
        self.observation = observation
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.decoder = decoder if decoder is not None else nn.Identity()

        self.pgm = self._build_model()

        # The guide q(z | input)
        guide = {"latents": {"z": self._build_guide()}}
        self.setup_inference(
            inference,
            {**guide, **(inference_kwargs or {})},
            train_inference,
            {**guide, **(train_inference_kwargs or {})},
        )

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def _resolve_graph(self) -> ConceptGraph:
        """Build the edgeless concept graph.

        A CBGM's concepts are conditionally independent given ``z``
        """
        labels = list(self.concept_names)
        return ConceptGraph(
            torch.zeros(len(labels), len(labels)),
            node_names=labels,
        )

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

    def default_query(self, c):
        """Query **every** variable, with the concepts teacher-forced.

        Overrides the base learner's concept-only query
        (:meth:`~torch_concepts.nn.modules.high.base.learner.BaseLearner.default_query`).
        :class:`~torch_concepts.nn.VariationalInference` requires all variables
        in the query — observed ones with values, latents absent or ``None`` —
        and the generative loss terms need the ones it would otherwise leave out:
        ``input`` for the reconstruction and ``mixing``/``unknown`` for the
        orthogonality penalty.
        """
        return {
            **{name: None for name in self.pgm.variables},
            **self.fully_observed_query(c),
        }

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------

    def _build_guide(self) -> ParametricCPD:
        """The variational posterior ``q(z | input)``, a Normal CPD on ``z``.

        The feature extractor is the CPD's **trunk**, not part of either
        parameter's head: ``loc`` and ``scale`` are two small linear readouts of
        the same features, so the backbone runs once per step. Sharing is safe
        here because both heads are independently learnable ``Linear`` layers —
        put the *scoring* layer in a trunk instead and the scale would collapse
        to a fixed function of the location.
        """
        z = self.pgm.variables["z"]
        observed = self.pgm.variables["input"]

        # Width of the trunk's output: the encoder's if it declares one (MLP,
        # nn.Linear), else the backbone's — `encoder` defaults to nn.Identity.
        width = (getattr(self.encoder, "out_features", None)
                 or getattr(self.backbone, "out_features", None))
        if width is None:
            raise ValueError(
                f"{type(self).__name__}: cannot size the guide's readout — neither "
                "`encoder` nor `backbone` declares `out_features`. Set that attribute "
                "on one of them, or pass an encoder that does (e.g. MLP, nn.Linear)."
            )
        return ParametricCPD(
            variable=z,
            parents=[observed],
            trunk=nn.Sequential(self.backbone, self.encoder),
            parametrization=self._flexible_parametrization(
                variable=z,
                first=nn.Linear(width, z.size),
                second=nn.Linear(width, z.size),
            ),
        )

    def _build_model(self) -> BayesianNetwork:
        """Assemble the CBGM Bayesian network.

        ``z → {embeddings, unknown} → concepts → context → input``: the standard
        Normal prior on ``z`` produces one embedding matrix per concept group
        plus one unsupervised context; each group's concepts are decoded from
        their embeddings; the concept bottleneck layer mixes them and appends the
        unsupervised context; the decoder turns that bottleneck into the
        observation's distribution parameters.
        """

        # --- variables ---
        observed = EmbeddingVariable("input", distribution=self.observation, shape=self.input_size)
        latent = EmbeddingVariable("z", distribution=Normal, size=self.latent_size)
        # Concepts and their embeddings share the grouping, hence align 1:1.
        concepts = self.build_concept_variables(self.concept_names, plate_name="concepts")
        embeddings = self.build_concept_embedding_variables(
            self.concept_names,
            self.embedding_size,
            plate_name="embeddings",
        )
        # The pre-defined concepts are incomplete in a generative setting, so the
        # bottleneck carries one extra, unsupervised context embedding.
        unknown = EmbeddingVariable(
            "unknown",
            distribution=Delta,
            shape=(1, self.embedding_size),
        )
        ordered_names = [m for cvar in concepts for m in cvar.members]
        reordered_axis = self.concept_annotations.subset(ordered_names)
        n_concepts = len(ordered_names)
        mixing = EmbeddingVariable(
            "mixing",
            distribution=Delta,
            shape=(n_concepts, self.embedding_size),
        )
    

        # --- factors ---
        # p(z) = N(0, I): fixed, not learned, so the guide has a fixed target.
        latent_cpd = ParametricCPD(
            latent,
            parents=[],
            parametrization={
                "loc": FixedPrior(torch.zeros(self.latent_size)),
                "scale": FixedPrior(torch.ones(self.latent_size)),
            },
        )
        # z -> embeddings: one batched encoder per group, plus the unsupervised one.
        emb_encoders = ParametricCPD(
            variable=[*embeddings, unknown],
            parents=[latent],
            parametrization=[
                {"value": LinearEmbeddingEncoder(
                    in_features=self.latent_size,
                    out_features=self.embedding_size,
                    n_embeddings=e.shape[0],
                )}
                for e in [*embeddings, unknown]
            ],
        )
        # embeddings -> concepts: decode one score per state embedding (per group).
        def concept_head():
            """One raw score per state embedding: ``(B, C, m) -> (B, C)``."""
            return pyc.nn.Sequential(
                LinearEmbeddingToConcept(
                    in_embeddings=self.embedding_size, # (B, C, emb_dim)
                    out_concepts=1, # (B, C, 1)
                ),
                # Collapse the (n_states, 1) score dims -> n_states
                nn.Flatten(start_dim=-2), # (B, C, 1) -> (B, C)
            )

        c_encoders = [
            ParametricCPD(
                variable=cvar,
                parents=[evar],
                parametrization=self._flexible_parametrization(
                    variable=cvar,
                    first=concept_head(),
                    second=concept_head(),
                ),
            )
            for cvar, evar in zip(concepts, embeddings)
        ]

        # Embeddings concatenate along the *states* axis,
        def mix_parents(concepts, embeddings):
            return {
                "concepts": torch.cat(list(concepts.values()), dim=-1),
                "embeddings": torch.cat(list(embeddings.values()), dim=-2),
            }

        mixing_cpd = ParametricCPD(
            variable=mixing,
            parents=[*concepts, *embeddings],
            parametrization={"value": MixConceptEmbeddingToEmbedding(
                in_concepts=reordered_axis, # require Annotations as in_concepts
                in_embeddings=self.embedding_size,
            )},
            aggregate=mix_parents,
        )

        def cat_embeddings(embeddings):
            return torch.cat(list(embeddings.values()), dim=-2)
        
        def decoder_head(decoder):
            return pyc.nn.Sequential(nn.Flatten(start_dim=-2), decoder)

        # A continuous observation needs `loc` and `scale` from *independent*
        # nets — sharing them through a trunk would make the scale a function of
        # the location. So the scale gets its own copy of the decoder, which is
        # only allocated when the observation family actually has a scale.
        needs_scale = "loc" in observed.param_sizes
        decoder_cpd = ParametricCPD(
            variable=observed,
            parents=[mixing, unknown],
            parametrization=self._flexible_parametrization(
                variable=observed,
                first=decoder_head(self.decoder),
                second=decoder_head(copy.deepcopy(self.decoder)) if needs_scale else None,
            ),
            aggregate=cat_embeddings,
        )

        return BayesianNetwork(
            variables=[latent, *embeddings, unknown, *concepts, mixing, observed],
            factors=[latent_cpd, *emb_encoders, *c_encoders, mixing_cpd, decoder_cpd],
        )
