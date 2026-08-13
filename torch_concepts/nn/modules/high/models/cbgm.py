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
from ...low.dense_layers import LinearEmbeddingEncoder, MLPEmbeddingEncoder
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.mix import MixConceptEmbeddings
from ...low.priors import FixedPrior
from ...low.scales import GlobalScale
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
    (:class:`~torch_concepts.nn.MixConceptEmbeddings`) sits between the
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
        The model composes the observation parameter's activation on top —
        identity for a ``Normal``'s ``loc``, a sigmoid for a ``Bernoulli``'s
        ``probs`` — so a decoder that squashes its own output is activated twice.
    latent_size : int, default 64
        Dimensionality of ``z``.
    embedding_size : int, default 16
        Width ``m`` of a single context embedding.
    use_unknown : bool, default True
        Whether the bottleneck carries the *unsupervised* context. ``True`` is
        the paper's ``w = [w_1, ..., w_k, w_{k+1}]``. ``False`` removes the slot
        entirely, shrinking the bottleneck to ``m * k`` and making the
        orthogonality penalty vacuous — the ablation in Table 3 of the paper,
        where it costs both steerability and sample quality.

        The unsupervised context is always ``embedding_size`` wide: the
        orthogonality penalty is a cosine similarity against the concept
        contexts (Eq. 5), which is undefined between vectors of different widths.
    context_hidden_size : int or None, default None
        Width of a hidden layer in the context networks ``z -> w``. ``None``
        builds them as :class:`~torch_concepts.nn.LinearEmbeddingEncoder`, which
        is the reference's layout — but the reference is a VAE-*GAN*, whose
        discriminator supplies the missing nonlinearity elsewhere. In the pure
        VAE built here a linear context network plus a shallow decoder makes the
        whole path ``z -> pixels`` close to affine, and an affine generator
        reproduces the data mean near the codes it was trained on and diverges
        away from them — sharp reconstructions, incoherent prior samples. Set it
        to swap in :class:`~torch_concepts.nn.MLPEmbeddingEncoder`.
    context_norm : str or None, default None
        Normalisation on the context embeddings: ``'layer'``, ``'batch'`` or
        ``None`` for none. Only meaningful alongside ``context_hidden_size`` —
        passing it alone raises rather than being dropped, since a
        ``LinearEmbeddingEncoder`` takes no normalisation. The reference uses
        ``BatchNorm1d``, which bounds the embeddings when ``z`` wanders off the
        posterior; ``'layer'`` does the same without depending on the batch, so
        a single generated sample decodes the way a batch of them does.
    observation : type, default ``torch.distributions.Normal``
        Distribution family of the generated variable. ``Bernoulli`` for images
        in ``[0, 1]``; ``Normal`` needs a ``scale`` too — see ``global_scale``.
    global_scale : bool, default True
        Only consulted when ``observation`` has a ``scale``. ``True``: one
        learnable sigma shared by every pixel and sample
        (:class:`~torch_concepts.nn.GlobalScale`). ``False``: ``scale`` gets its
        own copy of ``decoder``.
    scale_init : float, default 1.0
        The ``global_scale`` standard deviation — its starting point when
        ``scale_learnable``, its fixed value otherwise. It sets the weight of the
        reconstruction term: the Gaussian NLL's gradient carries a
        ``1 / scale**2`` factor, so halving it quadruples reconstruction relative
        to the KL. For images in ``[0, 1]``, values below ~0.3 make the KL
        negligible unless its loss weight is raised to compensate.
    scale_learnable : bool, default True
        Whether ``global_scale`` is trained. A learned scale settles at the
        residual RMS, which shrinks as the fit improves and therefore keeps
        *raising* the effective reconstruction weight — annealing the KL away.
        Set ``False`` to pin the trade-off at ``scale_init``.
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
    * The reference context generators end in ``BatchNorm1d``. Here they default
      to a bare :class:`~torch_concepts.nn.LinearEmbeddingEncoder`; set
      ``context_hidden_size`` and ``context_norm='batch'`` to approach it, or
      ``'layer'`` for the batch-independent equivalent.
    * The reference concept head is a per-concept ``Linear(bins * m, bins)``.
      Every concept here keeps the CEM head instead — a ``Linear(m, 1)`` shared
      across state embeddings — including a binary one, whose second state
      ``w-`` is derived inside the mixture rather than encoded.
    * What the mixture reads is the engine's ``p_int``, not a setting here. At
      ``1.0`` the concepts are teacher-forced on the ground-truth labels: that
      grounds the concept channel, but an intervention then *selects* a state
      rather than forming the convex combination of Section 3.1, and the
      unselected state embedding never sees reconstruction gradient. At ``0.0``
      the concepts are sampled through the relaxation (Gumbel-Softmax, endorsed
      by Appendix A of the paper, with the engine's temperature schedule
      controlling it) and the mixture reads a score in ``(0, 1)``, as the
      reference does. **Between** the two is CEM's RandInt, which is what makes
      the model steerable: the decoder is trained on concept values it did not
      itself predict, exactly what an intervention hands it. Set it on the train
      engine and leave evaluation at ``0.0``.

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
    ...     # Raw: the observation's activation (identity for `loc`, a sigmoid
    ...     # for a Bernoulli's `probs`) is composed on top for you.
    ...     decoder=MLP(3 * 8, 128, 784),
    ...     latent_size=32, embedding_size=8, observation=Bernoulli,
    ... )  # doctest: +SKIP
    >>> out = model(query=list(model.pgm.variables), input=torch.rand(4, 784))  # doctest: +SKIP

    See Also
    --------
    torch_concepts.nn.MixConceptEmbeddings : the concept bottleneck layer
    torch_concepts.nn.functional.concept_orthogonality : the orthogonality penalty
    torch_concepts.nn.GlobalScale : the default ``scale`` head for a Normal observation
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
        use_unknown: bool = True,
        context_hidden_size: Optional[int] = None,
        context_norm: Optional[str] = None,
        observation: Type = Normal,
        global_scale: bool = True,
        scale_init: float = 1.0,
        scale_learnable: bool = True,
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
        self.use_unknown = bool(use_unknown)
        self.context_hidden_size = context_hidden_size
        self.context_norm = context_norm
        if context_hidden_size is None and context_norm is not None:
            raise ValueError(
                f"{type(self).__name__}: `context_norm={context_norm!r}` needs "
                "`context_hidden_size` too. Without it the context networks are "
                "LinearEmbeddingEncoders, which take no normalisation — the "
                "setting would be silently dropped."
            )
        self.observation = observation
        self.global_scale = global_scale
        self.scale_init = scale_init
        self.scale_learnable = scale_learnable
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.decoder = decoder if decoder is not None else nn.Identity()

        self.pgm = self._build_model()

        # The guide q(z | input). Whether the engine's discrete draws are soft or
        # hard is not set here: it follows from the concepts' declared family
        # (see `variable_distributions`).
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
        """Query **every** variable, supplying the concepts' ground truth.

        Overrides the base learner's concept-only query
        (:meth:`~torch_concepts.nn.modules.high.base.learner.BaseLearner.default_query`).
        :class:`~torch_concepts.nn.VariationalInference` requires all variables
        in the query — observed ones with values, latents absent or ``None`` —
        and the generative loss terms need the ones it would otherwise leave out:
        ``input`` for the reconstruction and ``mixing``/``unknown`` for the
        orthogonality penalty.

        The ground truth is always supplied; how often it is actually used is the
        engine's ``p_int``. Always supplying it is what gives RandInt a target to
        force *to* — a query that dropped the concepts could only ever sample
        them. Their supervision is unaffected either way: the concept loss scores
        the reported ``probs``, which are the CPD's predictions no matter which
        value propagates.
        """
        return {
            **{name: None for name in self.pgm.variables},
            **self.fully_observed_query(c),
        }

    def default_extra(self, evidence):
        """Publish the evidence so :class:`~torch_concepts.nn.ReconstructionLoss`
        can score the observed variable (e.g. ``input``) against it."""
        return {"evidence": evidence}

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

        With ``use_unknown=False`` the unsupervised context is left out of the
        graph entirely — variable, encoder and decoder parent alike — so the
        decoder reads the ``k`` mixed concept contexts and nothing else.
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
        # bottleneck carries one extra, unsupervised context embedding. Held as a
        # list -- empty under the `use_unknown=False` ablation -- so each of the
        # places it participates can splat it and stay branch-free.
        unknowns = [
            EmbeddingVariable(
                "unknown",
                distribution=Delta,
                shape=(1, self.embedding_size),
            )
        ] if self.use_unknown else []
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
        # z -> embeddings: one batched encoder per group, plus the unsupervised
        # one. Linear is the reference's layout; `context_hidden_size` swaps in
        # the non-linear encoder instead (see the class docstring for when).
        def context_network(n_embeddings: int) -> nn.Module:
            if self.context_hidden_size is None:
                return LinearEmbeddingEncoder(
                    in_features=self.latent_size,
                    out_features=self.embedding_size,
                    n_embeddings=n_embeddings,
                )
            return MLPEmbeddingEncoder(
                in_features=self.latent_size,
                out_features=self.embedding_size,
                n_embeddings=n_embeddings,
                hidden_size=self.context_hidden_size,
                norm=self.context_norm,
            )

        emb_encoders = ParametricCPD(
            variable=[*embeddings, *unknowns],
            parents=[latent],
            parametrization=[
                {"value": context_network(e.shape[0])}
                for e in [*embeddings, *unknowns]
            ],
        )
        # embeddings → concepts: one score per state embedding (per group).
        c_encoders = [
            ParametricCPD(
                variable=cvar,
                parents=[evar],
                parametrization=self._flexible_parametrization(
                    variable=cvar,
                    first=pyc.nn.Sequential(
                        LinearEmbeddingToConcept(
                            in_embeddings=self.embedding_size,
                            out_concepts=1,
                        ),
                        # Collapse the (n_concepts, 1) score dims -> n_concepts
                        nn.Flatten(start_dim=-2),
                    ),
                    second="copy",
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
            parametrization={
                "value": MixConceptEmbeddings(
                    in_concepts=reordered_axis, # require Annotations as in_concepts
                    in_embeddings=self.embedding_size,
            )},
            aggregate=mix_parents,
        )

        def cat_embeddings(embeddings):
            return torch.cat(list(embeddings.values()), dim=-2)
        
        def decoder_head(decoder):
            return pyc.nn.Sequential(nn.Flatten(start_dim=-2), decoder)

        # `loc` always comes from the decoder. A `scale` — only allocated when the
        # family has one — is either a single learnable sigma (homoscedastic, the
        # default) or its OWN copy of the decoder: sharing a trunk with `loc` would
        # make the spread a fixed function of the mean.
        if "loc" not in observed.param_sizes:
            scale_head = None
        elif self.global_scale:
            scale_head = GlobalScale(observed.size, init=self.scale_init,
                                     learnable=self.scale_learnable)
        else:
            scale_head = decoder_head(copy.deepcopy(self.decoder))
        decoder_cpd = ParametricCPD(
            variable=observed,
            parents=[mixing, *unknowns],
            parametrization=self._flexible_parametrization(
                variable=observed,
                first=decoder_head(self.decoder),
                second=scale_head,
            ),
            aggregate=cat_embeddings,
        )

        return BayesianNetwork(
            variables=[latent, *embeddings, *unknowns, *concepts, mixing, observed],
            factors=[latent_cpd, *emb_encoders, *c_encoders, mixing_cpd, decoder_cpd],
        )
