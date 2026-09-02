"""Conditional Variational Autoencoder (CVAE), conditioned on a set of concepts.

CVAE conditions a VAE on an observed variable ``c``: ``q(z | x, c)`` encodes, ``p(x | z, c)`` decodes.
In the concept-based setting, the concepts are the condition, and the CVAE 
is a generative model that factorizes the joint as follows
``p(c) p(z | c) p(x | z, c)``.

The prior defaults to the paper's **conditional** ``p(z | c)``: ``z`` is drawn per
condition rather than from a fixed ``N(0, I)``, so the latent only has to carry
what ``c`` does not. It reads the concepts through the same embedding the guide
and the decoder use, so ``c`` has one learned representation across the whole
model. ``conditional_prior=False`` swaps in a plain ``N(0, I)`` instead — the
ablation, and the setting under which the KL target is fixed.

The guide is always ``q(z | x, c)``, the paper's recognition network.

One deviation from the paper remains, in the direction of the simpler
formulation the practitioner's version uses:

* No GSNN / hybrid objective (paper Sec. 4.2): those address the train/test
  mismatch of a *predictive* CVAE, whose condition is an input image. Here the
  condition is a low-dimensional concept vector supplied identically at training
  and at generation, so the mismatch does not arise.

The guide needs no matching branch either. The reference's ``guide`` falls back
to its ``prior_net`` when the output is unavailable at test time; here ``c`` is
always supplied as evidence and the unknown quantity is ``input``, which
generation draws from the model rather than from the guide.

References
----------
Sohn et al. "Learning Structured Output Representation using Deep
Conditional Generative Models", NeurIPS 2015.
"""
from typing import List, Optional

import torch
import torch.nn as nn

from torch.distributions import Bernoulli, Normal, OneHotCategorical

import torch_concepts as pyc
from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from .....distributions import Delta
from ...low.dense_layers import MLP
from ...low.priors import FixedPrior, LearnablePrior
from ...mid.inference.base import BaseInference
from ...mid.inference.pyro.variational import VariationalInference
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.factors.cpd import ParametricCPD
from ...mid.variable import EmbeddingVariable
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ..base.graph import DirectedGraphModel


class ConceptEmbedding(nn.Module):
    """One learnable embedding per concept: ``k`` separate ``Linear(size_i, m)``.
    Takes the concepts concatenated on the last axis and returns a tensor of 
    shape ``[..., n_concepts * m]``.

    Parameters
    ----------
    sizes : list of int
        Width of each concept's value, in the order they are concatenated —
        ``1`` for a binary or continuous concept, ``cardinality`` for a
        categorical one. Per *member*, so a plate contributes one entry per
        member rather than one for the whole plate.
    embedding_size : int
        Width ``m`` of every concept's embedding.
    """

    def __init__(self, sizes: List[int], embedding_size: int):
        super().__init__()
        self.sizes = [int(s) for s in sizes]
        self.embeddings = nn.ModuleList(
            nn.Linear(size, embedding_size) for size in self.sizes
        )
        self.out_features = len(self.sizes) * embedding_size

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        weight = self.embeddings[0].weight
        parts = torch.split(concepts.to(weight.dtype), self.sizes, dim=-1)
        return torch.cat(
            [embed(part) for embed, part in zip(self.embeddings, parts)], dim=-1
        )


class ConditionedInput(nn.Module):
    """Join a feature vector with the embedded condition."""

    def __init__(self, embedder: ConceptEmbedding, encoder: Optional[nn.Module] = None):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder

    def forward(self, embeddings, concepts=None):
        features = embeddings if self.encoder is None else self.encoder(embeddings)
        if concepts is None:
            return features
        return torch.cat([features, self.embedder(concepts)], dim=-1)


class ConditionalVariationalAutoencoder(DirectedGraphModel):
    """Conditional VAE whose condition is the concept set.

    Generative process ``p(c) p(z | c) p(input | z, c)`` — or ``p(z)`` in place of
    ``p(z | c)`` when ``conditional_prior`` is False — trained as a VAE through a
    variational guide ``q(z | input, c)``. Intervening on a concept changes the
    decoder's input directly, with no bottleneck in between.

    Parameters
    ----------
    input_size : int or tuple
        Shape of one observation (the generated variable).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types). Every concept is a
        conditioning variable; there are no task variables.
    encoder : nn.Module
        The guide's feature extractor, mapping an observation to a vector. It
        runs after ``backbone`` and must declare ``out_features`` (an
        :class:`~torch_concepts.nn.MLP` or ``nn.Linear`` does); the embedded
        concepts are appended to its output and two linear readouts produce
        ``loc`` and ``scale``.
    decoder : nn.Module
        The generative network, mapping ``latent_size + condition_size`` values
        to ``input_size`` values. The observation is a ``Delta``, whose ``value``
        takes no activation, so this output **is** the reconstruction: a decoder
        for images in ``[0, 1]`` has to land there itself.
    latent_size : int, default 64
        Dimensionality of ``z``.
    embedding_size : int, default 16
        Width ``m`` of one concept's embedding. Every concept gets its own
        ``Linear(size_i, m)`` (:class:`ConceptEmbedding`), so the condition the
        decoder reads is ``n_concepts * m`` wide however the concepts are
        distributed — one binary and one 100-way categorical contribute ``m``
        each, not 1 and 100.
    conditional_prior : bool, default True
        Which prior over ``z``. ``True`` is the paper's ``p(z | c)``, built over
        ``prior_encoder``: ``z`` is drawn per condition, so it only has to carry
        what ``c`` does not. ``False`` is a fixed ``p(z) = N(0, I)`` — the plain
        VAE prior, the ablation that shows what conditioning buys, and the way to
        remove the collapse mode the Notes describe, since the KL then has a
        constant target. The guide is ``q(z | input, c)`` either way.
    prior_encoder : nn.Module, optional
        The conditional prior's feature extractor, mapping the embedded condition
        (``condition_size``) to a vector; two linear readouts over it produce
        ``p(z | c)``'s ``loc`` and ``scale``. Must declare ``out_features`` (an
        :class:`~torch_concepts.nn.MLP` or ``nn.Linear`` does). Defaults to
        ``MLP(condition_size, latent_size)``. The hidden layer is the point: read
        linearly, ``loc`` would be a *sum* of per-concept contributions, so the
        prior could not tell "digit 7 in red" from digit-7 plus red. Unused when
        ``conditional_prior`` is False.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration. Defaults to
        :class:`~torch_concepts.nn.VariationalInference`, with the guide on
        ``z`` injected into ``inference_kwargs['latents']``.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default None
        Per-level plate preference (see :class:`BaseModel`). ``None``/``True``
        group homogeneous concepts into the minimum number of plates; ``False``
        gives one variable per concept, which is what per-concept interventions
        (and the steerability metric) address.
    **kwargs
        Forwarded to :class:`BaseModel`.

    Attributes
    ----------
    condition_size : int
        Width of the condition the decoder reads,
        ``n_concepts * embedding_size``. Known from the annotations alone, so the
        decoder can be sized before construction.
    condition_embedding : ConceptEmbedding
        The per-concept embedding layers, shared by the decoder, the guide and the
        conditional prior.

    Notes
    -----
    The concept loss trains the **marginal** ``p(c)``, not a concept predictor:
    this model reads its concepts and never infers them from the observation.
    Concept accuracy is therefore at the majority-class rate by construction, and
    is not a number to compare against a CBM/CBGM's. What *is* comparable is the
    steerability of the generations and their FID.

    With ``conditional_prior`` the prior is learned, so the KL has no fixed target:
    ``KL(q(z | x, c) ‖ p(z | c))`` can be driven to zero by ``p`` drifting toward
    ``q`` rather than by ``q`` becoming informative. Because ``p`` sees only ``c``, the zero-KL solution
    is ``q`` ignoring ``input`` altogether — ``z`` then carries nothing and the
    model degenerates into a ``c → input`` map that reconstructs the conditional
    mean. Watch the KL term; ``KLDivergenceLoss(latents=['z'], free_bits=...)``
    puts a floor under each latent dimension if it collapses, and
    ``conditional_prior=False`` removes the mode outright.

    Any registered distribution family works for a concept, via
    ``variable_distributions``: the plain discrete families (the defaults), their
    relaxed and straight-through variants — the latter making a *sampled* concept
    an exact bit / one-hot row rather than a soft Concrete draw — ``Normal``, and
    ``MultivariateNormal`` for a vector-valued continuous concept (which needs
    ``plate=False``, since a Cholesky factor is not a per-element parameter).

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.annotations import Annotations
    >>> from torch_concepts.nn import ConditionalVariationalAutoencoder, MLP
    >>>
    >>> ann = Annotations(labels=['digit', 'color'], cardinalities=[10, 2],
    ...                   types=['categorical', 'categorical'])
    >>> condition = len(ann.labels) * 8  # n_concepts * embedding_size
    >>> model = ConditionalVariationalAutoencoder(
    ...     input_size=784, annotations=ann,
    ...     encoder=MLP(784, 128, 32),
    ...     # The decoder's output is the reconstruction, unactivated.
    ...     decoder=MLP(32 + condition, 128, 784),
    ...     latent_size=32, embedding_size=8,
    ... )  # doctest: +SKIP
    >>> # Concepts are evidence, so the query supplies them.
    >>> c = torch.tensor([[3, 1]])
    >>> out = model(query=model.default_query(c), input=torch.rand(1, 784))  # doctest: +SKIP

    See Also
    --------
    torch_concepts.nn.ConceptBottleneckGenerativeModel : the model this baselines
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    # The concepts are conditioning values, and a value the decoder can read is a
    # probability (or a one-hot row), not a logit — the same convention CBGM uses,
    # which also keeps the two models' loss configs interchangeable.
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
        conditional_prior: bool = True,
        prior_encoder: nn.Module = None,
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
        self.conditional_prior = bool(conditional_prior)
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.decoder = decoder if decoder is not None else nn.Identity()

        self.condition_size = len(self.concept_names) * embedding_size
        # p(z | c)'s trunk, over the *embedded* condition. The default is an MLP
        # rather than a bare readout because a linear map off the embedding makes
        # `loc` additive across concepts — the prior could then not tell "digit 7
        # in red" from digit-7 plus red. Sized from what the model already knows.
        self.prior_encoder = None
        if self.conditional_prior:
            self.prior_encoder = (
                prior_encoder if prior_encoder is not None
                else MLP(self.condition_size, self.latent_size)
            )

        self.pgm = self._build_model()

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

        The condition's components are modelled as mutually independent: nothing
        in a CVAE relates one concept to another, and the marginal ``p(c)`` that
        stands in for their joint is a product of per-concept factors.
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

        :class:`~torch_concepts.nn.VariationalInference` requires all variables
        in the query — observed ones with values, latents absent or ``None`` —
        and the generative loss terms need the ones the base learner's
        concept-only query would leave out (``input``, for the reconstruction).
        """
        return {
            **{name: None for name in self.pgm.variables},
            **self.fully_observed_query(c),
        }

    def default_extra(self, evidence, query=None):
        """Publish the evidence so :class:`~torch_concepts.nn.ReconstructionLoss`
        can score the observed variable (e.g. ``input``) against it."""
        return {"evidence": evidence}

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------
    def _concept_variables(self) -> List:
        """The condition's variables, in the order they are concatenated."""
        return [v for v in self.pgm.variables.values() if v.variable_type == "concept"]

    @staticmethod
    def _prior_heads(variable) -> dict:
        """``first``/``second`` :class:`LearnablePrior` heads for a marginal ``p(v)``.

        Sized from the variable's own ``param_sizes`` rather than from its
        ``size``, because the two differ: a ``MultivariateNormal``'s
        ``scale_tril`` needs the ``size * (size + 1) // 2`` free entries of a
        Cholesky factor, not ``size``. Everything else here is one scalar per
        event element, so this reduces to ``size`` for the discrete families, a
        ``Delta``'s ``value`` and a ``Normal``'s ``loc``/``scale``.
        """
        sizes = variable.param_sizes
        if "loc" not in sizes:
            # Discrete (probs/logits) or Delta (value) — a single head, and every
            # candidate parameter has the same width.
            return {"first": LearnablePrior(variable.size), "second": None}
        scale_param = next(param for param in sizes if param != "loc")
        return {
            "first": LearnablePrior(sizes["loc"]),
            "second": LearnablePrior(sizes[scale_param]),
        }

    @staticmethod
    def _readout_width(width, what: str, culprit: str) -> int:
        """Validate a trunk's declared output width, or say which module lacks it.

        A CPD's ``loc``/``scale`` heads are ``Linear(width, size)``, so a trunk
        that does not advertise ``out_features`` cannot be built behind. Raising
        here names the module to fix rather than failing later on a shape.
        """
        if width is None:
            raise ValueError(
                f"ConditionalVariationalAutoencoder: cannot size {what} — "
                f"neither {culprit} `out_features`. Set that attribute, or pass a "
                "module that declares it (e.g. MLP, nn.Linear)."
            )
        return int(width)

    def _build_guide(self) -> ParametricCPD:
        """The variational posterior ``q(z | input, c)``, a Normal CPD on ``z``.

        The feature extractor plus the embedded concepts is the CPD's **trunk**,
        not part of either parameter's head: ``loc`` and ``scale`` are two small
        linear readouts of the same features, so the backbone runs once per step.
        Sharing is safe here because both heads are independently learnable
        ``Linear`` layers — put the *scoring* layer in a trunk instead and the
        scale would collapse to a fixed function of the location.

        The embedding layers are :attr:`condition_embedding` itself, not a copy:
        the guide and the decoder read ``c`` through one learned representation.
        """
        z = self.pgm.variables["z"]
        observed = self.pgm.variables["input"]

        # Width of the trunk's output: the encoder's if it declares one (MLP,
        # nn.Linear), else the backbone's — `encoder` defaults to nn.Identity.
        width = self._readout_width(
            getattr(self.encoder, "out_features", None)
            or getattr(self.backbone, "out_features", None),
            "the guide's readout",
            "`encoder` nor `backbone` declares",
        )
        conditioning = self._concept_variables()
        width = int(width) + self.condition_size
        return ParametricCPD(
            variable=z,
            parents=[observed, *conditioning],
            trunk=ConditionedInput(
                self.condition_embedding,
                encoder=nn.Sequential(self.backbone, self.encoder),
            ),
            parametrization=self._flexible_parametrization(
                variable=z,
                first=nn.Linear(width, z.size),
                second=nn.Linear(width, z.size),
            ),
        )

    def _build_model(self) -> BayesianNetwork:
        """Assemble the CVAE Bayesian network.

        ``concepts → z → input``: one learnable marginal per concept (group), a
        prior over ``z`` (conditional on the embedded condition, or a fixed
        ``N(0, I)`` when ``conditional_prior`` is False), and a decoder
        reading ``z`` concatenated with the *embedded* concepts. The decoder's input is
        therefore ``latent_size + condition_size`` wide, ``z`` first, then one
        ``embedding_size``-wide block per concept in annotation order.
        """
        # --- variables ---
        observed = EmbeddingVariable("input", distribution=Delta, shape=self.input_size)
        latent = EmbeddingVariable("z", distribution=Normal, size=self.latent_size)
        concepts = self.build_concept_variables(self.concept_names, plate_name="concepts")

        # One embedding per concept, sized to that concept's own value: 1 column
        # for a binary or continuous one, `cardinality` for a categorical one.
        # Widths are taken per MEMBER so a plate contributes one entry per member,
        # matching how the parent values arrive concatenated.
        self.condition_embedding = ConceptEmbedding(
            sizes=[cvar.member_size for cvar in concepts for _ in cvar.members],
            embedding_size=self.embedding_size,
        )

        # --- factors ---
        # p(z | c): the paper's conditional prior. The condition is read through
        # `condition_embedding` — the same instance the guide and the decoder use,
        # so `c` has one learned representation across the whole model — and the
        # two heads are independent readouts over that shared trunk, exactly as in
        # the guide. `prior_encoder` supplies the nonlinearity between them.
        if self.conditional_prior:
            prior_width = self._readout_width(
                getattr(self.prior_encoder, "out_features", None),
                "the conditional prior's readout",
                "`prior_encoder` declares",
            )
            latent_cpd = ParametricCPD(
                variable=latent,
                parents=[*concepts],
                trunk=pyc.nn.Sequential(self.condition_embedding, self.prior_encoder),
                parametrization=self._flexible_parametrization(
                    variable=latent,
                    first=nn.Linear(prior_width, latent.size),
                    second=nn.Linear(prior_width, latent.size),
                ),
            )
        else:
            # p(z) = N(0, I): parent-less and fixed, so the KL has a constant
            # target and `z` carries everything `c` does not by construction.
            latent_cpd = ParametricCPD(
                latent,
                parents=[],
                parametrization={
                    "loc": FixedPrior(torch.zeros(self.latent_size)),
                    "scale": FixedPrior(torch.ones(self.latent_size)),
                },
            )
        # p(c): one parent-less parameter per concept (per member, for a plate),
        # activated into its own domain — a sigmoid for a Bernoulli, a per-member
        # softmax for a categorical, softplus on a continuous concept's scale.
        # Unused while the concepts are observed; fitted by the concept loss so
        # that an *unconditional* draw produces a plausible condition.
        concept_cpds = [
            ParametricCPD(
                variable=cvar,
                parents=[],
                parametrization=self._flexible_parametrization(
                    variable=cvar,
                    **self._prior_heads(cvar),
                ),
            )
            for cvar in concepts
        ]

        # The parents reach the head as `z` (an embedding) and the raw concepts,
        # split by type — so the head embeds the concepts and concatenates,
        # producing the [z | embedded c] layout `condition_size` documents.
        def decoder_head(decoder: nn.Module) -> nn.Module:
            return pyc.nn.Sequential(
                ConditionedInput(self.condition_embedding), decoder
            )

        decoder_cpd = ParametricCPD(
            variable=observed,
            parents=[latent, *concepts],
            # A Delta has a single `value` parameter, so `second` is not needed:
            # the decoder's output IS the reconstruction.
            parametrization=self._flexible_parametrization(
                variable=observed,
                first=decoder_head(self.decoder),
            ),
        )

        return BayesianNetwork(
            variables=[latent, *concepts, observed],
            factors=[latent_cpd, *concept_cpds, decoder_cpd],
        )

