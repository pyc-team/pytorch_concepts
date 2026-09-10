"""Conditional Variational Autoencoder (CVAE), conditioned on a set of concepts.

A VAE conditioned on an observed ``c``: here the concepts are the condition, and
the joint factorises as ``p(c) p(z | c) p(x | z, c)``. The guide is always
``q(z | x, c)``, the paper's recognition network.

The prior defaults to the paper's conditional ``p(z | c)``, read through the same
concept embedding the guide and the decoder use, so ``c`` has one learned
representation across the model; ``conditional_prior=False`` swaps in a fixed
``N(0, I)``, under which the KL target is constant.

Deviation from the paper: no GSNN / hybrid objective (Sec. 4.2). That addresses
the train/test mismatch of a *predictive* CVAE whose condition is an image; here
the condition is a concept vector supplied identically at training and at
generation, so the mismatch does not arise — and for the same reason the guide
needs no ``prior_net`` fallback.

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
    """One learnable ``Linear(size_i, m)`` per concept.

    Concepts arrive concatenated on the last axis; the result is
    ``[..., n_concepts * m]``.

    Parameters
    ----------
    sizes : list of int
        Width of each concept's value in concatenation order — ``1`` for binary
        or continuous, ``cardinality`` for categorical. Per *member*, so a plate
        contributes one entry per member.
    embedding_size : int
        Width ``m`` of every embedding.
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


class ConditionalVAE(DirectedGraphModel):
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
        The guide's trunk, mapping an observation (``input_size``) to the
        features its ``loc``/``scale`` readouts share — the embedded concepts are
        appended to its output. Any feature extractor goes here — this model
        takes no ``backbone``, since the guide reads the raw observation.
    decoder : nn.Module
        Maps ``latent_size + condition_size`` to ``input_size``. The observation
        is a ``Delta``, so this output **is** the reconstruction — nothing is
        composed on top and it must not be squashed here.
    latent_size : int, default 64
        Dimensionality of ``z``.
    embedding_size : int, default 16
        Width ``m`` of one concept's embedding (:class:`ConceptEmbedding`). The
        condition is ``n_concepts * m`` wide however the concepts are
        distributed: a binary and a 100-way concept contribute ``m`` each.
    conditional_prior : bool, default True
        ``True`` is the paper's ``p(z | c)`` over ``prior_encoder``, so ``z``
        carries only what ``c`` does not. ``False`` is a fixed ``N(0, I)``: the
        ablation, and the way to remove the collapse mode in the Notes. The guide
        is ``q(z | input, c)`` either way.
    prior_encoder : nn.Module, optional
        Pre-built trunk for ``p(z | c)``, mapping the embedded condition
        (``condition_size``) to a vector two linear readouts turn into
        ``loc``/``scale``. Must declare ``out_features``. Defaults to an ``MLP``
        configured by ``prior_net_kwargs``. Unused when ``conditional_prior`` is
        False.
    prior_net_kwargs : dict, optional
        Arguments for that default ``MLP`` — ``hidden_size`` (defaults to
        ``latent_size``), ``n_layers``, ``activation``, ``dropout``. Its hidden
        layer is the point: read linearly, ``loc`` would be a *sum* of
        per-concept contributions, so the prior could not tell "digit 7 in red"
        from digit-7 plus red.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration. Defaults to
        :class:`~torch_concepts.nn.VariationalInference`, with the guide on
        ``z`` injected into ``inference_kwargs['latents']``.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default False.
    **kwargs
        Forwarded to :class:`BaseModel`.

    Attributes
    ----------
    condition_size : int
        ``n_concepts * embedding_size``. Known from the annotations alone, so the
        decoder can be sized before construction.
    condition_embedding : ConceptEmbedding
        The per-concept embeddings, shared by decoder, guide and prior.

    Notes
    -----
    The concept loss fits the **marginal** ``p(c)``, not a concept predictor: this
    model reads its concepts and never infers them. Concept accuracy is therefore
    at the majority-class rate by construction and is not comparable to a
    CBM/CBGM's; steerability and FID are.

    With ``conditional_prior`` the KL has no fixed target, so it can be driven to
    zero by ``p`` drifting toward ``q`` instead of ``q`` becoming informative.
    Since ``p`` sees only ``c``, that solution is ``q`` ignoring ``input``: ``z``
    carries nothing and the model degenerates into a ``c → input`` map.
    ``KLDivergenceLoss(free_bits=...)`` floors each latent dimension against it;
    ``conditional_prior=False`` removes the mode outright.

    Any registered family works for a concept via ``variable_distributions``: the
    plain discrete ones (the defaults), their relaxed and straight-through
    variants, ``Normal``, and ``MultivariateNormal`` (which needs ``plate=False``,
    a Cholesky factor not being a per-element parameter).

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.annotations import Annotations
    >>> from torch_concepts.nn import ConditionalVAE, MLP
    >>>
    >>> ann = Annotations(labels=['digit', 'color'], cardinalities=[10, 2],
    ...                   types=['categorical', 'categorical'])
    >>> condition = len(ann.labels) * 8  # n_concepts * embedding_size
    >>> model = ConditionalVAE(
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
    torch_concepts.nn.ConceptBottleneckVAE : the model this baselines
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    param_for_discrete_var = "logits"

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
        prior_net_kwargs: Optional[dict] = None,
        inference: Optional[BaseInference] = VariationalInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        plate: Optional[bool] = False,
        **kwargs,
    ):
        if kwargs.pop("backbone", None) is not None:
            raise TypeError(
                f"{type(self).__name__} does not accept a `backbone` parameter. "
                "The guide reads the raw observation directly, so any feature "
                "extractor belongs in `encoder`."
            )
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
        # p(z | c)'s trunk, over the *embedded* condition. An MLP rather than a
        # bare readout: read linearly, `loc` would be additive across concepts.
        self.prior_encoder = None
        if self.conditional_prior:
            self.prior_encoder = prior_encoder or MLP(
                self.condition_size,
                **{"hidden_size": self.latent_size, **(prior_net_kwargs or {})},
            )

        self.pgm = self._build_model()

        guide = {"latents": {"z": self._build_guide()}}
        self.setup_inference(
            inference,
            {**guide, **(inference_kwargs or {})},
            train_inference,
            {**guide, **(train_inference_kwargs or {})},
        )

    def __repr__(self):
        # The base reports `backbone`, which this model has none of: the guide's
        # trunk is `encoder`, and the condition's width is what sizes `decoder`.
        fields = (
            f"input_size={self.input_size}, "
            f"latent_size={self.latent_size}, "
            f"n_concepts={len(self.concept_names)}, "
            f"embedding_size={self.embedding_size}, "
            f"conditional_prior={self.conditional_prior}, "
            f"encoder={self.encoder.__class__.__name__}, "
            f"decoder={self.decoder.__class__.__name__}"
        )
        if self.plate:
            fields += f", plate={self.plate}"
        return f"{self.__class__.__name__}({fields})"

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
    def default_query(self, c, step='train'):
        """Query **every** variable, always observing the concepts.

        Evaluation does **not** withhold them the way the base query
        (:meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.default_query`)
        and the CBGM do: here the concepts are the *condition*, an input rather
        than something the model infers, and the guide ``q(z | input, c)`` reads
        them as parents — withholding them leaves that CPD without a value. For
        the same reason there is no RandInt knob on this model.

        :class:`~torch_concepts.nn.VariationalInference` requires all variables
        in the query — observed ones with values, latents absent or ``None`` —
        and the generative loss terms need the ones the base concept-only
        query would leave out (``input``, for the reconstruction).
        """
        return {
            **{name: None for name in self.pgm.variables},
            **self.fully_observed_query(c),
        }

    def default_extra(self, evidence, query=None):
        """Publish the evidence so :class:`~torch_concepts.nn.MSEReconstructionLoss`
        can score the observed variable (e.g. ``input``) against it."""
        return {"evidence": evidence}

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------
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

        # Width of the trunk's output: declared when the encoder exposes it
        # (MLP, nn.Linear), else measured with a dry run — the trick
        # `backbone.py` uses for torchvision models.
        width = getattr(self.encoder, "out_features", None)
        if width is None:
            with torch.no_grad():
                width = self.encoder(torch.zeros(1, *observed.shape)).shape[-1]
        # The condition's variables, in the order they are concatenated.
        conditioning = [v for v in self.pgm.variables.values()
                        if v.variable_type == "concept"]
        width = int(width) + self.condition_size
        return ParametricCPD(
            variable=z,
            parents=[observed, *conditioning],
            trunk=ConditionedInput(self.condition_embedding, encoder=self.encoder),
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
            # Same sizing as the guide's trunk: declared, else measured.
            prior_width = getattr(self.prior_encoder, "out_features", None)
            if prior_width is None:
                with torch.no_grad():
                    prior_width = self.prior_encoder(
                        torch.zeros(1, self.condition_size)).shape[-1]
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
        # Heads are sized from `param_sizes`, not `size`: a MultivariateNormal's
        # `scale_tril` needs the Cholesky factor's size*(size+1)//2 entries, while
        # everything else is one scalar per event element.
        def prior_heads(variable) -> dict:
            sizes = variable.param_sizes
            if "loc" not in sizes:
                # Discrete (probs/logits) or Delta (value) — a single head, and
                # every candidate parameter has the same width.
                return {"first": LearnablePrior(variable.size), "second": None}
            scale_param = next(param for param in sizes if param != "loc")
            return {
                "first": LearnablePrior(sizes["loc"]),
                "second": LearnablePrior(sizes[scale_param]),
            }

        concept_cpds = [
            ParametricCPD(
                variable=cvar,
                parents=[],
                parametrization=self._flexible_parametrization(
                    variable=cvar,
                    **prior_heads(cvar),
                ),
            )
            for cvar in concepts
        ]

        decoder_cpd = ParametricCPD(
            variable=observed,
            parents=[latent, *concepts],
            # A Delta has a single `value` parameter, so `second` is not needed:
            # the decoder's output IS the reconstruction.
            parametrization=self._flexible_parametrization(
                variable=observed,
                # ConditionedInput embeds the concepts and concatenates, giving
                # the [z | embedded c] layout `condition_size` documents.
                first=pyc.nn.Sequential(
                    ConditionedInput(self.condition_embedding), self.decoder
                ),
            ),
        )

        return BayesianNetwork(
            variables=[latent, *concepts, observed],
            factors=[latent_cpd, *concept_cpds, decoder_cpd],
        )

