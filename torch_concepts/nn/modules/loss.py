"""Loss functions for concept-based models."""
import inspect
import warnings
from typing import List, Optional, Union, Dict, Sequence, Mapping
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn

from .utils import TYPES, by_type, check_collection
from .outputs import CONTINUOUS_QUANTITIES, ModelOutput, supervised_subset
from ..functional import concept_orthogonality
from ...concept_graph import ConceptGraph
from ...distributions import Delta


def _get_forward_signature(module: nn.Module):
    """Introspect forward() to get accepted parameter names and whether it has **kwargs.
    
    Returns:
        Tuple[set, bool]: (set of parameter names, has_var_keyword)
    """
    params = inspect.signature(module.forward).parameters
    names = set()
    has_var_keyword = False
    for name, param in params.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        else:
            names.add(name)
    return names, has_var_keyword


def _filter_kwargs(kwargs: dict, accepted: set, has_var_keyword: bool) -> dict:
    """The subset of ``kwargs`` a term's ``forward`` declares — the same contract
    as ``torchmetrics.Metric._filter_kwargs``: declare what you want and get
    exactly that, or take ``**kwargs`` and get everything. Signatures are read
    once at construction, so no introspection happens here.
    """
    if has_var_keyword:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _unique_names(terms) -> List[str]:
    """Class name per term, suffixed where one class appears twice — colliding
    keys would drop a term from :meth:`CompositeLoss.breakdown` silently."""
    names = [type(t).__name__ for t in terms]
    unique = []
    for i, name in enumerate(names):
        if names.count(name) > 1:
            unique.append(f"{name}_{names[:i].count(name)}")
        else:
            unique.append(name)
    return unique


def _plain(tensor):
    """``tensor`` without its annotation. Terms get plain tensors: the annotation
    has done its job (aligning to the target by name) and every ``torch.*`` call
    inside a term would otherwise re-enter ``__torch_function__`` to strip it
    again — the largest cost in this module. ``ConceptMetrics`` does the same.
    """
    return getattr(tensor, "tensor", tensor)


def _normalize_loss_terms(terms, weights):
    """Normalize loss terms and weights to consistent list form.
    
    Args:
        terms: A single nn.Module, a list of nn.Module, or None.
        weights: A list of floats, or None.
        
    Returns:
        Tuple of (list_of_modules, list_of_weights), or (None, None) if terms is None.
    """
    if terms is None:
        return None, None
    if isinstance(terms, nn.Module):
        terms = [terms]
    if not isinstance(terms, (list, tuple)):
        raise TypeError(
            f"Loss terms must be an nn.Module or a list of nn.Module, got {type(terms)}"
        )
    if weights is None:
        weights = [1.0] * len(terms)
    if len(weights) != len(terms):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of loss terms ({len(terms)})."
        )
    return list(terms), list(weights)


def subset_output(output: ModelOutput, names: List[str]) -> ModelOutput:
    """A :class:`ModelOutput` restricted to the concepts in ``names``.

    Every quantity tensor and the target are sliced by concept name via their
    annotations, so the result carries whatever quantities those concepts report
    (``logits``/``probs`` and/or ``loc``/``scale``). Used by the composite losses
    to route a shared output to their sub-losses.
    """
    sub = ModelOutput(extra=output.extra)
    if output.target is not None:
        present = [n for n in names if n in output.target.annotation.label_to_index]
        sub.target = output.target[present]
    for quantity, tensor in output.params.items():
        present = [n for n in names if n in tensor.annotation.label_to_index]
        if present:
            sub.params[quantity] = tensor[present]
    return sub


#: Quantities a *discrete* concept may be reported under, in fallback order. The
#: same rule :class:`~torch_concepts.nn.ConceptMetrics` applies, so a loss and a
#: metric on one model read the same quantity without either being told which.
DISCRETE_QUANTITIES = ("logits", "probs")


def _resolve_quantity(params, configured: Optional[str], candidates: Sequence[str]):
    """The tensor a concept type is scored on, or ``None`` if the output has none.

    A model already declares the quantity it emits (``param_for_discrete_var`` is
    ``'logits'`` on a CBM, ``'probs'`` on a CBGM), so naming it again in the loss
    is the one place the two sides can disagree. Default (``configured=None``):
    take the first candidate the output reports.
    """
    if configured is not None:
        return params.get(configured)
    for quantity in candidates:
        tensor = params.get(quantity)
        if tensor is not None:
            return tensor
    return None


#: Parameter-name signature -> distribution family, used to build a variable's
#: distribution from a :class:`ModelOutput` alone. A loss sees annotated tensors
#: keyed by variable name, never the ``Variable`` objects, so it cannot go
#: through ``build_distribution``; the reported quantities identify the family
#: unambiguously for every registered one.
_FAMILY_BY_PARAMS = {
    frozenset({"probs"}): dist.Bernoulli,
    frozenset({"logits"}): dist.Bernoulli,
    frozenset({"loc", "scale"}): dist.Normal,
    frozenset({"loc", "scale_tril"}): dist.MultivariateNormal,
    frozenset({"value"}): Delta,
}


def _family_from_params(params: Mapping[str, torch.Tensor], context: str) -> type:
    """Infer a variable's distribution family from the quantities it reports."""
    family = _FAMILY_BY_PARAMS.get(frozenset(params))
    if family is None:
        raise ValueError(
            f"{context}: cannot infer a distribution family from the reported "
            f"parameters {sorted(params)}. Pass `distribution=` explicitly."
        )
    return family


def _variable_params(
    params: Mapping[str, torch.Tensor], name: str
) -> Dict[str, torch.Tensor]:
    """The quantities reported for one variable, as plain tensors.

    ``ModelOutput.params`` is quantity-keyed with every queried variable
    concatenated on the annotated axis; ``params[name]`` is the variable-first
    view of the same tensors (see :class:`ParamsDict`). The annotation is
    dropped here: these go straight into a ``torch.distributions`` constructor.
    """
    return {q: _plain(t) for q, t in params[name].items()}


class PyCLoss(nn.Module):
    """Base for every loss that is scored on a whole :class:`ModelOutput`.

    ``forward(output, target=None)`` rather than the usual
    ``forward(input, target)``: a PyC loss finds what it needs on the output
    itself — by concept name, by concept type, or by quantity — instead of being
    handed a pair of tensors. The learner checks for this base class to know it
    can pass the output straight through.

    Individual *terms* (``BCEWithLogitsLoss``, :class:`L1LogitRegularizer`, any
    custom module) are **not** subclasses: they take plain tensors, and
    :class:`ConceptLoss` is what dispatches to them.
    """


class CompositeLoss(PyCLoss):
    """Weighted sum of any number of :class:`PyCLoss` terms.

    The building block for objectives that are not a single concept term — an
    ELBO, for instance, is a reconstruction term plus a KL term plus whatever
    supervision and regularisation the model adds. Each term is handed the same
    :class:`ModelOutput`, so terms stay independent and reusable.

    Every term is called with ``target`` only if its ``forward`` accepts one, so
    terms with either signature compose freely.

    Args:
        terms (list of nn.Module): The loss terms to sum.
        weights (list of float, optional): Per-term weights. Defaults to all
            ``1.0``. A plain mutable list, so a schedule such as
            :class:`~torch_concepts.nn.LossWeightWarmup` can rewrite one entry
            mid-training.
        names (list of str, optional): Term names for the ``repr`` and for
            :meth:`breakdown`. Defaults to each term's class name.

    Terms here see the **whole** output, so this is the place for a penalty on
    something shared (a latent, an embedding, the evidence). A penalty on one
    concept type's predictions belongs in that type's list on
    :class:`ConceptLoss` instead.

    Example:
        Concept supervision plus one output-level penalty:

        >>> import torch
        >>> from torch_concepts.nn import CompositeLoss, ConceptLoss, OrthogonalityLoss
        >>> loss_fn = CompositeLoss(
        ...     terms=[ConceptLoss(binary=torch.nn.BCEWithLogitsLoss()),
        ...            OrthogonalityLoss(variables=['mixing', 'unknown'])],
        ...     weights=[1.0, 0.5],
        ... )
        >>> loss_fn
        CompositeLoss(ConceptLoss + 0.5*OrthogonalityLoss)

        An ELBO — independent terms, named so the ``repr`` and
        :meth:`breakdown` read the way you think about them:

        >>> from torch_concepts.nn import (KLDivergenceLoss, NLLProbLoss,
        ...                                ReconstructionLoss)
        >>> loss_fn = CompositeLoss(
        ...     terms=[ReconstructionLoss('input'),
        ...            KLDivergenceLoss(['z']),
        ...            ConceptLoss(categorical=NLLProbLoss())],
        ...     weights=[1.0, 1.0, 5.0],
        ...     names=['recon', 'kl', 'concepts'],
        ... )
        >>> loss_fn
        CompositeLoss(recon + kl + 5.0*concepts)
    """

    def __init__(
        self,
        terms: Union[nn.Module, List[nn.Module]],
        weights: Optional[List[float]] = None,
        names: Optional[List[str]] = None,
    ):
        super().__init__()
        terms, weights = _normalize_loss_terms(terms, weights)
        if not terms:
            raise ValueError("CompositeLoss: `terms` must not be empty.")
        if names is not None and len(names) != len(terms):
            raise ValueError(
                f"Number of names ({len(names)}) must match "
                f"number of loss terms ({len(terms)})."
            )
        self.terms = nn.ModuleList(terms)
        self.weights = list(weights)
        self.term_names = list(names) if names is not None else _unique_names(terms)
        self._takes_target = [
            "target" in sig or has_var_kw
            for sig, has_var_kw in (_get_forward_signature(t) for t in terms)
        ]

    def __repr__(self) -> str:
        parts = [
            f"{w}*{name}" if w != 1.0 else name
            for name, w in zip(self.term_names, self.weights)
        ]
        return f"{self.__class__.__name__}({' + '.join(parts)})"

    def breakdown(self, output: ModelOutput, target=None) -> Dict[str, torch.Tensor]:
        """Each term's **weighted** contribution, keyed by term name.

        :meth:`forward` returns exactly ``sum(breakdown(...).values())``, so the
        two are interchangeable: call the loss for the scalar, or this for the
        same computation with the addends kept apart. Use it when a total is not
        enough — an ELBO whose KL has collapsed still looks fine summed.
        """
        return {
            name: weight * (term(output, target) if takes_target else term(output))
            for name, term, weight, takes_target in zip(
                self.term_names, self.terms, self.weights, self._takes_target
            )
        }

    def forward(self, output: ModelOutput, target=None) -> torch.Tensor:
        return sum(self.breakdown(output, target).values())


class ReconstructionLoss(PyCLoss):
    """Negative log-likelihood of an **observed** variable under its own CPD.

    The generative half of an ELBO: the model predicts the parameters of
    them. Family-agnostic — a ``Bernoulli`` observation gives the usual
    binary cross-entropy, a ``Normal`` one a Gaussian NLL, and a ``Delta`` — a
    deterministic decoder, which is what the generative models use — the squared
    error. It works for any observed variable of any registered family, not just
    an image.

    The observed value is read from ``output.extra['evidence']``, which a
    learner publishes by overriding
    :meth:`~torch_concepts.nn.modules.high.base.learner.BaseLearner.default_extra`
    to return ``{'evidence': evidence}`` — the base learner merges nothing by
    default, since a purely discriminative model has no observation to score.

    Args:
        variable (str): Name of the observed variable to score. Default
            ``'input'``.
        distribution (type, optional): Distribution family. Inferred from the
            reported parameter names when omitted.
        reduction (str): ``'mean'`` (default) averages the per-sample NLL over
            the batch; ``'sum'`` sums it.

    Example:
        The usual case — the family is read off the reported parameters:

        >>> from torch_concepts.nn import ReconstructionLoss
        >>> loss_fn = ReconstructionLoss(variable='input')
        >>> loss_fn
        ReconstructionLoss(variable='input')

        Inside an ELBO, with the family pinned and the batch summed rather than
        averaged (which changes the reconstruction-to-KL ratio):

        >>> from torch.distributions import Normal
        >>> from torch_concepts.nn import CompositeLoss, KLDivergenceLoss
        >>> loss_fn = CompositeLoss(
        ...     terms=[ReconstructionLoss('input', distribution=Normal, reduction='sum'),
        ...            KLDivergenceLoss(['z'])],
        ...     weights=[1.0, 1.0],
        ... )
        >>> loss_fn
        CompositeLoss(ReconstructionLoss + KLDivergenceLoss)
    """

    def __init__(
        self,
        variable: str = "input",
        distribution: Optional[type] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.variable = variable
        self.distribution = distribution
        self.reduction = reduction

    def extra_repr(self) -> str:
        return f"variable={self.variable!r}"

    def forward(self, output: ModelOutput, target=None) -> torch.Tensor:
        extra = output.extra or {}
        evidence = extra.get("evidence") or {}
        observed = evidence.get(self.variable)
        if observed is None:
            raise ValueError(
                f"ReconstructionLoss: no observed value for {self.variable!r}. "
                "It must be supplied as evidence — the learner forwards its "
                "evidence dict to the loss under `output.extra['evidence']`."
            )

        params = _variable_params(output.params, self.variable)
        family = self.distribution or _family_from_params(
            params, f"ReconstructionLoss({self.variable!r})"
        )
        # Parameters are flat ``(*leading, size)``; the observed value may still
        # carry its event shape (an image stays ``(B, C, H, W)``), so reshape it
        # to the parameters' layout before scoring.
        reference = next(iter(params.values()))
        flat = observed.reshape(reference.shape).to(reference.dtype)
        if family is Delta:
            # A point mass has no spread to score: its NLL is degenerate, and
            # this Delta's ``log_prob`` is a gradient-free constant 0. The
            # squared error is the sigma=1 Gaussian NLL minus its constant —
            # identical gradients — so a model moved from ``Normal`` to
            # ``Delta`` keeps whatever reconstruction weight it was tuned with.
            nll = 0.5 * (params["value"] - flat).pow(2).sum(-1)
        else:
            # ``validate_args=False``: a Bernoulli likelihood over grey levels in
            # [0, 1] is the standard VAE reconstruction term (it is exactly
            # ``binary_cross_entropy``), but those values are outside Bernoulli's
            # declared {0, 1} support and strict validation would reject them.
            d = dist.Independent(family(**params, validate_args=False), 1)
            nll = -d.log_prob(flat)
        return nll.sum() if self.reduction == "sum" else nll.mean()


class KLDivergenceLoss(PyCLoss):
    """``KL(q ‖ p)`` between a variational guide and the model, per latent.

    The regularising half of an ELBO. The guide's parameters come from
    ``output.guide_params`` and the model's prior from ``output.params``, both
    keyed by variable name, so this works for any model with a registered guide
    — nothing here is specific to a particular architecture.

    Args:
        latents (list of str): Latent variable names to score. Default
            ``['z']``.
        distribution (type, optional): Distribution family shared by guide and
            prior. Inferred from the reported parameter names when omitted.
        free_bits (float): Per-dimension floor, in nats, below which a
            dimension's KL stops being penalised. Default ``0.0`` (off, the
            plain ELBO term).

            Raise it when a *generative* model reconstructs well but samples
            badly. Nothing in the ELBO stops a dimension from collapsing to the
            prior and carrying no information, and once enough of them have, the
            aggregate posterior occupies a thin region of a latent space the
            prior spreads mass over uniformly — so a draw from ``p(z)`` lands
            where the decoder has never been trained. A floor of ~0.5 nats keeps
            every dimension in use and the two distributions closer in shape.

    Example:
        The plain ELBO term, one latent:

        >>> from torch_concepts.nn import KLDivergenceLoss
        >>> loss_fn = KLDivergenceLoss(latents=['z'])
        >>> loss_fn
        KLDivergenceLoss(latents=['z'], free_bits=0.0)

        Several latents, with a floor that stops any single dimension from
        collapsing onto the prior:

        >>> loss_fn = KLDivergenceLoss(latents=['z', 'style'], free_bits=0.5)
        >>> loss_fn
        KLDivergenceLoss(latents=['z', 'style'], free_bits=0.5)
    """

    def __init__(
        self,
        latents: Sequence[str] = ("z",),
        distribution: Optional[type] = None,
        free_bits: float = 0.0,
    ):
        super().__init__()
        self.latents = list(latents)
        self.distribution = distribution
        self.free_bits = float(free_bits)

    def extra_repr(self) -> str:
        return f"latents={self.latents}, free_bits={self.free_bits}"

    def forward(self, output: ModelOutput, target=None) -> torch.Tensor:
        total = None
        for name in self.latents:
            q_params = _variable_params(output.guide_params, name)
            p_params = _variable_params(output.params, name)
            family = self.distribution or _family_from_params(
                q_params, f"KLDivergenceLoss({name!r})"
            )
            kl = dist.kl_divergence(family(**q_params), family(**p_params))
            if self.free_bits:
                # Clamp each dimension's BATCH-AVERAGED KL, then sum: a
                # dimension already below the floor contributes a constant and
                # its gradient vanishes, while the rest are still pulled down.
                # Averaging first (rather than clamping each sample) is the
                # usual formulation and is the gentler one — it lets an
                # individual sample sit under the floor as long as the dimension
                # is carrying information overall. Clamping the summed total
                # instead would simply switch the whole term off.
                value = kl.reshape(-1, kl.shape[-1]).mean(0).clamp_min(self.free_bits).sum()
            else:
                # One scalar per latent dimension: sum the event, average the batch.
                value = kl.sum(-1).mean()
            total = value if total is None else total + value
        return total


class OrthogonalityLoss(PyCLoss):
    """Push a concept bottleneck's contexts away from the unsupervised one.

    A thin :class:`PyCLoss` wrapper around
    :func:`~torch_concepts.nn.functional.concept_orthogonality`, which
    penalises the absolute cosine similarity between each supervised concept
    context and the unsupervised context that follows them. Applies to any model
    whose bottleneck ends in an unsupervised slot.

    Args:
        variables (list of str): Deterministic (``Delta``) variables whose
            values are concatenated, in order, into the bottleneck, the
            **unsupervised context last**. Default ``['mixing', 'unknown']``. If
            that last variable is not in the bottleneck the penalty has nothing
            to push against, so it warns once and contributes ``0`` rather than
            failing — see
            :class:`~torch_concepts.nn.ConceptBottleneckGenerativeModel`'s
            ``use_unknown``.
        n_concepts (int, optional): Number of supervised concepts. Inferred from
            the target's annotation when omitted.

    Example:
        The bottleneck's variables, unsupervised context last:

        >>> from torch_concepts.nn import OrthogonalityLoss
        >>> loss_fn = OrthogonalityLoss(variables=['mixing', 'unknown'])
        >>> loss_fn
        OrthogonalityLoss(variables=['mixing', 'unknown'], n_concepts=None)

        With the concept count pinned — needed when the loss is called without a
        target to infer it from — and weighted into an objective:

        >>> from torch_concepts.nn import CompositeLoss, ConceptLoss, NLLProbLoss
        >>> loss_fn = CompositeLoss(
        ...     terms=[ConceptLoss(categorical=NLLProbLoss()),
        ...            OrthogonalityLoss(variables=['mixing', 'unknown'], n_concepts=2)],
        ...     weights=[5.0, 1.0],
        ...     names=['concepts', 'orthogonality'],
        ... )
        >>> loss_fn
        CompositeLoss(5.0*concepts + orthogonality)
    """

    def __init__(
        self,
        variables: Sequence[str] = ("mixing", "unknown"),
        n_concepts: Optional[int] = None,
    ):
        super().__init__()
        self.variables = list(variables)
        self.n_concepts = n_concepts
        self._warned_vacuous = False

    def extra_repr(self) -> str:
        return f"variables={self.variables}, n_concepts={self.n_concepts}"

    def forward(self, output: ModelOutput, target=None) -> torch.Tensor:
        value = output.params.get("value")
        if value is None:
            raise ValueError(
                "OrthogonalityLoss: the output reports no 'value' quantity; the "
                f"bottleneck variables {self.variables} must be Delta variables "
                "included in the query."
            )

        # The penalty is a similarity *against* the unsupervised context, which
        # the ``variables`` contract puts last. A model built without one (CBGM's
        # ``use_unknown=False`` ablation) makes the term vacuous rather than wrong, so
        # it contributes zero — that is what lets the ablation be a single flag
        # instead of also needing the loss config swapped.
        unsupervised = self.variables[-1]
        annotation = value.annotation
        if (unsupervised not in annotation.label_to_index
                and unsupervised not in annotation.label_groups):
            if not self._warned_vacuous:
                warnings.warn(
                    f"OrthogonalityLoss: {unsupervised!r} is not in the bottleneck, "
                    "so the penalty is vacuous and contributes 0. This is expected "
                    "when the model was built without an unsupervised context "
                    "(e.g. ConceptBottleneckGenerativeModel(use_unknown=False)); if you "
                    "did not intend that, check the model and the `variables` "
                    "argument.",
                    stacklevel=2,
                )
                self._warned_vacuous = True
            return torch.zeros((), device=value.device, dtype=value.dtype)

        context = torch.cat([value[name] for name in self.variables], dim=-1)

        n_concepts = self.n_concepts
        if n_concepts is None:
            reference = target if target is not None else output.target
            if reference is None:
                raise ValueError(
                    "OrthogonalityLoss: `n_concepts` was not given and there is "
                    "no target to infer it from. Pass n_concepts=..."
                )
            n_concepts = len(reference.annotation.labels)
        return concept_orthogonality(context, n_concepts)


class NLLProbLoss(nn.Module):
    """Categorical negative log-likelihood for a model that reports ``probs``.

    ``CrossEntropyLoss`` expects logits and ``NLLLoss`` expects log-probabilities,
    so neither can score a head that already emits normalised probabilities.
    This takes the log first, with a floor so a zero probability does not become
    ``-inf``. Use it as the ``categorical`` term of a
    :class:`ConceptLoss` configured with ``categorical_param='probs'``.

    Args:
        eps (float): Lower bound applied before the log. Default ``1e-8``.

    Example:
        The quantity is inferred, so a model reporting ``probs`` needs only the
        matching term:

        >>> from torch_concepts.nn import ConceptLoss, NLLProbLoss
        >>> loss_fn = ConceptLoss(categorical=NLLProbLoss())
        >>> loss_fn
        ConceptLoss(categorical=NLLProbLoss)

        Stacked with a regulariser, and with a lower floor before the log:

        >>> from torch_concepts.nn import L1LogitRegularizer
        >>> loss_fn = ConceptLoss(
        ...     categorical=[NLLProbLoss(eps=1e-6), L1LogitRegularizer(scale=0.01)],
        ...     categorical_weights=[1.0, 0.1],
        ... )
        >>> loss_fn
        ConceptLoss(categorical=[NLLProbLoss + 0.1*L1LogitRegularizer])
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def extra_repr(self) -> str:
        return f"eps={self.eps}"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Padded columns arrive as -inf (see ConceptLoss._prepare_categorical);
        # clamp_min lifts them to eps, and their target is never that class.
        return F.nll_loss(input.clamp_min(self.eps).log(), target.long())


class ConceptLoss(PyCLoss):
    """
    Concept loss for concept-based models.

    Routes to the appropriate loss function based on the concept type
    (binary, categorical, continuous) read from the annotated model output.
    Each type accepts either a single loss module or a list of loss modules
    with optional per-term weights, enabling type-specific composition (e.g.
    adding a regularizer only to binary concepts).

    Args:
        binary (nn.Module or list of nn.Module, optional): Loss function(s)
            for binary concepts. A single module (e.g. ``BCEWithLogitsLoss()``)
            or a list of modules to be summed.
        categorical (nn.Module or list of nn.Module, optional): Loss function(s)
            for categorical concepts. A single module (e.g.
            ``CrossEntropyLoss()``) or a list of modules to be summed.
        continuous (nn.Module or list of nn.Module, optional): Loss function(s)
            for continuous concepts. A single module (e.g. ``MSELoss()``) or a
            list of modules to be summed.
        binary_weights (list of float, optional): Per-term weights when
            ``binary`` is a list. Defaults to ``[1.0, ...]``.
        categorical_weights (list of float, optional): Per-term weights when
            ``categorical`` is a list. Defaults to ``[1.0, ...]``.
        continuous_weights (list of float, optional): Per-term weights when
            ``continuous`` is a list. Defaults to ``[1.0, ...]``.
        binary_param (str, optional): Output quantity to read binary predictions
            from. ``None`` (default) takes the first of ``('logits', 'probs')``
            the model reports; pass a name to force one.
        categorical_param (str, optional): Same, for categorical predictions.
        continuous_param (str, optional): Same, for continuous predictions, over
            ``('loc', 'value')`` — ``loc`` for a ``Normal``, ``value`` for a
            ``Delta``.
        annotations (Annotations, optional): When given, the configured types are
            checked against the concept types that actually exist, at
            construction rather than at the first step.

    Example:
        One loss per type — types the data does not have need no entry:

        >>> from torch_concepts.nn import ConceptLoss, L1LogitRegularizer
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        >>> loss_fn = ConceptLoss(
        ...     binary=BCEWithLogitsLoss(),
        ...     categorical=CrossEntropyLoss(),
        ... )
        >>> loss_fn
        ConceptLoss(binary=BCEWithLogitsLoss, categorical=CrossEntropyLoss)

        Several terms on one type, weighted — here an L1 penalty added to the
        binary concepts only, at half weight:

        >>> loss_fn = ConceptLoss(
        ...     binary=[BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
        ...     binary_weights=[1.0, 0.5],
        ...     categorical=CrossEntropyLoss(),
        ... )
        >>> loss_fn
        ConceptLoss(binary=[BCEWithLogitsLoss + 0.5*L1LogitRegularizer], categorical=CrossEntropyLoss)
    """
    def __init__(
        self,
        binary: Optional[Union[nn.Module, List[nn.Module]]] = None,
        categorical: Optional[Union[nn.Module, List[nn.Module]]] = None,
        continuous: Optional[Union[nn.Module, List[nn.Module]]] = None,
        binary_weights: Optional[List[float]] = None,
        categorical_weights: Optional[List[float]] = None,
        continuous_weights: Optional[List[float]] = None,
        binary_param: Optional[str] = None,
        categorical_param: Optional[str] = None,
        continuous_param: Optional[str] = None,
        annotations=None,
    ):
        super().__init__()

        binary, binary_weights = _normalize_loss_terms(binary, binary_weights)
        categorical, categorical_weights = _normalize_loss_terms(categorical, categorical_weights)
        continuous, continuous_weights = _normalize_loss_terms(continuous, continuous_weights)
        self.terms_by_type = by_type(binary, categorical, continuous)
        if annotations is not None:
            self.terms_by_type = check_collection(annotations, self.terms_by_type, 'loss')

        # Register modules, weights, and signatures per type
        self._type_weights = {}
        self._type_signatures = {}
        weights_map = {
            'binary': binary_weights,
            'categorical': categorical_weights,
            'continuous': continuous_weights,
        }
        for type_name in TYPES:
            terms = self.terms_by_type.get(type_name)
            if terms is not None:
                # Register as nn.ModuleList for proper parameter tracking
                setattr(self, f'_{type_name}_terms', nn.ModuleList(terms))
                self._type_weights[type_name] = weights_map[type_name]
                # fill each loss type with (set of parameter names, has_var_keyword)
                self._type_signatures[type_name] = [
                    _get_forward_signature(m) for m in terms
                ]

        self.binary_param = binary_param
        self.categorical_param = categorical_param
        self.continuous_param = continuous_param

        # The categorical padding mask is only built when a term asks for it: it
        # costs an allocation the size of the padded logits every step, and the
        # usual term (``CrossEntropyLoss``) does not accept one.
        self._wants_padding_mask = any(
            'padding_mask' in sig or has_var_kw
            for sig, has_var_kw in self._type_signatures.get('categorical', ())
        )

        # Static categorical padding layout, keyed by the tuple of per-concept
        # cardinalities (see _prepare_categorical). Data-independent, so rebuilt
        # only when a new cardinality signature is seen.
        self._cat_pad_cache = {}

    def __repr__(self) -> str:
        parts = []
        for t in TYPES:
            terms = self.terms_by_type.get(t)
            if terms is not None:
                weights = self._type_weights[t]
                if len(terms) == 1 and weights[0] == 1.0:
                    name = terms[0].__class__.__name__
                    parts.append(f"{t}={name}")
                else:
                    term_strs = []
                    for m, w in zip(terms, weights):
                        n = m.__class__.__name__
                        term_strs.append(f"{w}*{n}" if w != 1.0 else n)
                    parts.append(f"{t}=[{' + '.join(term_strs)}]")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _compute_type_loss(self, type_name: str, kwargs: dict) -> torch.Tensor:
        """Weighted sum of one type's terms, each called with only the kwargs its
        ``forward`` declares (see :func:`_filter_kwargs`)."""
        terms = getattr(self, f'_{type_name}_terms')
        weights = self._type_weights[type_name]
        signatures = self._type_signatures[type_name]

        has_padding = 'padding_mask' in kwargs
        total = None

        for module, weight, (sig, has_var_kw) in zip(terms, weights, signatures):
            term_kwargs = _filter_kwargs(kwargs, sig, has_var_kw)
            if (has_padding and not has_var_kw
                    and 'padding_mask' not in sig and 'target' not in sig):
                # The term sees -inf padding without being told which columns it
                # is; it may or may not care, but silence would be worse.
                warnings.warn(
                    f"{module.__class__.__name__} does not accept a "
                    f"'padding_mask' parameter, so it cannot tell real "
                    f"categorical logits from the -inf padding added for "
                    f"concepts below the maximum cardinality. Add "
                    f"'padding_mask' to its forward() if that matters.",
                    stacklevel=2,
                )
            contribution = weight * module(**term_kwargs)
            total = contribution if total is None else total + contribution

        return total

    def _prepare_categorical(self, cat_logits: torch.Tensor, cat_target: torch.Tensor):
        """Pad and stack categorical logits/targets for CrossEntropy-style terms.

        ``cat_logits`` (logit-space) and ``cat_target`` (concept-space) are the
        categorical slices from :meth:`AnnotatedTensor.categorical`, already in
        the same concept order; per-concept widths come from ``cat_logits``'s
        annotation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor or None]:
                ``(padded_logits, targets, padding_mask)`` ready for loss
                functions like ``CrossEntropyLoss``. ``padding_mask`` is ``True``
                for real logit positions and ``False`` for padding. It is ``None``
                when there is nothing to mask (every concept has the same
                cardinality) or when no term asked for it — building it costs an
                allocation the size of ``padded_logits`` on every step.
        """
        cards = list(cat_logits.annotation.cardinalities)

        # Unwrap to plain tensors and fold any leading (batch-like) dimensions
        # into one batch axis, so the layout below is always (batch, width).
        # Both are flattened the same way, so their rows stay aligned.
        cat_logits = getattr(cat_logits, "tensor", cat_logits).reshape(-1, cat_logits.shape[-1])
        cat_target = getattr(cat_target, "tensor", cat_target).reshape(-1, cat_target.shape[-1])

        # The padding layout (max width and which columns are real per concept)
        # depends only on the cardinalities, so cache it per cardinality signature.
        key = tuple(cards)
        template = self._cat_pad_cache.get(key)
        if template is None:
            max_card = max(cards)
            # col_valid[i, j] is True where column j is a real class of concept i.
            col_valid = torch.zeros(len(cards), max_card, dtype=torch.bool)
            for i, c in enumerate(cards):
                col_valid[i, :c] = True
            template = (max_card, col_valid)
            self._cat_pad_cache[key] = template
        max_card, col_valid = template

        padded_logits = [
            nn.functional.pad(logits, (0, max_card - logits.shape[1]), value=float('-inf'))
            for logits in torch.split(_plain(cat_logits), cards, dim=1)
        ]
        cat_logits_out = torch.cat(padded_logits, dim=0)
        cat_targets = _plain(cat_target).T.reshape(-1).long()

        cat_mask = None
        if self._wants_padding_mask and len(set(cards)) > 1:
            # Repeat each concept's column-validity row over the batch to match
            # the concept-major row order of ``cat_logits_out``.
            batch = cat_logits.shape[0]
            cat_mask = col_valid.to(cat_logits_out.device).repeat_interleave(batch, dim=0)
        return cat_logits_out, cat_targets, cat_mask

    def forward(self, output: ModelOutput, target=None) -> torch.Tensor:
        """Total loss across all concept types.

        Each type is read from the quantity the model reports for it (or from the
        explicit ``*_param``) and aligned to the target by concept name;
        variables the target has no truth for are skipped.

        Args:
            output (ModelOutput): The model's output, carrying ``params``,
                ``target`` and optionally ``extra``.
            target (AnnotatedTensor, optional): Concept-space ground truth.
                Defaults to ``output.target``.

        Returns:
            torch.Tensor: Scalar loss.

        Raises:
            ValueError: If no configured type matched anything in the output — a
                zero loss with no gradient is never what the caller wanted.
        """
        extra = dict(output.extra) if output.extra else {}
        target = target if target is not None else output.target

        # Binary and categorical are sliced by type out of their (shared) discrete
        # quantity; continuous is taken whole. The per-type accessors are memoised
        # on the stable annotation, so each resolves at most once and stays warm.
        discrete = _resolve_quantity(
            output.params, self.binary_param, DISCRETE_QUANTITIES)
        binary = supervised_subset(
            discrete.binary() if discrete is not None else None, target)
        discrete = _resolve_quantity(
            output.params, self.categorical_param, DISCRETE_QUANTITIES)
        categorical = supervised_subset(
            discrete.categorical() if discrete is not None else None, target)
        continuous = supervised_subset(
            _resolve_quantity(
                output.params, self.continuous_param, CONTINUOUS_QUANTITIES),
            target,
        )

        contributions = []

        if self.terms_by_type.get('binary') and binary is not None:
            contributions.append(self._compute_type_loss('binary', {
                'input': _plain(binary),
                'target': _plain(target[binary.annotation.labels]).float(),
                **extra
            }))

        if self.terms_by_type.get('categorical') and categorical is not None:
            cat_logits, cat_targets, cat_mask = self._prepare_categorical(
                categorical, target[categorical.annotation.labels]
            )
            kwargs = {'input': cat_logits, 'target': cat_targets, **extra}
            # Offer the key only when padding actually exists, i.e. the concepts
            # have different cardinalities. Otherwise there is nothing to mask and
            # a term that ignores the mask has nothing to be warned about.
            if len(set(categorical.annotation.cardinalities)) > 1:
                kwargs['padding_mask'] = cat_mask
            contributions.append(self._compute_type_loss('categorical', kwargs))

        if self.terms_by_type.get('continuous') and continuous is not None:
            kwargs = {
                'input': _plain(continuous),
                'target': _plain(target[continuous.annotation.labels]),
                **extra,
            }
            if output.scale is not None:
                kwargs['scale'] = _plain(output.scale)
            contributions.append(self._compute_type_loss('continuous', kwargs))

        if not contributions:
            raise ValueError(
                f"ConceptLoss has terms for {sorted(self.terms_by_type)} but "
                f"scored nothing: the output reports {tuple(output.params)} and the "
                f"target covers "
                f"{sorted(set(target.annotation.types)) if target is not None else None}. "
                "Check that the model reports a quantity for those types (see its "
                "`param_for_discrete_var`) and that the target covers them."
            )
        return sum(contributions)


class ConceptSubset(PyCLoss):
    """``loss`` applied to a subset of the concepts, selected by name.

    The piece that turns any concept loss into a *group* loss: wrap it, name the
    group, and give the group a weight in a :class:`CompositeLoss`. That is how
    concepts are weighted differently from tasks, or shallow concepts from deep
    ones, without either idea needing its own machinery.

    Args:
        loss (nn.Module): The loss to apply to the subset. Must accept
            ``(output, target)`` — :class:`ConceptLoss` and :class:`CompositeLoss`
            both do.
        names (list of str, optional): The concepts in the group.
        exclude (list of str, optional): The concepts *not* in the group; the
            group is everything else the target covers. Use this for an
            open-ended group (all concepts that are not tasks), since which
            concepts exist is only known once a target arrives.

    Exactly one of ``names`` / ``exclude`` is required.

    Example:
        One group, scored on its own:

        >>> from torch_concepts.nn import CompositeLoss, ConceptLoss, ConceptSubset
        >>> from torch.nn import BCEWithLogitsLoss
        >>> tasks = ConceptSubset(ConceptLoss(binary=BCEWithLogitsLoss()),
        ...                       names=['cancer'])

        Two groups with their own weights — concepts against tasks. ``exclude``
        makes the first group open-ended, so a concept added to the data later
        lands in it without the loss being reconfigured:

        >>> loss_fn = CompositeLoss(
        ...     terms=[ConceptSubset(ConceptLoss(binary=BCEWithLogitsLoss()),
        ...                          exclude=['cancer']),
        ...            tasks],
        ...     weights=[0.5, 1.0],
        ...     names=['concepts', 'tasks'],
        ... )
        >>> loss_fn
        CompositeLoss(0.5*concepts + tasks)
    """

    def __init__(self, loss: nn.Module, names=None, exclude=None):
        super().__init__()
        if (names is None) == (exclude is None):
            raise ValueError(
                "ConceptSubset: pass exactly one of `names` or `exclude`."
            )
        self.loss = loss
        self.names = list(names) if names is not None else None
        self.exclude = list(exclude) if exclude is not None else None

    def extra_repr(self) -> str:
        if self.names is not None:
            return f"names={self.names}"
        return f"exclude={self.exclude}"

    def forward(self, output: ModelOutput, target=None) -> torch.Tensor:
        target = target if target is not None else output.target
        names = self.names
        if names is None:
            names = [n for n in target.annotation.labels if n not in self.exclude]

        sub = subset_output(output, names)
        if not sub.params:
            # None of this group's concepts is in the output — a depth level the
            # query left out, say. It contributes nothing rather than making the
            # whole objective fail.
            reference = next(iter(output.params.values()), None)
            return torch.zeros((), device=None if reference is None else reference.device)

        present = [n for n in names if n in target.annotation.label_to_index]
        return self.loss(sub, target[present])


class WeightedConceptLoss(CompositeLoss):
    """Concepts and tasks weighted separately.

    Two :class:`ConceptSubset` groups — everything that is not a task, and the
    tasks — summed with one weight each. The same loss configuration is used for
    both groups.

    Args:
        concept_weight (float): Weight for the concept group.
        task_weight (float): Weight for the task group.
        task_names (List[str]): Names of the task concepts.
        **loss_kwargs: Passed to :class:`ConceptLoss` for both groups (``binary``,
            ``categorical``, ``continuous``, their ``*_weights`` and ``*_param``).

    Example:
        Binary concepts, tasks down-weighted relative to concepts:

        >>> from torch_concepts.nn import WeightedConceptLoss
        >>> from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
        >>> loss_fn = WeightedConceptLoss(
        ...     concept_weight=0.7, task_weight=0.3,
        ...     task_names=['task'], binary=BCEWithLogitsLoss()
        ... )
        >>> loss_fn
        WeightedConceptLoss(0.7*concepts + 0.3*tasks)

        Mixed concept types, and the task worth twice the concepts. Both groups
        get the same per-type configuration:

        >>> loss_fn = WeightedConceptLoss(
        ...     concept_weight=1.0, task_weight=2.0, task_names=['cancer'],
        ...     binary=BCEWithLogitsLoss(),
        ...     categorical=CrossEntropyLoss(),
        ...     continuous=MSELoss(),
        ... )
        >>> loss_fn
        WeightedConceptLoss(concepts + 2.0*tasks)
    """

    def __init__(
        self,
        concept_weight: float,
        task_weight: float,
        task_names: List[str],
        **loss_kwargs,
    ):
        task_names = list(task_names)
        super().__init__(
            terms=[
                ConceptSubset(ConceptLoss(**loss_kwargs), exclude=task_names),
                ConceptSubset(ConceptLoss(**loss_kwargs), names=task_names),
            ],
            weights=[concept_weight, task_weight],
            names=['concepts', 'tasks'],
        )
        self.concept_weight = concept_weight
        self.task_weight = task_weight
        self.task_names = task_names


class DepthWeightedConceptLoss(CompositeLoss):
    """Concept weights decaying with depth in a DAG.

    One :class:`ConceptSubset` group per depth level of ``graph``, weighted
    ``source_weight * depth_decay ** d``. Values below 1 for ``depth_decay``
    down-weight deeper concepts, above 1 up-weight them.

    Concepts absent from the graph are scored at depth 0, so a graph that covers
    only some of the annotated concepts still works.

    Args:
        graph (ConceptGraph): DAG defining structure among concepts.
        source_weight (float): Weight at depth 0 (the graph sources). Default ``1.0``.
        depth_decay (float): Factor applied per additional depth level. Default ``0.5``.
        **loss_kwargs: Passed to :class:`ConceptLoss` for every level (``binary``,
            ``categorical``, ``continuous``, their ``*_weights`` and ``*_param``).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import DepthWeightedConceptLoss
        >>> from torch_concepts import ConceptGraph
        >>>
        >>> adj = torch.tensor([[0., 1., 0.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        >>>
        >>> # A -> B -> C, each level worth half the one above it.
        >>> loss_fn = DepthWeightedConceptLoss(
        ...     graph, source_weight=1.0, depth_decay=0.5,
        ...     binary=torch.nn.BCEWithLogitsLoss()
        ... )
        >>> loss_fn
        DepthWeightedConceptLoss(depth_0 + 0.5*depth_1 + 0.25*depth_2)

        A decay above 1 leans the other way — the downstream concepts matter
        most, the sources are only supporting them:

        >>> loss_fn = DepthWeightedConceptLoss(
        ...     graph, source_weight=1.0, depth_decay=2.0,
        ...     binary=torch.nn.BCEWithLogitsLoss(),
        ...     categorical=torch.nn.CrossEntropyLoss(),
        ... )
        >>> loss_fn
        DepthWeightedConceptLoss(depth_0 + 2.0*depth_1 + 4.0*depth_2)
    """

    def __init__(
        self,
        graph: ConceptGraph,
        source_weight: float = 1.0,
        depth_decay: float = 0.5,
        **loss_kwargs,
    ):
        levels = {d: list(names) for d, names in enumerate(graph.get_levels()) if names}
        levels.setdefault(0, [])  # depth 0 always exists: it absorbs the rest

        terms, weights, names = [], [], []
        for d in sorted(levels):
            # Depth 0 is defined by exclusion, so it also picks up the concepts the
            # graph does not mention; every other level lists its nodes.
            deeper = [n for level, ns in levels.items() if level > 0 for n in ns]
            group = (ConceptSubset(ConceptLoss(**loss_kwargs), exclude=deeper) if d == 0
                     else ConceptSubset(ConceptLoss(**loss_kwargs), names=levels[d]))
            terms.append(group)
            weights.append(source_weight * (depth_decay ** d))
            names.append(f"depth_{d}")

        super().__init__(terms=terms, weights=weights, names=names)
        self.depths = sorted(levels)
        self.source_weight = source_weight
        self.depth_decay = depth_decay


class L1LogitRegularizer(nn.Module):
    """Penalise large logit magnitudes via L1 regularisation.

    ``scale * mean(|input|)`` over the real (non-padded) positions. A *per-type*
    term: it reads one type's predictions, so it belongs in that type's list on
    :class:`ConceptLoss`, which hands it the ``padding_mask`` it declares.

    Args:
        scale (float): Multiplicative factor applied to the L1 mean.
            Default ``1.0``.

    Returns:
        torch.Tensor: Scalar regularisation loss.

    Example:
        On the binary concepts, at half the weight of the supervision:

        >>> from torch.nn import BCEWithLogitsLoss
        >>> from torch_concepts.nn import ConceptLoss, L1LogitRegularizer
        >>> loss_fn = ConceptLoss(
        ...     binary=[BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
        ...     binary_weights=[1.0, 0.5],
        ... )
        >>> loss_fn
        ConceptLoss(binary=[BCEWithLogitsLoss + 0.5*L1LogitRegularizer])

        On both discrete types, each with its own strength:

        >>> from torch.nn import CrossEntropyLoss
        >>> loss_fn = ConceptLoss(
        ...     binary=[BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.05)],
        ...     binary_weights=[1.0, 0.5],
        ...     categorical=[CrossEntropyLoss(), L1LogitRegularizer(scale=0.01)],
        ...     categorical_weights=[1.0, 0.3],
        ... )
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def extra_repr(self) -> str:
        return f"scale={self.scale}"

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            mask = padding_mask
        else:
            mask = torch.isfinite(input)
        if mask.any():
            return self.scale * input[mask].abs().mean()
        return torch.zeros((), device=input.device)