"""Single source of truth for the distribution families the mid level supports.

Every piece of family-specific knowledge the mid level needs lives in one
:class:`DistributionSpec` per family, collected in the :data:`SPECS` registry:

===========================  ==================================================
Field                        Answers the question
===========================  ==================================================
``param_sizes``              how many scalars must a network emit per parameter?
``valid_param_sets``         which parameter-name combinations are well formed?
``default_params``           which parameters does the single-module shorthand
                             expand to (and is the shorthand legal at all)?
``primary_param``            which parameter carries the canonical value used
                             for deterministic propagation?
``activations``              how is a supplied parameter mapped into the
                             canonical one's domain (``logits`` -> ``probs``)?
``param_activations``        which activation module turns a raw network output
                             into a *valid* value of this parameter?
``mode``                     what is its hard, most-likely value?
``is_discrete``              may it be a query/evidence variable of the
                             sampling estimators?
``wrap_independent``         does its event need reinterpreting as one event
                             axis of width ``size``?
``state_count``              how many states does it enumerate (belief
                             propagation), if any?
``relaxed``                  what is its reparameterisable surrogate?
``default_dist_kwargs``      which constructor kwargs does it always need
                             (e.g. a Concrete family's temperature)?
===========================  ==================================================

Adding a family is therefore a single registry entry rather than an edit spread
across the models and every inference backend.

A variable's distribution **must** be one of the registered families:
:meth:`Variable.__init__` resolves it through :func:`spec_for` and rejects
anything else at construction time. Because that is the single gate every
variable passes through, the rest of the mid level can look a spec up and rely
on it existing, with no "unknown family" branch to carry. To support a new
distribution, add its :class:`DistributionSpec` to :data:`SPECS` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn

from ....distributions.delta import Delta


# ---------------------------------------------------------------------------
# Small named helpers (kept out of the registry so the entries stay readable).
# ---------------------------------------------------------------------------
def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _per_element(size: int) -> int:
    """One scalar per event element — the common case."""
    return size


def _lower_triangular(size: int) -> int:
    """Entries of a lower-triangular Cholesky factor of a ``size x size`` matrix."""
    return size * (size + 1) // 2


def _binary_states(size: int) -> Optional[int]:
    """A single bit enumerates 2 states; a vector of bits is not one variable."""
    return 2 if size == 1 else None


def _categorical_states(size: int) -> Optional[int]:
    """One state per class."""
    return size


def _threshold(x: torch.Tensor) -> torch.Tensor:
    """Mode of a Bernoulli bit: set where the probability exceeds a half.

    Strict ``>``, so an exact 0.5 resolves to 0 — the same "lowest state wins"
    tie-break :func:`_argmax_one_hot` inherits from ``argmax``.
    """
    return (x > 0.5).to(x.dtype)


def _argmax_one_hot(x: torch.Tensor) -> torch.Tensor:
    """Mode of a categorical row: indicator of its top class (ties -> lowest)."""
    return torch.nn.functional.one_hot(x.argmax(-1), x.shape[-1]).to(x.dtype)


def _relaxed_bernoulli(params, temperature, validate_args):
    d = dist.RelaxedBernoulli(temperature=temperature, **params, validate_args=validate_args)
    return dist.Independent(d, 1, validate_args=validate_args)


def _relaxed_one_hot(params, temperature, validate_args):
    return dist.RelaxedOneHotCategorical(
        temperature=temperature, **params, validate_args=validate_args
    )


_softmax = partial(torch.softmax, dim=-1)


# ---------------------------------------------------------------------------
# Activation factories for :attr:`DistributionSpec.param_activations`.
#
# Each returns the ``nn.Module`` mapping a *raw* network output into one
# parameter's domain, given the variable's event ``size`` and — for a plate —
# its per-member width. Both are optional: only the categorical and Cholesky
# factories consult them. :class:`~torch_concepts.nn.DefaultActivation` is the
# single caller.
# ---------------------------------------------------------------------------
def _sigmoid_activation(size=None, member_size=None) -> nn.Module:
    """A Bernoulli's ``probs``: one independent probability per bit."""
    return nn.Sigmoid()


def _softplus_activation(size=None, member_size=None) -> nn.Module:
    """A Normal's ``scale``: positive, one per event element."""
    return nn.Softplus()


def _softmax_activation(size=None, member_size=None) -> nn.Module:
    """A categorical's ``probs``: each *member*'s states sum to one.

    A plate stacks ``size // member_size`` members along the last axis, so the
    normalisation happens per member rather than over the flattened width. A
    lone variable is one member wide (``member_size == size``) and collapses to
    a plain softmax.
    """
    if not size or not member_size or member_size == size:
        return nn.Softmax(dim=-1)
    return nn.Sequential(
        nn.Unflatten(-1, (size // member_size, member_size)),
        nn.Softmax(dim=-1),
        nn.Flatten(start_dim=-2),
    )


def _tril_activation(size=None, member_size=None) -> nn.Module:
    """A MultivariateNormal's ``scale_tril``: a positive-diagonal Cholesky factor."""
    # Deferred like ``ParametricCPD._instantiate_lazy``'s LazyConstructor import,
    # so this registry never pulls in the low level at module-import time.
    from ..low.scales import TrilActivation

    if not size:
        raise ValueError(
            "DefaultActivation('scale_tril'): a MultivariateNormal's Cholesky factor "
            "needs the event `size` to know the matrix side length. Pass size=..., "
            "or use DefaultActivation.for_variable(variable, 'scale_tril')."
        )
    return TrilActivation(size)


@dataclass(frozen=True)
class DistributionSpec:
    """Everything the mid level needs to know about one distribution family.

    Parameters
    ----------
    param_sizes : mapping
        Parameter name -> ``size -> n_scalars``. The number of scalar network
        outputs the parameter needs, given the variable's total event size. Most
        parameters are one-scalar-per-element; ``MultivariateNormal``'s
        ``scale_tril`` is the notable exception.
    valid_param_sets : tuple of frozenset
        The parameter-name combinations a CPD may supply. A parametrization must
        match exactly one of them (e.g. ``probs`` XOR ``logits`` for Bernoulli).
    default_params : tuple of str
        What the single-``nn.Module`` shorthand expands to. More than one entry
        means the shorthand is rejected: the family needs several distinct
        parameters and the user must pass a dict.
    primary_param : str
        The parameter holding the canonical value propagated in deterministic
        mode (``loc`` for Normal, ``probs`` for Bernoulli, ``value`` for Delta).
    activations : mapping
        Parameter name -> activation mapping *this* parameter into the
        :attr:`primary_param`'s domain (e.g. ``logits`` -> ``sigmoid`` yields
        ``probs``). The primary parameter maps to itself, so
        ``activations['probs']`` is the identity. Consumed at inference time by
        :func:`~torch_concepts.nn.modules.mid.inference.torch.utils._activate`.
    param_activations : mapping
        Parameter name -> ``(size, member_size) -> nn.Module`` building the
        activation that turns a *raw, unconstrained* network output into a valid
        value of that parameter (``probs`` -> ``Sigmoid``, ``scale`` ->
        ``Softplus``). The complement of ``activations``, which assumes the
        parameter is already valid: here ``probs`` is what needs squashing and
        ``logits`` is what does not. A missing entry means the parameter is
        unconstrained (``logits``, ``loc``, a Delta's ``value``), so
        :class:`~torch_concepts.nn.DefaultActivation` resolves it to
        ``nn.Identity``.
    mode : callable, optional
        Maps an *activated* parameter to the family's hard mode, operating on
        the last axis and preserving its width — a Bernoulli's ``probs`` to
        ``0.``/``1.`` bits, a categorical's row to a one-hot. ``None`` means the
        family's ``primary_param`` already *is* the mode (a Normal's ``loc``, a
        Delta's ``value``), so it is propagated unchanged. Because the parameter
        is activated first the rule is parametrization-agnostic:
        ``sigmoid(logits) > 0.5`` is ``logits > 0``, and ``argmax`` is invariant
        under ``softmax``. Used by
        :class:`~torch_concepts.nn.MAPForwardInference` through
        :func:`~torch_concepts.nn.modules.mid.inference.torch.utils.mode_value`;
        ``torch.distributions``' own ``.mode`` is unusable here (the relaxed
        families, ``Delta`` and a categorical plate's ``TransformedDistribution``
        all raise, and ``Categorical.mode`` returns a class index).
    is_discrete : bool
        Whether values are discrete, so exact equality matching is meaningful
        and the variable may be a query/evidence variable of the sampling
        estimators. Relaxed families count as discrete: a variable declared
        ``RelaxedBernoulli`` is conceptually a binary node.
    wrap_independent : bool
        Whether the exact distribution has a *univariate* event that must be
        reinterpreted (via ``Independent``) as a single event axis of width
        ``size``, keeping ``batch_shape == (*batch,)``.
    state_count : callable, optional
        ``size -> number of discrete states``, or ``None`` when this family
        cannot be enumerated at all. The callable itself returns ``None`` when
        this particular ``size`` is not enumerable (a vector of independent
        bits is not one variable). Used by belief propagation.
    relaxed : callable, optional
        ``(params, temperature, validate_args) -> Distribution`` building the
        reparameterisable surrogate. ``None`` means the exact distribution is
        already reparameterisable and is used directly.
    no_relaxed_reason : str, optional
        Set when the family has no usable surrogate; the message explains what
        to do instead.
    default_dist_kwargs : mapping, optional
        Constructor kwargs this family needs but cannot infer — currently the
        relaxation ``temperature`` of the Concrete families. Collected across
        the registry into :data:`DEFAULT_DIST_KWARGS`, which the high level uses
        to populate a variable's ``dist_kwargs``.
    """

    param_sizes: Mapping[str, Callable[[int], int]]
    valid_param_sets: Tuple[frozenset, ...]
    default_params: Tuple[str, ...]
    primary_param: str
    activations: Mapping[str, Callable[[torch.Tensor], torch.Tensor]]
    param_activations: Mapping[str, Callable[..., nn.Module]] = field(
        default_factory=dict
    )
    mode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    is_discrete: bool = False
    wrap_independent: bool = False
    state_count: Optional[Callable[[int], Optional[int]]] = None
    relaxed: Optional[Callable[..., dist.Distribution]] = None
    no_relaxed_reason: Optional[str] = None
    default_dist_kwargs: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_enumerable(self) -> bool:
        """Whether this family can be enumerated by belief propagation at all."""
        return self.state_count is not None

    @property
    def is_per_element(self) -> bool:
        """Whether every parameter is one scalar per event element.

        Plate members are addressed by slicing a contiguous column block out of
        each parameter, which is only meaningful when this holds.
        """
        return all(fn(3) == 3 for fn in self.param_sizes.values())


# ---------------------------------------------------------------------------
# The registry. Ordered most-specific-first: lookup falls back to an
# ``issubclass`` scan, so a user subclass resolves to its nearest base family.
# ---------------------------------------------------------------------------
_PROBS_OR_LOGITS = (frozenset({"probs"}), frozenset({"logits"}))

SPECS: Dict[type, DistributionSpec] = {
    Delta: DistributionSpec(
        param_sizes={"value": _per_element},
        valid_param_sets=(frozenset({"value"}),),
        default_params=("value",),
        primary_param="value",
        activations={"value": _identity},
        # No ``param_activations``: a point mass constrains nothing, so a raw
        # network output is already a valid ``value``.
        # A point mass has no extra batch dims to reinterpret, and our Delta is
        # built with ``batch_shape == ()`` already.
        wrap_independent=False,
    ),
    dist.RelaxedBernoulli: DistributionSpec(
        param_sizes={"probs": _per_element, "logits": _per_element},
        valid_param_sets=_PROBS_OR_LOGITS,
        default_params=("probs",),
        primary_param="probs",
        activations={"probs": _identity, "logits": torch.sigmoid},
        param_activations={"probs": _sigmoid_activation},
        mode=_threshold,
        is_discrete=True,
        wrap_independent=True,
        state_count=_binary_states,
        relaxed=_relaxed_bernoulli,
        default_dist_kwargs={"temperature": 0.5},
    ),
    dist.Bernoulli: DistributionSpec(
        param_sizes={"probs": _per_element, "logits": _per_element},
        valid_param_sets=_PROBS_OR_LOGITS,
        default_params=("probs",),
        primary_param="probs",
        activations={"probs": _identity, "logits": torch.sigmoid},
        param_activations={"probs": _sigmoid_activation},
        mode=_threshold,
        is_discrete=True,
        wrap_independent=True,
        state_count=_binary_states,
        relaxed=_relaxed_bernoulli,
    ),
    dist.RelaxedOneHotCategorical: DistributionSpec(
        param_sizes={"probs": _per_element, "logits": _per_element},
        valid_param_sets=_PROBS_OR_LOGITS,
        default_params=("probs",),
        primary_param="probs",
        activations={"probs": _identity, "logits": _softmax},
        param_activations={"probs": _softmax_activation},
        mode=_argmax_one_hot,
        is_discrete=True,
        state_count=_categorical_states,
        relaxed=_relaxed_one_hot,
        default_dist_kwargs={"temperature": 0.5},
    ),
    dist.OneHotCategorical: DistributionSpec(
        param_sizes={"probs": _per_element, "logits": _per_element},
        valid_param_sets=_PROBS_OR_LOGITS,
        default_params=("probs",),
        primary_param="probs",
        activations={"probs": _identity, "logits": _softmax},
        param_activations={"probs": _softmax_activation},
        mode=_argmax_one_hot,
        is_discrete=True,
        state_count=_categorical_states,
        relaxed=_relaxed_one_hot,
    ),
    dist.Categorical: DistributionSpec(
        param_sizes={"probs": _per_element, "logits": _per_element},
        valid_param_sets=_PROBS_OR_LOGITS,
        default_params=("probs",),
        primary_param="probs",
        activations={"probs": _identity, "logits": _softmax},
        param_activations={"probs": _softmax_activation},
        # A plain Categorical's *value* is encoded as a one-hot of width
        # ``size`` here, not as a class index — the same encoding
        # ``BeliefPropagation._encode_states`` uses — so that it matches the
        # ``(*leading, size)`` layout every cached value and child CPD expects.
        mode=_argmax_one_hot,
        is_discrete=True,
        state_count=_categorical_states,
        no_relaxed_reason=(
            "plain Categorical cannot be sampled with gradient flow. Declare it "
            "as OneHotCategorical instead, or always supply this variable as evidence."
        ),
    ),
    dist.Normal: DistributionSpec(
        param_sizes={"loc": _per_element, "scale": _per_element},
        valid_param_sets=(frozenset({"loc", "scale"}),),
        default_params=("loc", "scale"),
        primary_param="loc",
        activations={"loc": _identity, "scale": _identity},
        param_activations={"scale": _softplus_activation},
        wrap_independent=True,
    ),
    dist.MultivariateNormal: DistributionSpec(
        param_sizes={"loc": _per_element, "scale_tril": _lower_triangular},
        valid_param_sets=(frozenset({"loc", "scale_tril"}),),
        default_params=("loc", "scale_tril"),
        primary_param="loc",
        activations={"loc": _identity, "scale_tril": _identity},
        param_activations={"scale_tril": _tril_activation},
    ),
}

#: ``{family: default constructor kwargs}``, derived from the registry. The
#: high-level models seed each variable's ``dist_kwargs`` from this, so a new
#: family that needs a temperature (or any other fixed kwarg) only has to
#: declare ``default_dist_kwargs`` on its spec.
DEFAULT_DIST_KWARGS: Dict[type, dict] = {
    D: dict(spec.default_dist_kwargs)
    for D, spec in SPECS.items()
    if spec.default_dist_kwargs
}


# Families identified by name rather than identity. Pyro ships its own ``Delta``
# which is not a subclass of ours but plays the same role, and the Pyro backend
# has always matched it by name.
_NAME_ALIASES: Dict[str, type] = {"Delta": Delta}


@lru_cache(maxsize=None)
def _lookup(distribution: type) -> Optional[DistributionSpec]:
    """The spec for ``distribution``, or ``None`` if the family is unknown.

    Resolution order: exact class, then nearest registered base class
    (``issubclass``, so a Pyro distribution resolves to its torch base), then a
    name alias. Internal — every caller goes through :func:`spec_for`, which
    turns the ``None`` into a helpful error.
    """
    spec = SPECS.get(distribution)
    if spec is not None:
        return spec
    for base, candidate in SPECS.items():
        if issubclass(distribution, base):
            return candidate
    alias = _NAME_ALIASES.get(getattr(distribution, "__name__", ""))
    return SPECS[alias] if alias is not None else None


def spec_for(distribution: type, context: str = "") -> DistributionSpec:
    """The spec for ``distribution``; raises ``ValueError`` when unsupported.

    Every distribution reaching the mid level must be a registered family —
    :meth:`Variable.__init__` enforces this at the boundary — so this is the
    only lookup callers need.

    Parameters
    ----------
    distribution : type
        The distribution family to look up.
    context : str, optional
        Prefixed to the error message so the caller can name the variable or
        factor that triggered the lookup.

    Returns
    -------
    DistributionSpec
        The registered spec for this family.

    Raises
    ------
    ValueError
        If the family is not in :data:`SPECS`.
    """
    spec = _lookup(distribution)
    if spec is None:
        prefix = f"{context}: " if context else ""
        supported = ", ".join(sorted(d.__name__ for d in SPECS))
        raise ValueError(
            f"{prefix}distribution {getattr(distribution, '__name__', distribution)!r} "
            f"is not a supported family. Supported families: {supported}. "
            "Register a DistributionSpec for it in "
            "torch_concepts.nn.modules.mid.distributions to add support."
        )
    return spec
