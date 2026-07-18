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
``activations``              how is a raw parameter mapped into its domain?
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
across the models and every inference backend. Consumers look a family up with
:func:`spec_for` (raises when unsupported) or :func:`maybe_spec_for` (returns
``None``, for the call sites that tolerate a user-supplied custom distribution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.distributions as dist

from .....distributions.delta import Delta


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


def _relaxed_bernoulli(params, temperature, validate_args):
    d = dist.RelaxedBernoulli(temperature=temperature, **params, validate_args=validate_args)
    return dist.Independent(d, 1, validate_args=validate_args)


def _relaxed_one_hot(params, temperature, validate_args):
    return dist.RelaxedOneHotCategorical(
        temperature=temperature, **params, validate_args=validate_args
    )


_softmax = partial(torch.softmax, dim=-1)


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
        Parameter name -> activation mapping a raw network output into the
        parameter's natural domain (e.g. ``logits`` -> ``sigmoid``).
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
        wrap_independent=True,
    ),
    dist.MultivariateNormal: DistributionSpec(
        param_sizes={"loc": _per_element, "scale_tril": _lower_triangular},
        valid_param_sets=(frozenset({"loc", "scale_tril"}),),
        default_params=("loc", "scale_tril"),
        primary_param="loc",
        activations={"loc": _identity, "scale_tril": _identity},
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
def maybe_spec_for(distribution: type) -> Optional[DistributionSpec]:
    """The spec for ``distribution``, or ``None`` if the family is unknown.

    Resolution order: exact class, then nearest registered base class
    (``issubclass``), then a name alias. Use this at the call sites that
    tolerate a user-supplied custom distribution; use :func:`spec_for`
    where a spec is genuinely required.
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

    ``context`` is prefixed to the error message so the caller can name the
    variable or factor that triggered the lookup.
    """
    spec = maybe_spec_for(distribution)
    if spec is None:
        prefix = f"{context}: " if context else ""
        supported = ", ".join(sorted(d.__name__ for d in SPECS))
        raise ValueError(
            f"{prefix}distribution {getattr(distribution, '__name__', distribution)!r} "
            f"is not a supported family. Supported families: {supported}. "
            "Register a DistributionSpec for it in "
            "torch_concepts.nn.modules.mid.models.distributions to add support."
        )
    return spec
