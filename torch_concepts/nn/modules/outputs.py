"""Output containers for PGM inference engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch

from ...tensor import AnnotatedTensor


#: One variable's distribution parameters, e.g. ``{'loc': ..., 'scale': ...}``.
#: The *intermediate* form every engine builds per variable before
#: :meth:`BaseInference._assemble_params` turns it into the quantity-keyed
#: annotated tensors an :class:`InferenceOutput` exposes.
ParamDict = Dict[str, torch.Tensor]


#: The distribution-parameter names an engine may report, in the order the
#: attribute sugar is generated. Each becomes an ``InferenceOutput`` property
#: (``out.probs``, ``out.logits``, ...) reading ``params[<name>]``.
QUANTITIES = ("probs", "logits", "loc", "scale", "scale_tril", "value")

#: Quantities a *continuous* concept may be reported under. A continuous concept
#: is scored on its family's primary parameter, and which family models the type
#: is the model's choice: ``loc`` for a ``Normal``, ``value`` for a ``Delta`` (a
#: deterministic point estimate). Consumers with no per-instance quantity config
#: (e.g. :class:`~torch_concepts.nn.ConceptMetrics`) try both, mirroring the
#: ``logits``/``probs`` fallback already used for discrete concepts.
CONTINUOUS_QUANTITIES = ("loc", "value")


class _Unset:
    """Default for the quantity constructor arguments.

    Distinguishes "the caller said nothing about this quantity" from
    ``logits=None``, which means "drop it". Without that distinction the
    generated ``__init__`` — which assigns *every* field, defaults included —
    would erase the ``params`` entries it was just handed.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


UNSET = _Unset()


def _quantity_property(name: str) -> property:
    """Build the ``out.<name>`` <-> ``out.params[<name>]`` accessor pair."""

    def getter(self) -> Optional[AnnotatedTensor]:
        return self.params.get(name)

    def setter(self, value) -> None:
        if isinstance(value, _Unset):
            return
        if value is None:
            self.params.pop(name, None)
        else:
            self.params[name] = value

    getter.__name__ = name
    return property(getter, setter, doc=f"``params[{name!r}]``, or ``None`` when absent.")


class ParamsDict(Dict[str, AnnotatedTensor]):
    """Quantity-keyed parameter storage with variable-first indexing sugar.

    Storage, iteration, ``get`` and ``in`` are exactly a plain
    ``{quantity: AnnotatedTensor}`` dict — that is what the engines build and
    what the ``out.probs`` / ``out.logits`` properties read. On top of that,
    ``__getitem__`` also accepts a variable (or plate / plate-member) name and
    returns that variable's parameters as ``{quantity: AnnotatedTensor}``,
    where each value is a *view* into the corresponding quantity tensor:

    >>> out.params['logits']            # one tensor spanning all variables
    >>> out.params['c1']                # {'logits': <c1's columns, a view>}
    >>> out.params['c1']['logits']      # == out.logits['c1']

    Quantity keys take priority on the (pathological) collision where a
    variable is named like a parameter — don't call a concept ``logits``.
    """

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        views: Dict[str, AnnotatedTensor] = {}
        for quantity, tensor in dict.items(self):
            annotation = tensor.annotation
            if (
                key in annotation.label_to_index
                or key in annotation.label_groups
            ):
                views[quantity] = tensor[key]
        if not views:
            raise KeyError(
                f"{key!r} is neither a reported quantity {tuple(self)} nor a "
                f"queried variable {InferenceOutput._addressable(self)}."
            )
        return views


def supervised_subset(tensor, target):
    """``tensor`` restricted to the variables ``target`` provides truth for.

    A quantity spans every queried variable that reports it, which need not be
    only the supervised concepts: a generative model queried for all its
    variables also reports ``probs`` for the reconstructed observation. Those
    have no ground truth, so a loss or metric drops them rather than looking
    them up in the target and failing. Returns ``None`` when nothing survives,
    and the tensor itself when everything does (the common case, no copy).
    """
    if tensor is None or target is None:
        return tensor
    labels = list(tensor.annotation.labels)
    keep = [n for n in labels if n in target.annotation.label_to_index]
    if len(keep) == len(labels):
        return tensor
    return tensor[keep] if keep else None


# ---------------------------------------------------------------------------
# InferenceOutput
# ---------------------------------------------------------------------------

@dataclass
class InferenceOutput:
    """Return value of every inference engine.

    **Uniform contract.** An engine fills only the attributes it can produce —
    a deterministic forward pass leaves ``samples`` empty, a rejection sampler
    fills only ``probabilities`` — but whenever an attribute *is* filled, its
    contents mean the same thing across every engine. Consumers can therefore
    swap engines without rewriting their losses and metrics.

    **Layout.** Results are keyed by *quantity* (the distribution-parameter
    name), not by variable: ``params['logits']`` is a single
    :class:`~torch_concepts.tensor.AnnotatedTensor` holding the logits of every
    queried variable that has them, concatenated along the **last** axis and
    labelled by variable (or plate-member) name. A variable's own slice is
    recovered by label — ``out.logits['c1']`` — which is a *view* into that one
    tensor, so nothing is stored twice. Variables whose family reports a
    different quantity simply appear under a different key
    (``params['loc']`` / ``params['scale']`` for a Normal).

    ``params`` additionally supports **variable-first** indexing:
    ``params['c1']`` returns ``{'logits': <c1's columns>}`` — the same views,
    grouped per variable (see :class:`ParamsDict`). The quantity-first form and
    the attribute sugar remain the canonical access paths.

    Every tensor is shaped ``(*leading, width)``: any number of leading
    (batch-like) dimensions followed by the flat annotated axis. Event shapes
    are flattened into that last axis, so a variable declared with
    ``shape=(3, 4)`` contributes a label of width 12.

    Attributes
    ----------
    params : dict[str, AnnotatedTensor]
        Quantity -> annotated tensor of the queried variables' distribution
        parameters. Engines that compute a posterior marginal (e.g.
        :class:`~torch_concepts.nn.BeliefPropagation`) report it in this same
        parametrization rather than as a state-space belief. Only queried
        variables appear; fully-observed ones emit no parameters.
    guide_params : dict[str, AnnotatedTensor]
        Parameters of the variational guide's latents, same layout as ``params``.
    samples : AnnotatedTensor or None
        Per-variable realisations, one annotated tensor labelled the same way as
        ``params``. Filled only by engines that actually draw samples.
    probabilities : torch.Tensor or None
        ``(*leading,)`` estimate of ``P(query | evidence)`` for a fully realised
        query batch. Filled only by the sampling estimators.

    Examples
    --------
    >>> out = engine.query(query=['c1', 'c2'], evidence={'x': x})
    >>> out.logits.shape                  # (*leading, width of c1 + c2)
    >>> out.logits['c1']                  # just c1's columns (a view)
    >>> out.logits.binary()               # binary concepts' columns (or None)
    """

    params: Dict[str, AnnotatedTensor] = field(default_factory=dict)
    guide_params: Dict[str, AnnotatedTensor] = field(default_factory=dict)
    samples: Optional[AnnotatedTensor] = None
    probabilities: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        # Engines hand over plain dicts; wrap them so ``params`` (and the
        # guide's) gain the variable-first indexing without copying anything.
        if not isinstance(self.params, ParamsDict):
            self.params = ParamsDict(self.params)
        if not isinstance(self.guide_params, ParamsDict):
            self.guide_params = ParamsDict(self.guide_params)

    # Declared as fields so they can be passed to the
    # constructor (``ModelOutput(logits=...)``), but each is turned into a
    # property below, so assigning one — in ``__init__`` or later — writes
    # straight through to ``params`` and there is still only one storage
    # location per quantity. ``repr=False`` keeps them out of the generated
    # repr, which already shows ``params``.
    probs: Optional[Union[torch.Tensor, AnnotatedTensor]] = field(default=UNSET, repr=False)
    logits: Optional[Union[torch.Tensor, AnnotatedTensor]] = field(default=UNSET, repr=False)
    loc: Optional[Union[torch.Tensor, AnnotatedTensor]] = field(default=UNSET, repr=False)
    scale: Optional[Union[torch.Tensor, AnnotatedTensor]] = field(default=UNSET, repr=False)
    scale_tril: Optional[Union[torch.Tensor, AnnotatedTensor]] = field(default=UNSET, repr=False)
    value: Optional[Union[torch.Tensor, AnnotatedTensor]] = field(default=UNSET, repr=False)

    @property
    def quantities(self) -> tuple:
        """The quantity names this output actually carries."""
        return tuple(self.params)

    @staticmethod
    def _addressable(params: Dict[str, AnnotatedTensor]) -> tuple:
        """Every name that slices something out of a quantity-keyed dict.

        That is each label plus the variable it came from, so a query naming a
        plate exposes both the plate and each of its members. Ranges over every
        quantity, since one alone may not cover all the variables — a Bernoulli
        variable reports ``probs``/``logits`` while a Normal one reports
        ``loc``/``scale``.
        """
        names: Dict[str, None] = {}
        for tensor in params.values():
            annotation = tensor.annotation
            names.update(
                dict.fromkeys(annotation.label_groups)
            )
            names.update(dict.fromkeys(annotation.labels))
        return tuple(names)

    @property
    def variables(self) -> tuple:
        """Every name addressable in ``params``. See :meth:`_addressable`."""
        return self._addressable(self.params)

    @property
    def guide_variables(self) -> tuple:
        """Every name addressable in ``guide_params``. See :meth:`_addressable`."""
        return self._addressable(self.guide_params)


# Replace each quantity field with a property backed by ``params``. Attached
# after ``@dataclass`` has run so the decorator saw a plain ``UNSET`` default;
# because a property is a data descriptor on the type, the generated
# ``__init__``'s ``self.logits = logits`` now routes through the setter, and a
# quantity passed to the constructor lands in ``params`` like any other.
for _q in QUANTITIES:
    setattr(InferenceOutput, _q, _quantity_property(_q))
del _q


@dataclass
class ModelOutput(InferenceOutput):
    """Structured output from a high-level model's ``forward()`` method.

    An :class:`InferenceOutput` — same contract, same quantity-keyed layout —
    plus the extra fields the high level attaches around a query.

    Attributes
    ----------
    target : torch.Tensor or None
        The prepared ground-truth tensor aligned with the annotated axis of
        ``logits``.
    extra : dict[str, torch.Tensor] or None
        Model-specific extras that do not belong to the inference contract.
    """

    target: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, torch.Tensor]] = None
