"""Fitted scalers packaged as an :class:`torch.nn.Module`.

A :class:`~torch_concepts.data.base.scaler.Scaler` is a plain Python object: its
fitted statistics are ordinary tensors that nothing moves to the GPU and nothing
writes into a checkpoint. :class:`ScalerModule` wraps the *fitted* scalers of a
run so both come for free — the statistics are registered as buffers, so
``.to(device)``, ``.half()`` and ``state_dict()`` all reach them.

It is the object the datamodule hands to the model: the datamodule fits it on the
**train split only** (see :meth:`ConceptDataModule.setup
<torch_concepts.data.base.datamodule.ConceptDataModule.setup>`) and the learner
applies it around the model's forward pass (see :meth:`BaseLearner.shared_step
<torch_concepts.nn.modules.high.base.learner.BaseLearner.shared_step>`).

This module deliberately knows nothing about ``AnnotatedTensor`` or
``ModelOutput``: it works on plain tensors plus a list of concept labels, and the
annotation handling stays on the learner side.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

from ..base.scaler import Scaler


class ScalerModule(nn.Module):
    """Fitted input / concept scalers, with their statistics as buffers.

    Continuous concepts get **one scaler instance each**, fitted on that concept's
    own column. That keeps un-scaling generic over any :class:`Scaler` subclass
    (nothing here assumes an affine transform) and makes alignment work *by name*
    — which is required, because a continuous concept's position among the
    predicted columns is not its position in the concept axis: a categorical
    concept ahead of it widens the logit space, and a task-only model reports just
    a subset of the concepts.

    Build it with :meth:`fit` rather than calling the constructor directly.

    Parameters
    ----------
    input_scaler : Scaler, optional
        Already-fitted scaler for the model's input tensor.
    concept_scalers : dict[str, Scaler], optional
        Already-fitted scaler per continuous concept, keyed by concept name.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.data.scalers import ScalerModule, StandardScaler
    >>> from torch_concepts.annotations import Annotations
    >>> ann = Annotations(labels=['a', 'b'], cardinalities=[1, 1],
    ...                   types=['continuous', 'continuous'])
    >>> c = torch.tensor([[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]])
    >>> scalers = ScalerModule.fit(annotations=ann, concepts=c,
    ...                            concept_scaler=StandardScaler())
    >>> scaled = scalers.transform_concepts(c, ['a', 'b'])
    >>> torch.allclose(scalers.inverse_concepts(scaled, ['a', 'b']), c, atol=1e-5)
    True
    """

    def __init__(
        self,
        input_scaler: Optional[Scaler] = None,
        concept_scalers: Optional[Dict[str, Scaler]] = None,
    ) -> None:
        super().__init__()
        self.input_scaler = input_scaler
        self.concept_scalers = dict(concept_scalers) if concept_scalers else {}

        # Statistics live on the scaler objects, which nn.Module does not track.
        # Mirror every tensor attribute into a buffer (so device/dtype moves and
        # checkpointing reach them) and keep an index to write them back.
        # Vectorised stand-ins for the per-concept scalers, keyed by label tuple
        # and rebuilt whenever the buffers move (see _sync).
        self._fused_cache: Dict[tuple, object] = {}

        self._stat_index: List[tuple] = []
        for key, scaler in self._all_scalers():
            for attr, value in list(vars(scaler).items()):
                if isinstance(value, Tensor):
                    buffer_name = f"stat__{key}__{attr}"
                    self.register_buffer(buffer_name, value)
                    self._stat_index.append((key, attr, buffer_name))
        self._sync()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        annotations,
        concepts: Optional[Tensor] = None,
        input_data: Optional[Tensor] = None,
        concept_scaler: Optional[Scaler] = None,
        input_scaler: Optional[Scaler] = None,
    ) -> "ScalerModule":
        """Fit *unfitted* scaler prototypes on training data.

        ``concepts`` and ``input_data`` must already be restricted to the training
        split — this method has no notion of splits and would happily leak
        validation statistics into the model if handed the full tensors.

        Parameters
        ----------
        annotations : Annotations
            Concept-axis annotations, used to find the continuous concepts.
        concepts : torch.Tensor, optional
            ``(n_train, n_concepts)`` concept-space ground truth.
        input_data : torch.Tensor, optional
            ``(n_train, *features)`` inputs, shaped as the model consumes them.
        concept_scaler : Scaler, optional
            Prototype deep-copied once per continuous concept and fitted on that
            concept's column. Ignored when there are no continuous concepts.
        input_scaler : Scaler, optional
            Prototype fitted on ``input_data`` as a whole.

        Returns
        -------
        ScalerModule
            Fitted module. Empty (both flags False) when no prototype was given.
        """
        fitted_concepts: Dict[str, Scaler] = {}
        if concept_scaler is not None and concepts is not None:
            for name in annotations.type_groups['continuous']['labels']:
                concept = annotations.concept(name)
                # A column-per-label layout is what makes name-based alignment
                # work in transform/inverse; a wider continuous concept would
                # silently misalign, so reject it here rather than there.
                if concept.cardinality != 1:
                    raise ValueError(
                        f"ScalerModule: continuous concept {name!r} has cardinality "
                        f"{concept.cardinality}; scaling supports cardinality 1 "
                        f"(one column per continuous concept)."
                    )
                column = concepts[:, concept.index].unsqueeze(-1).float()
                fitted_concepts[name] = deepcopy(concept_scaler).fit(column)

        fitted_input = None
        if input_scaler is not None and input_data is not None:
            fitted_input = deepcopy(input_scaler).fit(input_data.float())

        return cls(input_scaler=fitted_input, concept_scalers=fitted_concepts)

    # ------------------------------------------------------------------
    # Buffer bookkeeping
    # ------------------------------------------------------------------
    def _all_scalers(self):
        """``(key, scaler)`` for every scaler held, input first."""
        if self.input_scaler is not None:
            yield ('input', self.input_scaler)
        for name, scaler in self.concept_scalers.items():
            yield (f"concept_{name}", scaler)

    def _sync(self) -> None:
        """Write the (possibly moved/cast) buffers back onto the scaler objects."""
        by_key = dict(self._all_scalers())
        for key, attr, buffer_name in self._stat_index:
            setattr(by_key[key], attr, getattr(self, buffer_name))
        # The fused scalers hold tensors *derived* from the buffers, so they are
        # stale as soon as the buffers are replaced; rebuilt lazily on next use.
        self._fused_cache = {}

    def _apply(self, *args, **kwargs):
        # ``.to()`` / ``.cuda()`` / ``.half()`` all funnel through _apply and
        # replace the buffers; re-point the scalers at the new tensors.
        result = super()._apply(*args, **kwargs)
        result._sync()
        return result

    # ------------------------------------------------------------------
    # Fused fast path
    # ------------------------------------------------------------------
    def _fused_for(self, labels: tuple):
        """One scaler standing in for ``labels``' per-concept scalers, or None.

        Scaling concept-by-concept costs a kernel launch per concept and grows
        linearly with the concept count, which dominates the step for a model
        with many continuous concepts. Every per-concept scaler here was fitted
        the same way on a single column, so when their statistics are scalars
        they can be concatenated into one ``(k,)`` tensor that broadcasts over a
        ``(*leading, k)`` block — turning the whole loop into one vectorised call.

        Returns None when that equivalence cannot be established (scalers of
        different classes, non-scalar statistics, or differing non-tensor
        configuration), in which case the caller falls back to the loop. The
        result is cached per label tuple: the label sets are fixed across a run,
        so the stacking is paid once.
        """
        if labels in self._fused_cache:
            return self._fused_cache[labels]

        fused = self._build_fused(labels)
        self._fused_cache[labels] = fused
        return fused

    def _build_fused(self, labels: tuple):
        scalers = [self.concept_scalers[name] for name in labels]
        cls = type(scalers[0])
        if any(type(s) is not cls for s in scalers):
            return None

        attributes = vars(scalers[0]).keys()
        if any(vars(s).keys() != attributes for s in scalers):
            return None

        merged = {}
        for attr in attributes:
            values = [vars(s)[attr] for s in scalers]
            if isinstance(values[0], Tensor):
                # Only a per-column statistic can be concatenated into a vector
                # that broadcasts over the block.
                if any(not isinstance(v, Tensor) or v.numel() != 1 for v in values):
                    return None
                merged[attr] = torch.cat([v.reshape(-1) for v in values])
            else:
                # Shared configuration (an axis, a float bias): it must be the
                # same for every concept, or one fused call cannot represent them.
                if any(v != values[0] for v in values):
                    return None
                merged[attr] = values[0]

        # Bypass __init__: it takes constructor arguments, not fitted state.
        fused = cls.__new__(cls)
        vars(fused).update(merged)
        return fused

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------
    @property
    def has_input(self) -> bool:
        """Whether an input scaler was fitted."""
        return self.input_scaler is not None

    @property
    def has_concepts(self) -> bool:
        """Whether any continuous concept scaler was fitted."""
        return bool(self.concept_scalers)

    @property
    def concept_names(self) -> List[str]:
        """Names of the continuous concepts this module scales."""
        return list(self.concept_scalers)

    def __repr__(self) -> str:
        input_name = type(self.input_scaler).__name__ if self.has_input else None
        return (f"{type(self).__name__}(input={input_name}, "
                f"concepts={self.concept_names})")

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def transform_input(self, x: Tensor) -> Tensor:
        """Scale the model input. Identity when no input scaler was fitted."""
        if not self.has_input:
            return x
        return self.input_scaler.transform(x)

    def inverse_input(self, x: Tensor) -> Tensor:
        """Map a scaled input back to the original scale."""
        if not self.has_input:
            return x
        return self.input_scaler.inverse_transform(x)

    def transform_concepts(self, x: Tensor, labels: Sequence[str]) -> Tensor:
        """Scale ``x``, whose last-axis columns are ``labels`` in that order."""
        return self._apply_concepts(x, labels, inverse=False)

    def inverse_concepts(self, x: Tensor, labels: Sequence[str]) -> Tensor:
        """Map ``x`` back to the original scale, columns being ``labels``."""
        return self._apply_concepts(x, labels, inverse=True)

    def _apply_concepts(self, x: Tensor, labels: Sequence[str], inverse: bool) -> Tensor:
        """Scale ``x``'s columns with each label's own scaler.

        Takes the vectorised path when the per-concept scalers can be fused into
        one (see :meth:`_fused_for`), which is the case for every column-wise
        scaler; otherwise falls back to scaling one column at a time. Either way
        the caller may pass any subset of the continuous concepts, in any order.
        """
        labels = tuple(labels)
        if x.shape[-1] != len(labels):
            raise ValueError(
                f"ScalerModule: got {x.shape[-1]} columns for {len(labels)} labels "
                f"{list(labels)}; the last axis must hold exactly one column per label."
            )
        unknown = [name for name in labels if name not in self.concept_scalers]
        if unknown:
            raise KeyError(
                f"ScalerModule: no fitted scaler for {unknown}; this module covers "
                f"{self.concept_names}."
            )

        fused = self._fused_for(labels)
        if fused is not None:
            return fused.inverse_transform(x) if inverse else fused.transform(x)

        columns = []
        for i, name in enumerate(labels):
            scaler = self.concept_scalers[name]
            column = x[..., i:i + 1]
            columns.append(
                scaler.inverse_transform(column) if inverse else scaler.transform(column)
            )
        return torch.cat(columns, dim=-1)
