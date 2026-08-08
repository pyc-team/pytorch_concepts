"""Lightning callbacks that schedule parts of the objective during training."""

from typing import Optional

import pytorch_lightning as pl


class LossWeightWarmup(pl.Callback):
    """Ramp one :class:`~torch_concepts.nn.CompositeLoss` weight up over epochs.

    The weight starts at ``start`` and reaches the value the loss was configured
    with after ``epochs`` epochs, moving linearly. Stepping per epoch rather than
    per batch keeps the schedule identical across batch sizes.

    The usual reason to want this is the KL term of a VAE. Applied at full
    strength from the first step, the KL is trivially minimised by ignoring the
    input — the guide matches the prior, ``z`` carries nothing, and the decoder
    settles into predicting the data mean before the reconstruction term has
    taught it anything. Dimensions lost that way rarely come back. Ramping the
    weight lets the model first learn to use ``z``, and only then pays for it.
    See also ``free_bits`` on :class:`~torch_concepts.nn.KLDivergenceLoss`, which
    addresses the same failure from the other end.

    Args:
        term (int or str): Which weight to schedule — an index into the loss's
            ``terms``, or the class name of the term (e.g. ``'KLDivergenceLoss'``).
        epochs (int): Epochs taken to reach the configured weight. ``0``
            disables the schedule.
        start (float): Weight at epoch 0. Default ``0.0``.
        attribute (str): Where to find the loss on the module. Default
            ``'loss'``.

    Example:
        >>> from torch_concepts.nn import LossWeightWarmup
        >>> callback = LossWeightWarmup(term='KLDivergenceLoss', epochs=5)
    """

    def __init__(
        self,
        term,
        epochs: int,
        start: float = 0.0,
        attribute: str = "loss",
    ):
        super().__init__()
        self.term = term
        self.epochs = int(epochs)
        self.start = float(start)
        self.attribute = attribute
        self._target: Optional[float] = None
        self._index: Optional[int] = None

    def _resolve(self, loss) -> int:
        """The index of ``term`` in ``loss.terms``."""
        if isinstance(self.term, int):
            return self.term
        for index, candidate in enumerate(loss.terms):
            if type(candidate).__name__ == self.term:
                return index
        raise ValueError(
            f"LossWeightWarmup: no term named {self.term!r} in "
            f"{[type(t).__name__ for t in loss.terms]}."
        )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        loss = getattr(pl_module, self.attribute, None)
        if loss is None or not hasattr(loss, "weights"):
            raise ValueError(
                f"LossWeightWarmup: {type(pl_module).__name__}.{self.attribute} "
                "is not a CompositeLoss (it has no `weights`)."
            )
        if self._target is None:
            # Read the configured weight once: after the first epoch the stored
            # value is whatever this callback last wrote, not the target.
            self._index = self._resolve(loss)
            self._target = float(loss.weights[self._index])
        if self.epochs <= 0:
            return

        fraction = min(1.0, trainer.current_epoch / self.epochs)
        weight = self.start + fraction * (self._target - self.start)
        loss.weights[self._index] = weight
        pl_module.log(
            f"weight_{type(loss.terms[self._index]).__name__}",
            weight,
            on_epoch=True,
        )
