"""Lightning callbacks that schedule parts of the objective during training."""

import pytorch_lightning as pl


class LossWeightWarmup(pl.Callback):
    """Ramp one :class:`~torch_concepts.nn.CompositeLoss` weight up over epochs.

    The weight moves linearly from ``start`` to the value the loss was
    configured with, reaching it after ``epochs`` epochs. Per epoch rather than
    per batch, so the schedule does not shift with the batch size.

    The usual reason to want this is a VAE's KL term: at full strength from the
    first step it is trivially minimised by ignoring the input — the guide
    matches the prior, ``z`` carries nothing, and dimensions lost that way
    rarely come back. ``free_bits`` on
    :class:`~torch_concepts.nn.KLDivergenceLoss` addresses the same failure from
    the other end.

    Args:
        term (str or int): The term to schedule — a name from the loss's
            ``term_names`` (each term's class name unless the loss was given
            ``names``), or an index into them.
        epochs (int): Epochs taken to reach the configured weight. ``0``
            disables the schedule.
        start (float): Weight at epoch 0. Default ``0.0``.

    Example:
        >>> from torch_concepts.nn import LossWeightWarmup
        >>> callback = LossWeightWarmup(term='KLDivergenceLoss', epochs=5)
    """

    def __init__(self, term, epochs: int, start: float = 0.0):
        super().__init__()
        self.term = term
        self.epochs = int(epochs)
        self.start = float(start)
        self.index = self.target = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        loss = pl_module.loss
        if self.target is None:
            # Read the configured weight once: from the second epoch on, the
            # stored value is whatever this callback last wrote.
            self.index = (self.term if isinstance(self.term, int)
                          else loss.term_names.index(self.term))
            self.target = float(loss.weights[self.index])
        # `epochs=0` keeps the configured weight, so a sweep can include "no
        # warmup" as a point without swapping the callback out.
        fraction = min(1.0, trainer.current_epoch / self.epochs) if self.epochs else 1.0
        weight = self.start + fraction * (self.target - self.start)
        loss.weights[self.index] = weight
        pl_module.log(f"weight_{loss.term_names[self.index]}", weight)
