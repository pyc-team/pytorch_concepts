"""LossWeightWarmup's schedule, driven by hand rather than through a Trainer.

The two things that can break: the ramp itself, and the read-once target — the
callback writes into the same ``weights`` list it reads, so re-reading it after
the first epoch would latch the schedule at ``start`` forever.
"""
from torch_concepts.nn import (CompositeLoss, KLDivergenceLoss,
                               LossWeightWarmup, MSEReconstructionLoss)


class _Module:
    """The two members the callback touches on a LightningModule."""

    def __init__(self, loss):
        self.loss = loss
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = value


class _Trainer:
    current_epoch = 0


def _elbo():
    return CompositeLoss(
        terms=[MSEReconstructionLoss('input'), KLDivergenceLoss(['z'])], weights=[1.0, 4.0]
    )


def test_the_ramp_reaches_the_configured_weight_and_holds():
    loss = _elbo()
    module, trainer = _Module(loss), _Trainer()
    warmup = LossWeightWarmup(term="KLDivergenceLoss", epochs=4)

    seen = []
    for epoch in range(6):
        trainer.current_epoch = epoch
        warmup.on_train_epoch_start(trainer, module)
        seen.append(loss.weights[1])

    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0, 4.0]
    assert loss.weights[0] == 1.0  # the other term is untouched
    assert module.logged["weight_KLDivergenceLoss"] == 4.0


def test_zero_epochs_leaves_the_configured_weight_alone():
    loss = _elbo()
    LossWeightWarmup(term=1, epochs=0).on_train_epoch_start(_Trainer(), _Module(loss))
    assert loss.weights == [1.0, 4.0]
