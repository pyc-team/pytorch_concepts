"""Tests for LearnablePrior, FixedPrior, and TiedPrior."""
import torch
import torch.nn as nn

from torch_concepts.nn.modules.low.priors import (
    LearnablePrior,
    FixedPrior,
    TiedPrior,
)


class TestLearnablePrior:
    def test_forward_returns_parameter(self):
        prior = LearnablePrior(4)
        assert prior() is prior.param

    def test_default_broadcast_true(self):
        assert LearnablePrior(3).broadcast is True

    def test_broadcast_false(self):
        assert LearnablePrior(3, broadcast=False).broadcast is False


class TestFixedPrior:
    def test_forward_returns_values(self):
        values = torch.tensor([0.1, 0.5, 0.9])
        prior = FixedPrior(values)
        assert torch.equal(prior(), values)

    def test_values_are_cloned_not_aliased(self):
        values = torch.tensor([0.1, 0.5, 0.9])
        prior = FixedPrior(values)
        values[0] = 99.0
        assert prior()[0] == 0.1

    def test_no_gradient(self):
        prior = FixedPrior(torch.tensor([0.1, 0.5]))
        assert not prior.values.requires_grad
        assert list(prior.parameters()) == []

    def test_default_broadcast_true(self):
        assert FixedPrior(torch.zeros(2)).broadcast is True

    def test_broadcast_false(self):
        assert FixedPrior(torch.zeros(2), broadcast=False).broadcast is False


class TestTiedPrior:
    def test_forward_calls_source(self):
        embedding = nn.Embedding(5, 3)
        prior = TiedPrior(lambda: embedding.weight)
        assert prior() is embedding.weight

    def test_forward_reflects_live_updates(self):
        """No copy is made: mutating the source is visible on the next call."""
        source = torch.zeros(4)
        prior = TiedPrior(lambda: source)
        assert torch.equal(prior(), source)
        source[0] = 1.0
        assert prior()[0] == 1.0

    def test_nothing_registered(self):
        """The source owns its values; TiedPrior itself has no params/buffers."""
        embedding = nn.Embedding(5, 3)
        prior = TiedPrior(lambda: embedding.weight)
        assert list(prior.parameters()) == []
        assert list(prior.buffers()) == []

    def test_default_broadcast_true(self):
        prior = TiedPrior(lambda: torch.zeros(2))
        assert prior.broadcast is True

    def test_broadcast_false(self):
        prior = TiedPrior(lambda: torch.zeros(2), broadcast=False)
        assert prior.broadcast is False
