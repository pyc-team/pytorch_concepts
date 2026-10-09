"""``p_int`` on the Pyro engine: teacher forcing, and RandInt in between.

``ForwardInference`` has honoured ``p_int`` all along; ``VariationalInference``
passed query values straight to Pyro as ``obs=``, which is all-or-nothing. These
pin the three regimes on the Pyro path, and — importantly — that the default rate
leaves the ``obs=`` path untouched.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.low.priors import FixedPrior
from torch_concepts.nn.modules.mid.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn import VariationalInference

SIZE = 4


def _model():
    """``x (delta, root) -> c (bernoulli) -> echo (delta, identity)``.

    ``VariationalInference`` reports parameters, not samples, so the value ``c``
    actually *propagates* is read off ``echo`` — a Delta child that copies its
    parent. That is also the position that matters in practice: it is where
    CBGM's mixture sits.

    ``c``'s probs are pinned at 0.5 by the zero-initialised head, so a relaxed
    draw is almost never exactly 0 or 1 — which is what lets a test tell a forced
    row from an unforced one by equality alone.
    """
    x = EmbeddingVariable("x", distribution=Delta, size=SIZE)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=SIZE)
    echo = EmbeddingVariable("echo", distribution=Delta, size=SIZE)
    head = nn.Linear(SIZE, SIZE)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return BayesianNetwork(
        variables=[x, c, echo],
        factors=[
            ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(SIZE))}),
            ParametricCPD(
                variable=c,
                parametrization=nn.Sequential(head, nn.Sigmoid()),
                parents=[x],
            ),
            ParametricCPD(variable=echo, parametrization={"value": nn.Identity()}, parents=[c]),
        ],
    )


def _run(p_int, batch=64, seed=0):
    """Return (the value ``c`` propagated, ground truth) for one query."""
    torch.manual_seed(seed)
    engine = VariationalInference(_model(), p_int=p_int)
    gt = torch.ones(batch, SIZE)
    out = engine.query(
        query={"x": torch.zeros(batch, SIZE), "c": gt, "echo": None},
        evidence={},
    )
    return out.params["value"]["echo"], gt


def _forced_rows(sample, gt):
    """Boolean mask of rows that took the ground truth verbatim."""
    return (sample == gt).all(dim=-1)


class TestEndpoints:
    def test_p_int_1_forces_every_row(self):
        """The default rate, and it must match the old ``obs=`` behaviour."""
        sample, gt = _run(1.0)
        assert torch.equal(sample, gt)

    def test_p_int_0_forces_no_row(self):
        sample, gt = _run(0.0)
        assert not _forced_rows(sample, gt).any()

    def test_p_int_1_leaves_the_obs_path_untouched(self):
        """`teacher_forced` is empty at the default, so the site stays observed.

        The guard that keeps this change invisible to every caller that never
        sets a rate.
        """
        engine = VariationalInference(_model(), p_int=1.0)
        gt = torch.ones(8, SIZE)
        out = engine.query(
            query={"x": torch.zeros(8, SIZE), "c": gt, "echo": None}, evidence={}
        )
        assert torch.equal(out.params["value"]["echo"], gt)


class TestRandInt:
    def test_some_rows_forced_and_some_not(self):
        sample, gt = _run(0.5, batch=256)
        forced = _forced_rows(sample, gt)
        # Binomial(256, 0.5): the odds of landing outside this are nil, and the
        # seed is fixed anyway.
        assert 0 < int(forced.sum()) < 256

    def test_unforced_rows_are_genuine_draws(self):
        """A row that was not forced carries the model's own relaxed sample.

        Pins that the blend is a per-row *mask* and not an interpolation: an
        unforced row lands strictly inside ``(0, 1)`` — the Concrete draw — rather
        than somewhere partway towards the all-ones ground truth.
        """
        mixed, gt = _run(0.5, batch=256, seed=7)
        unforced = ~_forced_rows(mixed, gt)
        assert unforced.any()
        assert ((mixed[unforced] > 0.0) & (mixed[unforced] < 1.0)).all()

    def test_rate_controls_how_many_rows_are_forced(self):
        low = _forced_rows(*_run(0.2, batch=512)).float().mean()
        high = _forced_rows(*_run(0.8, batch=512)).float().mean()
        assert low < high
        assert low == pytest.approx(0.2, abs=0.08)
        assert high == pytest.approx(0.8, abs=0.08)

    @pytest.mark.parametrize("p_int", [1.0, 0.5, 0.0])
    def test_probs_are_reported_at_every_rate(self, p_int):
        """Turning an observed site latent must not cost it its parameters.

        ``ConceptLoss`` and ``ConceptMetrics`` look their target up in
        ``params['probs']``; a site that stopped reporting would surface there as
        a ``KeyError``, far from the cause.
        """
        torch.manual_seed(0)
        engine = VariationalInference(_model(), p_int=p_int)
        out = engine.query(
            query={"x": torch.zeros(8, SIZE), "c": torch.ones(8, SIZE), "echo": None},
            evidence={},
        )
        assert "probs" in out.params
        assert out.params["probs"]["c"].shape == (8, SIZE)


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_rate_outside_the_unit_interval_is_rejected(self, bad):
        with pytest.raises(ValueError, match="p_int"):
            VariationalInference(_model(), p_int=bad)
