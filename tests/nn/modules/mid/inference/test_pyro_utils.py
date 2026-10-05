"""build_relaxed_pyro_distribution: family dispatch, and the clear error for a
family Pyro cannot sample unobserved."""
import pyro
import pyro.distributions as pyro_dist
import pytest
import torch
import torch.distributions as td

from torch_concepts.nn.modules.mid.inference.pyro.utils import (
    build_relaxed_pyro_distribution,
)
from torch_concepts.nn.modules.mid.variable import ConceptVariable

T = torch.tensor(1.0)


def _base(d):
    """The family under Pyro's ``.to_event`` wrapper."""
    return d.base_dist if isinstance(d, td.Independent) else d


class TestFamilyDispatch:
    def test_soft_bernoulli_is_a_pyro_relaxed_bernoulli(self):
        v = ConceptVariable("b", distribution=td.Bernoulli, size=3)
        d = build_relaxed_pyro_distribution(v, {"probs": torch.full((5, 3), 0.5)}, T)
        base = _base(d)
        assert isinstance(base, pyro_dist.RelaxedBernoulli)
        assert not isinstance(base, pyro_dist.RelaxedBernoulliStraightThrough)
        assert d.batch_shape == (5,)

    def test_straight_through_is_picked_before_its_soft_base(self):
        """The ST class subclasses the soft one, so dispatch order matters."""
        v = ConceptVariable(
            "b", distribution=pyro_dist.RelaxedBernoulliStraightThrough, size=3
        )
        d = build_relaxed_pyro_distribution(v, {"probs": torch.full((5, 3), 0.5)}, T)
        assert isinstance(_base(d), pyro_dist.RelaxedBernoulliStraightThrough)

    def test_result_is_samplable_unobserved(self):
        """The reason this builder exists: a plain torch distribution is not
        callable, so ``pyro.sample`` rejects it on an unobserved site."""
        v = ConceptVariable("o", distribution=td.OneHotCategorical, size=4)
        d = build_relaxed_pyro_distribution(v, {"probs": torch.full((5, 4), 0.25)}, T)
        assert pyro.sample("o", d).shape == (5, 1, 4)


class TestUnsamplableFamily:
    def test_plain_categorical_raises_the_registry_reason(self):
        """Used to return a plain torch distribution, which failed later inside
        ``pyro.sample`` as an opaque "object is not callable"."""
        v = ConceptVariable("c", distribution=td.Categorical, size=4)
        with pytest.raises(ValueError, match="Declare it as OneHotCategorical"):
            build_relaxed_pyro_distribution(v, {"probs": torch.full((5, 4), 0.25)}, T)
