"""Owner-keyed parent resolution in ``ParametricCPD.forward`` and the
``ProbabilisticModel.extract`` / ``extract_params`` read primitives (see
PLATE_REFACTOR_INSTRUCTIONS.md §3.1, §3.2, §8.3)."""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.low.priors import LearnablePrior, FixedPrior
from torch_concepts.distributions import Delta


def _model():
    """x -> concepts{c1,c2,c3} ; y <- whole plate ; y1 <- member c1 only."""
    x = EmbeddingVariable("x", distribution=Delta, size=4)
    concepts = ConceptVariable("concepts", members=["c1", "c2", "c3"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli)
    y1 = ConceptVariable("y1", distribution=dist.Bernoulli)
    factors = [
        ParametricCPD(x, parametrization=LearnablePrior(4)),
        ParametricCPD(concepts, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(4, 3), nn.Sigmoid())),
        ParametricCPD(y, parents=[concepts],
                      parametrization=nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())),
        ParametricCPD(y1, parents=[concepts.member("c1")],
                      parametrization=nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())),
    ]
    return BayesianNetwork(variables=[x, concepts, y, y1], factors=factors)


class TestForwardParentResolution:
    def test_plate_parent_owner_key(self):
        pgm = _model()
        B = 5
        out = pgm.factors["y"](parent_values={"concepts": torch.rand(B, 3)})
        assert out["probs"].shape == (B, 1)

    def test_member_parent_owner_key_equals_exact_key(self):
        pgm = _model()
        B = 5
        c = torch.rand(B, 3)
        cpd_y1 = pgm.factors["y1"]
        owner = cpd_y1(parent_values={"concepts": c})          # owner key (new)
        exact = cpd_y1(parent_values={"c1": c[:, 0:1]})        # exact key (old contract)
        assert torch.allclose(owner["probs"], exact["probs"])

    def test_superset_cache_ignores_extra_keys(self):
        pgm = _model()
        B = 5
        c = torch.rand(B, 3)
        cpd_y1 = pgm.factors["y1"]
        base = cpd_y1(parent_values={"concepts": c})
        superset = cpd_y1(parent_values={"concepts": c, "x": torch.randn(B, 4), "y": torch.rand(B, 1)})
        assert torch.allclose(superset["probs"], base["probs"])

    def test_ordinary_parent_exact_key(self):
        pgm = _model()
        B = 5
        out = pgm.factors["concepts"](parent_values={"x": torch.randn(B, 4)})
        assert out["probs"].shape == (B, 3)

    def test_missing_parent_raises_keyerror_with_message(self):
        pgm = _model()
        cpd_y1 = pgm.factors["y1"]
        with pytest.raises(KeyError) as exc:
            cpd_y1(parent_values={"x": torch.randn(2, 4)})
        msg = str(exc.value)
        assert "y1" in msg and "c1" in msg and "concepts" in msg


class TestExtractHelpers:
    def test_extract_member_is_column(self):
        pgm = _model()
        c = torch.rand(6, 3)
        assert torch.allclose(pgm.extract("c2", {"concepts": c}), c[:, 1:2])

    def test_extract_whole_variable(self):
        pgm = _model()
        c = torch.rand(6, 3)
        assert torch.equal(pgm.extract("concepts", {"concepts": c}), c)

    def test_extract_ordinary_variable(self):
        pgm = _model()
        v = torch.randn(6, 4)
        assert torch.equal(pgm.extract("x", {"x": v}), v)

    def test_extract_params_member(self):
        pgm = _model()
        c = torch.rand(6, 3)
        got = pgm.extract_params("c3", {"concepts": {"probs": c}})
        assert torch.allclose(got["probs"], c[:, 2:3])

    def test_extract_returns_view(self):
        pgm = _model()
        c = torch.rand(6, 3)
        view = pgm.extract("c2", {"concepts": c})
        assert view.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()
