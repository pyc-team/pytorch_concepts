"""Cosine of centered cubes, including gradients and degenerate columns."""
import pytest
import torch
import torch.nn.functional as F

from torch_concepts.nn import CosineCubedLoss
from torch_concepts.nn.functional import cosine_cubed_similarity


def test_reference_formula_and_loss():
    rng = torch.Generator().manual_seed(2)
    x = torch.randn(12, 3, generator=rng)
    y = torch.randn(12, 3, generator=rng)
    expected = F.cosine_similarity((x - x.mean(0)) ** 3, (y - y.mean(0)) ** 3, dim=0)
    actual = cosine_cubed_similarity(x, y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(CosineCubedLoss()(x, y), -expected.mean())


def test_alignment_identities_and_constant_gradients():
    x = torch.tensor([[1., 5.], [2., 5.], [4., 5.]], requires_grad=True)
    torch.testing.assert_close(cosine_cubed_similarity(x, x), torch.tensor([1., 0.]))
    torch.testing.assert_close(cosine_cubed_similarity(x, -x), torch.tensor([-1., 0.]))
    CosineCubedLoss()(x, x.detach()).backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[:, 1] == 0).all()


def test_double_precision_gradients_and_scale_invariance():
    rng = torch.Generator().manual_seed(6)
    x = torch.randn(9, 2, generator=rng, dtype=torch.float64, requires_grad=True)
    y = torch.randn(9, 2, generator=rng, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(cosine_cubed_similarity, (x, y))
    # Cubing unscaled float32 activations of this magnitude would overflow.
    torch.testing.assert_close(cosine_cubed_similarity(x.float() * 1e15, y.float()),
                               cosine_cubed_similarity(x.float(), y.float()))


@pytest.mark.parametrize('x,y', [
    (torch.zeros(3), torch.zeros(3)), (torch.zeros(3, 2), torch.zeros(3, 1)),
    (torch.zeros(0, 2), torch.zeros(0, 2)), (torch.zeros(3, 0), torch.zeros(3, 0)),
    (torch.zeros(3, 2, dtype=torch.long), torch.zeros(3, 2)),
])
def test_invalid_alignment_inputs(x, y):
    with pytest.raises(ValueError):
        cosine_cubed_similarity(x, y)
