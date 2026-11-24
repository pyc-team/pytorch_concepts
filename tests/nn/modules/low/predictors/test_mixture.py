import torch

from torch_concepts.nn import (
    MixtureOfExpertsConceptToConcept,
    SymbolicConceptToConcept,
)


def _constant_experts():
    experts = [torch.nn.Linear(2, 1) for _ in range(2)]
    for value, expert in zip([1.0, 3.0], experts):
        torch.nn.init.zeros_(expert.weight)
        torch.nn.init.constant_(expert.bias, value)
    return MixtureOfExpertsConceptToConcept(experts)


def test_hard_selector_selects_one_expert():
    layer = _constant_experts()
    out = layer(torch.randn(2, 2), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(out, torch.tensor([[1.0], [3.0]]))


def test_soft_selector_averages_experts():
    layer = _constant_experts()
    out = layer(torch.randn(1, 2), torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(out, torch.tensor([[2.5]]))


def test_mixture_keeps_leading_dims():
    layer = MixtureOfExpertsConceptToConcept([torch.nn.Linear(3, 2)] * 4)
    selector = torch.softmax(torch.randn(5, 7, 4), -1)
    assert layer(torch.randn(5, 7, 3), selector).shape == (5, 7, 2)


def test_symbolic_values_and_broadcast_constant():
    layer = SymbolicConceptToConcept(["2.0*c1 + c2**2", "0"], ["c1", "c2"])
    out = layer(torch.tensor([[1.0, 3.0], [0.0, 2.0]]))
    assert torch.allclose(out, torch.tensor([[11.0, 0.0], [4.0, 0.0]]))
    assert layer(torch.randn(5, 3, 2)).shape == (5, 3, 2)


def test_symbolic_constants_train_but_exponents_do_not():
    layer = SymbolicConceptToConcept(["3*c1 + c2**2.0"], ["c1", "c2"])
    values = sorted(p.item() for p in layer.parameters())
    assert values == [3.0]  # the exponent 2.0 is fixed, not a parameter
    layer(torch.tensor([[1.0, 2.0]])).sum().backward()
    assert all(p.grad is not None for p in layer.parameters())


def test_symbolic_reports_trained_expressions():
    layer = SymbolicConceptToConcept(["2.0*c1"], ["c1"])
    with torch.no_grad():
        next(layer.parameters()).fill_(5.0)
    assert str(layer.sympy()[0]) == "5.0*c1"
