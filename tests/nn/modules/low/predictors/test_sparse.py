"""The real optional solver, plus its absence and input contract."""
import sys

import pytest
import torch

from torch_concepts import Annotations
from torch_concepts.nn import SparseLinearConceptToConcept


@pytest.fixture
def multiclass_data():
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(90, 5, generator=rng)
    y = x[:, :3].argmax(1)
    x = (x - x.mean(0)) / x.std(0)
    return x, y


def test_real_solver_fits_multiclass_and_freezes(multiclass_data):
    pytest.importorskip('glm_saga.elasticnet')
    x, y = multiclass_data
    classes = Annotations(labels=['kind'], states=[['third', 'first', 'second']])
    classifier = SparseLinearConceptToConcept(5, classes).fit(
        x[:60], y[:60], x[60:], y[60:], epochs=100, batch_size=20, lr=.2, lam=.001,
    )
    assert classifier(x).shape == (90, 3)
    assert (classifier(x).argmax(1) == y).float().mean() > .9
    assert classifier.diagnostics_['val_loss'] < 1.1
    assert all(torch.isfinite(p).all() and not p.requires_grad for p in classifier.parameters())
    assert not classifier.training


def test_l1_penalty_induces_sparsity(multiclass_data):
    pytest.importorskip('glm_saga.elasticnet')
    x, y = multiclass_data
    low = SparseLinearConceptToConcept(5, 3).fit(x, y, epochs=60, lr=.2, lam=0, alpha=1)
    high = SparseLinearConceptToConcept(5, 3).fit(x, y, epochs=60, lr=.2, lam=1, alpha=1)
    assert low.predictor.weight.abs().sum() > 1
    assert torch.count_nonzero(high.predictor.weight) == 0
    assert high.diagnostics_['regularization'] == 1
    assert high.diagnostics_['alpha'] == 1


def test_real_solver_determinism(multiclass_data):
    pytest.importorskip('glm_saga.elasticnet')
    x, y = multiclass_data
    a = SparseLinearConceptToConcept(5, 3).fit(x, y, epochs=10, batch_size=20, seed=8, alpha=0)
    b = SparseLinearConceptToConcept(5, 3).fit(x, y, epochs=10, batch_size=20, seed=8, alpha=0)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


def test_missing_dependency_has_actionable_error(monkeypatch, multiclass_data):
    monkeypatch.setitem(sys.modules, 'glm_saga', None)
    monkeypatch.delitem(sys.modules, 'glm_saga.elasticnet', raising=False)
    classifier = SparseLinearConceptToConcept(5, 3)
    assert classifier(multiclass_data[0]).shape == (90, 3)
    with pytest.raises(ImportError, match=r'pytorch_concepts\[sparse\]'):
        classifier.fit(*multiclass_data)


@pytest.mark.parametrize('kwargs', [
    {'alpha': -1}, {'alpha': 2}, {'lam': -1}, {'lam': float('nan')},
    {'epochs': 0}, {'batch_size': 0}, {'lr': float('inf')}, {'tolerance': -1},
    {'val_concepts': torch.zeros(2, 5)},
])
def test_invalid_solver_settings(multiclass_data, kwargs):
    pytest.importorskip('glm_saga.elasticnet')
    with pytest.raises(ValueError):
        SparseLinearConceptToConcept(5, 3).fit(*multiclass_data, **kwargs)


@pytest.mark.parametrize('x,y', [
    (torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
    (torch.zeros(4, 5), torch.zeros(4)),
    (torch.zeros(4, 5), torch.full((4,), 3, dtype=torch.long)),
    (torch.full((4, 5), float('nan')), torch.zeros(4, dtype=torch.long)),
])
def test_invalid_solver_data(x, y):
    pytest.importorskip('glm_saga.elasticnet')
    with pytest.raises(ValueError):
        SparseLinearConceptToConcept(5, 3).fit(x, y)


def test_pinned_solver_supports_double_precision_modules(multiclass_data):
    pytest.importorskip('glm_saga.elasticnet')
    x, y = multiclass_data
    model = SparseLinearConceptToConcept(5, 3).double().fit(x.double(), y, epochs=20)
    assert model(x.double()).dtype == torch.float64
    assert all(torch.isfinite(p).all() and not p.requires_grad for p in model.parameters())
