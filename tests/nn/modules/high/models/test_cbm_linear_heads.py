"""Installing external parameters without reaching into a model's graph."""
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from torch_concepts import Annotations
from torch_concepts.distributions import Delta
from torch_concepts.nn import ConceptBottleneckModel, ConceptEmbeddingModel


@pytest.fixture
def model():
    # An interleaved task and non-alphabetical labels expose ordering mistakes.
    ann = Annotations(labels=['zeta', 'kind', 'alpha'],
                      states=[['value'], ['last', 'first', 'middle'], ['value']],
                      types=['continuous', 'categorical', 'continuous'])
    return ConceptBottleneckModel(4, ann, ['kind'], variable_distributions={'continuous': Delta})


@pytest.fixture
def heads():
    return dict(concept_weight=torch.tensor([[1., 2., -1., .5], [3., -2., 1., 4.]]),
                concept_bias=torch.tensor([2., -3.]),
                task_weight=torch.tensor([[1., 5.], [3., -2.], [-4., 2.]]),
                task_bias=torch.tensor([.1, -.2, .7]))


@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('with_bias', [False, True])
def test_matches_explicit_equation_and_annotation_order(model, heads, normalize, with_bias):
    rng = torch.Generator().manual_seed(4)
    x = torch.randn(7, 4, generator=rng)
    if not with_bias:
        heads.pop('concept_bias')
        heads.pop('task_bias')
    stats = dict(concept_mean=torch.tensor([[3., -2.]]), concept_std=torch.tensor([[2., 4.]])) if normalize else {}
    model.eval().set_linear_heads(**heads, **stats)
    c = F.linear(x, heads['concept_weight'], heads.get('concept_bias'))
    if normalize:
        c = (c - stats['concept_mean']) / stats['concept_std']
    expected = F.linear(c, heads['task_weight'], heads.get('task_bias'))
    out = model(query=['zeta', 'kind', 'alpha'], input=x)
    torch.testing.assert_close(out.value[['zeta', 'alpha']].tensor, c)
    torch.testing.assert_close(out.logits['kind'].tensor, expected)
    # Named selection reorders values with their metadata.
    reordered = out.value[['alpha', 'zeta']]
    assert reordered.annotations.labels == ['alpha', 'zeta']
    torch.testing.assert_close(reordered.tensor, c[:, [1, 0]])
    assert model.intermediate_concept_names == ['zeta', 'alpha']
    assert model.concept_annotations.concept('kind').states == ['last', 'first', 'middle']
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('name,invalid', [
    ('concept_weight', torch.zeros(2, 3)), ('concept_bias', torch.zeros(3)),
    ('task_weight', torch.zeros(2, 3)), ('task_bias', torch.zeros(2)),
    ('task_bias', torch.full((3,), float('nan'))),
    ('concept_weight', torch.full((2, 4), float('inf'))),
    ('task_weight', torch.zeros(3, 2, dtype=torch.complex64)),
])
def test_invalid_heads_leave_model_unchanged(model, heads, name, invalid):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    heads[name] = invalid
    with pytest.raises(ValueError, match=name):
        model.set_linear_heads(**heads)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@pytest.mark.parametrize('stats', [
    {'concept_mean': torch.zeros(2)},
    {'concept_std': torch.ones(2)},
    {'concept_mean': torch.zeros(3), 'concept_std': torch.ones(2)},
    {'concept_mean': torch.zeros(2), 'concept_std': torch.tensor([0., 1.])},
    {'concept_mean': torch.zeros(2), 'concept_std': torch.tensor([-1., 1.])},
    {'concept_mean': torch.zeros(2), 'concept_std': torch.tensor([float('nan'), 1.])},
    {'concept_mean': torch.zeros(2), 'concept_std': torch.tensor([1e-45, 1.])},
])
def test_invalid_normalization_is_atomic(model, heads, stats):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        model.set_linear_heads(**heads, **stats)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [{'plate': False}, {'variable_distributions': {'continuous': Normal}}])
def test_unsupported_cbm_architectures(model, heads, kwargs):
    settings = dict(variable_distributions={'continuous': Delta})
    settings.update(kwargs)
    other = ConceptBottleneckModel(4, model.concept_annotations, ['kind'], **settings)
    with pytest.raises(ValueError, match='Linear heads require'):
        other.set_linear_heads(**heads)


def test_embedding_architecture_rejected(model, heads):
    other = ConceptEmbeddingModel(4, model.concept_annotations, ['kind'], embedding_size=2,
                                  variable_distributions={'continuous': Delta})
    # CEM is not a subclass of CBM and has no head-installation API.
    assert not hasattr(other, 'set_linear_heads')


def test_parameters_are_copied_and_dtype_converted(model, heads):
    model.double().set_linear_heads(**heads)
    x = torch.ones(2, 4, dtype=torch.float64)
    expected = model(query=['kind'], input=x).logits['kind'].tensor.clone()
    heads['concept_weight'].zero_()
    torch.testing.assert_close(model(query=['kind'], input=x).logits['kind'].tensor, expected)
