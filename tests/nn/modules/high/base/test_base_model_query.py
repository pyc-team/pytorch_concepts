"""
Comprehensive tests for BaseModel's fully-observed query machinery:
``_query_plan``, ``_query_segments``, and ``fully_observed_query``.

These target the type-agnostic rewrite that lets a concept variable (plate or
single concept) mix binary, continuous, and categorical members: cardinality-1
members (binary/continuous) are gathered directly in one vectorized op, while
cardinality>1 members (categorical) are one-hot encoded individually. Coverage
includes pure binary, pure continuous, pure categorical, arbitrary mixes (run-
length-encoding boundaries), multiple concept variables, non-concept variables
being excluded, cached_property caching, and both AnnotatedTensor and plain
torch.Tensor ground_truth inputs (a regression the rewrite introduced).
"""
from dataclasses import dataclass
from typing import List

import pytest
import torch

from torch_concepts.annotations import Annotations
from torch_concepts.nn.modules.high.base.model import BaseModel
from torch_concepts.tensor import AnnotatedTensor


class ConcreteModel(BaseModel):
    """Minimal concrete BaseModel subclass (forward is unused by these tests)."""

    def forward(self, x, query=None):
        return self.backbone(x)


@dataclass
class _FakeVariable:
    """Minimal stand-in for a PGM Variable: only the attributes _query_plan reads."""
    name: str
    members: List[str]
    variable_type: str = "concept"


class _FakePGM:
    """Minimal stand-in for a BayesianNetwork: only ``.variables`` is read."""

    def __init__(self, variables):
        self.variables = {v.name: v for v in variables}


def make_model(annotations, variables):
    model = ConcreteModel(input_size=10, annotations=annotations)
    model.pgm = _FakePGM(variables)
    return model


class TestQueryPlan:
    """_query_plan: per-concept-variable (name, [(index, cardinality), ...])."""

    def test_single_plate_all_binary(self):
        annotations = Annotations(
            labels=['c0', 'c1', 'c2'], cardinalities=[1, 1, 1],
            types=['binary'] * 3,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['c0', 'c1', 'c2'])])
        assert model._query_plan == [('concepts', [(0, 1), (1, 1), (2, 1)])]

    def test_cardinality_reflects_categorical_member(self):
        annotations = Annotations(labels=['color'], cardinalities=[3], types=['categorical'])
        model = make_model(annotations, [_FakeVariable('color', ['color'])])
        assert model._query_plan == [('color', [(0, 3)])]

    def test_excludes_non_concept_variables(self):
        annotations = Annotations(labels=['c0'], cardinalities=[1], types=['binary'])
        variables = [
            _FakeVariable('concepts', ['c0'], variable_type='concept'),
            _FakeVariable('latent', [], variable_type='embedding'),
        ]
        model = make_model(annotations, variables)
        names = [name for name, _ in model._query_plan]
        assert names == ['concepts']

    def test_is_cached_across_accesses(self):
        annotations = Annotations(labels=['c0'], cardinalities=[1], types=['binary'])
        model = make_model(annotations, [_FakeVariable('concepts', ['c0'])])
        plan1 = model._query_plan
        plan2 = model._query_plan
        assert plan1 is plan2


class TestQuerySegments:
    """_query_segments: run-length-encoded plain/one-hot gather plan."""

    def test_all_binary_collapses_to_one_plain_segment(self):
        annotations = Annotations(
            labels=['c0', 'c1', 'c2'], cardinalities=[1, 1, 1],
            types=['binary'] * 3,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['c0', 'c1', 'c2'])])
        segments = model._query_segments['concepts']

        assert len(segments) == 1
        kind, payload = segments[0]
        assert kind == 'plain'
        assert torch.equal(payload, torch.tensor([0, 1, 2]))

    def test_all_continuous_collapses_to_one_plain_segment(self):
        annotations = Annotations(
            labels=['x0', 'x1'], cardinalities=[1, 1], types=['continuous'] * 2,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['x0', 'x1'])])
        segments = model._query_segments['concepts']

        assert len(segments) == 1
        assert segments[0][0] == 'plain'
        assert torch.equal(segments[0][1], torch.tensor([0, 1]))

    def test_binary_and_continuous_mix_still_one_plain_segment(self):
        """Cardinality (not the type label) drives the fast path: binary and
        continuous members interleaved should still collapse into ONE gather,
        exactly like the pure-binary and pure-continuous cases."""
        annotations = Annotations(
            labels=['b0', 'x0', 'b1'], cardinalities=[1, 1, 1],
            types=['binary', 'continuous', 'binary'],
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['b0', 'x0', 'b1'])])
        segments = model._query_segments['concepts']

        assert len(segments) == 1
        assert segments[0][0] == 'plain'
        assert torch.equal(segments[0][1], torch.tensor([0, 1, 2]))

    def test_all_categorical_each_gets_its_own_onehot_segment(self):
        annotations = Annotations(
            labels=['c0', 'c1'], cardinalities=[3, 4], types=['categorical'] * 2,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['c0', 'c1'])])
        segments = model._query_segments['concepts']

        assert segments == [('onehot', (0, 3)), ('onehot', (1, 4))]

    def test_mixed_run_length_encoding(self):
        """b0,b1 (binary run) | cat (categorical) | x0,x1 (continuous run) | cat2"""
        annotations = Annotations(
            labels=['b0', 'b1', 'cat', 'x0', 'x1', 'cat2'],
            cardinalities=[1, 1, 3, 1, 1, 2],
            types=['binary', 'binary', 'categorical', 'continuous', 'continuous', 'categorical'],
        )
        model = make_model(
            annotations,
            [_FakeVariable('concepts', ['b0', 'b1', 'cat', 'x0', 'x1', 'cat2'])],
        )
        segments = model._query_segments['concepts']

        assert len(segments) == 4
        assert segments[0][0] == 'plain'
        assert torch.equal(segments[0][1], torch.tensor([0, 1]))
        assert segments[1] == ('onehot', (2, 3))
        assert segments[2][0] == 'plain'
        assert torch.equal(segments[2][1], torch.tensor([3, 4]))
        assert segments[3] == ('onehot', (5, 2))

    def test_leading_and_trailing_categorical_no_empty_runs(self):
        """Categorical first and last member: no stray empty 'plain' segment."""
        annotations = Annotations(
            labels=['cat0', 'b0', 'cat1'], cardinalities=[3, 1, 2],
            types=['categorical', 'binary', 'categorical'],
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['cat0', 'b0', 'cat1'])])
        segments = model._query_segments['concepts']

        assert len(segments) == 3
        assert segments[0] == ('onehot', (0, 3))
        assert segments[1][0] == 'plain'
        assert torch.equal(segments[1][1], torch.tensor([1]))
        assert segments[2] == ('onehot', (2, 2))

    def test_is_cached_across_accesses(self):
        annotations = Annotations(labels=['c0'], cardinalities=[1], types=['binary'])
        model = make_model(annotations, [_FakeVariable('concepts', ['c0'])])
        segs1 = model._query_segments
        segs2 = model._query_segments
        assert segs1 is segs2


class TestFullyObservedQuery:
    """fully_observed_query: end-to-end teacher-forcing tensor assembly."""

    def test_all_binary_values_match_raw_columns(self):
        annotations = Annotations(
            labels=['c0', 'c1', 'c2'], cardinalities=[1, 1, 1], types=['binary'] * 3,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['c0', 'c1', 'c2'])])
        gt = torch.tensor([[0., 1., 1.], [1., 0., 0.]])

        query = model.fully_observed_query(gt)

        assert set(query.keys()) == {'concepts'}
        assert torch.equal(query['concepts'], gt)

    def test_all_continuous_values_match_raw_columns(self):
        annotations = Annotations(
            labels=['x0', 'x1'], cardinalities=[1, 1], types=['continuous'] * 2,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['x0', 'x1'])])
        gt = torch.tensor([[0.3, -1.2], [2.5, 0.0]])

        query = model.fully_observed_query(gt)

        assert torch.allclose(query['concepts'], gt)

    def test_categorical_is_one_hot_encoded(self):
        annotations = Annotations(labels=['color'], cardinalities=[3], types=['categorical'])
        model = make_model(annotations, [_FakeVariable('color', ['color'])])
        gt = torch.tensor([[0.], [2.], [1.]])

        query = model.fully_observed_query(gt)

        expected = torch.tensor([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        assert torch.equal(query['color'], expected)

    def test_mixed_binary_categorical_continuous(self):
        annotations = Annotations(
            labels=['b0', 'cat', 'x0'], cardinalities=[1, 3, 1],
            types=['binary', 'categorical', 'continuous'],
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['b0', 'cat', 'x0'])])
        gt = torch.tensor([
            [1., 2., 0.5],
            [0., 0., -1.5],
        ])

        query = model.fully_observed_query(gt)

        expected = torch.tensor([
            [1., 0., 0., 1., 0.5],
            [0., 1., 0., 0., -1.5],
        ])
        assert torch.equal(query['concepts'], expected)

    def test_multiple_concept_variables_each_get_their_own_slice(self):
        annotations = Annotations(
            labels=['c0', 'c1', 't0'], cardinalities=[1, 1, 1],
            types=['binary', 'binary', 'binary'],
        )
        variables = [
            _FakeVariable('concepts', ['c0', 'c1']),
            _FakeVariable('tasks', ['t0']),
        ]
        model = make_model(annotations, variables)
        gt = torch.tensor([[1., 0., 1.], [0., 1., 0.]])

        query = model.fully_observed_query(gt)

        assert set(query.keys()) == {'concepts', 'tasks'}
        assert torch.equal(query['concepts'], gt[:, :2])
        assert torch.equal(query['tasks'], gt[:, 2:3])

    def test_accepts_annotated_tensor(self):
        annotations = Annotations(
            labels=['c0', 'c1'], cardinalities=[1, 1], types=['binary'] * 2,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['c0', 'c1'])])
        raw = torch.tensor([[1., 0.], [0., 1.]])
        gt = AnnotatedTensor(raw, annotations, axis=1)

        query = model.fully_observed_query(gt)

        assert torch.equal(query['concepts'], raw)

    def test_accepts_plain_tensor_not_just_annotated_tensor(self):
        """Regression guard: ground_truth need not be an AnnotatedTensor."""
        annotations = Annotations(
            labels=['c0', 'c1'], cardinalities=[1, 1], types=['binary'] * 2,
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['c0', 'c1'])])
        gt = torch.tensor([[1., 0.], [0., 1.]])
        assert not isinstance(gt, AnnotatedTensor)

        query = model.fully_observed_query(gt)

        assert torch.equal(query['concepts'], gt)

    def test_annotated_tensor_and_plain_tensor_give_identical_results(self):
        """Same underlying data, wrapped vs. unwrapped, must produce identical
        queries -- guards the isinstance(ground_truth, AnnotatedTensor) branch."""
        annotations = Annotations(
            labels=['b0', 'cat', 'x0'], cardinalities=[1, 3, 1],
            types=['binary', 'categorical', 'continuous'],
        )
        model = make_model(annotations, [_FakeVariable('concepts', ['b0', 'cat', 'x0'])])
        raw = torch.tensor([[1., 2., 0.5], [0., 0., -1.5]])

        # Ground truth is integer-coded (one raw column per label), so it must
        # be wrapped with the concept-space annotation (cardinality 1 per
        # label) -- matching how callers (e.g. BaseLearner.prepare_target)
        # annotate raw targets in practice.
        wrapped = AnnotatedTensor(raw, annotations.to_concept_space(), axis=1)

        query_plain = model.fully_observed_query(raw)
        query_annotated = model.fully_observed_query(wrapped)

        assert torch.equal(query_plain['concepts'], query_annotated['concepts'])
