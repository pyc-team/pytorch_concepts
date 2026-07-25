"""
Comprehensive tests for torch_concepts.nn.modules.high.base.bipartite.

Covers the bipartite concept->task graph construction that BipartiteMixin
derives from ``task_names``:
- intermediate_concept_names correctly excludes tasks (order preserved)
- _resolve_graph builds a complete concept->task bipartite adjacency
  (every concept connects to every task, no concept-concept, task-task, or
  task-concept edges)
- correct behavior regardless of where task_names fall in the label order
  (prefix, suffix, interspersed)
- edge count / edge weights are exactly as expected
- missing task_names raise a clear assertion error
"""
import pytest
import torch

from torch_concepts.nn.modules.high.base.bipartite import BipartiteMixin


class _ConcreteBipartite(BipartiteMixin):
    """Minimal concrete class exercising BipartiteMixin in isolation.

    Provides just what BipartiteMixin's _resolve_graph needs
    (``self.concept_names``) without pulling in a full GraphModel/PGM stack.
    """

    def __init__(self, concept_names, task_names):
        self.concept_names = concept_names
        super().__init__(task_names=task_names)


class TestIntermediateConceptNames:
    def test_excludes_single_task_preserves_order(self):
        m = _ConcreteBipartite(concept_names=['a', 'b', 'task1', 'c'], task_names='task1')
        assert m.intermediate_concept_names == ['a', 'b', 'c']

    def test_excludes_multiple_tasks(self):
        m = _ConcreteBipartite(concept_names=['a', 't1', 'b', 't2'], task_names=['t1', 't2'])
        assert m.intermediate_concept_names == ['a', 'b']

    def test_no_tasks_present_returns_all_concepts(self):
        # task_names must be an ensure_list-able value; use a task name absent
        # from concept_names to simulate "no concept is a task" without
        # violating the constructor's required task_names kwarg.
        m = _ConcreteBipartite(concept_names=['a', 'b', 'c'], task_names=['unrelated_task'])
        assert m.intermediate_concept_names == ['a', 'b', 'c']

    def test_string_task_names_is_wrapped_in_list(self):
        m = _ConcreteBipartite(concept_names=['a', 'task1'], task_names='task1')
        assert m.task_names == ['task1']


class TestResolveGraph:
    def test_every_concept_connects_to_every_task(self):
        m = _ConcreteBipartite(concept_names=['c0', 'c1', 't0', 't1'], task_names=['t0', 't1'])
        graph = m._resolve_graph()

        assert graph.node_names == ['c0', 'c1', 't0', 't1']
        for c in ['c0', 'c1']:
            for t in ['t0', 't1']:
                assert graph.has_edge(c, t), f"expected edge {c} -> {t}"

    def test_no_extraneous_edges(self):
        m = _ConcreteBipartite(concept_names=['c0', 'c1', 't0', 't1'], task_names=['t0', 't1'])
        graph = m._resolve_graph()

        # no concept-concept edges
        assert not graph.has_edge('c0', 'c1')
        assert not graph.has_edge('c1', 'c0')
        # no task-task edges
        assert not graph.has_edge('t0', 't1')
        assert not graph.has_edge('t1', 't0')
        # no reverse (task-concept) edges
        assert not graph.has_edge('t0', 'c0')
        assert not graph.has_edge('t1', 'c1')

    def test_edge_count_equals_concepts_times_tasks(self):
        m = _ConcreteBipartite(
            concept_names=['c0', 'c1', 'c2', 't0', 't1'],
            task_names=['t0', 't1'],
        )
        graph = m._resolve_graph()
        assert graph.edge_index.shape[1] == 3 * 2

    def test_edge_weights_all_ones(self):
        m = _ConcreteBipartite(concept_names=['c0', 'c1', 't0'], task_names=['t0'])
        graph = m._resolve_graph()
        assert torch.equal(graph.edge_weight, torch.ones(graph.edge_index.shape[1]))

    def test_single_concept_single_task(self):
        m = _ConcreteBipartite(concept_names=['c0', 't0'], task_names=['t0'])
        graph = m._resolve_graph()
        assert graph.has_edge('c0', 't0')
        assert graph.edge_index.shape[1] == 1

    def test_tasks_interspersed_among_concepts(self):
        """task_names need not be a suffix (or prefix) of the label list."""
        m = _ConcreteBipartite(
            concept_names=['t0', 'c0', 'c1', 't1', 'c2'],
            task_names=['t0', 't1'],
        )
        graph = m._resolve_graph()

        assert graph.node_names == ['t0', 'c0', 'c1', 't1', 'c2']
        for c in ['c0', 'c1', 'c2']:
            for t in ['t0', 't1']:
                assert graph.has_edge(c, t)
        assert graph.edge_index.shape[1] == 3 * 2
        # still no concept-concept / task-task edges
        assert not graph.has_edge('c0', 'c1')
        assert not graph.has_edge('t0', 't1')

    def test_tasks_as_prefix(self):
        m = _ConcreteBipartite(concept_names=['t0', 't1', 'c0', 'c1'], task_names=['t0', 't1'])
        graph = m._resolve_graph()
        for t in ['t0', 't1']:
            for c in ['c0', 'c1']:
                assert graph.has_edge(c, t)
        assert graph.edge_index.shape[1] == 2 * 2

    def test_single_task_many_concepts(self):
        concepts = [f'c{i}' for i in range(5)]
        m = _ConcreteBipartite(concept_names=concepts + ['t0'], task_names=['t0'])
        graph = m._resolve_graph()
        assert graph.edge_index.shape[1] == 5
        for c in concepts:
            assert graph.has_edge(c, 't0')

    def test_missing_task_name_raises_assertion_error(self):
        m = _ConcreteBipartite(concept_names=['c0', 'c1'], task_names=['missing_task'])
        with pytest.raises(AssertionError, match="missing_task"):
            m._resolve_graph()

    def test_partially_missing_task_names_raises(self):
        m = _ConcreteBipartite(concept_names=['c0', 't0'], task_names=['t0', 'missing_task'])
        with pytest.raises(AssertionError, match="missing_task"):
            m._resolve_graph()

    def test_resolve_graph_is_deterministic_across_calls(self):
        """_resolve_graph is called fresh each time (not cached); repeated
        calls on the same instance should yield an identical graph."""
        m = _ConcreteBipartite(concept_names=['c0', 'c1', 't0'], task_names=['t0'])
        g1 = m._resolve_graph()
        g2 = m._resolve_graph()
        assert g1.node_names == g2.node_names
        assert torch.equal(g1.edge_index, g2.edge_index)
        assert torch.equal(g1.edge_weight, g2.edge_weight)
