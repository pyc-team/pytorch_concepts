"""
Comprehensive tests for torch_concepts.nn.modules.low.graph

Tests graph learning modules (WANDA).
"""
import unittest
import torch
from torch_concepts.nn.modules.low.graph.wanda import WANDAGraphLearner


class TestWANDAGraphLearner(unittest.TestCase):
    """Test WANDAGraphLearner."""

    def test_initialization(self):
        """Test WANDA graph learner initialization."""
        concepts = ['c1', 'c2', 'c3', 'c4', 'c5']
        wanda = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts,
            priority_var=1.0,
            hard_threshold=True
        )
        self.assertEqual(wanda.n_labels, 5)
        self.assertEqual(wanda.priority_var, 1.0 / (2 ** 0.5))
        self.assertTrue(wanda.hard_threshold)

    def test_weighted_adj_shape(self):
        """Test weighted adjacency matrix shape."""
        concepts = ['c1', 'c2', 'c3']
        wanda = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts
        )
        adj_matrix = wanda.weighted_adj
        self.assertEqual(adj_matrix.shape, (3, 3))

    def test_acyclic_property(self):
        """Test that learned graph is acyclic."""
        concepts = ['c1', 'c2', 'c3', 'c4']
        wanda = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts
        )
        adj_matrix = wanda.weighted_adj

        # Check diagonal is zero (no self-loops)
        diagonal = torch.diag(adj_matrix)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))

    def test_soft_vs_hard_threshold(self):
        """Test soft vs hard thresholding."""
        concepts = ['c1', 'c2', 'c3']

        wanda_hard = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts,
            hard_threshold=True
        )

        wanda_soft = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts,
            hard_threshold=False
        )

        adj_hard = wanda_hard.weighted_adj
        adj_soft = wanda_soft.weighted_adj

        self.assertEqual(adj_hard.shape, adj_soft.shape)

    def test_gradient_flow(self):
        """Test gradient flow through graph learner."""
        concepts = ['c1', 'c2', 'c3']
        wanda = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts,
            hard_threshold=True
        )

        adj_matrix = wanda.weighted_adj
        loss = adj_matrix.sum()
        loss.backward()

        # Check that np_params has gradients (threshold doesn't get gradients with hard thresholding)
        self.assertIsNotNone(wanda.np_params.grad)

    def test_gradient_flow_soft_threshold(self):
        """Test gradient flow through graph learner with soft thresholding."""
        concepts = ['c1', 'c2', 'c3']
        wanda = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts,
            hard_threshold=False
        )

        adj_matrix = wanda.weighted_adj
        loss = adj_matrix.sum()
        loss.backward()

        # With soft thresholding, both parameters should receive gradients
        self.assertIsNotNone(wanda.np_params.grad)

    def test_priority_parameters(self):
        """Test priority parameter properties."""
        concepts = ['c1', 'c2', 'c3', 'c4']
        wanda = WANDAGraphLearner(
            row_labels=concepts,
            col_labels=concepts,
            priority_var=2.0
        )

        # Priority params should be learnable
        self.assertTrue(wanda.np_params.requires_grad)
        self.assertEqual(wanda.np_params.shape, (4, 1))

    def test_different_row_col_labels(self):
        """Test with different row and column labels - should fail since they must be equal."""
        row_concepts = ['c1', 'c2', 'c3']
        col_concepts = ['c1', 'c2']  # Different length

        # WANDA requires row_labels and col_labels to have same length
        with self.assertRaises(AssertionError):
            WANDAGraphLearner(
                row_labels=row_concepts,
                col_labels=col_concepts
            )


if __name__ == '__main__':
    unittest.main()
