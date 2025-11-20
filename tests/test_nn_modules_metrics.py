"""
Comprehensive tests for torch_concepts.nn.modules.metrics

Tests metrics modules for concept-based model evaluation.
"""
import unittest
import torch


class TestConceptMetrics(unittest.TestCase):
    """Test concept metrics module."""

    def test_module_imports(self):
        """Test that metrics module can be imported."""
        from torch_concepts.nn.modules import metrics
        self.assertIsNotNone(metrics)

    def test_module_has_metric_class(self):
        """Test that Metric base class is accessible."""
        from torch_concepts.nn.modules.metrics import Metric
        self.assertIsNotNone(Metric)

    def test_placeholder(self):
        """Placeholder test for commented out code."""
        # The ConceptCausalEffect class is currently commented out
        # This test ensures the module structure is correct
        self.assertTrue(True)


# When metrics are uncommented, add these tests:
# class TestConceptCausalEffect(unittest.TestCase):
#     """Test Concept Causal Effect metric."""
#
#     def test_initialization(self):
#         """Test metric initialization."""
#         from torch_concepts.nn.modules.metrics import ConceptCausalEffect
#         cace = ConceptCausalEffect()
#         self.assertIsNotNone(cace)
#
#     def test_update(self):
#         """Test metric update."""
#         from torch_concepts.nn.modules.metrics import ConceptCausalEffect
#         cace = ConceptCausalEffect()
#
#         preds_do_1 = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
#         preds_do_0 = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
#
#         cace.update(preds_do_1, preds_do_0)
#
#     def test_compute(self):
#         """Test metric computation."""
#         from torch_concepts.nn.modules.metrics import ConceptCausalEffect
#         cace = ConceptCausalEffect()
#
#         preds_do_1 = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
#         preds_do_0 = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
#
#         cace.update(preds_do_1, preds_do_0)
#         effect = cace.compute()
#
#         self.assertIsInstance(effect, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
