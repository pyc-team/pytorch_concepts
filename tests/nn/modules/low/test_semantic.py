"""
Comprehensive tests for torch_concepts.semantic

Tests all semantic operations and t-norms.
"""
import unittest
import torch
from torch_concepts.nn.modules.low.semantic import (
    Semantic,
    CMRSemantic,
    ProductTNorm,
    GodelTNorm
)


class TestCMRSemantic(unittest.TestCase):
    """Test CMR Semantic operations."""

    def setUp(self):
        """Set up semantic instance."""
        self.semantic = CMRSemantic()

    def test_conjunction_two_tensors(self):
        """Test conjunction with two tensors."""
        a = torch.tensor([0.5, 0.8, 0.3])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = self.semantic.conj(a, b)
        expected = a * b
        self.assertTrue(torch.allclose(result, expected))

    def test_conjunction_multiple_tensors(self):
        """Test conjunction with multiple tensors."""
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.6, 0.4])
        c = torch.tensor([0.7, 0.9])
        result = self.semantic.conj(a, b, c)
        expected = a * b * c
        self.assertTrue(torch.allclose(result, expected))

    def test_disjunction_two_tensors(self):
        """Test disjunction with two tensors."""
        a = torch.tensor([0.5, 0.8, 0.3])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = self.semantic.disj(a, b)
        expected = a + b
        self.assertTrue(torch.allclose(result, expected))

    def test_disjunction_multiple_tensors(self):
        """Test disjunction with multiple tensors."""
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.6, 0.4])
        c = torch.tensor([0.7, 0.9])
        result = self.semantic.disj(a, b, c)
        expected = a + b + c
        self.assertTrue(torch.allclose(result, expected))

    def test_negation(self):
        """Test negation operation."""
        a = torch.tensor([0.3, 0.7, 0.5, 1.0, 0.0])
        result = self.semantic.neg(a)
        expected = torch.tensor([0.7, 0.3, 0.5, 0.0, 1.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_iff_two_tensors(self):
        """Test biconditional with two tensors."""
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.6, 0.4])
        result = self.semantic.iff(a, b)
        # iff(a, b) = conj(disj(neg(a), b), disj(a, neg(b)))
        expected = self.semantic.conj(
            self.semantic.disj(self.semantic.neg(a), b),
            self.semantic.disj(a, self.semantic.neg(b))
        )
        self.assertTrue(torch.allclose(result, expected))

    def test_iff_multiple_tensors(self):
        """Test biconditional with multiple tensors."""
        a = torch.tensor([0.5])
        b = torch.tensor([0.6])
        c = torch.tensor([0.7])
        result = self.semantic.iff(a, b, c)
        self.assertIsNotNone(result)


class TestProductTNorm(unittest.TestCase):
    """Test Product t-norm operations."""

    def setUp(self):
        """Set up semantic instance."""
        self.semantic = ProductTNorm()

    def test_conjunction_product(self):
        """Test conjunction uses product."""
        a = torch.tensor([0.5, 0.8, 0.3])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = self.semantic.conj(a, b)
        expected = a * b
        self.assertTrue(torch.allclose(result, expected))

    def test_conjunction_multiple(self):
        """Test conjunction with multiple tensors."""
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.6, 0.4])
        c = torch.tensor([0.7, 0.9])
        result = self.semantic.conj(a, b, c)
        expected = a * b * c
        self.assertTrue(torch.allclose(result, expected))

    def test_disjunction_probabilistic_sum(self):
        """Test disjunction uses probabilistic sum: a + b - a*b."""
        a = torch.tensor([0.5, 0.8, 0.3])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = self.semantic.disj(a, b)
        expected = a + b - a * b
        self.assertTrue(torch.allclose(result, expected))

    def test_disjunction_multiple(self):
        """Test disjunction with multiple tensors."""
        a = torch.tensor([0.3, 0.5])
        b = torch.tensor([0.4, 0.6])
        c = torch.tensor([0.2, 0.7])
        result = self.semantic.disj(a, b, c)
        # Should apply probabilistic sum iteratively
        temp = a + b - a * b
        expected = temp + c - temp * c
        self.assertTrue(torch.allclose(result, expected))

    def test_negation(self):
        """Test negation operation."""
        a = torch.tensor([0.3, 0.7, 0.5, 1.0, 0.0])
        result = self.semantic.neg(a)
        expected = torch.tensor([0.7, 0.3, 0.5, 0.0, 1.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_iff_operation(self):
        """Test biconditional operation."""
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.6, 0.4])
        result = self.semantic.iff(a, b)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, a.shape)

    def test_boundary_values(self):
        """Test with boundary values 0 and 1."""
        a = torch.tensor([0.0, 1.0, 0.0, 1.0])
        b = torch.tensor([0.0, 0.0, 1.0, 1.0])

        conj_result = self.semantic.conj(a, b)
        self.assertTrue(torch.allclose(conj_result, torch.tensor([0.0, 0.0, 0.0, 1.0])))

        disj_result = self.semantic.disj(a, b)
        self.assertTrue(torch.allclose(disj_result, torch.tensor([0.0, 1.0, 1.0, 1.0])))


class TestGodelTNorm(unittest.TestCase):
    """Test Gödel t-norm operations."""

    def setUp(self):
        """Set up semantic instance."""
        self.semantic = GodelTNorm()

    def test_conjunction_minimum(self):
        """Test conjunction uses minimum."""
        a = torch.tensor([0.5, 0.8, 0.3])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = self.semantic.conj(a, b)
        expected = torch.tensor([0.5, 0.4, 0.3])
        self.assertTrue(torch.allclose(result, expected))

    def test_conjunction_multiple(self):
        """Test conjunction with multiple tensors."""
        a = torch.tensor([0.5, 0.8, 0.9])
        b = torch.tensor([0.6, 0.4, 0.7])
        c = torch.tensor([0.7, 0.9, 0.3])
        result = self.semantic.conj(a, b, c)
        expected = torch.tensor([0.5, 0.4, 0.3])
        self.assertTrue(torch.allclose(result, expected))

    def test_disjunction_maximum(self):
        """Test disjunction uses maximum."""
        a = torch.tensor([0.5, 0.8, 0.3])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = self.semantic.disj(a, b)
        expected = torch.tensor([0.6, 0.8, 0.9])
        self.assertTrue(torch.allclose(result, expected))

    def test_disjunction_multiple(self):
        """Test disjunction with multiple tensors."""
        a = torch.tensor([0.5, 0.8, 0.9])
        b = torch.tensor([0.6, 0.4, 0.7])
        c = torch.tensor([0.7, 0.9, 0.3])
        result = self.semantic.disj(a, b, c)
        expected = torch.tensor([0.7, 0.9, 0.9])
        self.assertTrue(torch.allclose(result, expected))

    def test_negation(self):
        """Test negation operation."""
        a = torch.tensor([0.3, 0.7, 0.5, 1.0, 0.0])
        result = self.semantic.neg(a)
        expected = torch.tensor([0.7, 0.3, 0.5, 0.0, 1.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_iff_operation(self):
        """Test biconditional operation."""
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.6, 0.4])
        result = self.semantic.iff(a, b)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, a.shape)

    def test_boundary_values(self):
        """Test with boundary values 0 and 1."""
        a = torch.tensor([0.0, 1.0, 0.0, 1.0])
        b = torch.tensor([0.0, 0.0, 1.0, 1.0])

        conj_result = self.semantic.conj(a, b)
        self.assertTrue(torch.allclose(conj_result, torch.tensor([0.0, 0.0, 0.0, 1.0])))

        disj_result = self.semantic.disj(a, b)
        self.assertTrue(torch.allclose(disj_result, torch.tensor([0.0, 1.0, 1.0, 1.0])))

    def test_idempotency(self):
        """Test idempotency property for Gödel t-norm."""
        a = torch.tensor([0.3, 0.7, 0.5])
        # For Gödel: conj(a, a) = a and disj(a, a) = a
        conj_result = self.semantic.conj(a, a)
        disj_result = self.semantic.disj(a, a)
        self.assertTrue(torch.allclose(conj_result, a))
        self.assertTrue(torch.allclose(disj_result, a))


class TestSemanticGradients(unittest.TestCase):
    """Test gradient flow through semantic operations."""

    def test_cmr_gradient_flow(self):
        """Test gradients flow through CMR semantic."""
        semantic = CMRSemantic()
        a = torch.tensor([0.5, 0.8], requires_grad=True)
        b = torch.tensor([0.6, 0.4], requires_grad=True)

        result = semantic.conj(a, b)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

    def test_product_tnorm_gradient_flow(self):
        """Test gradients flow through Product t-norm."""
        semantic = ProductTNorm()
        a = torch.tensor([0.5, 0.8], requires_grad=True)
        b = torch.tensor([0.6, 0.4], requires_grad=True)

        result = semantic.disj(a, b)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

    def test_godel_tnorm_gradient_flow(self):
        """Test gradients flow through Gödel t-norm."""
        semantic = GodelTNorm()
        a = torch.tensor([0.5, 0.8], requires_grad=True)
        b = torch.tensor([0.6, 0.4], requires_grad=True)

        result = semantic.conj(a, b)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)


class TestSemanticBatchOperations(unittest.TestCase):
    """Test semantic operations with batched tensors."""

    def test_cmr_batch_operations(self):
        """Test CMR semantic with batched tensors."""
        semantic = CMRSemantic()
        a = torch.rand(4, 5)
        b = torch.rand(4, 5)

        conj_result = semantic.conj(a, b)
        disj_result = semantic.disj(a, b)
        neg_result = semantic.neg(a)

        self.assertEqual(conj_result.shape, (4, 5))
        self.assertEqual(disj_result.shape, (4, 5))
        self.assertEqual(neg_result.shape, (4, 5))

    def test_product_tnorm_batch_operations(self):
        """Test Product t-norm with batched tensors."""
        semantic = ProductTNorm()
        a = torch.rand(3, 7)
        b = torch.rand(3, 7)

        conj_result = semantic.conj(a, b)
        disj_result = semantic.disj(a, b)

        self.assertEqual(conj_result.shape, (3, 7))
        self.assertEqual(disj_result.shape, (3, 7))

    def test_godel_tnorm_batch_operations(self):
        """Test Gödel t-norm with batched tensors."""
        semantic = GodelTNorm()
        a = torch.rand(2, 10)
        b = torch.rand(2, 10)

        conj_result = semantic.conj(a, b)
        disj_result = semantic.disj(a, b)

        self.assertEqual(conj_result.shape, (2, 10))
        self.assertEqual(disj_result.shape, (2, 10))


if __name__ == '__main__':
    unittest.main()

