import unittest
import torch

from torch_concepts.nn.semantics import ProductTNorm, GodelTNorm


class TestProductTNorm(unittest.TestCase):
    def setUp(self):
        self.model = ProductTNorm()

    def test_conj(self):
        a = torch.tensor([[0.5, 0.8], [0.3, 0.9]])
        result = self.model.conj(a)
        expected = torch.tensor([[0.4], [0.27]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_conj_pair(self):
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.3, 0.9])
        result = self.model.conj_pair(a, b)
        expected = torch.tensor([0.15, 0.72])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_disj(self):
        a = torch.tensor([[0.5, 0.8], [0.3, 0.9]])
        result = self.model.disj(a)
        expected = torch.tensor([[0.9], [0.93]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_disj_pair(self):
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.3, 0.9])
        result = self.model.disj_pair(a, b)
        expected = torch.tensor([0.65, 0.98])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_neg(self):
        a = torch.tensor([0.5, 0.8])
        result = self.model.neg(a)
        expected = torch.tensor([0.5, 0.2])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_iff_pair(self):
        a = torch.tensor([1, 0])
        b = torch.tensor([0, 0])
        result = self.model.iff_pair(a, b)
        expected = torch.tensor([0, 1])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_predict_proba(self):
        a = torch.tensor([[0.5], [0.8]])
        result = self.model.predict_proba(a)
        expected = torch.tensor([0.5, 0.8])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))


class TestGodelTNorm(unittest.TestCase):
    def setUp(self):
        self.model = GodelTNorm()

    def test_conj(self):
        a = torch.tensor([[0.5, 0.8], [0.3, 0.9]])
        result = self.model.conj(a)
        expected = torch.tensor([[0.5], [0.3]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_conj_pair(self):
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.3, 0.9])
        result = self.model.conj_pair(a, b)
        expected = torch.tensor([0.3, 0.8])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_disj(self):
        a = torch.tensor([[0.5, 0.8], [0.3, 0.9]])
        result = self.model.disj(a)
        expected = torch.tensor([[0.8], [0.9]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_disj_pair(self):
        a = torch.tensor([0.5, 0.8])
        b = torch.tensor([0.3, 0.9])
        result = self.model.disj_pair(a, b)
        expected = torch.tensor([0.5, 0.9])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_neg(self):
        a = torch.tensor([0.5, 0.8])
        result = self.model.neg(a)
        expected = torch.tensor([0.5, 0.2])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_iff_pair(self):
        a = torch.tensor([0, 1])
        b = torch.tensor([0, 1])
        result = self.model.iff_pair(a, b)
        expected = torch.tensor([1, 1])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_predict_proba(self):
        a = torch.tensor([[0.5], [0.8]])
        result = self.model.predict_proba(a)
        expected = torch.tensor([0.5, 0.8])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
