"""
Tests for seed setting and reproducibility.

This test suite verifies that seed_everything correctly sets seeds for all
random number generators and ensures reproducible results.
"""
import unittest
import os
import torch
import numpy as np
import random

from torch_concepts.utils import seed_everything


class TestSeedEverything(unittest.TestCase):
    """Test suite for seed_everything function."""
    
    def test_seed_returns_value(self):
        """Test that seed_everything returns the seed value."""
        seed = 42
        result = seed_everything(seed)
        self.assertEqual(result, seed, "seed_everything should return the seed value")
    
    def test_python_random_reproducibility(self):
        """Test that Python's random module produces reproducible results."""
        seed = 12345
        
        # First run
        seed_everything(seed)
        random_values_1 = [random.random() for _ in range(10)]
        
        # Second run with same seed
        seed_everything(seed)
        random_values_2 = [random.random() for _ in range(10)]
        
        self.assertEqual(random_values_1, random_values_2,
                        "Python random should produce same values with same seed")
    
    def test_numpy_random_reproducibility(self):
        """Test that NumPy random produces reproducible results."""
        seed = 54321
        
        # First run
        seed_everything(seed)
        np_values_1 = np.random.randn(10)
        
        # Second run with same seed
        seed_everything(seed)
        np_values_2 = np.random.randn(10)
        
        np.testing.assert_array_equal(np_values_1, np_values_2,
                                     "NumPy random should produce same values with same seed")
    
    def test_torch_cpu_reproducibility(self):
        """Test that PyTorch CPU random produces reproducible results."""
        seed = 99999
        
        # First run
        seed_everything(seed)
        torch_values_1 = torch.randn(10)
        
        # Second run with same seed
        seed_everything(seed)
        torch_values_2 = torch.randn(10)
        
        self.assertTrue(torch.equal(torch_values_1, torch_values_2),
                       "PyTorch CPU random should produce same values with same seed")
    
    def test_torch_cuda_reproducibility(self):
        """Test that PyTorch CUDA random produces reproducible results."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        seed = 77777
        
        # First run
        seed_everything(seed)
        torch_cuda_values_1 = torch.randn(10, device='cuda')
        
        # Second run with same seed
        seed_everything(seed)
        torch_cuda_values_2 = torch.randn(10, device='cuda')
        
        self.assertTrue(torch.equal(torch_cuda_values_1, torch_cuda_values_2),
                       "PyTorch CUDA random should produce same values with same seed")
    
    def test_pythonhashseed_environment_variable(self):
        """Test that PYTHONHASHSEED environment variable is set."""
        seed = 33333
        seed_everything(seed)
        
        self.assertIn('PYTHONHASHSEED', os.environ,
                     "PYTHONHASHSEED should be set in environment variables")
        self.assertEqual(os.environ['PYTHONHASHSEED'], str(seed),
                        "PYTHONHASHSEED should match the seed value")
    
    def test_pl_global_seed_environment_variable(self):
        """Test that PL_GLOBAL_SEED environment variable is set by Lightning."""
        seed = 66666
        seed_everything(seed)
        
        self.assertIn('PL_GLOBAL_SEED', os.environ,
                     "PL_GLOBAL_SEED should be set by PyTorch Lightning")
        self.assertEqual(os.environ['PL_GLOBAL_SEED'], str(seed),
                        "PL_GLOBAL_SEED should match the seed value")
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different random values."""
        # First seed
        seed_everything(42)
        torch_values_1 = torch.randn(10)
        np_values_1 = np.random.randn(10)
        random_values_1 = [random.random() for _ in range(10)]
        
        # Different seed
        seed_everything(123)
        torch_values_2 = torch.randn(10)
        np_values_2 = np.random.randn(10)
        random_values_2 = [random.random() for _ in range(10)]
        
        self.assertFalse(torch.equal(torch_values_1, torch_values_2),
                        "Different seeds should produce different PyTorch values")
        self.assertFalse(np.array_equal(np_values_1, np_values_2),
                        "Different seeds should produce different NumPy values")
        self.assertNotEqual(random_values_1, random_values_2,
                           "Different seeds should produce different Python random values")
    
    def test_workers_parameter(self):
        """Test that workers parameter is accepted."""
        seed = 11111
        # Should not raise an error
        result = seed_everything(seed, workers=True)
        self.assertEqual(result, seed)
        
        result = seed_everything(seed, workers=False)
        self.assertEqual(result, seed)
    
    def test_neural_network_reproducibility(self):
        """Test that neural network training is reproducible with same seed."""
        seed = 88888
        
        # Create simple model and data
        def train_step():
            model = torch.nn.Linear(10, 5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)
            
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            
            return loss.item(), model.weight.data.clone()
        
        # First run
        seed_everything(seed)
        loss_1, weights_1 = train_step()
        
        # Second run with same seed
        seed_everything(seed)
        loss_2, weights_2 = train_step()
        
        self.assertAlmostEqual(loss_1, loss_2, places=6,
                              msg="Loss should be identical with same seed")
        self.assertTrue(torch.allclose(weights_1, weights_2, atol=1e-6),
                       "Model weights should be identical with same seed")


if __name__ == '__main__':
    unittest.main()
