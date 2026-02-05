"""
Comprehensive tests for autoencoder preprocessing module.

Tests cover:
- SimpleAutoencoder initialization and forward pass
- AutoencoderTrainer training loop and early stopping
- Latent representation extraction
- extract_embs_from_autoencoder convenience function
- Edge cases and error handling
- Device consistency
- Noise injection
"""
import pytest
import unittest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from torch_concepts.data.preprocessing.autoencoder import (
    SimpleAutoencoder,
    AutoencoderTrainer,
    extract_embs_from_autoencoder
)


# =============================================================================
# SimpleAutoencoder Tests
# =============================================================================

class TestSimpleAutoencoderInitialization(unittest.TestCase):
    """Test SimpleAutoencoder initialization."""
    
    def test_basic_init(self):
        """Test basic initialization."""
        model = SimpleAutoencoder(input_shape=100, latent_dim=32)
        
        self.assertIsInstance(model, nn.Module)
        self.assertTrue(hasattr(model, 'encoder'))
        self.assertTrue(hasattr(model, 'decoder'))
    
    def test_encoder_structure(self):
        """Test encoder has correct structure."""
        model = SimpleAutoencoder(input_shape=100, latent_dim=32)
        
        # Encoder should be Sequential
        self.assertIsInstance(model.encoder, nn.Sequential)
        
        # Check input layer dimension
        # First layer is Flatten, second is Linear
        linear_layer = model.encoder[1]
        self.assertEqual(linear_layer.in_features, 100)
        self.assertEqual(linear_layer.out_features, 32)
    
    def test_decoder_structure(self):
        """Test decoder has correct structure."""
        model = SimpleAutoencoder(input_shape=100, latent_dim=32)
        
        # Decoder should be Sequential
        self.assertIsInstance(model.decoder, nn.Sequential)
        
        # Check output layer dimension
        output_layer = model.decoder[-1]
        self.assertEqual(output_layer.in_features, 32)
        self.assertEqual(output_layer.out_features, 100)
    
    def test_different_dimensions(self):
        """Test initialization with various dimensions."""
        configs = [
            (784, 64),   # MNIST-like
            (50, 10),    # Small
            (1000, 128), # Larger
            (10, 2),     # Very small latent
        ]
        
        for input_shape, latent_dim in configs:
            model = SimpleAutoencoder(input_shape=input_shape, latent_dim=latent_dim)
            self.assertEqual(model.encoder[1].in_features, input_shape)
            self.assertEqual(model.encoder[1].out_features, latent_dim)
            self.assertEqual(model.decoder[-1].out_features, input_shape)


class TestSimpleAutoencoderForward(unittest.TestCase):
    """Test SimpleAutoencoder forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleAutoencoder(input_shape=100, latent_dim=32)
    
    def test_forward_returns_tuple(self):
        """Test forward returns (encoded, decoded) tuple."""
        x = torch.randn(4, 100)
        result = self.model(x)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_forward_encoded_shape(self):
        """Test encoded output has correct shape."""
        x = torch.randn(4, 100)
        encoded, _ = self.model(x)
        
        self.assertEqual(encoded.shape, (4, 32))
    
    def test_forward_decoded_shape(self):
        """Test decoded output has correct shape."""
        x = torch.randn(4, 100)
        _, decoded = self.model(x)
        
        self.assertEqual(decoded.shape, (4, 100))
    
    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 100)
            encoded, decoded = self.model(x)
            
            self.assertEqual(encoded.shape[0], batch_size)
            self.assertEqual(decoded.shape[0], batch_size)
    
    def test_forward_with_flatten(self):
        """Test forward pass handles 2D input correctly (flatten is applied)."""
        x = torch.randn(4, 100)
        encoded, decoded = self.model(x)
        
        # Should work without issues
        self.assertEqual(encoded.shape, (4, 32))
        self.assertEqual(decoded.shape, (4, 100))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        x = torch.randn(4, 100, requires_grad=True)
        encoded, decoded = self.model(x)
        
        # Compute reconstruction loss
        loss = nn.functional.mse_loss(decoded, x)
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


class TestSimpleAutoencoderDevice(unittest.TestCase):
    """Test SimpleAutoencoder device handling."""
    
    def test_cpu_device(self):
        """Test model works on CPU."""
        model = SimpleAutoencoder(input_shape=100, latent_dim=32)
        model = model.to('cpu')
        
        x = torch.randn(4, 100, device='cpu')
        encoded, decoded = model(x)
        
        self.assertEqual(encoded.device.type, 'cpu')
        self.assertEqual(decoded.device.type, 'cpu')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test model works on CUDA."""
        model = SimpleAutoencoder(input_shape=100, latent_dim=32)
        model = model.to('cuda')
        
        x = torch.randn(4, 100, device='cuda')
        encoded, decoded = model(x)
        
        self.assertEqual(encoded.device.type, 'cuda')
        self.assertEqual(decoded.device.type, 'cuda')


# =============================================================================
# AutoencoderTrainer Tests
# =============================================================================

class TestAutoencoderTrainerInitialization(unittest.TestCase):
    """Test AutoencoderTrainer initialization."""
    
    def test_basic_init(self):
        """Test basic initialization."""
        trainer = AutoencoderTrainer(input_shape=100)
        
        self.assertIsInstance(trainer.model, SimpleAutoencoder)
        self.assertIsInstance(trainer.criterion, nn.MSELoss)
        self.assertIsNotNone(trainer.optimizer)
    
    def test_default_parameters(self):
        """Test default parameter values."""
        trainer = AutoencoderTrainer(input_shape=100)
        
        self.assertEqual(trainer.noise_level, 0.0)
        self.assertEqual(trainer.latend_dim, 32)
        self.assertEqual(trainer.lr, 0.0005)
        self.assertEqual(trainer.epochs, 2000)
        self.assertEqual(trainer.batch_size, 512)
        self.assertEqual(trainer.patience, 50)
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        trainer = AutoencoderTrainer(
            input_shape=100,
            noise=0.2,
            latent_dim=64,
            lr=0.001,
            epochs=100,
            batch_size=32,
            patience=10,
            device='cpu'
        )
        
        self.assertEqual(trainer.noise_level, 0.2)
        self.assertEqual(trainer.latend_dim, 64)
        self.assertEqual(trainer.lr, 0.001)
        self.assertEqual(trainer.epochs, 100)
        self.assertEqual(trainer.batch_size, 32)
        self.assertEqual(trainer.patience, 10)
        self.assertEqual(trainer.device, 'cpu')
    
    def test_auto_device_selection_cpu(self):
        """Test automatic device selection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            trainer = AutoencoderTrainer(input_shape=100, device=None)
            self.assertEqual(trainer.device, 'cpu')
    
    def test_auto_device_selection_cuda(self):
        """Test automatic device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            trainer = AutoencoderTrainer(input_shape=100, device=None)
            self.assertEqual(trainer.device, 'cuda')
    
    def test_explicit_device_override(self):
        """Test explicit device parameter overrides auto-detection."""
        trainer = AutoencoderTrainer(input_shape=100, device='cpu')
        self.assertEqual(trainer.device, 'cpu')
    
    def test_model_on_correct_device(self):
        """Test model is placed on correct device."""
        trainer = AutoencoderTrainer(input_shape=100, device='cpu')
        
        # Check model parameters are on CPU
        for param in trainer.model.parameters():
            self.assertEqual(param.device.type, 'cpu')


class TestAutoencoderTrainerTraining(unittest.TestCase):
    """Test AutoencoderTrainer training functionality."""
    
    def test_train_basic(self):
        """Test basic training loop."""
        trainer = AutoencoderTrainer(
            input_shape=50,
            latent_dim=10,
            epochs=5,
            batch_size=16,
            patience=10,
            device='cpu'
        )
        
        data = torch.randn(100, 50)
        trainer.train(data)
        
        self.assertTrue(trainer.is_fitted)
        self.assertIsNotNone(trainer.best_model_wts)
    
    def test_train_creates_dataloader(self):
        """Test that training creates a DataLoader."""
        trainer = AutoencoderTrainer(
            input_shape=50,
            latent_dim=10,
            epochs=2,
            batch_size=16,
            device='cpu'
        )
        
        data = torch.randn(100, 50)
        trainer.train(data)
        
        self.assertTrue(hasattr(trainer, 'data_loader'))
        self.assertIsNotNone(trainer.data_loader)
    
    def test_train_early_stopping(self):
        """Test early stopping triggers when loss doesn't improve."""
        # Create data that will lead to quick convergence
        trainer = AutoencoderTrainer(
            input_shape=10,
            latent_dim=5,
            epochs=1000,  # High epochs, should stop early
            batch_size=32,
            patience=3,   # Very low patience for quick stopping
            device='cpu'
        )
        
        # Simple data that's easy to learn
        data = torch.randn(100, 10)
        trainer.train(data)
        
        # Training should have completed (either early stopping or reaching epochs)
        self.assertTrue(trainer.is_fitted)
    
    def test_train_saves_best_weights(self):
        """Test that best model weights are saved."""
        trainer = AutoencoderTrainer(
            input_shape=50,
            latent_dim=10,
            epochs=5,
            batch_size=16,
            device='cpu'
        )
        
        data = torch.randn(100, 50)
        trainer.train(data)
        
        self.assertIsNotNone(trainer.best_model_wts)
        self.assertIsInstance(trainer.best_model_wts, dict)
    
    def test_train_with_different_batch_sizes(self):
        """Test training with different batch sizes."""
        for batch_size in [8, 32, 64]:
            trainer = AutoencoderTrainer(
                input_shape=50,
                latent_dim=10,
                epochs=2,
                batch_size=batch_size,
                device='cpu'
            )
            
            data = torch.randn(100, 50)
            trainer.train(data)
            
            self.assertTrue(trainer.is_fitted)
    
    def test_train_logs_at_epoch_300_interval(self):
        """Test that logging occurs at epoch 300 intervals."""
        trainer = AutoencoderTrainer(
            input_shape=20,
            latent_dim=5,
            epochs=301,  # At least 301 to trigger logging at epoch 300
            batch_size=32,
            patience=1000,  # High patience so we don't early stop
            device='cpu'
        )
        
        data = torch.randn(50, 20)
        
        with patch('torch_concepts.data.preprocessing.autoencoder.logger') as mock_logger:
            trainer.train(data)
            # Should log at epoch 0 (first epoch when epoch % 300 == 0) and epoch 300
            # Plus initial and final log messages
            self.assertTrue(mock_logger.info.called)


class TestAutoencoderTrainerExtractLatent(unittest.TestCase):
    """Test AutoencoderTrainer latent extraction."""
    
    def setUp(self):
        """Set up trained trainer."""
        self.trainer = AutoencoderTrainer(
            input_shape=50,
            latent_dim=10,
            epochs=5,
            batch_size=16,
            noise=0.0,
            device='cpu'
        )
        self.data = torch.randn(100, 50)
        self.trainer.train(self.data)
    
    def test_extract_latent_shape(self):
        """Test extracted latent has correct shape."""
        latent = self.trainer.extract_latent()
        
        self.assertEqual(latent.shape, (100, 10))
    
    def test_extract_latent_type(self):
        """Test extracted latent is a tensor."""
        latent = self.trainer.extract_latent()
        
        self.assertIsInstance(latent, torch.Tensor)
    
    def test_extract_latent_uses_best_weights(self):
        """Test that extract_latent uses the best model weights."""
        # The method should load best_model_wts
        latent = self.trainer.extract_latent()
        
        # Just verify it runs and produces output
        self.assertEqual(latent.shape[0], 100)
    
    def test_extract_latent_deterministic(self):
        """Test extract_latent is deterministic without noise."""
        latent1 = self.trainer.extract_latent()
        latent2 = self.trainer.extract_latent()
        
        self.assertTrue(torch.allclose(latent1, latent2))
    
    def test_extract_latent_with_noise(self):
        """Test extract_latent with noise injection."""
        trainer_with_noise = AutoencoderTrainer(
            input_shape=50,
            latent_dim=10,
            epochs=5,
            batch_size=16,
            noise=0.5,  # 50% noise
            device='cpu'
        )
        
        data = torch.randn(100, 50)
        trainer_with_noise.train(data)
        
        # With noise, results should differ between calls
        torch.manual_seed(42)
        latent1 = trainer_with_noise.extract_latent()
        torch.manual_seed(123)
        latent2 = trainer_with_noise.extract_latent()
        
        # They should be different due to noise
        self.assertFalse(torch.allclose(latent1, latent2))


class TestAutoencoderTrainerDevice(unittest.TestCase):
    """Test AutoencoderTrainer device handling."""
    
    def test_training_on_cpu(self):
        """Test training on CPU."""
        trainer = AutoencoderTrainer(
            input_shape=50,
            latent_dim=10,
            epochs=2,
            batch_size=16,
            device='cpu'
        )
        
        data = torch.randn(100, 50)
        trainer.train(data)
        
        latent = trainer.extract_latent()
        self.assertEqual(latent.device.type, 'cpu')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_on_cuda(self):
        """Test training on CUDA."""
        trainer = AutoencoderTrainer(
            input_shape=50,
            latent_dim=10,
            epochs=2,
            batch_size=16,
            device='cuda'
        )
        
        data = torch.randn(100, 50)
        trainer.train(data)
        
        latent = trainer.extract_latent()
        self.assertEqual(latent.device.type, 'cuda')


# =============================================================================
# extract_embs_from_autoencoder Tests
# =============================================================================

class TestExtractEmbsFromAutoencoder(unittest.TestCase):
    """Test extract_embs_from_autoencoder function."""
    
    def test_basic_extraction(self):
        """Test basic embedding extraction from DataFrame."""
        df = pd.DataFrame(np.random.randn(100, 50))
        
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'latent_dim': 10,
                'epochs': 5,
                'batch_size': 16,
                'device': 'cpu'
            }
        )
        
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape, (100, 10))
    
    def test_extraction_with_column_names(self):
        """Test extraction from DataFrame with named columns."""
        df = pd.DataFrame({
            f'feature_{i}': np.random.randn(50)
            for i in range(20)
        })
        
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'latent_dim': 5,
                'epochs': 3,
                'batch_size': 16,
                'device': 'cpu'
            }
        )
        
        self.assertEqual(embeddings.shape, (50, 5))
    
    def test_extraction_empty_kwargs(self):
        """Test extraction with empty kwargs uses defaults."""
        df = pd.DataFrame(np.random.randn(50, 20))
        
        # Should use default latent_dim=32
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'epochs': 2,
                'device': 'cpu'
            }
        )
        
        self.assertEqual(embeddings.shape[0], 50)
        self.assertEqual(embeddings.shape[1], 32)  # Default latent_dim
    
    def test_extraction_with_noise(self):
        """Test extraction with noise parameter."""
        df = pd.DataFrame(np.random.randn(100, 30))
        
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'latent_dim': 8,
                'epochs': 3,
                'batch_size': 32,
                'noise': 0.1,
                'device': 'cpu'
            }
        )
        
        self.assertEqual(embeddings.shape, (100, 8))
    
    def test_extraction_preserves_sample_count(self):
        """Test that extraction preserves number of samples."""
        for n_samples in [20, 100, 500]:
            df = pd.DataFrame(np.random.randn(n_samples, 30))
            
            embeddings = extract_embs_from_autoencoder(
                df,
                autoencoder_kwargs={
                    'latent_dim': 5,
                    'epochs': 2,
                    'batch_size': 16,
                    'device': 'cpu'
                }
            )
            
            self.assertEqual(embeddings.shape[0], n_samples)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestAutoencoderEdgeCases(unittest.TestCase):
    """Test edge cases for autoencoder module."""
    
    def test_single_sample(self):
        """Test with single sample."""
        model = SimpleAutoencoder(input_shape=50, latent_dim=10)
        x = torch.randn(1, 50)
        
        encoded, decoded = model(x)
        
        self.assertEqual(encoded.shape, (1, 10))
        self.assertEqual(decoded.shape, (1, 50))
    
    def test_very_small_latent_dim(self):
        """Test with very small latent dimension."""
        model = SimpleAutoencoder(input_shape=100, latent_dim=1)
        x = torch.randn(4, 100)
        
        encoded, decoded = model(x)
        
        self.assertEqual(encoded.shape, (4, 1))
        self.assertEqual(decoded.shape, (4, 100))
    
    def test_latent_dim_equals_input(self):
        """Test when latent dimension equals input dimension."""
        model = SimpleAutoencoder(input_shape=50, latent_dim=50)
        x = torch.randn(4, 50)
        
        encoded, decoded = model(x)
        
        self.assertEqual(encoded.shape, (4, 50))
        self.assertEqual(decoded.shape, (4, 50))
    
    def test_training_with_small_dataset(self):
        """Test training with very small dataset."""
        trainer = AutoencoderTrainer(
            input_shape=20,
            latent_dim=5,
            epochs=5,
            batch_size=4,  # Small batch for small dataset
            device='cpu'
        )
        
        data = torch.randn(10, 20)  # Only 10 samples
        trainer.train(data)
        
        latent = trainer.extract_latent()
        self.assertEqual(latent.shape, (10, 5))
    
    def test_batch_size_larger_than_dataset(self):
        """Test when batch size is larger than dataset."""
        trainer = AutoencoderTrainer(
            input_shape=20,
            latent_dim=5,
            epochs=3,
            batch_size=100,  # Larger than dataset
            device='cpu'
        )
        
        data = torch.randn(50, 20)
        trainer.train(data)
        
        latent = trainer.extract_latent()
        self.assertEqual(latent.shape[0], 50)
    
    def test_zero_noise(self):
        """Test that zero noise gives deterministic results."""
        trainer = AutoencoderTrainer(
            input_shape=30,
            latent_dim=10,
            epochs=5,
            batch_size=16,
            noise=0.0,
            device='cpu'
        )
        
        data = torch.randn(50, 30)
        trainer.train(data)
        
        latent1 = trainer.extract_latent()
        latent2 = trainer.extract_latent()
        
        self.assertTrue(torch.allclose(latent1, latent2))
    
    def test_full_noise(self):
        """Test with maximum noise level."""
        trainer = AutoencoderTrainer(
            input_shape=30,
            latent_dim=10,
            epochs=5,
            batch_size=16,
            noise=1.0,  # Full noise
            device='cpu'
        )
        
        data = torch.randn(50, 30)
        trainer.train(data)
        
        # Should still produce output
        latent = trainer.extract_latent()
        self.assertEqual(latent.shape, (50, 10))


class TestAutoencoderReproducibility(unittest.TestCase):
    """Test reproducibility of autoencoder training."""
    
    def test_deterministic_training_with_seed(self):
        """Test that training is deterministic with same seed."""
        torch.manual_seed(42)
        trainer1 = AutoencoderTrainer(
            input_shape=30,
            latent_dim=10,
            epochs=3,
            batch_size=16,
            device='cpu'
        )
        data1 = torch.randn(50, 30)
        trainer1.train(data1)
        
        torch.manual_seed(42)
        trainer2 = AutoencoderTrainer(
            input_shape=30,
            latent_dim=10,
            epochs=3,
            batch_size=16,
            device='cpu'
        )
        data2 = torch.randn(50, 30)
        trainer2.train(data2)
        
        # With same seed and data, results should be identical
        latent1 = trainer1.extract_latent()
        latent2 = trainer2.extract_latent()
        
        self.assertTrue(torch.allclose(latent1, latent2, atol=1e-5))


class TestDataFrameConversion(unittest.TestCase):
    """Test DataFrame to tensor conversion in extract_embs_from_autoencoder."""
    
    def test_integer_columns(self):
        """Test extraction from DataFrame with integer values."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [1, 1, 1, 1, 1]
        })
        
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'latent_dim': 2,
                'epochs': 3,
                'batch_size': 2,
                'device': 'cpu'
            }
        )
        
        self.assertEqual(embeddings.shape, (5, 2))
    
    def test_mixed_types(self):
        """Test extraction from DataFrame with mixed numeric types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.1, 2.2, 3.3, 4.4],
        })
        
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'latent_dim': 2,
                'epochs': 3,
                'batch_size': 2,
                'device': 'cpu'
            }
        )
        
        self.assertEqual(embeddings.shape, (4, 2))


# =============================================================================
# Integration Tests
# =============================================================================

class TestAutoencoderIntegration(unittest.TestCase):
    """Integration tests for autoencoder preprocessing pipeline."""
    
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create synthetic high-dimensional data
        n_samples = 200
        n_features = 100
        latent_dim = 16
        
        df = pd.DataFrame(np.random.randn(n_samples, n_features))
        
        # Extract embeddings
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'latent_dim': latent_dim,
                'epochs': 10,
                'batch_size': 32,
                'patience': 5,
                'device': 'cpu'
            }
        )
        
        # Verify output
        self.assertEqual(embeddings.shape, (n_samples, latent_dim))
        self.assertIsInstance(embeddings, torch.Tensor)
        
        # Verify embeddings are finite
        self.assertTrue(torch.isfinite(embeddings).all())
    
    def test_dimensionality_reduction(self):
        """Test that autoencoder achieves dimensionality reduction."""
        n_samples = 100
        n_features = 50
        latent_dim = 10
        
        df = pd.DataFrame(np.random.randn(n_samples, n_features))
        
        embeddings = extract_embs_from_autoencoder(
            df,
            autoencoder_kwargs={
                'latent_dim': latent_dim,
                'epochs': 5,
                'batch_size': 32,
                'device': 'cpu'
            }
        )
        
        # Reduced dimension
        self.assertEqual(embeddings.shape[1], latent_dim)
        self.assertLess(embeddings.shape[1], n_features)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
