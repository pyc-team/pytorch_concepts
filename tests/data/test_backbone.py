"""
Extended tests for torch_concepts.data.backbone to increase coverage.
"""
import unittest
import torch
from torch import nn
import tempfile
import os
from torch_concepts.data.backbone import compute_backbone_embs
from torch.utils.data import Dataset


class TestBackboneExtended:
    """Extended tests for backbone utilities."""

    def test_compute_backbone_embs_with_eval_mode_preserved(self):
        """Test that compute_backbone_embs preserves model's eval mode."""
        from torch_concepts.data.backbone import compute_backbone_embs
        from torch_concepts.data.datasets.toy import ToyDataset

        backbone = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
        backbone.eval()

        dataset = ToyDataset('xor', n_gen=20)
        embeddings = compute_backbone_embs(dataset, backbone, batch_size=10, device='cpu', verbose=False)

        assert embeddings.shape[0] == 20
        assert not backbone.training  # Should still be in eval mode

    def test_compute_backbone_embs_with_training_mode_preserved(self):
        """Test that compute_backbone_embs preserves model's training mode."""
        from torch_concepts.data.backbone import compute_backbone_embs
        from torch_concepts.data.datasets.toy import ToyDataset

        backbone = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
        backbone.train()

        dataset = ToyDataset('xor', n_gen=20)
        embeddings = compute_backbone_embs(dataset, backbone, batch_size=10, device='cpu', verbose=False)

        assert embeddings.shape[0] == 20
        assert backbone.training  # Should still be in training mode

    def test_compute_backbone_embs_auto_device_detection(self):
        """Test compute_backbone_embs with automatic device detection (None)."""
        from torch_concepts.data.backbone import compute_backbone_embs
        from torch_concepts.data.datasets.toy import ToyDataset

        backbone = nn.Linear(2, 5)
        dataset = ToyDataset('xor', n_gen=10)

        # Pass device=None to test auto-detection
        embeddings = compute_backbone_embs(dataset, backbone, batch_size=5, device=None, verbose=False)

        assert embeddings.shape[0] == 10

    def test_compute_backbone_embs_with_verbose(self):
        """Test compute_backbone_embs with verbose output."""
        from torch_concepts.data.backbone import compute_backbone_embs
        from torch_concepts.data.datasets.toy import ToyDataset

        backbone = nn.Linear(2, 5)
        dataset = ToyDataset('xor', n_gen=10)

        # Test with verbose=True
        embeddings = compute_backbone_embs(dataset, backbone, batch_size=5, device='cpu', verbose=True)

        assert embeddings.shape[0] == 10

    def test_get_backbone_embs_compute_and_cache(self):
        """Test get_backbone_embs computes and caches embeddings."""
        from torch_concepts.data.backbone import get_backbone_embs
        from torch_concepts.data.datasets.toy import ToyDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')

            backbone = nn.Linear(2, 5)
            dataset = ToyDataset('xor', n_gen=20)

            # First call should compute and save
            embeddings1 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=10,
                force_recompute=False,
                device='cpu',
                verbose=False
            )

            assert os.path.exists(cache_path)
            assert embeddings1.shape[0] == 20

            # Second call should load from cache
            embeddings2 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=10,
                force_recompute=False,
                device='cpu',
                verbose=False
            )

            assert torch.allclose(embeddings1, embeddings2)

    def test_get_backbone_embs_force_recompute(self):
        """Test get_backbone_embs with force_recompute=True."""
        from torch_concepts.data.backbone import get_backbone_embs
        from torch_concepts.data.datasets.toy import ToyDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')

            backbone = nn.Linear(2, 5)
            dataset = ToyDataset('xor', n_gen=20)

            # First compute
            embeddings1 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=10,
                force_recompute=True,
                device='cpu',
                verbose=False
            )

            # Force recompute even though cache exists
            embeddings2 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=10,
                force_recompute=True,
                device='cpu',
                verbose=False
            )

            assert embeddings1.shape == embeddings2.shape

    def test_get_backbone_embs_verbose_logging(self):
        """Test get_backbone_embs with verbose logging."""
        from torch_concepts.data.backbone import get_backbone_embs
        from torch_concepts.data.datasets.toy import ToyDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')

            backbone = nn.Linear(2, 5)
            dataset = ToyDataset('xor', n_gen=10)

            # Test verbose output during computation
            embeddings = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=5,
                device='cpu',
                verbose=True  # This should trigger logging
            )

            assert embeddings.shape[0] == 10

class SimpleDictDataset(Dataset):
    """Simple dataset that returns dict with 'x' key."""
    def __init__(self, n_samples=20, n_features=2):
        self.data = torch.randn(n_samples, n_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'x': self.data[idx]}


class NestedDictDataset(Dataset):
    """Dataset that returns nested dict with 'inputs'.'x' structure."""
    def __init__(self, n_samples=20, n_features=2):
        self.data = torch.randn(n_samples, n_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'inputs': {'x': self.data[idx]}}


class TestComputeBackboneEmbsComprehensive:
    """Comprehensive tests for compute_backbone_embs function."""

    def test_compute_with_simple_dict_dataset(self):
        """Test compute_backbone_embs with dataset returning {'x': tensor}."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Linear(2, 5)
        dataset = SimpleDictDataset(n_samples=20, n_features=2)

        embs = compute_backbone_embs(
            dataset, backbone, batch_size=8, workers=0, device='cpu', verbose=False
        )

        assert embs.shape == (20, 5)
        assert embs.dtype == torch.float32

    def test_compute_with_nested_dict_dataset(self):
        """Test compute_backbone_embs with dataset returning {'inputs': {'x': tensor}}."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Linear(2, 5)
        dataset = NestedDictDataset(n_samples=20, n_features=2)

        embs = compute_backbone_embs(
            dataset, backbone, batch_size=8, workers=0, device='cpu', verbose=False
        )

        assert embs.shape == (20, 5)

    def test_compute_preserves_eval_mode(self):
        """Test that compute_backbone_embs preserves model's eval mode."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
        backbone.eval()

        dataset = SimpleDictDataset(n_samples=20)

        embs = compute_backbone_embs(
            dataset, backbone, batch_size=8, device='cpu', verbose=False
        )

        # Model should remain in eval mode after computation
        assert not backbone.training

    def test_compute_preserves_training_mode(self):
        """Test that compute_backbone_embs preserves model's training mode."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
        backbone.train()

        dataset = SimpleDictDataset(n_samples=20)

        embs = compute_backbone_embs(
            dataset, backbone, batch_size=8, device='cpu', verbose=False
        )

        # Model should be back in training mode after computation
        assert backbone.training

    def test_compute_auto_device_detection_cpu(self):
        """Test compute_backbone_embs with automatic device detection (None)."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Linear(2, 5)
        dataset = SimpleDictDataset(n_samples=10)

        # device=None should auto-detect
        embs = compute_backbone_embs(
            dataset, backbone, batch_size=10, device=None, verbose=False
        )

        assert embs.shape == (10, 5)
        assert embs.device.type == 'cpu'

    def test_compute_with_verbose_enabled(self):
        """Test compute_backbone_embs with verbose output."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Linear(2, 5)
        dataset = SimpleDictDataset(n_samples=10)

        # Should not raise any errors with verbose=True
        embs = compute_backbone_embs(
            dataset, backbone, batch_size=5, device='cpu', verbose=True
        )

        assert embs.shape == (10, 5)

    def test_compute_large_batch_size(self):
        """Test compute_backbone_embs with batch size larger than dataset."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Linear(2, 5)
        dataset = SimpleDictDataset(n_samples=10)

        # Batch size larger than dataset
        embs = compute_backbone_embs(
            dataset, backbone, batch_size=100, device='cpu', verbose=False
        )

        assert embs.shape == (10, 5)

    def test_compute_embeddings_correctly(self):
        """Test that embeddings are computed correctly."""
        from torch_concepts.data.backbone import compute_backbone_embs

        # Use a deterministic backbone
        backbone = nn.Linear(2, 5)
        torch.manual_seed(42)
        nn.init.constant_(backbone.weight, 1.0)
        nn.init.constant_(backbone.bias, 0.0)

        dataset = SimpleDictDataset(n_samples=5)
        dataset.data = torch.ones(5, 2)  # All ones

        embs = compute_backbone_embs(
            dataset, backbone, batch_size=5, device='cpu', verbose=False
        )

        # Each embedding should be sum of weights = 2.0 for each output dim
        expected = torch.full((5, 5), 2.0)
        assert torch.allclose(embs, expected)

    def test_compute_with_workers(self):
        """Test compute_backbone_embs with multiple workers."""
        from torch_concepts.data.backbone import compute_backbone_embs

        backbone = nn.Linear(2, 5)
        dataset = SimpleDictDataset(n_samples=20)

        # Test with workers (set to 0 to avoid multiprocessing issues in tests)
        embs = compute_backbone_embs(
            dataset, backbone, batch_size=8, workers=0, device='cpu', verbose=False
        )

        assert embs.shape == (20, 5)


class TestGetBackboneEmbsComprehensive:
    """Comprehensive tests for get_backbone_embs function with caching."""

    def test_get_embs_compute_and_cache(self):
        """Test get_backbone_embs computes and caches embeddings."""
        from torch_concepts.data.backbone import get_backbone_embs

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')

            backbone = nn.Linear(2, 5)
            dataset = SimpleDictDataset(n_samples=20)

            # First call should compute and save
            embs1 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=8,
                force_recompute=False,
                workers=0,
                device='cpu',
                verbose=False
            )

            assert embs1.shape == (20, 5)
            assert os.path.exists(cache_path)

            # Modify backbone to verify caching
            backbone2 = nn.Linear(2, 5)
            nn.init.constant_(backbone2.weight, 0.0)

            # Second call should load from cache (not recompute)
            embs2 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone2,
                batch_size=8,
                force_recompute=False,
                workers=0,
                device='cpu',
                verbose=False
            )

            # Should be same as first (cached)
            assert torch.allclose(embs1, embs2)

    def test_get_embs_force_recompute(self):
        """Test get_backbone_embs with force_recompute=True."""
        from torch_concepts.data.backbone import get_backbone_embs

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')

            backbone = nn.Linear(2, 5)
            torch.manual_seed(42)
            nn.init.constant_(backbone.weight, 1.0)
            nn.init.constant_(backbone.bias, 0.0)

            dataset = SimpleDictDataset(n_samples=20)
            dataset.data = torch.ones(20, 2)

            # First call
            embs1 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=8,
                force_recompute=False,
                workers=0,
                device='cpu',
                verbose=False
            )

            # Modify backbone
            backbone2 = nn.Linear(2, 5)
            nn.init.constant_(backbone2.weight, 2.0)
            nn.init.constant_(backbone2.bias, 0.0)

            # Force recompute with new backbone
            embs2 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone2,
                batch_size=8,
                force_recompute=True,
                workers=0,
                device='cpu',
                verbose=False
            )

            # Should be different (recomputed with new backbone)
            assert not torch.allclose(embs1, embs2)
            assert torch.allclose(embs2, torch.full((20, 5), 4.0))

    def test_get_embs_verbose_logging(self):
        """Test get_backbone_embs with verbose logging."""
        from torch_concepts.data.backbone import get_backbone_embs

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')

            backbone = nn.Linear(2, 5)
            dataset = SimpleDictDataset(n_samples=10)

            # Test with verbose=True (should log messages)
            embs = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=5,
                force_recompute=False,
                workers=0,
                device='cpu',
                verbose=True
            )

            assert embs.shape == (10, 5)
            assert os.path.exists(cache_path)

    def test_get_embs_loads_from_cache(self):
        """Test that get_backbone_embs loads from cache when available."""
        from torch_concepts.data.backbone import get_backbone_embs

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')

            # Create and save some embeddings manually
            manual_embs = torch.randn(15, 7)
            torch.save(manual_embs, cache_path)

            backbone = nn.Linear(2, 5)
            dataset = SimpleDictDataset(n_samples=10)

            # Should load the manually saved embeddings
            loaded_embs = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=5,
                force_recompute=False,
                workers=0,
                device='cpu',
                verbose=False
            )

            assert torch.allclose(loaded_embs, manual_embs)
            assert loaded_embs.shape == (15, 7)  # Not (10, 5) because loaded from cache

    def test_get_embs_creates_directory(self):
        """Test that get_backbone_embs creates directory if it doesn't exist."""
        from torch_concepts.data.backbone import get_backbone_embs

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested path that doesn't exist
            cache_path = os.path.join(tmpdir, 'nested', 'dir', 'embeddings.pt')

            backbone = nn.Linear(2, 5)
            dataset = SimpleDictDataset(n_samples=10)

            # Should create directory structure
            embs = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone=backbone,
                batch_size=5,
                force_recompute=False,
                workers=0,
                device='cpu',
                verbose=False
            )

            assert os.path.exists(cache_path)
            assert embs.shape == (10, 5)


class TestBackboneTrainingStatePreservation(unittest.TestCase):
    """Test that compute_backbone_embs preserves the training state of the model."""

    def setUp(self):
        # Create a simple backbone model
        self.backbone = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU()
        )
        # Create a simple dataset
        X = torch.randn(20, 10)
        self.dataset = [{'x': X[i]} for i in range(len(X))]

    def test_preserves_training_mode(self):
        """Test that a model in training mode is restored to training mode."""
        self.backbone.train()
        self.assertTrue(self.backbone.training, "Model should start in training mode")

        _ = compute_backbone_embs(
            self.dataset,
            self.backbone,
            batch_size=4,
            verbose=False
        )

        self.assertTrue(
            self.backbone.training,
            "Model should be restored to training mode after compute_backbone_embs"
        )

    def test_preserves_eval_mode(self):
        """Test that a model in eval mode remains in eval mode."""
        self.backbone.eval()
        self.assertFalse(self.backbone.training, "Model should start in eval mode")

        _ = compute_backbone_embs(
            self.dataset,
            self.backbone,
            batch_size=4,
            verbose=False
        )

        self.assertFalse(
            self.backbone.training,
            "Model should remain in eval mode after compute_backbone_embs"
        )

    def test_embeddings_computed_correctly(self):
        """Test that embeddings are computed with correct shape."""
        embs = compute_backbone_embs(
            self.dataset,
            self.backbone,
            batch_size=4,
            verbose=False
        )

        self.assertEqual(embs.shape[0], len(self.dataset), "Should have one embedding per sample")
        self.assertEqual(embs.shape[1], 5, "Embedding dimension should match backbone output")


if __name__ == '__main__':
    unittest.main()
