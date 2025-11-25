"""
Extended tests for torch_concepts.data.backbone to increase coverage.
"""
import pytest
import torch
from torch import nn
import tempfile
import os


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
