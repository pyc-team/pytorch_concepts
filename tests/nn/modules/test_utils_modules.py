"""
Tests for the indices_to_mask helper function.

This tests the conversion from index-based interventions to mask-based format.
"""
import torch
import pytest
from torch_concepts.nn.modules.utils import indices_to_mask


class TestIndicesToMask:
    """Test suite for indices_to_mask function."""

    def test_basic_conversion(self):
        """Test basic index to mask conversion."""
        c_idxs = [0, 2]
        c_vals = [1.0, 0.5]
        n_concepts = 5
        batch_size = 2

        mask, target = indices_to_mask(c_idxs, c_vals, n_concepts, batch_size)

        # Check shapes
        assert mask.shape == (2, 5)
        assert target.shape == (2, 5)

        # Check mask: 0 at intervention indices, 1 elsewhere
        expected_mask = torch.tensor([[0., 1., 0., 1., 1.],
                                      [0., 1., 0., 1., 1.]])
        assert torch.allclose(mask, expected_mask)

        # Check target: intervention values at specified indices
        expected_target = torch.tensor([[1.0, 0., 0.5, 0., 0.],
                                        [1.0, 0., 0.5, 0., 0.]])
        assert torch.allclose(target, expected_target)

    def test_tensor_inputs(self):
        """Test with tensor inputs instead of lists."""
        c_idxs = torch.tensor([1, 3])
        c_vals = torch.tensor([0.8, 0.2])
        n_concepts = 4
        batch_size = 3

        mask, target = indices_to_mask(c_idxs, c_vals, n_concepts, batch_size)

        assert mask.shape == (3, 4)
        assert target.shape == (3, 4)

        # Check first batch
        expected_mask_row = torch.tensor([1., 0., 1., 0.])
        assert torch.allclose(mask[0], expected_mask_row)

        expected_target_row = torch.tensor([0., 0.8, 0., 0.2])
        assert torch.allclose(target[0], expected_target_row)

    def test_per_batch_values(self):
        """Test with different intervention values per batch."""
        c_idxs = [0, 1]
        c_vals = torch.tensor([[1.0, 0.0],  # batch 0
                               [0.5, 0.5],  # batch 1
                               [0.0, 1.0]]) # batch 2
        n_concepts = 3
        batch_size = 3

        mask, target = indices_to_mask(c_idxs, c_vals, n_concepts, batch_size)

        assert mask.shape == (3, 3)
        assert target.shape == (3, 3)

        # Mask should be same for all batches
        expected_mask = torch.tensor([[0., 0., 1.],
                                      [0., 0., 1.],
                                      [0., 0., 1.]])
        assert torch.allclose(mask, expected_mask)

        # Target values differ per batch
        expected_target = torch.tensor([[1.0, 0.0, 0.],
                                        [0.5, 0.5, 0.],
                                        [0.0, 1.0, 0.]])
        assert torch.allclose(target, expected_target)

    def test_empty_interventions(self):
        """Test with no interventions (empty indices)."""
        c_idxs = []
        c_vals = []
        n_concepts = 4
        batch_size = 2

        mask, target = indices_to_mask(c_idxs, c_vals, n_concepts, batch_size)

        # Should return all-ones mask and zeros target
        assert torch.allclose(mask, torch.ones(2, 4))
        assert torch.allclose(target, torch.zeros(2, 4))

    def test_single_concept_intervention(self):
        """Test intervention on a single concept."""
        c_idxs = [2]
        c_vals = [0.75]
        n_concepts = 5
        batch_size = 1

        mask, target = indices_to_mask(c_idxs, c_vals, n_concepts, batch_size)

        expected_mask = torch.tensor([[1., 1., 0., 1., 1.]])
        expected_target = torch.tensor([[0., 0., 0.75, 0., 0.]])

        assert torch.allclose(mask, expected_mask)
        assert torch.allclose(target, expected_target)

    def test_device_and_dtype(self):
        """Test that device and dtype parameters work correctly."""
        c_idxs = [0, 1]
        c_vals = [1.0, 0.5]
        n_concepts = 3
        batch_size = 2

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        dtype = torch.float64

        mask, target = indices_to_mask(
            c_idxs, c_vals, n_concepts, batch_size,
            device=device, dtype=dtype
        )

        assert mask.device.type == device.type
        assert target.device.type == device.type
        assert mask.dtype == dtype
        assert target.dtype == dtype

    def test_invalid_indices(self):
        """Test that invalid indices raise appropriate errors."""
        # Index out of range
        with pytest.raises(ValueError, match="All indices must be in range"):
            indices_to_mask([0, 5], [1.0, 0.5], n_concepts=5, batch_size=1)

        # Negative index
        with pytest.raises(ValueError, match="All indices must be in range"):
            indices_to_mask([-1, 2], [1.0, 0.5], n_concepts=5, batch_size=1)

    def test_mismatched_lengths(self):
        """Test that mismatched c_idxs and c_vals lengths raise errors."""
        with pytest.raises(ValueError, match="must match c_idxs length"):
            indices_to_mask([0, 1, 2], [1.0, 0.5], n_concepts=5, batch_size=1)

    def test_wrong_batch_size(self):
        """Test that wrong batch size in c_vals raises error."""
        c_vals = torch.tensor([[1.0, 0.5],
                               [0.0, 1.0]])  # 2 batches
        with pytest.raises(ValueError, match="must match batch_size"):
            indices_to_mask([0, 1], c_vals, n_concepts=3, batch_size=3)

    def test_integration_with_intervention(self):
        """Test that indices_to_mask works with intervention strategies."""
        import torch.nn as nn
        from torch_concepts.nn import DoIntervention

        # Create a simple model
        model = nn.Linear(5, 3)

        # Define index-based intervention
        c_idxs = [0, 2]
        c_vals = [1.0, 0.0]
        n_concepts = 3
        batch_size = 4

        # Convert to mask-based format
        mask, target = indices_to_mask(c_idxs, c_vals, n_concepts, batch_size)

        # Create intervention with constant values matching target
        intervention_vals = torch.tensor([1.0, 0.0, 0.0])
        strategy = DoIntervention(model, intervention_vals)

        # Create a simple wrapper to test
        class DummyModule(nn.Module):
            def forward(self, **kwargs):
                return torch.randn(batch_size, n_concepts)

        dummy = DummyModule()
        wrapped = strategy.query(dummy, mask)

        # Test that it runs without error
        output = wrapped()
        assert output.shape == (batch_size, n_concepts)

        # Check that intervened positions match target values
        # (within the mask: where mask is 0, output should match target)
        intervened_mask = (mask == 0)
        for i in range(batch_size):
            for j in range(n_concepts):
                if intervened_mask[i, j]:
                    assert torch.isclose(output[i, j], target[i, j], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

