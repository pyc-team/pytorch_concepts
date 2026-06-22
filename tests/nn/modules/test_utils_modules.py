"""
Tests for the indices_to_mask helper function and GroupConfig.

This tests the conversion from index-based interventions to mask-based format,
plus GroupConfig utility methods.
"""
import torch
import pytest
from torch_concepts.nn.modules.utils import indices_to_mask, GroupConfig


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
        """Test that indices_to_mask output is compatible with DoIntervention."""
        from torch_concepts.nn import DoIntervention

        c_idxs = [0, 2]
        c_vals = [1.0, 0.0]
        n_concepts = 3
        batch_size = 4

        mask, target = indices_to_mask(c_idxs, c_vals, n_concepts, batch_size)

        # DoIntervention replaces concept predictions with constants
        intervention_vals = torch.tensor([1.0, 0.0, 0.0])
        strategy = DoIntervention(intervention_vals)

        concept_preds = torch.randn(batch_size, n_concepts)
        output = strategy(concept_preds)
        assert output.shape == (batch_size, n_concepts)

        # Every row should equal the intervention constants
        for i in range(batch_size):
            assert torch.allclose(output[i], intervention_vals, atol=1e-5)


class TestGroupConfig:
    """Test suite for GroupConfig utility class."""

    def test_len(self):
        """__len__ returns the number of configured groups (line 86)."""
        gc = GroupConfig(binary="bce", categorical="ce")
        assert len(gc) == 2

    def test_len_empty(self):
        """__len__ returns 0 for empty config."""
        gc = GroupConfig()
        assert len(gc) == 0

    def test_values(self):
        """values() returns dict values (line 102)."""
        gc = GroupConfig(binary="bce", categorical="ce")
        vals = list(gc.values())
        assert "bce" in vals
        assert "ce" in vals

    def test_to_dict(self):
        """to_dict() returns a plain copy of the internal dict (line 110)."""
        gc = GroupConfig(binary="bce")
        d = gc.to_dict()
        assert isinstance(d, dict)
        assert d == {"binary": "bce"}
        # Modifying the returned dict should not affect gc
        d["binary"] = "changed"
        assert gc["binary"] == "bce"

    def test_from_dict(self):
        """from_dict() creates GroupConfig from a plain dict (line 122)."""
        d = {"binary": "bce", "categorical": "ce"}
        gc = GroupConfig.from_dict(d)
        assert gc["binary"] == "bce"
        assert gc["categorical"] == "ce"
        assert len(gc) == 2

    def test_from_dict_roundtrip(self):
        """from_dict -> to_dict roundtrip preserves values."""
        d = {"binary": "bce"}
        gc = GroupConfig.from_dict(d)
        assert gc.to_dict() == d


class TestIndicesToMaskErrors:
    """Additional error-path tests for indices_to_mask."""

    def test_non_1d_c_idxs_raises(self):
        """2-D c_idxs tensor raises ValueError (line 308)."""
        c_idxs = torch.tensor([[0, 1]])  # shape (1, 2) — not 1-D
        c_vals = torch.tensor([1.0, 0.5])
        with pytest.raises(ValueError, match="c_idxs must be 1-D"):
            indices_to_mask(c_idxs, c_vals, n_concepts=5, batch_size=1)

    def test_2d_c_vals_wrong_k_raises(self):
        """2-D c_vals with mismatched K raises ValueError (line 322)."""
        c_idxs = torch.tensor([0, 1])        # K=2
        c_vals = torch.tensor([[1.0, 0.5, 0.3],
                               [0.0, 0.2, 0.8]])  # shape (2, 3) — K=3 mismatch
        with pytest.raises(ValueError, match="c_vals second dim"):
            indices_to_mask(c_idxs, c_vals, n_concepts=5, batch_size=2)

    def test_3d_c_vals_raises(self):
        """3-D c_vals raises ValueError (line 326)."""
        c_idxs = torch.tensor([0, 1])
        c_vals = torch.ones(2, 2, 2)  # 3-D
        with pytest.raises(ValueError, match="c_vals must be 1-D or 2-D"):
            indices_to_mask(c_idxs, c_vals, n_concepts=5, batch_size=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

