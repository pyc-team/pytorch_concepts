"""
Extended tests for torch_concepts.data.scalers to increase coverage.
"""
import pytest
import torch


class TestZerosToOne:
    """Tests for zeros_to_one_ helper function."""

    def test_zeros_to_one_scalar_zero(self):
        """Test zeros_to_one_ with scalar zero value."""
        from torch_concepts.data.scalers.standard import zeros_to_one_

        # Test with scalar zero - should return 1.0
        result = zeros_to_one_(0.0)
        assert result == 1.0

    def test_zeros_to_one_scalar_nonzero(self):
        """Test zeros_to_one_ with scalar non-zero value."""
        from torch_concepts.data.scalers.standard import zeros_to_one_

        # Test with scalar non-zero - should return the value
        result = zeros_to_one_(2.5)
        assert result == 2.5

    def test_zeros_to_one_scalar_near_zero(self):
        """Test zeros_to_one_ with scalar near-zero value."""
        from torch_concepts.data.scalers.standard import zeros_to_one_

        # Test with scalar very small value - should return 1.0
        result = zeros_to_one_(1e-20)
        assert result == 1.0

    def test_zeros_to_one_tensor(self):
        """Test zeros_to_one_ with tensor input."""
        from torch_concepts.data.scalers.standard import zeros_to_one_

        scales = torch.tensor([1.0, 0.0, 2.5, 1e-20])
        result = zeros_to_one_(scales)

        # Zeros and near-zeros should be 1.0
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] == 2.5
        assert result[3] == 1.0


class TestStandardScalerExtended:
    """Extended tests for StandardScaler."""

    def test_standard_scaler_fit_transform(self):
        """Test StandardScaler fit and transform."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scaler = StandardScaler()
        data = torch.randn(100, 5) * 10 + 5

        # Fit the scaler
        scaler.fit(data)

        # Transform the data
        transformed = scaler.transform(data)

        # Check that mean is close to 0 and std is close to 1
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(5), atol=0.1)
        assert torch.allclose(transformed.std(dim=0), torch.ones(5), atol=0.1)

    def test_standard_scaler_inverse_transform(self):
        """Test StandardScaler inverse transform."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scaler = StandardScaler()
        data = torch.randn(100, 5) * 10 + 5

        scaler.fit(data)
        transformed = scaler.transform(data)
        reconstructed = scaler.inverse_transform(transformed)

        assert torch.allclose(data, reconstructed, atol=0.01)

    def test_standard_scaler_1d_data(self):
        """Test StandardScaler with 1D data."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scaler = StandardScaler()
        data = torch.randn(100) * 10 + 5

        scaler.fit(data)
        transformed = scaler.transform(data)

        assert transformed.shape == data.shape

    def test_standard_scaler_constant_feature(self):
        """Test StandardScaler with constant feature (zero variance)."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scaler = StandardScaler()
        # Create data with one constant feature
        data = torch.randn(100, 3)
        data[:, 1] = 5.0  # Constant feature

        scaler.fit(data)
        transformed = scaler.transform(data)

        # Constant feature should remain constant (std = 1 from zeros_to_one_)
        assert torch.allclose(transformed[:, 1], torch.zeros(100), atol=0.01)

    def test_standard_scaler_fit_transform_chaining(self):
        """Test StandardScaler fit_transform method chaining."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scaler = StandardScaler()
        data = torch.randn(100, 5) * 10 + 5

        # fit() should return self for chaining
        result = scaler.fit(data)
        assert result is scaler

        # Now we can transform
        transformed = scaler.transform(data)
        assert transformed.shape == data.shape

    def test_standard_scaler_different_axis(self):
        """Test StandardScaler with different axis parameter."""
        from torch_concepts.data.scalers.standard import StandardScaler

        scaler = StandardScaler(axis=1)
        data = torch.randn(10, 100)

        scaler.fit(data)
        transformed = scaler.transform(data)

        # Should normalize along axis 1
        assert transformed.shape == data.shape
