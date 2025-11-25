"""
Comprehensive tests for torch_concepts.data.base.scaler to increase coverage.
"""
import pytest
import torch
from torch_concepts.data.base.scaler import Scaler


class ConcreteScaler(Scaler):
    """Concrete implementation of Scaler for testing."""

    def fit(self, x, dim=0):
        """Fit by computing mean and std."""
        self.mean = x.mean(dim=dim, keepdim=True)
        self.std = x.std(dim=dim, keepdim=True)
        return self

    def transform(self, x):
        """Transform using mean and std."""
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, x):
        """Inverse transform."""
        return x * (self.std + 1e-8) + self.mean


class MinimalScaler(Scaler):
    """Minimal scaler that does nothing."""

    def fit(self, x, dim=0):
        return self

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class TestScalerAbstractBase:
    """Tests for Scaler abstract base class."""

    def test_scaler_cannot_be_instantiated(self):
        """Test that Scaler abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            scaler = Scaler()

    def test_concrete_scaler_can_be_instantiated(self):
        """Test that concrete implementation can be instantiated."""
        scaler = ConcreteScaler()
        assert isinstance(scaler, Scaler)

    def test_scaler_default_initialization(self):
        """Test Scaler initialization with default values."""
        scaler = ConcreteScaler()
        assert scaler.bias == 0.0
        assert scaler.scale == 1.0

    def test_scaler_custom_initialization(self):
        """Test Scaler initialization with custom values."""
        scaler = ConcreteScaler(bias=5.0, scale=2.0)
        assert scaler.bias == 5.0
        assert scaler.scale == 2.0

    def test_concrete_scaler_fit_method(self):
        """Test that fit method works correctly."""
        scaler = ConcreteScaler()
        data = torch.randn(100, 5)

        result = scaler.fit(data, dim=0)

        # fit should return self for chaining
        assert result is scaler
        assert hasattr(scaler, 'mean')
        assert hasattr(scaler, 'std')

    def test_concrete_scaler_transform_method(self):
        """Test that transform method works correctly."""
        scaler = ConcreteScaler()
        data = torch.randn(100, 5)

        scaler.fit(data, dim=0)
        transformed = scaler.transform(data)

        assert transformed.shape == data.shape
        # Transformed data should have mean ~0 and std ~1
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(5), atol=1e-5)
        assert torch.allclose(transformed.std(dim=0), torch.ones(5), atol=1e-1)

    def test_concrete_scaler_inverse_transform_method(self):
        """Test that inverse_transform method works correctly."""
        scaler = ConcreteScaler()
        data = torch.randn(100, 5)

        scaler.fit(data, dim=0)
        transformed = scaler.transform(data)
        recovered = scaler.inverse_transform(transformed)

        # Should recover original data
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_scaler_fit_transform_method(self):
        """Test that fit_transform method works correctly."""
        scaler = ConcreteScaler()
        data = torch.randn(100, 5)

        transformed = scaler.fit_transform(data, dim=0)

        assert transformed.shape == data.shape
        assert hasattr(scaler, 'mean')
        assert hasattr(scaler, 'std')
        # Should be same as calling fit then transform
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(5), atol=1e-5)

    def test_scaler_fit_transform_different_dims(self):
        """Test fit_transform with different dim parameter."""
        scaler = ConcreteScaler()
        data = torch.randn(10, 20, 5)

        # Fit along dim=1
        transformed = scaler.fit_transform(data, dim=1)

        assert transformed.shape == data.shape
        assert scaler.mean.shape[1] == 1  # Reduced along dim=1

    def test_minimal_scaler_identity(self):
        """Test minimal scaler that does identity transformation."""
        scaler = MinimalScaler()
        data = torch.randn(50, 3)

        transformed = scaler.fit_transform(data)

        # Should be identity
        assert torch.allclose(transformed, data)

    def test_scaler_preserves_dtype(self):
        """Test that scaler preserves tensor dtype."""
        scaler = MinimalScaler()

        # Test with float32
        data_f32 = torch.randn(10, 5, dtype=torch.float32)
        result_f32 = scaler.fit_transform(data_f32)
        assert result_f32.dtype == torch.float32

        # Test with float64
        data_f64 = torch.randn(10, 5, dtype=torch.float64)
        result_f64 = scaler.fit_transform(data_f64)
        assert result_f64.dtype == torch.float64

    def test_scaler_with_1d_tensor(self):
        """Test scaler with 1D tensor."""
        scaler = ConcreteScaler()
        data = torch.randn(100)

        transformed = scaler.fit_transform(data, dim=0)

        assert transformed.shape == data.shape

    def test_scaler_with_3d_tensor(self):
        """Test scaler with 3D tensor."""
        scaler = ConcreteScaler()
        data = torch.randn(10, 20, 30)

        transformed = scaler.fit_transform(data, dim=0)

        assert transformed.shape == data.shape

    def test_scaler_method_chaining(self):
        """Test that fit returns self for method chaining."""
        scaler = ConcreteScaler()
        data = torch.randn(100, 5)

        # Should be able to chain fit().transform()
        result = scaler.fit(data).transform(data)

        assert result is not None
        assert result.shape == data.shape


class TestScalerEdgeCases:
    """Tests for edge cases in Scaler implementations."""

    def test_scaler_with_constant_data(self):
        """Test scaler with constant data (zero std)."""
        scaler = ConcreteScaler()
        data = torch.ones(100, 5) * 3.0  # All values are 3.0

        scaler.fit(data, dim=0)
        transformed = scaler.transform(data)

        # Should handle zero std gracefully (due to epsilon)
        assert not torch.isnan(transformed).any()
        assert not torch.isinf(transformed).any()

    def test_scaler_with_single_sample(self):
        """Test scaler with single sample."""
        scaler = MinimalScaler()
        data = torch.randn(1, 5)

        transformed = scaler.fit_transform(data, dim=0)

        assert transformed.shape == data.shape

    def test_scaler_with_empty_metadata(self):
        """Test that scaler works without using bias/scale attributes."""
        scaler = ConcreteScaler(bias=0.0, scale=1.0)
        data = torch.randn(50, 3)

        # Just verify it doesn't break with these attributes
        assert scaler.bias == 0.0
        assert scaler.scale == 1.0

        scaler.fit_transform(data)

    def test_scaler_roundtrip_consistency(self):
        """Test that transform -> inverse_transform is consistent."""
        scaler = ConcreteScaler()

        # Test multiple times with different data
        for _ in range(5):
            data = torch.randn(100, 10)
            scaler.fit(data, dim=0)

            transformed = scaler.transform(data)
            recovered = scaler.inverse_transform(transformed)

            assert torch.allclose(recovered, data, atol=1e-4)


class TestScalerSubclassRequirements:
    """Tests that verify subclass implementations."""

    def test_incomplete_scaler_raises_error(self):
        """Test that incomplete implementation raises TypeError."""

        class IncompleteScaler(Scaler):
            # Missing all abstract methods
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            scaler = IncompleteScaler()

    def test_partial_scaler_raises_error(self):
        """Test that partially implemented scaler raises TypeError."""

        class PartialScaler(Scaler):
            def fit(self, x, dim=0):
                return self
            # Missing transform and inverse_transform

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            scaler = PartialScaler()

    def test_all_methods_required(self):
        """Test that all abstract methods must be implemented."""

        # This should work - all methods implemented
        class CompleteScaler(Scaler):
            def fit(self, x, dim=0):
                return self

            def transform(self, x):
                return x

            def inverse_transform(self, x):
                return x

        scaler = CompleteScaler()
        assert isinstance(scaler, Scaler)

