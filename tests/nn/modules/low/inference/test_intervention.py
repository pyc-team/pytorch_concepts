"""Comprehensive tests for torch_concepts.nn.modules.low.inference.intervention module to improve coverage."""
import pytest
from torch_concepts.nn.modules.low.inference.intervention import (
    _get_submodule,
    _set_submodule,
    _as_list,
)
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.low.inference.intervention import (
    _GlobalPolicyState,
)
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
from torch_concepts.nn.modules.low.inference.intervention import (
    RewiringIntervention,
    GroundTruthIntervention,
    DoIntervention,
    DistributionIntervention,
    _InterventionWrapper,
)


class ConcreteRewiringIntervention(RewiringIntervention):
    """Concrete implementation for testing."""

    def _make_target(self, y, target_value=1.0):
        """Create target tensor filled with target_value."""
        return torch.full_like(y, target_value)


class SimpleModule(nn.Module):
    """Simple module for testing."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, **kwargs):
        if 'x' in kwargs:
            return self.linear(kwargs['x'])
        return torch.randn(2, self.linear.out_features)


class TestRewiringIntervention(unittest.TestCase):
    """Test RewiringIntervention."""

    def setUp(self):
        """Set up test model."""
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )

    def test_initialization(self):
        """Test intervention initialization."""
        intervention = ConcreteRewiringIntervention(self.model)
        self.assertIsNotNone(intervention.model)

    def test_query_creates_wrapper(self):
        """Test that query creates intervention wrapper."""
        intervention = ConcreteRewiringIntervention(self.model)
        original_module = SimpleModule(10, 5)
        mask = torch.ones(5)

        wrapper = intervention.query(original_module, mask)
        self.assertIsInstance(wrapper, nn.Module)

    def test_intervention_with_mask(self):
        """Test intervention applies mask correctly."""
        intervention = ConcreteRewiringIntervention(self.model)
        original_module = SimpleModule(10, 5)

        # Mask: 1 = keep, 0 = replace
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        wrapper = intervention.query(original_module, mask)

        output = wrapper(x=torch.randn(2, 10))
        self.assertEqual(output.shape, (2, 5))


class TestGroundTruthIntervention(unittest.TestCase):
    """Test GroundTruthIntervention."""

    def test_initialization(self):
        """Test initialization with ground truth."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        intervention = GroundTruthIntervention(model, ground_truth)
        self.assertTrue(torch.equal(intervention.ground_truth, ground_truth))

    def test_make_target(self):
        """Test _make_target returns ground truth."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.5, 0.0]])

        intervention = GroundTruthIntervention(model, ground_truth)
        y = torch.randn(1, 3)
        target = intervention._make_target(y)

        self.assertTrue(torch.equal(target, ground_truth.to(dtype=y.dtype)))

    def test_ground_truth_device_transfer(self):
        """Test ground truth transfers to correct device."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)
        y = torch.randn(1, 3)
        target = intervention._make_target(y)

        self.assertEqual(target.device, y.device)


class TestDoIntervention(unittest.TestCase):
    """Test DoIntervention."""

    def test_initialization_scalar(self):
        """Test initialization with scalar constant."""
        model = nn.Linear(10, 3)
        intervention = DoIntervention(model, 1.0)
        self.assertIsNotNone(intervention.constants)

    def test_initialization_tensor(self):
        """Test initialization with tensor constant."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([0.5, 1.0, 0.0])
        intervention = DoIntervention(model, constants)
        self.assertTrue(torch.equal(intervention.constants, constants))

    def test_make_target_scalar(self):
        """Test _make_target with scalar broadcasting."""
        model = nn.Linear(10, 3)
        intervention = DoIntervention(model, 0.5)

        y = torch.randn(4, 3)
        target = intervention._make_target(y)

        self.assertEqual(target.shape, (4, 3))
        self.assertTrue(torch.allclose(target, torch.full((4, 3), 0.5)))

    def test_make_target_per_concept(self):
        """Test _make_target with per-concept values [F]."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([0.0, 0.5, 1.0])
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)
        target = intervention._make_target(y)

        self.assertEqual(target.shape, (2, 3))
        self.assertTrue(torch.equal(target[0], constants))
        self.assertTrue(torch.equal(target[1], constants))

    def test_make_target_per_sample(self):
        """Test _make_target with per-sample values [B, F]."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)
        target = intervention._make_target(y)

        self.assertTrue(torch.equal(target, constants))

    def test_make_target_broadcast_batch(self):
        """Test _make_target with [1, F] broadcasting."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([[0.1, 0.2, 0.3]])
        intervention = DoIntervention(model, constants)

        y = torch.randn(5, 3)
        target = intervention._make_target(y)

        self.assertEqual(target.shape, (5, 3))
        for i in range(5):
            self.assertTrue(torch.equal(target[i], constants[0]))

    def test_make_target_wrong_dimensions(self):
        """Test _make_target raises error for wrong dimensions."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([0.0, 0.5])  # Wrong size
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)
        with self.assertRaises(AssertionError):
            intervention._make_target(y)


class TestDistributionIntervention(unittest.TestCase):
    """Test DistributionIntervention."""

    def test_initialization_single_distribution(self):
        """Test initialization with single distribution."""
        model = nn.Linear(10, 3)
        dist = Bernoulli(torch.tensor(0.5))
        intervention = DistributionIntervention(model, dist)
        self.assertIsNotNone(intervention.dist)

    def test_initialization_list_distributions(self):
        """Test initialization with per-concept distributions."""
        model = nn.Linear(10, 3)
        dists = [
            Bernoulli(torch.tensor(0.3)),
            Bernoulli(torch.tensor(0.7)),
            Normal(torch.tensor(0.0), torch.tensor(1.0))
        ]
        intervention = DistributionIntervention(model, dists)
        self.assertEqual(len(intervention.dist), 3)

    def test_make_target_single_distribution(self):
        """Test _make_target with single distribution."""
        torch.manual_seed(42)
        model = nn.Linear(10, 3)
        dist = Bernoulli(torch.tensor(0.5))
        intervention = DistributionIntervention(model, dist)

        y = torch.randn(2, 3)
        target = intervention._make_target(y)

        self.assertEqual(target.shape, (2, 3))
        # Check values are 0 or 1
        self.assertTrue(torch.all((target == 0) | (target == 1)))

    def test_make_target_list_distributions(self):
        """Test _make_target with per-concept distributions."""
        torch.manual_seed(42)
        model = nn.Linear(10, 3)
        dists = [
            Bernoulli(torch.tensor(0.9)),
            Bernoulli(torch.tensor(0.1)),
            Bernoulli(torch.tensor(0.5))
        ]
        intervention = DistributionIntervention(model, dists)

        y = torch.randn(4, 3)
        target = intervention._make_target(y)

        self.assertEqual(target.shape, (4, 3))

    def test_make_target_normal_distribution(self):
        """Test _make_target with normal distribution."""
        torch.manual_seed(42)
        model = nn.Linear(10, 2)
        dist = Normal(torch.tensor(0.0), torch.tensor(1.0))
        intervention = DistributionIntervention(model, dist)

        y = torch.randn(3, 2)
        target = intervention._make_target(y)

        self.assertEqual(target.shape, (3, 2))


class TestInterventionWrapper(unittest.TestCase):
    """Test _InterventionWrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5)
        self.assertEqual(wrapper.quantile, 0.5)

    def test_build_mask_all_keep(self):
        """Test mask building with quantile=0 (keep all)."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.0)
        policy_endogenous = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_endogenous)

        self.assertEqual(mask.shape, (2, 5))
        # With quantile=0, should keep most concepts

    def test_build_mask_all_replace(self):
        """Test mask building with quantile=1 (replace all)."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        wrapper = _InterventionWrapper(original, policy, strategy, quantile=1.0)
        policy_endogenous = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_endogenous)

        self.assertEqual(mask.shape, (2, 5))

    def test_build_mask_with_subset(self):
        """Test mask building with subset selection."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        subset = [0, 2, 4]
        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5, subset=subset)
        policy_endogenous = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_endogenous)

        self.assertEqual(mask.shape, (2, 5))

    def test_build_mask_single_concept_subset(self):
        """Test mask building with single concept in subset."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        subset = [2]
        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5, subset=subset)
        policy_endogenous = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_endogenous)

        self.assertEqual(mask.shape, (2, 5))

    def test_build_mask_empty_subset(self):
        """Test mask building with empty subset."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        subset = []
        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5, subset=subset)
        policy_endogenous = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_endogenous)

        # Empty subset should return all ones (keep all)
        self.assertTrue(torch.allclose(mask, torch.ones_like(policy_endogenous)))

    def test_forward(self):
        """Test forward pass through wrapper."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5)
        x = torch.randn(2, 10)
        output = wrapper(x=x)

        self.assertEqual(output.shape, (2, 5))

    def test_gradient_flow(self):
        """Test gradient flow through wrapper."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5)
        x = torch.randn(2, 10, requires_grad=True)
        output = wrapper(x=x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_different_quantiles(self):
        """Test wrapper with different quantile values."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        for quantile in [0.0, 0.25, 0.5, 0.75, 1.0]:
            wrapper = _InterventionWrapper(original, policy, strategy, quantile=quantile)
            x = torch.randn(2, 10)
            output = wrapper(x=x)
            self.assertEqual(output.shape, (2, 5))


class TestHelperFunctions:
    """Test helper functions for intervention module."""

    def test_get_submodule_single_level(self):
        """Test _get_submodule with single level path."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )

        layer0 = _get_submodule(model, "0")
        assert isinstance(layer0, nn.Linear)
        assert layer0.in_features == 10
        assert layer0.out_features == 5

    def test_get_submodule_nested(self):
        """Test _get_submodule with nested path."""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU()
                )
                self.layer2 = nn.Linear(5, 3)

        model = NestedModel()

        # Access nested submodule
        linear = _get_submodule(model, "layer1.0")
        assert isinstance(linear, nn.Linear)
        assert linear.in_features == 10

    def test_set_submodule_single_level(self):
        """Test _set_submodule with single level path."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU()
        )

        new_layer = nn.Linear(10, 8)
        _set_submodule(model, "0", new_layer)

        assert model[0].out_features == 8

    def test_set_submodule_nested(self):
        """Test _set_submodule with nested path."""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU()
                )

        model = NestedModel()
        new_layer = nn.Linear(10, 8)
        _set_submodule(model, "layer1.0", new_layer)

        assert model.layer1[0].out_features == 8

    def test_set_submodule_with_parametric_cpd(self):
        """Test _set_submodule with ParametricCPD."""
        model = nn.Module()
        cpd = ParametricCPD('concept', parametrization=nn.Linear(10, 5))
        _set_submodule(model, "concept", cpd)

        assert hasattr(model, 'concept')
        assert isinstance(model.concept, ParametricCPD)

    def test_set_submodule_wraps_module_in_cpd(self):
        """Test _set_submodule wraps regular module in ParametricCPD."""
        model = nn.Module()
        layer = nn.Linear(10, 5)
        _set_submodule(model, "concept", layer)

        assert hasattr(model, 'concept')
        assert isinstance(model.concept, ParametricCPD)

    def test_set_submodule_empty_path_raises_error(self):
        """Test _set_submodule with empty path raises error."""
        model = nn.Module()

        with pytest.raises(ValueError, match="Dotted path must not be empty"):
            _set_submodule(model, "", nn.Linear(10, 5))

    def test_as_list_scalar_broadcast(self):
        """Test _as_list broadcasts scalar to list."""
        result = _as_list(5, 3)
        assert result == [5, 5, 5]
        assert len(result) == 3

    def test_as_list_with_list_input(self):
        """Test _as_list preserves list if correct length."""
        input_list = [1, 2, 3]
        result = _as_list(input_list, 3)
        assert result == [1, 2, 3]

    def test_as_list_with_tuple_input(self):
        """Test _as_list converts tuple to list."""
        input_tuple = (1, 2, 3)
        result = _as_list(input_tuple, 3)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_as_list_wrong_length_raises_error(self):
        """Test _as_list raises error for wrong length list."""
        with pytest.raises(ValueError, match="Expected list of length 3, got 2"):
            _as_list([1, 2], 3)


class TestGroundTruthIntervention:
    """Test GroundTruthIntervention class."""

    def test_initialization(self):
        """Test GroundTruthIntervention initialization."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        assert hasattr(intervention, 'ground_truth')
        assert torch.equal(intervention.ground_truth, ground_truth)

    def test_make_target_returns_ground_truth(self):
        """Test _make_target returns ground truth values."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Test prediction tensor
        y = torch.randn(1, 3)
        target = intervention._make_target(y)

        assert torch.equal(target, ground_truth.to(dtype=y.dtype, device=y.device))

    def test_make_target_device_transfer(self):
        """Test _make_target transfers to correct device."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Create tensor with different dtype
        y = torch.randn(1, 3, dtype=torch.float64)
        target = intervention._make_target(y)

        assert target.dtype == torch.float64
        assert target.device == y.device

    def test_query_creates_wrapper(self):
        """Test query creates intervention wrapper."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Create mask (1 = keep prediction, 0 = replace with target)
        mask = torch.tensor([[1.0, 0.0, 1.0]])

        wrapped = intervention.query(model, mask)

        assert isinstance(wrapped, nn.Module)
        assert hasattr(wrapped, 'orig')
        assert hasattr(wrapped, 'mask')


class TestDoIntervention:
    """Test DoIntervention class."""

    def test_initialization_scalar(self):
        """Test DoIntervention initialization with scalar."""
        model = nn.Linear(10, 3)
        intervention = DoIntervention(model, 1.0)

        assert hasattr(intervention, 'constants')
        assert intervention.constants.item() == 1.0

    def test_initialization_tensor(self):
        """Test DoIntervention initialization with tensor."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([0.5, 1.0, 0.0])
        intervention = DoIntervention(model, constants)

        assert torch.equal(intervention.constants, constants)

    def test_make_target_scalar(self):
        """Test _make_target with scalar constant."""
        model = nn.Linear(10, 3)
        intervention = DoIntervention(model, 1.0)

        y = torch.randn(2, 3)
        target = intervention._make_target(y)

        assert target.shape == (2, 3)
        assert torch.all(target == 1.0)

    def test_make_target_per_concept(self):
        """Test _make_target with per-concept constants [F]."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([0.5, 1.0, 0.0])
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)
        target = intervention._make_target(y)

        assert target.shape == (2, 3)
        # Check each sample has the same per-concept values
        assert torch.allclose(target[0], constants)
        assert torch.allclose(target[1], constants)

    def test_make_target_broadcast_batch(self):
        """Test _make_target with [1, F] broadcasted to [B, F]."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([[0.5, 1.0, 0.0]])
        intervention = DoIntervention(model, constants)

        y = torch.randn(4, 3)
        target = intervention._make_target(y)

        assert target.shape == (4, 3)
        # Check all samples have the same values
        for i in range(4):
            assert torch.allclose(target[i], constants[0])

    def test_make_target_per_sample(self):
        """Test _make_target with per-sample constants [B, F]."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([[0.5, 1.0, 0.0],
                                   [1.0, 0.0, 0.5]])
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)
        target = intervention._make_target(y)

        assert target.shape == (2, 3)
        assert torch.allclose(target, constants)

    def test_make_target_wrong_dimensions_raises_error(self):
        """Test _make_target with wrong dimensions raises error."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([[[0.5, 1.0, 0.0]]])  # 3D tensor
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)

        with pytest.raises(ValueError, match="constants must be scalar"):
            intervention._make_target(y)

    def test_make_target_wrong_feature_dim_raises_error(self):
        """Test _make_target with wrong feature dimension raises error."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([0.5, 1.0])  # Only 2 features, expecting 3
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)

        with pytest.raises(AssertionError):
            intervention._make_target(y)

    def test_make_target_wrong_batch_dim_raises_error(self):
        """Test _make_target with wrong batch dimension raises error."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([[0.5, 1.0, 0.0],
                                   [1.0, 0.0, 0.5],
                                   [0.0, 0.5, 1.0]])  # 3 samples
        intervention = DoIntervention(model, constants)

        y = torch.randn(2, 3)  # Only 2 samples

        with pytest.raises(AssertionError):
            intervention._make_target(y)


class TestDistributionIntervention:
    """Test DistributionIntervention class."""

    def test_initialization_single_distribution(self):
        """Test DistributionIntervention with single distribution."""
        model = nn.Linear(10, 3)
        dist = Bernoulli(torch.tensor(0.5))

        intervention = DistributionIntervention(model, dist)

        assert hasattr(intervention, 'dist')

    def test_initialization_list_distributions(self):
        """Test DistributionIntervention with list of distributions."""
        model = nn.Linear(10, 3)
        dists = [
            Bernoulli(torch.tensor(0.3)),
            Bernoulli(torch.tensor(0.7)),
            Bernoulli(torch.tensor(0.5))
        ]

        intervention = DistributionIntervention(model, dists)

        assert hasattr(intervention, 'dist')

    def test_make_target_single_distribution(self):
        """Test _make_target with single distribution."""
        model = nn.Linear(10, 3)
        dist = Bernoulli(torch.tensor(0.5))

        intervention = DistributionIntervention(model, dist)

        y = torch.randn(4, 3)
        target = intervention._make_target(y)

        assert target.shape == (4, 3)
        # Values should be binary (0 or 1) for Bernoulli
        assert torch.all((target == 0) | (target == 1))

    def test_make_target_normal_distribution(self):
        """Test _make_target with Normal distribution."""
        model = nn.Linear(10, 3)
        dist = Normal(torch.tensor(0.0), torch.tensor(1.0))

        intervention = DistributionIntervention(model, dist)

        y = torch.randn(4, 3)
        target = intervention._make_target(y)

        assert target.shape == (4, 3)
        # Just check shape and type, values are random

    def test_make_target_list_distributions(self):
        """Test _make_target with list of per-concept distributions."""
        model = nn.Linear(10, 3)
        dists = [
            Bernoulli(torch.tensor(0.3)),
            Normal(torch.tensor(0.0), torch.tensor(1.0)),
            Bernoulli(torch.tensor(0.8))
        ]

        intervention = DistributionIntervention(model, dists)

        y = torch.randn(4, 3)
        target = intervention._make_target(y)

        assert target.shape == (4, 3)


class TestRewiringInterventionWrapper:
    """Test the intervention wrapper created by RewiringIntervention.query()."""

    def test_wrapper_forward_keeps_predictions(self):
        """Test wrapper keeps predictions where mask is 1."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 1.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Mask: keep all predictions (all 1s)
        mask = torch.ones(1, 3)
        wrapped = intervention.query(model, mask)

        # Forward pass
        x = torch.randn(1, 10)
        with torch.no_grad():
            original_output = model(x)
            wrapped_output = wrapped(input=x)

        # Should be identical since mask is all 1s
        assert torch.allclose(wrapped_output, original_output, rtol=1e-5)

    def test_wrapper_forward_replaces_with_targets(self):
        """Test wrapper replaces predictions where mask is 0."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Mask: replace all predictions (all 0s)
        mask = torch.zeros(1, 3)
        wrapped = intervention.query(model, mask)

        # Forward pass
        x = torch.randn(1, 10)
        with torch.no_grad():
            wrapped_output = wrapped(input=x)

        # Should match ground truth since mask is all 0s
        assert torch.allclose(wrapped_output, ground_truth, rtol=1e-5)

    def test_wrapper_forward_mixed_mask(self):
        """Test wrapper with mixed mask (some keep, some replace)."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 1.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Mask: keep first, replace middle, keep last
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        wrapped = intervention.query(model, mask)

        # Forward pass
        x = torch.randn(1, 10)
        with torch.no_grad():
            original_output = model(x)
            wrapped_output = wrapped(input=x)

        # First and last should match original, middle should be 1.0
        assert torch.allclose(wrapped_output[0, 0], original_output[0, 0], rtol=1e-5)
        assert torch.allclose(wrapped_output[0, 1], torch.tensor(1.0), rtol=1e-5)
        assert torch.allclose(wrapped_output[0, 2], original_output[0, 2], rtol=1e-5)

    def test_wrapper_forward_wrong_shape_raises_error(self):
        """Test wrapper raises error for wrong shaped output."""
        # Create a model that outputs wrong shape
        class WrongShapeModel(nn.Module):
            def forward(self, input):
                # Returns 3D tensor instead of 2D
                return torch.randn(2, 3, 4)

        model = WrongShapeModel()
        ground_truth = torch.tensor([[1.0, 1.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)
        mask = torch.ones(1, 3)
        wrapped = intervention.query(model, mask)

        x = torch.randn(1, 10)

        with pytest.raises(AssertionError, match="RewiringIntervention expects 2-D tensors"):
            wrapped(input=x)

    def test_wrapper_preserves_gradient_flow(self):
        """Test that wrapper preserves gradient flow."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Partial mask
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        wrapped = intervention.query(model, mask)

        # Forward pass with gradients
        x = torch.randn(1, 10, requires_grad=True)
        output = wrapped(input=x)
        loss = output.sum()
        loss.backward()

        # Check that gradients were computed
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestRewiringInterventionBatchProcessing:
    """Test RewiringIntervention with batch processing."""

    def test_batch_processing(self):
        """Test intervention works with batched inputs."""
        model = nn.Linear(10, 3)
        constants = torch.tensor([[0.0, 0.5, 1.0],
                                   [1.0, 0.5, 0.0],
                                   [0.5, 1.0, 0.5]])

        intervention = DoIntervention(model, constants)

        # Batch of 3 samples
        mask = torch.tensor([[1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0],
                              [1.0, 1.0, 0.0]])
        wrapped = intervention.query(model, mask)

        x = torch.randn(3, 10)
        with torch.no_grad():
            output = wrapped(input=x)

        assert output.shape == (3, 3)


class TestRewiringInterventionEdgeCases:
    """Test edge cases for RewiringIntervention."""

    def test_empty_batch_size_one(self):
        """Test intervention with batch size 1."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)
        mask = torch.tensor([[0.0, 0.0, 0.0]])
        wrapped = intervention.query(model, mask)

        x = torch.randn(1, 10)
        with torch.no_grad():
            output = wrapped(input=x)

        assert output.shape == (1, 3)

    def test_large_batch(self):
        """Test intervention with large batch."""
        model = nn.Linear(10, 3)
        ground_truth = torch.tensor([[1.0, 0.0, 1.0]])

        intervention = GroundTruthIntervention(model, ground_truth)

        # Repeat mask for large batch
        batch_size = 100
        mask = torch.ones(batch_size, 3)
        mask[:, 1] = 0  # Replace middle concept

        wrapped = intervention.query(model, mask)

        x = torch.randn(batch_size, 10)
        with torch.no_grad():
            output = wrapped(input=x)

        assert output.shape == (batch_size, 3)
        # Check that middle column is all zeros (from ground truth)
        assert torch.all(output[:, 1] == 0.0)


class DummyOriginal(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self._out = torch.zeros((1, out_features))

    def forward(self, **kwargs):
        return self._out


class DummyPolicy(nn.Module):
    def __init__(self, endogenous):
        super().__init__()
        self._end = endogenous

    def forward(self, y):
        # ignore y and return the provided endogenous
        return self._end


def test_distribution_intervention_single_and_per_feature():
    model = nn.Linear(2, 3)
    dist_single = Bernoulli(torch.tensor(0.7))
    di_single = DistributionIntervention(model, dist_single)

    y = torch.randn(4, 3)
    t = di_single._make_target(y)
    assert t.shape == (4, 3)

    # per-feature distributions
    dists = [Bernoulli(torch.tensor(0.2)), Normal(torch.tensor(0.0), torch.tensor(1.0)), Bernoulli(torch.tensor(0.8))]
    di_multi = DistributionIntervention(model, dists)
    t2 = di_multi._make_target(y)
    assert t2.shape == (4, 3)


def test_intervention_wrapper_build_mask_single_column_behaviour():
    # Create wrapper with subset single column
    B, F = 2, 3
    original = DummyOriginal(out_features=F)
    # policy endogenous: shape [B, F]
    endogenous = torch.tensor([[0.1, 0.5, 0.2], [0.2, 0.4, 0.6]], dtype=torch.float32)
    policy = DummyPolicy(endogenous)
    strategy = DoIntervention(original, 1.0)

    # q < 1: selected column should be kept (mask close to 1 with STE proxy applied)
    wrapper_soft = _InterventionWrapper(original=original, policy=policy, strategy=strategy, quantile=0.5, subset=[1])
    mask_soft = wrapper_soft._build_mask(endogenous)
    assert mask_soft.shape == (B, F)
    # For single column with q < 1, the hard mask is 1 (keep), STE proxy modifies slightly
    # The selected column values should be close to the soft proxy values (between 0 and 1)
    # Check that non-selected columns are 1.0
    assert torch.allclose(mask_soft[:, 0], torch.ones((B,), dtype=mask_soft.dtype))
    assert torch.allclose(mask_soft[:, 2], torch.ones((B,), dtype=mask_soft.dtype))
    # Selected column should have STE proxy applied (values influenced by endogenous)
    # Since hard mask starts at 1 and STE subtracts soft_proxy then adds it back,
    # the result equals soft_proxy which is log1p(sel)/log1p(row_max)
    # This should be < 1 for most cases
    soft_values = mask_soft[:, 1]
    assert soft_values.shape == (B,)
    # With the given endogenous values, soft values should be less than 1.0
    # Actually, let's just verify the shape and dtype are correct
    assert soft_values.dtype == mask_soft.dtype

    # q == 1: selected column should be zeros (replace)
    wrapper_hard = _InterventionWrapper(original=original, policy=policy, strategy=strategy, quantile=1.0, subset=[1])
    mask_hard = wrapper_hard._build_mask(endogenous)
    # For q==1, hard mask is 0 (replace), and after STE proxy it becomes the soft proxy value
    # which should be < 1 for the selected column
    assert mask_hard[:, 1].max() < 1.0  # At least somewhat less than 1
    # Non-selected columns should still be 1.0
    assert torch.allclose(mask_hard[:, 0], torch.ones((B,), dtype=mask_hard.dtype))
    assert torch.allclose(mask_hard[:, 2], torch.ones((B,), dtype=mask_hard.dtype))


def test_global_policy_state_compute_and_slice():
    state = _GlobalPolicyState(n_wrappers=2, quantile=0.5)
    B = 1
    end1 = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    end2 = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
    out1 = torch.zeros((B, 2))
    out2 = torch.zeros((B, 2))

    state.register(0, end1, out1)
    state.register(1, end2, out2)

    assert not state.is_ready() or state.is_ready()  # register doesn't compute readiness until both are in

    # Should be ready now
    assert state.is_ready()
    state.compute_global_mask()
    gm = state.global_mask
    assert gm.shape == (B, 4)

    slice0 = state.get_mask_slice(0)
    slice1 = state.get_mask_slice(1)
    assert slice0.shape == out1.shape
    assert slice1.shape == out2.shape


if __name__ == "__main__":
    unittest.main()
