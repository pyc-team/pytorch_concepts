"""
Comprehensive tests for torch_concepts.nn.modules.low.inference

Tests inference and intervention modules.
"""
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
        policy_logits = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_logits)

        self.assertEqual(mask.shape, (2, 5))
        # With quantile=0, should keep most concepts

    def test_build_mask_all_replace(self):
        """Test mask building with quantile=1 (replace all)."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        wrapper = _InterventionWrapper(original, policy, strategy, quantile=1.0)
        policy_logits = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_logits)

        self.assertEqual(mask.shape, (2, 5))

    def test_build_mask_with_subset(self):
        """Test mask building with subset selection."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        subset = [0, 2, 4]
        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5, subset=subset)
        policy_logits = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_logits)

        self.assertEqual(mask.shape, (2, 5))

    def test_build_mask_single_concept_subset(self):
        """Test mask building with single concept in subset."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        subset = [2]
        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5, subset=subset)
        policy_logits = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_logits)

        self.assertEqual(mask.shape, (2, 5))

    def test_build_mask_empty_subset(self):
        """Test mask building with empty subset."""
        original = SimpleModule(10, 5)
        policy = nn.Linear(5, 5)
        model = nn.Linear(10, 5)
        strategy = ConcreteRewiringIntervention(model)

        subset = []
        wrapper = _InterventionWrapper(original, policy, strategy, quantile=0.5, subset=subset)
        policy_logits = torch.randn(2, 5)
        mask = wrapper._build_mask(policy_logits)

        # Empty subset should return all ones (keep all)
        self.assertTrue(torch.allclose(mask, torch.ones_like(policy_logits)))

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


if __name__ == '__main__':
    unittest.main()
