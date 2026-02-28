import unittest
from unittest.mock import patch, MagicMock
import pytest
from torch_concepts.nn.modules.low.inference.intervention import _GlobalPolicyInterventionWrapper
from torch.distributions import Normal
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept
from torch.nn import Linear, Identity
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical
from torch_concepts.data.datasets import ToyDataset
from torch_concepts import InputVariable, EndogenousVariable, ExogenousVariable, Annotations, AxisAnnotation, ConceptGraph
from torch_concepts.nn import AncestralSamplingInference, DeterministicInference, WANDAGraphLearner, GraphModel, LazyConstructor, LinearLatentToExogenous, \
    LinearExogenousToConcept, HyperlinearConceptExogenousToConcept
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference, LazyForwardInference
from torch_concepts.distributions import Delta


class SimpleForwardInference(ForwardInference):
    """Concrete implementation of ForwardInference for testing."""

    def get_results(self, results, parent_variable):
        """Simple implementation that samples from distributions."""
        if isinstance(parent_variable.distribution, type) and issubclass(parent_variable.distribution, Bernoulli):
            return torch.bernoulli(torch.sigmoid(results))
        elif isinstance(parent_variable.distribution, type) and issubclass(parent_variable.distribution, Categorical):
            return torch.argmax(results, dim=-1, keepdim=True).float()
        elif isinstance(parent_variable.distribution, type) and issubclass(parent_variable.distribution, Normal):
            return results
        else:
            return results

    def ground_truth_to_evidence(self, value: torch.Tensor, cardinality: int) -> torch.Tensor:
        """Convert ground truth to sample format (same as AncestralSamplingInference)."""
        if cardinality > 1:
            return torch.nn.functional.one_hot(
                value.long(), num_classes=cardinality
            ).float()
        else:
            return value.unsqueeze(-1).float()

class TestForwardInferenceQuery:
    """Test query functionality of ForwardInference."""

    def test_query_single_concept(self):
        """Test querying a single concept."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        # Query single concept
        batch_input = torch.randn(4, 10)
        result = inference.query(['A'], {'input': batch_input})

        assert result.shape == (4, 3)

    def test_query_multiple_concepts(self):
        """Test querying multiple concepts."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        # Query multiple concepts
        batch_input = torch.randn(4, 10)
        result = inference.query(['A', 'B'], {'input': batch_input})

        # Should concatenate A (3 features) and B (2 features)
        assert result.shape == (4, 5)

    def test_query_with_specific_order(self):
        """Test that query respects the order of concepts."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)

        # Query in different orders
        result_AB = inference.query(['A', 'B'], {'input': batch_input})
        result_BA = inference.query(['B', 'A'], {'input': batch_input})

        assert result_AB.shape == (4, 5)
        assert result_BA.shape == (4, 5)

    def test_query_missing_concept_raises_error(self):
        """Test that querying a non-existent concept raises error."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)

        with pytest.raises(ValueError, match="Query concept 'NonExistent' was requested"):
            inference.query(['NonExistent'], {'input': batch_input})

    def test_query_empty_list(self):
        """Test querying with empty list raises AssertionError."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)
        with pytest.raises(AssertionError, match="Query list cannot be empty"):
            inference.query([], {'input': batch_input})

    def test_query_with_debug_mode(self):
        """Test query with debug mode enabled."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)
        result = inference.query(['A'], {'input': batch_input}, debug=True)

        assert result.shape == (4, 3)


class TestForwardInferencePredictDevices:
    """Test predict method with different device configurations."""

    def test_predict_device_cpu(self):
        """Test predict with explicit CPU device."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)
        result = inference.query(['A'], {'input': batch_input}, device='cpu')

        # A has size=3
        assert result.shape == (4, 3)

    def test_predict_device_auto(self):
        """Test predict with auto device detection."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)
        result = inference.query(['A'], {'input': batch_input}, device='auto')

        # A has size=1
        assert result.shape == (4, 1)

    def test_predict_device_invalid_raises_error(self):
        """Test that invalid device raises error."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)

        with pytest.raises(ValueError, match="Invalid device 'invalid_device'"):
            inference.query(['A'], {'input': batch_input}, device='invalid_device')

    def test_predict_with_parallel_branches(self):
        """Test predict with parallel branches for CPU threading."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=2)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 2))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)
        result = inference.query(['A', 'B', 'C'], {'input': batch_input}, device='cpu')

        # A (size=3) + B (size=2) + C (size=1) = 6
        assert result.shape == (4, 6)


class TestForwardInferenceComputeSingleVariable:
    """Test _compute_single_variable method."""

    def test_compute_root_variable_missing_input_raises_error(self):
        """Test that computing root variable without external input raises error."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())

        model = ProbabilisticModel(
            variables=[input_var],
            parametric_cpds=[cpd_input]
        )

        inference = SimpleForwardInference(model)

        # Try to compute without providing external input
        with pytest.raises(ValueError, match="Root variable 'input' requires an input tensor"):
            inference._compute_single_variable(input_var, {}, {})

    def test_compute_missing_cpd_raises_error(self):
        """Test that computing variable without CPD raises error."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        # Intentionally not adding cpd_A

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)
        results = {'input': batch_input}

        with pytest.raises(RuntimeError, match="Missing parametric_cpd for variable/concept: A"):
            inference._compute_single_variable(var_A, {'input': batch_input}, results)


class TestForwardInferenceAvailableQueryVars:
    """Test available_query_vars property."""

    def test_available_query_vars(self):
        """Test that available_query_vars returns correct set."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(3, 2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        available = inference.available_query_vars

        assert isinstance(available, set)
        assert 'input' in available
        assert 'A' in available
        assert 'B' in available
        assert len(available) == 3


class TestForwardInferenceGetParentKwargs:
    """Test get_parent_kwargs method."""

    def test_get_parent_kwargs_with_endogenous_only(self):
        """Test get_parent_kwargs with only concept parents."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        parent_concepts = [torch.randn(4, 1)]
        kwargs = inference.get_parent_kwargs(cpd_B, [], parent_concepts)

        assert 'concepts' in kwargs
        assert kwargs['concepts'].shape == (4, 1)

    def test_get_parent_kwargs_with_input_and_endogenous(self):
        """Test get_parent_kwargs with both latent and concept parents."""
        from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept

        # Create a module that accepts both latent and concepts
        class CustomLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_latent = nn.Linear(10, 5)
                self.linear_concepts = nn.Linear(1, 5)

            def forward(self, latent, concepts):
                return self.linear_latent(latent) + self.linear_concepts(concepts)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['input', 'A'], distribution=Delta, size=5)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=CustomLinear())

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        parent_latent = [torch.randn(4, 10)]
        parent_concepts = [torch.randn(4, 1)]
        kwargs = inference.get_parent_kwargs(cpd_B, parent_latent, parent_concepts)

        assert 'latent' in kwargs
        assert 'concepts' in kwargs


class TestForwardInferenceCycleDetection:
    """Test that cycles are detected properly."""

    def test_cyclic_graph_raises_error(self):
        """Test that cyclic graphs raise an error during initialization."""
        # Create variables with a cycle: A -> B -> C -> A
        var_A = EndogenousVariable('A', parents=['C'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['B'], distribution=Bernoulli, size=1)

        cpd_A = ParametricCPD('A', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))

        model = ProbabilisticModel(
            variables=[var_A, var_B, var_C],
            parametric_cpds=[cpd_A, cpd_B, cpd_C]
        )

        with pytest.raises(RuntimeError, match="contains cycles"):
            inference = SimpleForwardInference(model)


class TestForwardInferenceComplexHierarchy:
    """Test complex hierarchical structures."""

    def test_diamond_structure(self):
        """Test diamond structure: input -> A, B -> C."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['A', 'B'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=2, out_concepts=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C]
        )

        inference = SimpleForwardInference(model)

        # Check levels structure
        assert len(inference.levels) == 3
        assert len(inference.levels[0]) == 1  # input
        assert len(inference.levels[1]) == 2  # A and B
        assert len(inference.levels[2]) == 1  # C

        # Test prediction
        batch_input = torch.randn(4, 10)
        result = inference.query(['C'], {'input': batch_input})

        # C has size=1
        assert result.shape == (4, 1)

    def test_multi_level_hierarchy(self):
        """Test multi-level hierarchy."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['B'], distribution=Bernoulli, size=1)
        var_D = EndogenousVariable('D', parents=['C'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))
        cpd_D = ParametricCPD('D', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D]
        )

        inference = SimpleForwardInference(model)

        # Check levels
        assert len(inference.levels) == 5

        # Test prediction
        batch_input = torch.randn(4, 10)
        result = inference.query(['A', 'B', 'C', 'D'], {'input': batch_input})

        # A, B, C, D each size=1 = 4 total
        assert result.shape == (4, 4)


class TestForwardInferenceDebugMode:
    """Test debug mode functionality."""

    def test_predict_debug_mode_sequential(self):
        """Test that debug mode runs sequentially."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        batch_input = torch.randn(4, 10)
        result = inference.query(['A', 'B'], {'input': batch_input}, debug=True)

        # A (size=3) + B (size=2) = 5
        assert result.shape == (4, 5)


class SimpleForwardInference(ForwardInference):
    """Concrete implementation of ForwardInference for testing."""

    def get_results(self, results, parent_variable):
        """Simple implementation that samples from Bernoulli distributions."""
        if isinstance(parent_variable.distribution, type) and issubclass(parent_variable.distribution, Bernoulli):
            # For Bernoulli, sample
            return torch.bernoulli(torch.sigmoid(results))
        elif isinstance(parent_variable.distribution, type) and issubclass(parent_variable.distribution, Categorical):
            # For Categorical, take argmax
            return torch.argmax(results, dim=-1, keepdim=True).float()
        else:
            # For other distributions (like Delta), return as-is
            return results


class TestForwardInferenceBasic:
    """Test basic functionality of ForwardInference."""

    def test_initialization_simple_model(self):
        """Test ForwardInference initialization with a simple model."""
        # Create a simple model: input -> A
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        assert len(inference.sorted_variables) == 2
        assert len(inference.levels) == 2
        assert inference.concept_map['input'] == input_var
        assert inference.concept_map['A'] == var_A

    def test_initialization_chain_model(self):
        """Test ForwardInference with a chain model: input -> A -> B -> C."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['B'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        # Use LinearConceptToConcept for endogenous-only parents
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C]
        )

        inference = SimpleForwardInference(model)

        # Check topological order
        assert len(inference.sorted_variables) == 4
        assert inference.sorted_variables[0].concept == 'input'
        assert inference.sorted_variables[1].concept == 'A'
        assert inference.sorted_variables[2].concept == 'B'
        assert inference.sorted_variables[3].concept == 'C'

        # Check levels
        assert len(inference.levels) == 4

    def test_initialization_parallel_model(self):
        """Test ForwardInference with parallel branches: input -> [A, B, C]."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C]
        )

        inference = SimpleForwardInference(model)

        # Check that A, B, C are in the same level (can be computed in parallel)
        assert len(inference.levels) == 2
        assert len(inference.levels[0]) == 1  # input
        assert len(inference.levels[1]) == 3  # A, B, C in parallel

    def test_topological_sort_diamond(self):
        """Test topological sort with diamond pattern: input -> [A, B] -> C."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['A', 'B'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        # Use LinearConceptToConcept for multiple endogenous parents
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=2, out_concepts=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C]
        )

        inference = SimpleForwardInference(model)

        # Check levels
        assert len(inference.levels) == 3
        assert len(inference.levels[0]) == 1  # input
        assert len(inference.levels[1]) == 2  # A, B
        assert len(inference.levels[2]) == 1  # C


class TestForwardInferencePredict:
    """Test the predict method of ForwardInference."""

    def test_predict_simple_model(self):
        """Test predict with a simple model."""
        torch.manual_seed(42)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        # Create input
        batch_size = 5
        external_inputs = {'input': torch.randn(batch_size, 10)}

        # Predict
        results = inference.query(['A'], external_inputs)

        # A has size=1
        assert results.shape == (batch_size, 1)

    def test_predict_chain_model(self):
        """Test predict with a chain model."""
        torch.manual_seed(42)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        # Use LinearConceptToConcept for endogenous parent
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        batch_size = 3
        external_inputs = {'input': torch.randn(batch_size, 10)}

        results = inference.query(['A', 'B'], external_inputs)

        # A (size=1) + B (size=1) = 2
        assert results.shape == (batch_size, 2)

    def test_predict_debug_mode(self):
        """Test predict with debug=True (sequential execution)."""
        torch.manual_seed(42)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        external_inputs = {'input': torch.randn(2, 10)}

        # Predict with debug mode
        results = inference.query(['A', 'B'], external_inputs, debug=True)

        # A (size=1) + B (size=1) = 2
        assert results.shape == (2, 2)

    def test_predict_device_cpu(self):
        """Test predict with explicit CPU device."""
        torch.manual_seed(42)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        external_inputs = {'input': torch.randn(2, 5)}
        results = inference.query(['A'], external_inputs, device='cpu')

        assert results.device.type == 'cpu'

    def test_predict_device_auto(self):
        """Test predict with device='auto'."""
        torch.manual_seed(42)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        external_inputs = {'input': torch.randn(2, 5)}
        results = inference.query(['A'], external_inputs, device='auto')

        # Should work regardless of CUDA availability
        assert results.shape == (2, 1)

    def test_predict_invalid_device(self):
        """Test predict with invalid device raises error."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        external_inputs = {'input': torch.randn(2, 5)}

        with pytest.raises(ValueError, match="Invalid device"):
            inference.query(['A'], external_inputs, device='invalid_device')

    def test_predict_missing_external_input(self):
        """Test predict with missing external input raises error."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A]
        )

        inference = SimpleForwardInference(model)

        # Missing 'input' in external_inputs
        external_inputs = {}

        with pytest.raises(AssertionError, match="Evidence must contain an 'input' key"):
            inference.query(['A'], external_inputs)


class TestForwardInferenceEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_cpd_raises_error(self):
        """Test that missing CPD raises RuntimeError during prediction."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)

        # Only provide CPD for input, not for A
        cpd_input = ParametricCPD('input', parametrization=nn.Identity())

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input]
        )

        inference = SimpleForwardInference(model)

        external_inputs = {'input': torch.randn(2, 5)}

        with pytest.raises(RuntimeError, match="Missing parametric_cpd for variable/concept"):
            inference.query(['A'], external_inputs)

    def test_parallel_execution_with_multiple_variables(self):
        """Test parallel execution with multiple variables at same level."""
        torch.manual_seed(42)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Bernoulli, size=1)
        var_D = EndogenousVariable('D', parents=['input'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(10, 1))
        cpd_D = ParametricCPD('D', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D]
        )

        inference = SimpleForwardInference(model)

        # Should have 4 variables in parallel at level 1
        assert len(inference.levels[1]) == 4

        external_inputs = {'input': torch.randn(3, 10)}
        results = inference.query(['A', 'B', 'C', 'D'], external_inputs, device='cpu')

        # A, B, C, D each size=1 = 4 total
        assert results.shape == (3, 4)

    def test_complex_dag_structure(self):
        """Test complex DAG with multiple dependencies."""
        torch.manual_seed(42)

        # Create structure: input -> [A, B] -> C -> D
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['A', 'B'], distribution=Bernoulli, size=1)
        var_D = EndogenousVariable('D', parents=['C'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        # Use LinearConceptToConcept for multiple endogenous parents
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=2, out_concepts=1))
        cpd_D = ParametricCPD('D', parametrization=LinearConceptToConcept(in_concepts=1, out_concepts=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D]
        )

        inference = SimpleForwardInference(model)

        # Check levels
        assert len(inference.levels) == 4

        external_inputs = {'input': torch.randn(2, 10)}
        results = inference.query(['A', 'B', 'C', 'D'], external_inputs)

        # A, B, C, D each size=1 = 4 total
        assert results.shape == (2, 4)

class SimpleForwardInference(ForwardInference):
    """Concrete implementation for testing."""

    def get_results(self, results: torch.Tensor, parent_variable: Variable):
        """Simple pass-through implementation."""
        return results


class TestForwardInference(unittest.TestCase):
    """Test ForwardInference class."""

    def test_initialization_simple_model(self):
        """Test initialization with simple model."""
        # Create simple model: latent -> A
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)
        self.assertIsNotNone(inference.sorted_variables)
        self.assertIsNotNone(inference.levels)
        self.assertEqual(len(inference.sorted_variables), 2)

    def test_topological_sort(self):
        """Test topological sorting of variables."""
        # Create chain: latent -> A -> B
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = EndogenousVariable('B', parents=[var_a], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(1, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b],
            parametric_cpds=[latent_factor, cpd_a, cpd_b]
        )

        inference = SimpleForwardInference(pgm)

        # Check topological order
        sorted_names = [v.concept for v in inference.sorted_variables]
        self.assertEqual(sorted_names, ['input', 'A', 'B'])

    def test_levels_computation(self):
        """Test level-based grouping for parallel computation."""
        # Create diamond structure
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = EndogenousVariable('B', parents=[input_var], distribution=Bernoulli, size=1)
        var_c = EndogenousVariable('C', parents=[var_a, var_b], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_c = ParametricCPD('C', parametrization=nn.Linear(2, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b, var_c],
            parametric_cpds=[latent_factor, cpd_a, cpd_b, cpd_c]
        )

        inference = SimpleForwardInference(pgm)

        # Check levels
        self.assertEqual(len(inference.levels), 3)
        # Level 0: latent
        self.assertEqual(len(inference.levels[0]), 1)
        # Level 1: A and B (can be computed in parallel)
        self.assertEqual(len(inference.levels[1]), 2)
        # Level 2: C
        self.assertEqual(len(inference.levels[2]), 1)

    def test_predict_simple_chain(self):
        """Test predict method with simple chain."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        # Run prediction
        external_inputs = {'input': torch.randn(4, 10)}
        results = inference.query(['A'], external_inputs)

        # A has size=1
        self.assertEqual(results.shape, (4, 1))

    def test_predict_with_debug_mode(self):
        """Test predict with debug mode (sequential execution)."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = inference.query(['A'], external_inputs, debug=True)

        # A has size=1
        self.assertEqual(results.shape, (4, 1))

    def test_predict_diamond_structure(self):
        """Test predict with diamond structure (parallel computation)."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[input_var], distribution=Bernoulli, size=1)
        var_c = Variable('C', parents=[var_a, var_b], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_c = ParametricCPD('C', parametrization=nn.Linear(2, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b, var_c],
            parametric_cpds=[latent_factor, cpd_a, cpd_b, cpd_c]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = inference.query(['A', 'B', 'C'], external_inputs)

        # A (size=1) + B (size=1) + C (size=1) = 3
        self.assertEqual(results.shape, (4, 3))

    def test_compute_single_variable_root(self):
        """Test _compute_single_variable for root variable."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())

        pgm = ProbabilisticModel(
            variables=[input_var],
            parametric_cpds=[latent_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = {}

        concept_name, output = inference._compute_single_variable(
            input_var, external_inputs, results
        )

        self.assertEqual(concept_name, 'input')
        self.assertEqual(output.shape[0], 4)

    def test_compute_single_variable_child(self):
        """Test _compute_single_variable for child variable."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = {'input': torch.randn(4, 10)}

        concept_name, output = inference._compute_single_variable(
            var_a, external_inputs, results
        )

        self.assertEqual(concept_name, 'A')
        self.assertIsNotNone(output)

    def test_missing_external_input(self):
        """Test error when root variable missing from external_inputs."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())

        pgm = ProbabilisticModel(
            variables=[input_var],
            parametric_cpds=[latent_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {}  # Missing 'input'
        results = {}

        with self.assertRaises(ValueError):
            inference._compute_single_variable(input_var, external_inputs, results)

    def test_missing_parent_result(self):
        """Test error when parent hasn't been computed yet."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = {}  # Missing 'input' in results

        with self.assertRaises(RuntimeError):
            inference._compute_single_variable(var_a, external_inputs, results)

    def test_get_parent_kwargs(self):
        """Test get_parent_kwargs method."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        parent_input = [torch.randn(4, 10)]
        parent_endogenous = []

        kwargs = inference.get_parent_kwargs(cpd_a, parent_input, parent_endogenous)
        self.assertIsInstance(kwargs, dict)

    def test_concept_map(self):
        """Test concept_map creation."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        self.assertIn('input', inference.concept_map)
        self.assertIn('A', inference.concept_map)
        self.assertEqual(inference.concept_map['input'], input_var)

    def test_categorical_parent(self):
        """Test with categorical parent variable."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Categorical, size=3)
        var_b = Variable('B', parents=[var_a], distribution=Bernoulli, size=1)

        latent_cpd = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(3, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b],
            parametric_cpds=[latent_cpd, cpd_a, cpd_b]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = inference.query(['B'], external_inputs)

        # B has size=1
        self.assertEqual(results.shape, (4, 1))

    def test_multiple_children_same_parent(self):
        """Test multiple children depending on same parent."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[input_var], distribution=Bernoulli, size=1)
        var_c = Variable('C', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_c = ParametricCPD('C', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b, var_c],
            parametric_cpds=[latent_factor, cpd_a, cpd_b, cpd_c]
        )

        inference = SimpleForwardInference(pgm)

        # All three children should be in the same level
        self.assertEqual(len(inference.levels[1]), 3)

    def test_missing_factor(self):
        """Test error when factor is missing for a variable."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        # Missing cpd_a

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}

        with self.assertRaises(RuntimeError):
            inference.query(['A'], external_inputs)

    def test_unroll_pgm(self):
        latent_dims = 20
        n_epochs = 1000
        n_samples = 1000
        concept_reg = 0.5

        dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
        x_train = dataset.input_data
        concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
        task_idx = list(dataset.graph.edge_index[1].unique().numpy())
        c_train = dataset.concepts[:, concept_idx]
        y_train = dataset.concepts[:, task_idx]

        c_train = torch.cat([c_train, y_train], dim=1)
        y_train = deepcopy(c_train)
        cy_train = torch.cat([c_train, y_train], dim=1)
        c_train_one_hot = torch.cat(
            [cy_train[:, :2], torch.nn.functional.one_hot(cy_train[:, 2].long(), num_classes=2).float()], dim=1)
        cy_train_one_hot = torch.cat([c_train_one_hot, c_train_one_hot], dim=1)

        concept_names = ['c1', 'c2', 'xor']
        task_names = ['c1_copy', 'c2_copy', 'xor_copy']
        cardinalities = [1, 1, 2, 1, 1, 2]
        metadata = {
            'c1': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1'},
            'c2': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2'},
            'xor': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical', 'description': 'XOR Task'},
            'c1_copy': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1 Copy'},
            'c2_copy': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2 Copy'},
            'xor_copy': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical',
                         'description': 'XOR Task Copy'},
        }
        annotations = Annotations(
            {1: AxisAnnotation(concept_names + task_names, cardinalities=cardinalities, metadata=metadata)})

        model_graph = ConceptGraph(torch.tensor([[0, 0, 0, 0, 1, 1],
                                                 [0, 0, 0, 1, 0, 1],
                                                 [0, 0, 0, 1, 1, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0]]), list(annotations.get_axis_annotation(1).labels))

        # ProbabilisticModel Initialization
        encoder = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU())
        concept_model = GraphModel(model_graph=model_graph,
                                   input_size=latent_dims,
                                   annotations=annotations,
                                   source_exogenous=LazyConstructor(LinearLatentToExogenous, out_exogenous=11),
                                   internal_exogenous=LazyConstructor(LinearLatentToExogenous, out_exogenous=7),
                                   encoder=LazyConstructor(LinearExogenousToConcept),
                                   predictor=LazyConstructor(HyperlinearConceptExogenousToConcept, hidden_size=20))

        # graph learning init
        graph_learner = WANDAGraphLearner(concept_names, task_names)

        inference_engine = AncestralSamplingInference(concept_model.probabilistic_model, graph_learner, temperature=0.1)
        query_concepts = ["c1", "c2", "xor", "c1_copy", "c2_copy", "xor_copy"]

        emb = encoder(x_train)
        cy_pred_before_unrolling = inference_engine.query(query_concepts, evidence={'input': emb}, debug=True)

        concept_model_new = inference_engine.unrolled_probabilistic_model()

        # identify available query concepts in the unrolled model
        query_concepts = [c for c in query_concepts if c in inference_engine.available_query_vars]
        concept_idx = {v: i for i, v in enumerate(concept_names)}
        reverse_c2t_mapping = dict(zip(task_names, concept_names))
        query_concepts = sorted(query_concepts, key=lambda x: concept_idx[x] if x in concept_idx else concept_idx[reverse_c2t_mapping[x]])

        inference_engine = AncestralSamplingInference(concept_model_new, temperature=0.1)
        cy_pred_after_unrolling = inference_engine.query(query_concepts, evidence={'input': emb}, debug=True)

        self.assertTrue(cy_pred_after_unrolling.shape == c_train_one_hot.shape)


class TestDeterministicInference(unittest.TestCase):
    """Unit tests for DeterministicInference class."""
    
    def setUp(self):
        """Set up test fixtures with a simple PGM."""
        # Create simple PGM: input -> A -> B
        from torch_concepts import LatentVariable, ConceptVariable
        
        self.input_var = LatentVariable('input', parents=[], distribution=Delta, size=10)
        self.var_A = ConceptVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        self.var_B = ConceptVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        
        # Define CPDs
        cpd_input = ParametricCPD('input', parametrization=Identity())
        cpd_A = ParametricCPD('A', parametrization=Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(1, 1))
        
        self.pgm = ProbabilisticModel(
            variables=[self.input_var, self.var_A, self.var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )
        
        self.inference = DeterministicInference(self.pgm)
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, 10)
    
    # --- Test get_results ---
    
    def test_get_results_returns_raw_output(self):
        """Test that get_results returns raw output without sampling."""
        # Create some test logits
        logits = torch.randn(self.batch_size, 1)
        
        result = self.inference.get_results(logits, self.var_A)
        
        # Should return the same tensor (no sampling)
        torch.testing.assert_close(result, logits)
    
    def test_get_results_preserves_gradients(self):
        """Test that get_results preserves gradient flow."""
        logits = torch.randn(self.batch_size, 1, requires_grad=True)
        
        result = self.inference.get_results(logits, self.var_A)
        
        # Should still require grad
        self.assertTrue(result.requires_grad)
        
        # Gradient should flow through
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(logits.grad)
    
    def test_get_results_with_different_shapes(self):
        """Test get_results works with various tensor shapes."""
        # Single feature
        result1 = self.inference.get_results(torch.randn(4, 1), self.var_A)
        self.assertEqual(result1.shape, (4, 1))
        
        # Multiple features
        result2 = self.inference.get_results(torch.randn(4, 5), self.var_A)
        self.assertEqual(result2.shape, (4, 5))
        
        # Larger batch
        result3 = self.inference.get_results(torch.randn(32, 3), self.var_A)
        self.assertEqual(result3.shape, (32, 3))
    
    # --- Test ground_truth_to_evidence (binary) ---
    
    def test_ground_truth_to_evidence_binary(self):
        """Test ground_truth_to_evidence for binary (cardinality=1) concepts."""
        # Binary ground truth: 0 or 1
        gt = torch.tensor([0., 1., 0., 1.])
        
        evidence = self.inference.ground_truth_to_evidence(gt, cardinality=1)
        
        # Should return logits
        self.assertEqual(evidence.shape, (4, 1))
        
        # 0 -> negative logit, 1 -> positive logit
        self.assertTrue(evidence[0, 0] < 0)  # 0 -> large negative logit
        self.assertTrue(evidence[1, 0] > 0)  # 1 -> large positive logit
    
    def test_ground_truth_to_evidence_binary_extreme_values(self):
        """Test that binary evidence uses clamping to avoid inf logits."""
        gt = torch.tensor([0., 1., 0., 1.])
        
        evidence = self.inference.ground_truth_to_evidence(gt, cardinality=1)
        
        # Should not have inf values due to clamping
        self.assertFalse(torch.isinf(evidence).any())
        self.assertFalse(torch.isnan(evidence).any())
    
    def test_ground_truth_to_evidence_binary_recovers_probs(self):
        """Test that sigmoid of evidence recovers ~original values."""
        gt = torch.tensor([0., 1., 0., 1.])
        
        evidence = self.inference.ground_truth_to_evidence(gt, cardinality=1)
        recovered = torch.sigmoid(evidence).squeeze()
        
        # Should be close to 0 and 1 (within clamping tolerance)
        self.assertTrue(recovered[0] < 0.01)  # ~0
        self.assertTrue(recovered[1] > 0.99)  # ~1
    
    # --- Test ground_truth_to_evidence (categorical) ---
    
    def test_ground_truth_to_evidence_categorical(self):
        """Test ground_truth_to_evidence for categorical (cardinality>1) concepts."""
        # Categorical ground truth: class indices
        gt = torch.tensor([0, 1, 2, 0])
        
        evidence = self.inference.ground_truth_to_evidence(gt, cardinality=3)
        
        # Should return logits with shape (batch, cardinality)
        self.assertEqual(evidence.shape, (4, 3))
    
    def test_ground_truth_to_evidence_categorical_correct_class(self):
        """Test that categorical evidence has highest logit for correct class."""
        gt = torch.tensor([0, 1, 2, 1])
        
        evidence = self.inference.ground_truth_to_evidence(gt, cardinality=3)
        
        # Argmax of each row should match ground truth
        predicted_classes = evidence.argmax(dim=1)
        torch.testing.assert_close(predicted_classes, gt)
    
    def test_ground_truth_to_evidence_categorical_no_inf_nan(self):
        """Test that categorical evidence has no inf/nan values."""
        gt = torch.tensor([0, 1, 2, 3, 4])
        
        evidence = self.inference.ground_truth_to_evidence(gt, cardinality=5)
        
        self.assertFalse(torch.isinf(evidence).any())
        self.assertFalse(torch.isnan(evidence).any())
    
    def test_ground_truth_to_evidence_categorical_softmax_recovers_onehot(self):
        """Test that softmax of evidence recovers ~one-hot encoding."""
        gt = torch.tensor([0, 2, 1])
        
        evidence = self.inference.ground_truth_to_evidence(gt, cardinality=3)
        recovered = torch.softmax(evidence, dim=1)
        
        # Should be close to one-hot
        expected = torch.nn.functional.one_hot(gt, num_classes=3).float()
        torch.testing.assert_close(recovered, expected, atol=0.02, rtol=0.02)
    
    # --- Integration tests with inference ---
    
    def test_deterministic_prediction_returns_logits(self):
        """Test that DeterministicInference.query returns logits, not samples."""
        results = self.inference.query(['A', 'B'], {'input': self.x})
        
        # Should have A (size=1) + B (size=1) = 2 features
        self.assertEqual(results.shape, (self.batch_size, 2))
        
        # Logits should be continuous real values
        unique_vals = results.unique()
        # More than 2 unique values means continuous, not binary
        self.assertGreater(len(unique_vals), 2)
    
    def test_deterministic_query_returns_logits(self):
        """Test that DeterministicInference.query returns concatenated logits."""
        output = self.inference.query(['A', 'B'], evidence={'input': self.x})
        
        # Should have shape (batch, 2) for two concepts
        self.assertEqual(output.shape, (self.batch_size, 2))
        
        # Values should be continuous logits
        unique_vals = output.unique()
        self.assertGreater(len(unique_vals), 2)
    
    def test_deterministic_gradient_flow(self):
        """Test that gradients flow through deterministic inference."""
        x = self.x.clone().requires_grad_(True)
        
        output = self.inference.query(['A', 'B'], evidence={'input': x})
        loss = output.sum()
        loss.backward()
        
        # Gradient should reach input
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))
    
    def test_deterministic_vs_sampling_difference(self):
        """Test that deterministic inference differs from ancestral sampling."""
        sampling_inference = AncestralSamplingInference(self.pgm)
        
        # Fix seed for reproducibility
        torch.manual_seed(42)
        
        det_output = self.inference.query(['A'], evidence={'input': self.x})
        samp_output = sampling_inference.query(['A'], evidence={'input': self.x})
        
        # Deterministic returns logits, sampling returns samples
        # Deterministic values should be continuous
        det_unique = det_output.unique()
        self.assertGreater(len(det_unique), 2)  # More than binary
        
        # Sampling from Bernoulli returns binary (0/1)
        samp_unique = samp_output.unique()
        # Sampling should give samples in [0, 1] range
        self.assertTrue(torch.all(samp_output >= 0))
        self.assertTrue(torch.all(samp_output <= 1))


# ---------------------------------------------------------------------------
# New targeted tests for 4 specific functionalities
# ---------------------------------------------------------------------------

class _SimpleForwardInferenceForNewTests(ForwardInference):
    """Minimal concrete ForwardInference for the new tests."""
    def get_results(self, results, parent_variable):
        return results


class _SimpleLazyForwardInference(LazyForwardInference):
    """Minimal concrete LazyForwardInference for the new tests."""
    def get_results(self, results, parent_variable):
        return results


class TestParallelisationIsHappening:
    """Verify that ThreadPoolExecutor is actually spawned for parallel levels."""

    def _build_parallel_model(self):
        """Build a model where A, B, C are in the same level (all depend on input)."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=2)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=2)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 2))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(5, 2))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(5, 2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C],
        )
        return model

    def test_thread_pool_used_on_cpu(self):
        """ThreadPoolExecutor must be instantiated when >1 variable in a level on CPU."""
        model = self._build_parallel_model()
        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 5)

        with patch(
            'torch_concepts.nn.modules.mid.inference.forward.ThreadPoolExecutor',
            wraps=ThreadPoolExecutor,
        ) as mock_pool:
            inference.query(['A', 'B', 'C'], {'input': batch_input}, device='cpu', debug=False)
            # Level 1 has 3 variables -> ThreadPoolExecutor should be called
            assert mock_pool.called, "ThreadPoolExecutor was never invoked for a parallel level"

    def test_thread_pool_not_used_in_debug(self):
        """ThreadPoolExecutor must NOT be used when debug=True."""
        model = self._build_parallel_model()
        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 5)

        with patch(
            'torch_concepts.nn.modules.mid.inference.forward.ThreadPoolExecutor',
            wraps=ThreadPoolExecutor,
        ) as mock_pool:
            inference.query(['A', 'B', 'C'], {'input': batch_input}, device='cpu', debug=True)
            assert not mock_pool.called, "ThreadPoolExecutor was invoked in debug mode"

    def test_parallel_results_match_sequential(self):
        """Parallel and sequential (debug) execution must produce identical results."""
        torch.manual_seed(0)
        model = self._build_parallel_model()
        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 5)

        # Sequential (debug)
        result_seq = inference.query(['A', 'B', 'C'], {'input': batch_input}, device='cpu', debug=True)
        # Parallel
        result_par = inference.query(['A', 'B', 'C'], {'input': batch_input}, device='cpu', debug=False)

        # Result is now a tensor of shape (4, 6) - A(2) + B(2) + C(2)
        torch.testing.assert_close(result_seq, result_par)


class TestExogenousVariableInference:
    """Verify that inference works correctly with ExogenousVariable parents."""

    def test_exogenous_parent_routed_as_latent_input(self):
        """An ExogenousVariable parent should be routed through parent_input (not parent_concepts)."""
        # Model: input (latent) -> exog (exogenous) -> A (concept)
        # exog uses Identity CPD, A uses a module with `exogenous` param
        input_var = InputVariable('input', parents=[], distribution=Delta, size=6)
        exog_var = ExogenousVariable('exog', parents=['input'], distribution=Delta, size=6)
        var_A = EndogenousVariable('A', parents=['exog'], distribution=Delta, size=3)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_exog = ParametricCPD('exog', parametrization=nn.Linear(6, 6))

        # A module that explicitly accepts `exogenous` kwarg
        class ExogToOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 3)

            def forward(self, exogenous):
                return self.fc(exogenous)

        cpd_A = ParametricCPD('A', parametrization=ExogToOutput())

        model = ProbabilisticModel(
            variables=[input_var, exog_var, var_A],
            parametric_cpds=[cpd_input, cpd_exog, cpd_A],
        )

        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 6)
        result = inference.query(['A'], {'input': batch_input}, device='cpu')

        assert result.shape == (4, 3)

    def test_mixed_concept_and_exogenous_parents(self):
        """A variable can have both ConceptVariable and ExogenousVariable parents."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=6)
        concept_var = EndogenousVariable('C', parents=['input'], distribution=Delta, size=2)
        exog_var = ExogenousVariable('E', parents=['input'], distribution=Delta, size=4)
        var_A = EndogenousVariable('A', parents=['C', 'E'], distribution=Delta, size=3)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(6, 2))
        cpd_E = ParametricCPD('E', parametrization=nn.Linear(6, 4))

        # A module that takes concepts + exogenous (like HyperlinearConceptExogenousToConcept)
        class ConceptExogModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 3)  # 2 (concept) + 4 (exog) = 6

            def forward(self, concepts, exogenous):
                return self.fc(torch.cat([concepts, exogenous], dim=-1))

        cpd_A = ParametricCPD('A', parametrization=ConceptExogModule())

        model = ProbabilisticModel(
            variables=[input_var, concept_var, exog_var, var_A],
            parametric_cpds=[cpd_input, cpd_C, cpd_E, cpd_A],
        )

        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 6)
        result = inference.query(['A'], {'input': batch_input}, device='cpu')

        assert result.shape == (4, 3)


class TestLazyInferenceSkipsDownstreamVariables:
    """Verify that LazyForwardInference does NOT compute variables outside the ancestor tree."""

    def _build_two_branch_model(self):
        """
        Build: input -> A -> B  (branch 1)
               input -> C -> D  (branch 2)
        Querying B should NOT compute C or D.
        """
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Delta, size=2)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Delta, size=3)
        var_D = EndogenousVariable('D', parents=['C'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 3))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(5, 3))
        cpd_D = ParametricCPD('D', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D],
        )
        return model

    def test_lazy_query_skips_unrelated_branch(self):
        """Querying B should compute input, A, B only — not C or D."""
        model = self._build_two_branch_model()
        inference = _SimpleLazyForwardInference(model)

        batch_input = torch.randn(4, 5)

        # Patch _compute_single_variable to track which variables are computed
        original_compute = inference._compute_single_variable
        computed_vars = []

        def tracking_compute(var, ext, res):
            computed_vars.append(var.concept)
            return original_compute(var, ext, res)

        with patch.object(inference, '_compute_single_variable', side_effect=tracking_compute):
            result = inference.query(['B'], evidence={'input': batch_input}, device='cpu')

        assert result.shape == (4, 2)
        assert 'input' in computed_vars, "Lazy inference must compute 'input'"
        assert 'A' in computed_vars, "Lazy inference must compute 'A' (ancestor of B)"
        assert 'B' in computed_vars, "Lazy inference must compute 'B' (queried)"
        assert 'C' not in computed_vars, "Lazy inference should NOT compute 'C'"
        assert 'D' not in computed_vars, "Lazy inference should NOT compute 'D'"

    def test_lazy_query_includes_shared_ancestors(self):
        """If two query concepts share an ancestor, it should be computed once."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Delta, size=2)
        var_C = EndogenousVariable('C', parents=['A'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 3))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C],
        )
        inference = _SimpleLazyForwardInference(model)

        batch_input = torch.randn(4, 5)
        result = inference.query(['B', 'C'], evidence={'input': batch_input}, device='cpu')

        # B(2) + C(2) = 4 features
        assert result.shape == (4, 4)

    def test_lazy_vs_full_same_result(self):
        """LazyForwardInference.query must produce the same output as ForwardInference.query."""
        model_lazy = self._build_two_branch_model()
        model_full = self._build_two_branch_model()

        # Share the same parameters
        model_full.load_state_dict(model_lazy.state_dict())

        lazy_inf = _SimpleLazyForwardInference(model_lazy)
        full_inf = _SimpleForwardInferenceForNewTests(model_full)

        torch.manual_seed(0)
        batch_input = torch.randn(4, 5)

        result_lazy = lazy_inf.query(['B'], evidence={'input': batch_input}, device='cpu')
        result_full = full_inf.query(['B'], evidence={'input': batch_input}, device='cpu')

        torch.testing.assert_close(result_lazy, result_full)

    def test_get_ancestors_returns_correct_set(self):
        """_get_ancestors for B should return {input, A, B}."""
        model = self._build_two_branch_model()
        inference = _SimpleLazyForwardInference(model)

        ancestors = inference._get_ancestors(['B'])
        assert ancestors == {'input', 'A', 'B'}


class TestEvidenceBypassSkipsCPD:
    """Verify that variables with provided evidence are NOT recomputed through their CPD."""

    def test_evidence_replaces_cpd_output(self):
        """When evidence is provided for a non-root variable, the CPD should be bypassed."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 3))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B],
        )
        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 5)
        evidence_A = torch.ones(4, 3) * 42.0  # fixed evidence for A

        # Provide evidence for A  ➜  A should use evidence, B should use it as parent
        result_A = inference.query(['A'], {'input': batch_input, 'A': evidence_A}, device='cpu')

        # A's result should be exactly the evidence tensor
        torch.testing.assert_close(result_A, evidence_A)

    def test_evidence_cpd_not_called(self):
        """The CPD forward of a variable with evidence must never be executed."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 3))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A],
        )
        inference = _SimpleForwardInferenceForNewTests(model)

        # Spy on cpd_A's forward
        original_forward = cpd_A.forward
        cpd_a_called = []

        def spy_forward(*args, **kwargs):
            cpd_a_called.append(True)
            return original_forward(*args, **kwargs)

        cpd_A.forward = spy_forward

        batch_input = torch.randn(4, 5)
        evidence_A = torch.ones(4, 3)
        _ = inference.query(['A'], {'input': batch_input, 'A': evidence_A}, device='cpu')

        assert len(cpd_a_called) == 0, "CPD.forward was called for a variable with evidence"

    def test_evidence_propagates_to_children(self):
        """Evidence for a mid-graph variable should propagate to downstream children."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 3))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B],
        )
        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 5)
        evidence_A = torch.ones(4, 3) * 5.0

        # Run twice: once without evidence, once with — B should differ
        result_no_ev = inference.query(['B'], {'input': batch_input}, device='cpu')
        result_with_ev = inference.query(['B'], {'input': batch_input, 'A': evidence_A}, device='cpu')

        # B should differ because A was overridden with evidence
        assert not torch.allclose(result_no_ev, result_with_ev), \
            "B should change when evidence is provided for its parent A"

    def test_root_variable_evidence_still_uses_cpd(self):
        """Evidence for a root (no-parent) variable should still pass through the CPD."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 3))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A],
        )
        inference = _SimpleForwardInferenceForNewTests(model)

        batch_input = torch.randn(4, 5)
        result_input = inference.query(['input'], {'input': batch_input}, device='cpu')

        # Root 'input' goes through Identity CPD -> output == input
        torch.testing.assert_close(result_input, batch_input)


class TestForwardVsLazyInferenceParity:
    """
    Comprehensive tests verifying that ForwardInference and LazyForwardInference
    produce identical results, and that Lazy skips unnecessary computations.
    """

    # -------------------- Helper builders for various graph topologies --------------------

    @staticmethod
    def _sync_model_weights(model_src: ProbabilisticModel, model_dst: ProbabilisticModel):
        """Copy state_dict from model_src to model_dst to ensure identical weights."""
        model_dst.load_state_dict(model_src.state_dict())

    @staticmethod
    def _build_linear_chain():
        """
        Linear chain: input -> A -> B -> C
        """
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=4)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Delta, size=3)
        var_C = EndogenousVariable('C', parents=['B'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 4))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=4, out_concepts=3))
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))

        return ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C],
        )

    @staticmethod
    def _build_diamond():
        """
        Diamond pattern:
              input
              /   \\
             A     B
              \\   /
                C
        """
        input_var = InputVariable('input', parents=[], distribution=Delta, size=6)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=3)
        var_C = EndogenousVariable('C', parents=['A', 'B'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(6, 3))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(6, 3))
        # C receives A + B as concepts
        cpd_C = ParametricCPD('C', parametrization=LinearConceptToConcept(in_concepts=6, out_concepts=2))

        return ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C],
        )

    @staticmethod
    def _build_two_branches():
        """
        Two independent branches:
            input -> A -> B
            input -> C -> D
        """
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=3)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Delta, size=2)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Delta, size=3)
        var_D = EndogenousVariable('D', parents=['C'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(5, 3))
        cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(5, 3))
        cpd_D = ParametricCPD('D', parametrization=LinearConceptToConcept(in_concepts=3, out_concepts=2))

        return ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D],
        )

    @staticmethod
    def _build_wide_tree():
        """
        Wide tree with many siblings:
            input -> A
            input -> B
            input -> C
            input -> D
            input -> E
        """
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=2)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=2)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Delta, size=2)
        var_D = EndogenousVariable('D', parents=['input'], distribution=Delta, size=2)
        var_E = EndogenousVariable('E', parents=['input'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 2))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 2))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(10, 2))
        cpd_D = ParametricCPD('D', parametrization=nn.Linear(10, 2))
        cpd_E = ParametricCPD('E', parametrization=nn.Linear(10, 2))

        return ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D, var_E],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D, cpd_E],
        )

    @staticmethod
    def _build_complex_dag():
        """
        Complex DAG:
                input
               /  |  \\
              A   B   C
              |   |\\  |
              D   | E---
               \\ |/
                 F
        
        Where:
        - D depends on A
        - E depends on B, C
        - F depends on D, B, E
        """
        input_var = InputVariable('input', parents=[], distribution=Delta, size=8)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Delta, size=2)
        var_B = EndogenousVariable('B', parents=['input'], distribution=Delta, size=2)
        var_C = EndogenousVariable('C', parents=['input'], distribution=Delta, size=2)
        var_D = EndogenousVariable('D', parents=['A'], distribution=Delta, size=2)
        var_E = EndogenousVariable('E', parents=['B', 'C'], distribution=Delta, size=2)
        var_F = EndogenousVariable('F', parents=['D', 'B', 'E'], distribution=Delta, size=2)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(8, 2))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(8, 2))
        cpd_C = ParametricCPD('C', parametrization=nn.Linear(8, 2))
        cpd_D = ParametricCPD('D', parametrization=LinearConceptToConcept(in_concepts=2, out_concepts=2))
        cpd_E = ParametricCPD('E', parametrization=LinearConceptToConcept(in_concepts=4, out_concepts=2))
        cpd_F = ParametricCPD('F', parametrization=LinearConceptToConcept(in_concepts=6, out_concepts=2))

        return ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D, var_E, var_F],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F],
        )

    # -------------------- Parity tests (same results) --------------------

    @pytest.mark.parametrize("builder,query_concepts", [
        ("_build_linear_chain", ["C"]),
        ("_build_linear_chain", ["B"]),
        ("_build_linear_chain", ["A", "C"]),
        ("_build_linear_chain", ["A", "B", "C"]),
        ("_build_diamond", ["C"]),
        ("_build_diamond", ["A"]),
        ("_build_diamond", ["A", "B"]),
        ("_build_diamond", ["A", "B", "C"]),
        ("_build_two_branches", ["B"]),
        ("_build_two_branches", ["D"]),
        ("_build_two_branches", ["B", "D"]),
        ("_build_two_branches", ["A", "B", "C", "D"]),
        ("_build_wide_tree", ["A"]),
        ("_build_wide_tree", ["A", "C", "E"]),
        ("_build_wide_tree", ["A", "B", "C", "D", "E"]),
        ("_build_complex_dag", ["F"]),
        ("_build_complex_dag", ["D"]),
        ("_build_complex_dag", ["E"]),
        ("_build_complex_dag", ["A", "F"]),
        ("_build_complex_dag", ["D", "E"]),
    ])
    def test_forward_and_lazy_produce_same_results(self, builder, query_concepts):
        """ForwardInference and LazyForwardInference must produce identical results."""
        torch.manual_seed(42)

        # Build two identical models
        model_full = getattr(self, builder)()
        model_lazy = getattr(self, builder)()
        self._sync_model_weights(model_full, model_lazy)

        full_inf = _SimpleForwardInferenceForNewTests(model_full)
        lazy_inf = _SimpleLazyForwardInference(model_lazy)

        batch_size = 4
        input_size = model_full.variables[0].out_features
        batch_input = torch.randn(batch_size, input_size)

        result_full = full_inf.query(query_concepts, evidence={'input': batch_input}, device='cpu')
        result_lazy = lazy_inf.query(query_concepts, evidence={'input': batch_input}, device='cpu')

        torch.testing.assert_close(
            result_full, result_lazy,
            msg=f"Mismatch for {builder} with query={query_concepts}"
        )

    # -------------------- Efficiency tests (lazy skips computations) --------------------

    def _count_computed_variables(self, inference, query_concepts, evidence):
        """Helper to count which variables were actually computed."""
        original_compute = inference._compute_single_variable
        computed_vars = []

        def tracking_compute(var, ext, res):
            computed_vars.append(var.concept)
            return original_compute(var, ext, res)

        with patch.object(inference, '_compute_single_variable', side_effect=tracking_compute):
            inference.query(query_concepts, evidence=evidence, device='cpu')

        return set(computed_vars)

    def test_lazy_chain_query_leaf_skips_nothing(self):
        """Querying leaf C in chain input->A->B->C must compute all."""
        model = self._build_linear_chain()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 5)

        computed = self._count_computed_variables(inference, ['C'], {'input': batch_input})
        assert computed == {'input', 'A', 'B', 'C'}

    def test_lazy_chain_query_mid_skips_downstream(self):
        """Querying B in chain input->A->B->C should skip C."""
        model = self._build_linear_chain()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 5)

        computed = self._count_computed_variables(inference, ['B'], {'input': batch_input})
        assert computed == {'input', 'A', 'B'}
        assert 'C' not in computed, "Lazy should skip C when querying only B"

    def test_lazy_two_branches_query_one_skips_other(self):
        """Querying B should skip branch C->D."""
        model = self._build_two_branches()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 5)

        computed = self._count_computed_variables(inference, ['B'], {'input': batch_input})
        assert computed == {'input', 'A', 'B'}
        assert 'C' not in computed and 'D' not in computed, "Lazy should skip C and D"

    def test_lazy_two_branches_query_both_computes_all(self):
        """Querying B and D should compute all variables."""
        model = self._build_two_branches()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 5)

        computed = self._count_computed_variables(inference, ['B', 'D'], {'input': batch_input})
        assert computed == {'input', 'A', 'B', 'C', 'D'}

    def test_lazy_wide_tree_query_one_skips_siblings(self):
        """Querying only A should skip B, C, D, E."""
        model = self._build_wide_tree()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 10)

        computed = self._count_computed_variables(inference, ['A'], {'input': batch_input})
        assert computed == {'input', 'A'}
        for skip in ['B', 'C', 'D', 'E']:
            assert skip not in computed, f"Lazy should skip {skip}"

    def test_lazy_wide_tree_query_subset_skips_rest(self):
        """Querying A, C should skip B, D, E."""
        model = self._build_wide_tree()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 10)

        computed = self._count_computed_variables(inference, ['A', 'C'], {'input': batch_input})
        assert computed == {'input', 'A', 'C'}
        for skip in ['B', 'D', 'E']:
            assert skip not in computed, f"Lazy should skip {skip}"

    def test_lazy_diamond_query_leaf_computes_both_branches(self):
        """Querying C in diamond must compute A and B (shared ancestors)."""
        model = self._build_diamond()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 6)

        computed = self._count_computed_variables(inference, ['C'], {'input': batch_input})
        assert computed == {'input', 'A', 'B', 'C'}

    def test_lazy_diamond_query_one_branch_skips_other(self):
        """Querying A in diamond should skip B and C."""
        model = self._build_diamond()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 6)

        computed = self._count_computed_variables(inference, ['A'], {'input': batch_input})
        assert computed == {'input', 'A'}
        assert 'B' not in computed and 'C' not in computed

    def test_lazy_complex_dag_query_D_skips_BCE_F(self):
        """Querying D in complex DAG should only compute input, A, D."""
        model = self._build_complex_dag()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 8)

        computed = self._count_computed_variables(inference, ['D'], {'input': batch_input})
        assert computed == {'input', 'A', 'D'}
        for skip in ['B', 'C', 'E', 'F']:
            assert skip not in computed, f"Lazy should skip {skip}"

    def test_lazy_complex_dag_query_E_skips_ADF(self):
        """Querying E should compute input, B, C, E (not A, D, F)."""
        model = self._build_complex_dag()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 8)

        computed = self._count_computed_variables(inference, ['E'], {'input': batch_input})
        assert computed == {'input', 'B', 'C', 'E'}
        for skip in ['A', 'D', 'F']:
            assert skip not in computed, f"Lazy should skip {skip}"

    def test_lazy_complex_dag_query_F_computes_all_ancestors(self):
        """Querying F should compute all variables (F depends on everything)."""
        model = self._build_complex_dag()
        inference = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 8)

        computed = self._count_computed_variables(inference, ['F'], {'input': batch_input})
        assert computed == {'input', 'A', 'B', 'C', 'D', 'E', 'F'}

    # -------------------- Edge case tests --------------------

    def test_lazy_query_all_equals_full_predict(self):
        """When all variables are queried, lazy should compute exactly what full does."""
        model_full = self._build_two_branches()
        model_lazy = self._build_two_branches()
        self._sync_model_weights(model_full, model_lazy)

        full_inf = _SimpleForwardInferenceForNewTests(model_full)
        lazy_inf = _SimpleLazyForwardInference(model_lazy)

        batch_input = torch.randn(4, 5)
        all_concepts = ['A', 'B', 'C', 'D']

        result_full = full_inf.query(all_concepts, evidence={'input': batch_input}, device='cpu')
        result_lazy = lazy_inf.query(all_concepts, evidence={'input': batch_input}, device='cpu')

        torch.testing.assert_close(result_full, result_lazy)

    def test_lazy_handles_single_variable_model(self):
        """Degenerate case: model with only root variable."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=5)
        cpd_input = ParametricCPD('input', parametrization=nn.Identity())

        model = ProbabilisticModel(
            variables=[input_var],
            parametric_cpds=[cpd_input],
        )

        lazy_inf = _SimpleLazyForwardInference(model)
        batch_input = torch.randn(4, 5)

        # Querying the root should work
        result = lazy_inf.query(['input'], evidence={'input': batch_input}, device='cpu')
        torch.testing.assert_close(result, batch_input)

    def test_lazy_vs_full_with_debug_mode(self):
        """Both should produce same results in debug mode (sequential execution)."""
        torch.manual_seed(42)
        model_full = self._build_diamond()
        model_lazy = self._build_diamond()
        self._sync_model_weights(model_full, model_lazy)

        full_inf = _SimpleForwardInferenceForNewTests(model_full)
        lazy_inf = _SimpleLazyForwardInference(model_lazy)

        batch_input = torch.randn(4, 6)

        result_full = full_inf.query(['C'], evidence={'input': batch_input}, device='cpu', debug=True)
        result_lazy = lazy_inf.query(['C'], evidence={'input': batch_input}, device='cpu', debug=True)

        torch.testing.assert_close(result_full, result_lazy)


if __name__ == "__main__":
    unittest.main()
