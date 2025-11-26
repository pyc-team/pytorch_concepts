import unittest
import pytest
from torch_concepts.nn.modules.low.inference.intervention import _GlobalPolicyInterventionWrapper
from torch.distributions import Normal
from torch_concepts.nn.modules.low.predictors.linear import LinearCC
from torch.nn import Linear, Identity
from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical
from torch_concepts.data.datasets import ToyDataset
from torch_concepts import InputVariable, EndogenousVariable, Annotations, AxisAnnotation, ConceptGraph
from torch_concepts.nn import AncestralSamplingInference, WANDAGraphLearner, GraphModel, LazyConstructor, LinearZU, \
    LinearUC, HyperLinearCUC
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference
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
        """Test querying with empty list returns empty tensor."""
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
        result = inference.query([], {'input': batch_input})

        assert result.shape == (0,)

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
        result = inference.predict({'input': batch_input}, device='cpu')

        assert 'A' in result
        assert result['A'].shape == (4, 3)

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
        result = inference.predict({'input': batch_input}, device='auto')

        assert 'A' in result
        assert result['A'].shape == (4, 1)

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
            inference.predict({'input': batch_input}, device='invalid_device')

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
        result = inference.predict({'input': batch_input}, device='cpu')

        assert 'A' in result and result['A'].shape == (4, 3)
        assert 'B' in result and result['B'].shape == (4, 2)
        assert 'C' in result and result['C'].shape == (4, 1)


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
        with pytest.raises(ValueError, match="Root variable 'input' requires an external input"):
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
        """Test get_parent_kwargs with only endogenous parents."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=LinearCC(in_features_endogenous=1, out_features=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        parent_endogenous = [torch.randn(4, 1)]
        kwargs = inference.get_parent_kwargs(cpd_B, [], parent_endogenous)

        assert 'endogenous' in kwargs
        assert kwargs['endogenous'].shape == (4, 1)

    def test_get_parent_kwargs_with_input_and_endogenous(self):
        """Test get_parent_kwargs with both input and endogenous parents."""
        from torch_concepts.nn.modules.low.predictors.linear import LinearCC

        # Create a module that accepts both input and endogenous
        class CustomLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_input = nn.Linear(10, 5)
                self.linear_endo = nn.Linear(1, 5)

            def forward(self, input, endogenous):
                return self.linear_input(input) + self.linear_endo(endogenous)

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

        parent_input = [torch.randn(4, 10)]
        parent_endogenous = [torch.randn(4, 1)]
        kwargs = inference.get_parent_kwargs(cpd_B, parent_input, parent_endogenous)

        assert 'input' in kwargs
        assert 'endogenous' in kwargs


class TestForwardInferenceCycleDetection:
    """Test that cycles are detected properly."""

    def test_cyclic_graph_raises_error(self):
        """Test that cyclic graphs raise an error during initialization."""
        # Create variables with a cycle: A -> B -> C -> A
        var_A = EndogenousVariable('A', parents=['C'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['B'], distribution=Bernoulli, size=1)

        cpd_A = ParametricCPD('A', parametrization=LinearCC(in_features_endogenous=1, out_features=1))
        cpd_B = ParametricCPD('B', parametrization=LinearCC(in_features_endogenous=1, out_features=1))
        cpd_C = ParametricCPD('C', parametrization=LinearCC(in_features_endogenous=1, out_features=1))

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
        cpd_C = ParametricCPD('C', parametrization=LinearCC(in_features_endogenous=2, out_features=1))

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
        result = inference.predict({'input': batch_input})

        assert 'C' in result
        assert result['C'].shape == (4, 1)

    def test_multi_level_hierarchy(self):
        """Test multi-level hierarchy."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        var_C = EndogenousVariable('C', parents=['B'], distribution=Bernoulli, size=1)
        var_D = EndogenousVariable('D', parents=['C'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_B = ParametricCPD('B', parametrization=LinearCC(in_features_endogenous=1, out_features=1))
        cpd_C = ParametricCPD('C', parametrization=LinearCC(in_features_endogenous=1, out_features=1))
        cpd_D = ParametricCPD('D', parametrization=LinearCC(in_features_endogenous=1, out_features=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D]
        )

        inference = SimpleForwardInference(model)

        # Check levels
        assert len(inference.levels) == 5

        # Test prediction
        batch_input = torch.randn(4, 10)
        result = inference.predict({'input': batch_input})

        assert all(k in result for k in ['input', 'A', 'B', 'C', 'D'])


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
        result = inference.predict({'input': batch_input}, debug=True)

        assert 'A' in result and result['A'].shape == (4, 3)
        assert 'B' in result and result['B'].shape == (4, 2)


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
        # Use LinearCC for endogenous-only parents
        cpd_B = ParametricCPD('B', parametrization=LinearCC(in_features_endogenous=1, out_features=1))
        cpd_C = ParametricCPD('C', parametrization=LinearCC(in_features_endogenous=1, out_features=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C]
        )

        inference = SimpleForwardInference(model)

        # Check topological order
        assert len(inference.sorted_variables) == 4
        assert inference.sorted_variables[0].concepts[0] == 'input'
        assert inference.sorted_variables[1].concepts[0] == 'A'
        assert inference.sorted_variables[2].concepts[0] == 'B'
        assert inference.sorted_variables[3].concepts[0] == 'C'

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
        # Use LinearCC for multiple endogenous parents
        cpd_C = ParametricCPD('C', parametrization=LinearCC(in_features_endogenous=2, out_features=1))

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
        results = inference.predict(external_inputs)

        assert 'input' in results
        assert 'A' in results
        assert results['A'].shape == (batch_size, 1)

    def test_predict_chain_model(self):
        """Test predict with a chain model."""
        torch.manual_seed(42)

        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        # Use LinearCC for endogenous parent
        cpd_B = ParametricCPD('B', parametrization=LinearCC(in_features_endogenous=1, out_features=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            parametric_cpds=[cpd_input, cpd_A, cpd_B]
        )

        inference = SimpleForwardInference(model)

        batch_size = 3
        external_inputs = {'input': torch.randn(batch_size, 10)}

        results = inference.predict(external_inputs)

        assert 'input' in results
        assert 'A' in results
        assert 'B' in results
        assert results['B'].shape == (batch_size, 1)

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
        results = inference.predict(external_inputs, debug=True)

        assert 'A' in results
        assert 'B' in results

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
        results = inference.predict(external_inputs, device='cpu')

        assert results['A'].device.type == 'cpu'

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
        results = inference.predict(external_inputs, device='auto')

        # Should work regardless of CUDA availability
        assert 'A' in results

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
            inference.predict(external_inputs, device='invalid_device')

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

        with pytest.raises(ValueError, match="Root variable 'input' requires an external input"):
            inference.predict(external_inputs)


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
            inference.predict(external_inputs)

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
        results = inference.predict(external_inputs, device='cpu')

        assert all(var in results for var in ['A', 'B', 'C', 'D'])

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
        # Use LinearCC for multiple endogenous parents
        cpd_C = ParametricCPD('C', parametrization=LinearCC(in_features_endogenous=2, out_features=1))
        cpd_D = ParametricCPD('D', parametrization=LinearCC(in_features_endogenous=1, out_features=1))

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_C, var_D],
            parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C, cpd_D]
        )

        inference = SimpleForwardInference(model)

        # Check levels
        assert len(inference.levels) == 4

        external_inputs = {'input': torch.randn(2, 10)}
        results = inference.predict(external_inputs)

        assert all(var in results for var in ['input', 'A', 'B', 'C', 'D'])
        assert results['D'].shape == (2, 1)

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
        sorted_names = [v.concepts[0] for v in inference.sorted_variables]
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
        results = inference.predict(external_inputs)

        self.assertIn('input', results)
        self.assertIn('A', results)
        self.assertEqual(results['A'].shape[0], 4)

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
        results = inference.predict(external_inputs, debug=True)

        self.assertIn('input', results)
        self.assertIn('A', results)

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
        results = inference.predict(external_inputs)

        self.assertEqual(len(results), 4)
        self.assertIn('C', results)

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
        var_a = Variable('A', parents=[], distribution=Categorical, size=3)
        var_b = Variable('B', parents=[var_a], distribution=Bernoulli, size=1)

        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(3, 1))

        pgm = ProbabilisticModel(
            variables=[var_a, var_b],
            parametric_cpds=[cpd_a, cpd_b]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'A': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertIn('B', results)

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
            inference.predict(external_inputs)

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
                                   source_exogenous=LazyConstructor(LinearZU, exogenous_size=11),
                                   internal_exogenous=LazyConstructor(LinearZU, exogenous_size=7),
                                   encoder=LazyConstructor(LinearUC),
                                   predictor=LazyConstructor(HyperLinearCUC, embedding_size=20))

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


if __name__ == "__main__":
    unittest.main()
