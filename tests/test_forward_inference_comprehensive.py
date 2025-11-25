"""Comprehensive tests for torch_concepts.nn.modules.mid.inference.forward module to improve coverage."""

import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Normal

from torch_concepts.nn.modules.mid.models.variable import Variable, EndogenousVariable, InputVariable
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference
from torch_concepts.distributions.delta import Delta
from torch_concepts.nn.modules.low.predictors.linear import LinearCC


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
