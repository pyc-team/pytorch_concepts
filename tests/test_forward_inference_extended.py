"""Extended tests for torch_concepts.nn.modules.mid.inference.forward module to improve coverage."""

import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.nn.modules.mid.models.variable import Variable, EndogenousVariable, InputVariable
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference
from torch_concepts.distributions.delta import Delta
from torch_concepts.nn.modules.low.predictors.linear import LinearCC


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
