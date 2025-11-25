import torch
import pytest
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.models.variable import Variable, EndogenousVariable
from torch.nn import Linear, Identity

# Minimal CPD mock
class DummyCPD:
    def __init__(self, name, parametrization=None):
        self.name = name
        self.parametrization = parametrization or Identity()
    def forward(self, **kwargs):
        # Return a tensor with shape (batch, 1)
        return torch.ones(2, 1) * 42

# Minimal ProbabilisticModel mock
class DummyProbModel:
    def __init__(self, variables, cpds):
        self.variables = variables
        self._cpds = {c.name: c for c in cpds}
    def get_module_of_concept(self, name):
        return self._cpds.get(name, None)

# Concrete ForwardInference for testing
class TestForwardInference(ForwardInference):
    def get_results(self, results, parent_variable):
        return results  # No-op for test

# Helper to create a simple acyclic model
@pytest.fixture
def acyclic_model():
    v1 = Variable('A', parents=[])
    v2 = EndogenousVariable('B', parents=[v1])
    cpd1 = DummyCPD('A')
    cpd2 = DummyCPD('B')
    model = DummyProbModel([v1, v2], [cpd1, cpd2])
    return model, v1, v2

# Helper to create a cyclic model
@pytest.fixture
def cyclic_model():
    v1 = Variable('A', parents=[])
    v2 = EndogenousVariable('B', parents=[v1])
    v1.parents = [v2]  # Introduce cycle
    cpd1 = DummyCPD('A')
    cpd2 = DummyCPD('B')
    model = DummyProbModel([v1, v2], [cpd1, cpd2])
    return model

def test_topological_sort_acyclic(acyclic_model):
    model, v1, v2 = acyclic_model
    inf = TestForwardInference(model)
    assert [v.concepts[0] for v in inf.sorted_variables] == ['A', 'B']
    assert len(inf.levels) == 2

def test_topological_sort_cycle(cyclic_model):
    with pytest.raises(RuntimeError):
        TestForwardInference(cyclic_model)

def test_compute_single_variable_root(acyclic_model):
    model, v1, v2 = acyclic_model
    inf = TestForwardInference(model)
    external_inputs = {'A': torch.ones(2, 1)}
    results = {}
    name, out = inf._compute_single_variable(v1, external_inputs, results)
    assert name == 'A'
    assert torch.all(out == 42)

def test_compute_single_variable_child(acyclic_model):
    model, v1, v2 = acyclic_model
    inf = TestForwardInference(model)
    external_inputs = {'A': torch.ones(2, 1)}
    results = {'A': torch.ones(2, 1)}
    name, out = inf._compute_single_variable(v2, external_inputs, results)
    assert name == 'B'
    assert torch.all(out == 42)

def test_missing_cpd_raises(acyclic_model):
    model, v1, v2 = acyclic_model
    model._cpds.pop('A')
    inf = TestForwardInference(model)
    with pytest.raises(RuntimeError):
        inf._compute_single_variable(v1, {'A': torch.ones(2, 1)}, {})

def test_missing_external_input_raises(acyclic_model):
    model, v1, v2 = acyclic_model
    inf = TestForwardInference(model)
    with pytest.raises(ValueError):
        inf._compute_single_variable(v1, {}, {})

def test_missing_parent_data_raises(acyclic_model):
    model, v1, v2 = acyclic_model
    inf = TestForwardInference(model)
    with pytest.raises(RuntimeError):
        inf._compute_single_variable(v2, {'A': torch.ones(2, 1)}, {})

def test_predict_debug_and_parallel(acyclic_model):
    model, v1, v2 = acyclic_model
    inf = TestForwardInference(model)
    # Debug mode (sequential)
    out = inf.predict({'A': torch.ones(2, 1)}, debug=True, device='cpu')
    assert 'A' in out and 'B' in out
    # Parallel mode (ThreadPoolExecutor)
    out2 = inf.predict({'A': torch.ones(2, 1)}, debug=False, device='cpu')
    assert 'A' in out2 and 'B' in out2

def test_predict_invalid_device(acyclic_model):
    model, v1, v2 = acyclic_model
    inf = TestForwardInference(model)
    with pytest.raises(ValueError):
        inf.predict({'A': torch.ones(2, 1)}, device='invalid')

# Additional tests for intervention and CUDA branches can be added with mocks if needed

