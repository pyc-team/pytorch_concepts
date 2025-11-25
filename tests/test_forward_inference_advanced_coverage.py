import torch
import pytest
from unittest import mock
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference
from torch_concepts.nn.modules.mid.models.variable import Variable, EndogenousVariable
from torch_concepts.nn.modules.low.inference.intervention import _GlobalPolicyInterventionWrapper

# Dummy parametrization with a forward method
class DummyParametrization:
    def forward(self, input=None, **kwargs):
        return torch.ones(2, 1) * 1

# Dummy CPD and ProbabilisticModel for advanced tests
class DummyCPD:
    def __init__(self, name, parametrization=None):
        self.name = name
        self.parametrization = parametrization or DummyParametrization()
    def forward(self, **kwargs):
        return self.parametrization.forward(**kwargs)

class DummyProbModel:
    def __init__(self, variables, cpds):
        self.variables = variables
        self._cpds = {c.name: c for c in cpds}
    def get_module_of_concept(self, name):
        return self._cpds.get(name, None)

class DummySharedState:
    def __init__(self):
        self.reset_called = False
    def is_ready(self):
        return True
    def reset(self):
        self.reset_called = True

class DummyGlobalPolicyInterventionWrapper(_GlobalPolicyInterventionWrapper):
    def __init__(self, original=None, policy=None, strategy=None, wrapper_id=None, shared_state=None):
        if shared_state is None:
            shared_state = DummySharedState()
        super().__init__(original, policy, strategy, wrapper_id, shared_state)
        self.shared_state = shared_state
    def apply_intervention(self, x):
        return x + 100
    def forward(self, **kwargs):
        return torch.ones(2, 1) * 1

class TestForwardInference(ForwardInference):
    def get_results(self, results, parent_variable):
        return results

@pytest.fixture
def model_with_global_policy():
    v1 = Variable('A', parents=[])
    v2 = EndogenousVariable('B', parents=[v1])
    dummy_policy = object()
    dummy_strategy = object()
    dummy_wrapper_id = 'dummy'
    dummy_shared_state = DummySharedState()
    cpd1 = DummyCPD('A')
    dummy_original = DummyParametrization()  # Fix: provide a valid object with .forward
    cpd2 = DummyCPD('B', parametrization=DummyGlobalPolicyInterventionWrapper(
        original=dummy_original, policy=dummy_policy, strategy=dummy_strategy, wrapper_id=dummy_wrapper_id, shared_state=dummy_shared_state
    ))
    model = DummyProbModel([v1, v2], [cpd1, cpd2])
    return model, v1, v2

def test_apply_global_interventions_for_level_debug(model_with_global_policy):
    model, v1, v2 = model_with_global_policy
    inf = TestForwardInference(model)
    results = {'B': torch.ones(2, 1)}
    level = [v2]
    # Should apply intervention and update results
    inf._apply_global_interventions_for_level(level, results, debug=True, use_cuda=False)
    assert torch.all(results['B'] == 101)

def test_apply_global_interventions_for_level_parallel(model_with_global_policy):
    model, v1, v2 = model_with_global_policy
    inf = TestForwardInference(model)
    results = {'B': torch.ones(2, 1)}
    level = [v2]
    # Should apply intervention and update results (parallel branch, but only one wrapper)
    inf._apply_global_interventions_for_level(level, results, debug=False, use_cuda=False)
    assert torch.all(results['B'] == 101)

def test_predict_cuda_branch(monkeypatch, model_with_global_policy):
    model, v1, v2 = model_with_global_policy
    inf = TestForwardInference(model)
    # Patch torch.cuda.is_available to True, patch torch.cuda.Stream and synchronize
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda device=None: mock.Mock())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    # Should run without error (simulate CUDA branch)
    out = inf.predict({'A': torch.ones(2, 1)}, debug=False, device='cuda')
    assert 'A' in out and 'B' in out
    assert torch.all(out['B'] == 101)

def test_predict_cuda_not_available(monkeypatch, model_with_global_policy):
    model, v1, v2 = model_with_global_policy
    inf = TestForwardInference(model)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError):
        inf.predict({'A': torch.ones(2, 1)}, device='cuda')

def test_apply_single_global_intervention(model_with_global_policy):
    model, v1, v2 = model_with_global_policy
    inf = TestForwardInference(model)
    results = {'B': torch.ones(2, 1)}
    dummy_original = DummyParametrization()  # Provide a valid object with .forward
    wrapper = DummyGlobalPolicyInterventionWrapper(original=dummy_original)
    name, out = inf._apply_single_global_intervention('B', wrapper, results)
    assert name == 'B'
    assert torch.all(out == 101)
