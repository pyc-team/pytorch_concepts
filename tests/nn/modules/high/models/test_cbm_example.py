import pytest
import torch
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel_Joint
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch.distributions import Categorical, Bernoulli

def test_cbm_docstring_example():
    ann = Annotations({
        1: AxisAnnotation(
            labels=['c1', 'task'],
            cardinalities=[2, 1],
            metadata={
                'c1': {'type': 'discrete', 'distribution': Categorical},
                'task': {'type': 'continuous', 'distribution': Bernoulli}
            }
        )
    })
    model = ConceptBottleneckModel_Joint(
        input_size=8,
        annotations=ann,
        task_names=['task'],
        variable_distributions=None
    )
    x = torch.randn(2, 8)
    out = model(x, query=['c1', 'task'])
    assert out.shape[0] == 2
    assert out.shape[1] == 3  # 2 for c1, 1 for task
