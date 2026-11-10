"""Generated concept scores annotated on the last axis (-1) as well as on axis 1."""
import pytest
import torch

from torch_concepts import Annotations
from torch_concepts.data.generation.base.pipeline import ConceptGenerationPipeline
from torch_concepts.data.generation.calibrators.sigmoid import SigmoidCalibrator
from torch_concepts.tensor import AnnotatedTensor


def scores(axis):
    """Three samples, two concepts on different baselines."""
    values = torch.tensor([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    return AnnotatedTensor(values, Annotations(labels=["red", "big"]), axis=axis)


def test_standardized_calibration_is_per_concept_on_either_axis():
    calibrator = SigmoidCalibrator(standardize=True)
    on_1, on_last = calibrator.calibrate(scores(1)), calibrator.calibrate(scores(-1))
    torch.testing.assert_close(on_last.tensor, on_1.tensor)
    # each concept standardized on its own: the two columns end up equal
    torch.testing.assert_close(on_last.tensor[:, 0], on_last.tensor[:, 1])


@pytest.mark.parametrize("axis", [1, -1])
def test_pipeline_accepts_two_dimensional_values_on_either_axis(axis):
    values = scores(axis)
    ConceptGenerationPipeline._validate_value("scores", values, values.annotations, dataset=list(range(3)))
