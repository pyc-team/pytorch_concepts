import torch
from torch.utils.data import TensorDataset

from torch_concepts import Annotations
from torch_concepts.data.base.annotator import Annotator
from torch_concepts.data.base.concept_generator import ConceptGenerator
from torch_concepts.data.base.concept_pipeline import ConceptSupervisionPipeline
from torch_concepts.data.concept_generators import LLMConceptGenerator


def test_generator_builds_annotations_with_current_type_api():
    generator = LLMConceptGenerator(
        llm=lambda prompt: (
            '[{"name": "striped"}, '
            '{"name": "color", "states": ["red", "green"]}]'
        ),
        prompt="Generate concepts.",
    )

    annotations = generator.generate()

    assert annotations.labels == ["striped", "color"]
    assert annotations.states == [["0"], ["red", "green"]]
    assert annotations.cardinalities == [1, 2]
    assert annotations.types == ["binary", "categorical"]


class _StaticGenerator(ConceptGenerator):
    def __init__(self, annotations):
        self.annotations = annotations

    def generate(self, dataset=None, class_names=None, **kwargs):
        return self.annotations


class _ZeroAnnotator(Annotator):
    def annotate(self, dataset, concepts, **kwargs):
        return torch.zeros(len(dataset), concepts.size)


def test_merged_pipeline_uses_current_annotations_api():
    binary = Annotations(
        labels=["striped"],
        cardinalities=[1],
        types=["binary"],
    )
    categorical = Annotations(
        labels=["color"],
        states=[["red", "green"]],
        types=["categorical"],
    )
    pipeline = ConceptSupervisionPipeline(
        generators=[
            _StaticGenerator(binary),
            _StaticGenerator(categorical),
        ],
        annotators=_ZeroAnnotator(),
        routing="merged",
    )

    values, annotations = pipeline(TensorDataset(torch.randn(3, 2)))

    axis = annotations["_ZeroAnnotator"]
    assert axis.labels == ["striped", "color"]
    assert axis.types == ["binary", "categorical"]
    assert values["_ZeroAnnotator"].shape == (3, 3)
