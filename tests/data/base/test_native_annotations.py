"""Native schemas remain available independently of generated supervision."""
import torch

from torch_concepts import Annotations, AnnotatedTensor
from torch_concepts.data.base import ConceptDataset


def test_native_schema_survives_generated_ground_truth_selection():
    native = Annotations(labels=['kind', 'z', 'a'],
                         states=[['second', 'first'], ['value'], ['value']],
                         types=['categorical', 'continuous', 'continuous'])
    dataset = ConceptDataset(torch.zeros(5, 2), torch.zeros(5, 3),
                             annotations=native, reorder_by_type=False)
    generated = AnnotatedTensor(torch.ones(5, 2), Annotations(labels=['generated_b', 'generated_a']))
    dataset.set_generated_concepts({'scores': generated}, use_as_gt=True)
    assert dataset.annotations.labels == ['generated_b', 'generated_a']
    assert dataset.native_annotations.labels == ['kind', 'z', 'a']
    assert dataset.native_annotations.concept('kind').states == ['second', 'first']
    assert dataset.native_annotations.to_dict() == native.to_dict()
    assert dataset.native_concepts.annotations.labels == native.labels
    assert dataset.native_concepts.shape == (5, 3)
    assert dataset.collate([dataset[0]])['concepts']['native'].annotations.labels == native.labels


def test_unlabelled_dataset_has_no_native_schema():
    dataset = ConceptDataset(torch.zeros(5, 2))
    assert dataset.native_annotations is None
    generated = AnnotatedTensor(torch.ones(5, 1), Annotations(labels=['generated']))
    dataset.set_generated_concepts({'scores': generated})
    assert dataset.annotations.labels == ['generated']
    assert dataset.native_annotations is None


def test_native_schema_follows_declared_subset_order():
    dataset = ConceptDataset(torch.zeros(5, 2), torch.zeros(5, 3),
                             Annotations(labels=['a', 'b', 'c']),
                             concept_names_subset=['c', 'a'])
    assert dataset.native_annotations.labels == ['c', 'a']
    assert dataset.native_annotations.labels == dataset.native_concepts.annotations.labels
