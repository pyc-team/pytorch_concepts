"""Public semantic vocabulary filtering with fixed, offline text embeddings."""
import pytest
import torch
from torch.nn import functional as F

from torch_concepts import AnnotatedTensor, Annotations
from torch_concepts.data.base import ConceptDataset
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.base.annotator import Annotator
from torch_concepts.data.generation.filters import SemanticConceptFilter
from torch_concepts.data.generation.generators import FixedConceptGenerator


def encoder(vectors):
    def encode(names):
        return F.normalize(torch.tensor([vectors[name] for name in names]).float(), dim=1)
    return encode


def test_default_compares_matching_class_and_concept_indices():
    encode = encoder({'class': [1, 0], 'alias': [1, 0], 'green': [0, 1]})
    concepts = Annotations(labels=['green', 'alias'], cardinalities=[1, 1])
    general = SemanticConceptFilter(['class'], encode, encode)
    assert general.filter(concepts).labels == ['green']
    # Reference compatibility skips class 0 vs concept 0 (alphabetical 'alias').
    reference = SemanticConceptFilter(['class'], encode, encode, reference_class_filter=True)
    assert reference.filter(concepts).labels == ['alias', 'green']


def test_exact_class_aliases_and_long_labels_are_removed_before_encoding():
    calls = []

    def encode(names):
        calls.append(list(names))
        return torch.tensor([[1., 0.] if name == 'bird' else [0., 1.] for name in names])

    concepts = Annotations(labels=['bird', 'a bird', 'the bird', 'BIRD', 'Bird',
                                  'an excessively long label', 'feathers'], cardinalities=[1] * 7)
    original = concepts.to_dict()
    result = SemanticConceptFilter(['bird'], encode, encode, max_length=12).filter(concepts)
    assert result.labels == ['feathers']
    assert calls[1] == ['feathers']
    assert all(set(names) <= {'bird', 'feathers'} for names in calls)
    assert concepts.to_dict() == original


def test_pairwise_ties_preserve_selected_definitions_and_groups():
    encode = encoder({'class': [1, 0], 'a': [1, 0], 'b': [0, 1], 'c': [0, 1]})
    concepts = Annotations(
        labels=['c', 'a', 'b'], states=[['dark', 'bright'], ['short', 'long'], ['rough', 'smooth', 'mixed']],
        types=['categorical'] * 3, groups={'textures': ['c', 'b']},
    )
    original = concepts.to_dict()
    result = SemanticConceptFilter(['class'], encode, encode).filter(concepts)
    assert result.labels == ['b']
    assert result.states == [['rough', 'smooth', 'mixed']]
    assert result.cardinalities == [3] and result.types == ['categorical']
    assert result.groups == {'textures': ['b']}
    assert concepts.to_dict() == original


def test_pairwise_pruning_prefers_more_general_concept():
    encode = encoder({'class': [-1, 0], 'a': [1, 0], 'b': [.8, .6], 'c': [0, 1]})
    concepts = Annotations(labels=['c', 'b', 'a'], cardinalities=[1] * 3)
    # a/b similarity is .8, b/c is .6, a/c is 0. Summed similarity favors b.
    result = SemanticConceptFilter(['class'], encode, encode, similarity_cutoff=.75).filter(concepts)
    assert result.labels == ['b', 'c']


def test_similarity_blend_normalizes_clip_and_uses_inclusive_cutoff():
    def clip(names):
        return torch.tensor([[10., 0.] if name == 'class' else [2., 0.] for name in names])

    sentence = encoder({'class': [1, 0], 'candidate': [0, 1]})
    concepts = Annotations(labels=['candidate'], cardinalities=[1])
    # CLIP cosine is 1, sentence similarity is 0: combined similarity is .75.
    assert SemanticConceptFilter(['class'], clip, sentence, class_cutoff=.76).filter(concepts).labels == ['candidate']
    with pytest.raises(ValueError, match='class-similarity'):
        SemanticConceptFilter(['class'], clip, sentence, class_cutoff=.75).filter(concepts)


def test_no_survivors_after_lexical_filtering_is_informative():
    def unused_encoder(names):
        pytest.fail('Text encoding should not run after every concept is removed.')

    concepts = Annotations(labels=['class', 'a very long concept'], cardinalities=[1, 1])
    with pytest.raises(ValueError, match='length and class-alias'):
        SemanticConceptFilter(['class'], unused_encoder, unused_encoder, max_length=5).filter(concepts)


def test_semantic_filter_is_a_pipeline_component():
    class Scores(Annotator):
        def annotate(self, dataset, concepts, **kwargs):
            return AnnotatedTensor(torch.ones(len(dataset), concepts.size), concepts, axis=1)

    encode = encoder({'class': [1, 0], 'a': [1, 0], 'b': [0, 1], 'c': [0, 1]})
    pipeline = ConceptGenerationPipeline(
        FixedConceptGenerator(['c', 'a', 'b']), Scores(),
        generator_filter=SemanticConceptFilter(['class'], encode, encode),
    )
    result = pipeline(ConceptDataset(torch.zeros(3, 2)))['Scores']
    assert result.annotations.labels == ['b'] and result.shape == (3, 1)
