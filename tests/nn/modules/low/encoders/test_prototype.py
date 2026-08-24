"""Tests for PrototypeEmbeddingToConcept."""
import pytest
import torch

from torch_concepts import Annotations
from torch_concepts.nn import PrototypeEmbeddingToConcept
from torch_concepts.nn.modules.low.encoders.prototype import (
    PrototypeEmbeddingToConcept as PrototypeEmbeddingToConceptDeep,
)


def _binary_annotations(n_concepts=2):
    return Annotations(
        labels=[f"concept_{i}" for i in range(n_concepts)],
        cardinalities=[1] * n_concepts,
        types=["binary"] * n_concepts,
    )


class TestPrototypeConstruction:
    def test_import_paths_agree(self):
        assert PrototypeEmbeddingToConcept is PrototypeEmbeddingToConceptDeep

    def test_buffers_are_frozen_and_sized_by_states(self):
        ann = Annotations(
            labels=["present", "color"],
            states=[["present"], ["red", "blue"]],
            types=["binary", "categorical"],
        )
        layer = PrototypeEmbeddingToConcept(4, ann, torch.randn(3, 4))

        assert layer.state_prototypes.shape == (3, 4)
        assert layer.temperature.item() == 12.0
        assert len(list(layer.parameters())) == 0
        assert layer.prototype_labels == ("present", "color:red", "color:blue")

    def test_rejects_invalid_configuration(self):
        ann = _binary_annotations()
        with pytest.raises(ValueError, match="shape"):
            PrototypeEmbeddingToConcept(4, ann, torch.randn(2, 3))
        with pytest.raises(ValueError, match="positive"):
            PrototypeEmbeddingToConcept(4, ann, torch.randn(2, 4), temperature=0)
        with pytest.raises(ValueError, match="prototype labels"):
            PrototypeEmbeddingToConcept(
                4, ann, torch.randn(2, 4), prototype_labels=["only one"]
            )

        continuous = Annotations(
            labels=["amount"], cardinalities=[1], types=["continuous"]
        )
        with pytest.raises(ValueError, match="binary and categorical"):
            PrototypeEmbeddingToConcept(4, continuous, torch.randn(1, 4))


class TestPrototypeForward:
    def test_cosine_logits_and_leading_dimensions(self):
        ann = _binary_annotations()
        prototypes = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        layer = PrototypeEmbeddingToConcept(2, ann, prototypes, temperature=2.0)
        x = torch.tensor([[[5.0, 0.0], [0.0, 6.0]]])

        logits = layer(x)
        expected = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        assert logits.shape == (1, 2, 2)
        torch.testing.assert_close(logits, expected)

    def test_custom_similarity_and_input_validation(self):
        ann = _binary_annotations(1)

        def dot(embeddings, prototypes):
            return embeddings @ prototypes.T

        layer = PrototypeEmbeddingToConcept(
            2, ann, torch.tensor([[1.0, 2.0]]), temperature=3.0, similarity=dot
        )
        torch.testing.assert_close(layer(torch.tensor([[2.0, 1.0]])), torch.tensor([[12.0]]))
        with pytest.raises(ValueError, match="final size"):
            layer(torch.randn(2, 3))

    def test_output_follows_input_dtype_and_gradients(self):
        ann = _binary_annotations(1)
        layer = PrototypeEmbeddingToConcept(3, ann, torch.randn(1, 3))
        x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        logits = layer(x)
        assert logits.dtype == torch.float64
        logits.sum().backward()
        assert x.grad is not None
        assert layer.state_prototypes.grad is None

    def test_checkpoint_restores_prototype_semantics(self):
        ann = _binary_annotations(2)
        layer = PrototypeEmbeddingToConcept(
            3, ann, torch.randn(2, 3), prototype_labels=["striped", "red"]
        )
        restored = PrototypeEmbeddingToConcept(3, ann, torch.zeros(2, 3))
        restored.load_state_dict(layer.state_dict())
        assert restored.prototype_labels == ("striped", "red")
        torch.testing.assert_close(restored.state_prototypes, layer.state_prototypes)
