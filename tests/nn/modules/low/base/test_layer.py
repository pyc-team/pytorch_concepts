"""Tests for torch_concepts.nn.modules.low.base.layer."""
import pytest
import torch
import torch.nn as nn

from torch_concepts.nn.modules.low.base.layer import BaseConceptLayer


# ---------------------------------------------------------------------------
# Helpers — concrete subclasses for testing
# ---------------------------------------------------------------------------

class _SimpleLayer(BaseConceptLayer):
    def __init__(self, out_concepts, in_concepts=None, in_embeddings=None):
        super().__init__(out_concepts=out_concepts,
                         in_concepts=in_concepts,
                         in_embeddings=in_embeddings)
        in_dim = (in_concepts or 0) + (in_embeddings or 0) or 1
        self.linear = nn.Linear(in_dim, out_concepts if isinstance(out_concepts, int) else 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestBaseConceptLayerConstruction:
    def test_out_concepts_stored(self):
        layer = _SimpleLayer(out_concepts=5)
        assert layer.out_concepts == 5

    def test_in_concepts_default_none(self):
        layer = _SimpleLayer(out_concepts=5)
        assert layer.in_concepts is None

    def test_in_embeddings_default_none(self):
        layer = _SimpleLayer(out_concepts=5)
        assert layer.in_embeddings is None

    def test_all_dims_stored(self):
        layer = _SimpleLayer(out_concepts=5, in_concepts=10, in_embeddings=8)
        assert layer.out_concepts == 5
        assert layer.in_concepts == 10
        assert layer.in_embeddings == 8

    def test_is_nn_module(self):
        layer = _SimpleLayer(out_concepts=5)
        assert isinstance(layer, nn.Module)


# ===========================================================================
# 2. Abstract forward
# ===========================================================================

class TestBaseConceptLayerAbstract:
    def test_forward_raises_not_implemented(self):
        layer = BaseConceptLayer(out_concepts=5)
        with pytest.raises(NotImplementedError):
            layer(torch.randn(2, 5))

    def test_subclass_can_implement_forward(self):
        layer = _SimpleLayer(out_concepts=5, in_concepts=10)
        out = layer(torch.randn(3, 10))
        assert out.shape == (3, 5)

    def test_output_in_valid_range_for_sigmoid(self):
        layer = _SimpleLayer(out_concepts=3, in_concepts=4)
        out = layer(torch.randn(5, 4))
        assert (out >= 0).all() and (out <= 1).all()


# ===========================================================================
# 3. shape attributes
# ===========================================================================

class TestBaseConceptLayerShapes:
    def test_in_concepts_shape_is_int(self):
        layer = _SimpleLayer(out_concepts=5, in_concepts=10)
        assert layer.in_concepts_shape == 10

    def test_in_embeddings_shape_is_int(self):
        layer = _SimpleLayer(out_concepts=5, in_embeddings=8)
        assert layer.in_embeddings_shape == 8

    def test_out_concepts_shape_is_int(self):
        layer = _SimpleLayer(out_concepts=7)
        assert layer.out_concepts_shape == 7

    def test_in_concepts_shape_none_when_not_provided(self):
        layer = _SimpleLayer(out_concepts=5)
        assert layer.in_concepts_shape is None

    def test_in_embeddings_shape_none_when_not_provided(self):
        layer = _SimpleLayer(out_concepts=5)
        assert layer.in_embeddings_shape is None


# ===========================================================================
# 4. prune raises NotImplementedError
# ===========================================================================

class TestBaseConceptLayerPrune:
    def test_prune_raises(self):
        layer = _SimpleLayer(out_concepts=5, in_concepts=4)
        with pytest.raises(NotImplementedError):
            layer.prune(torch.ones(5))


# ===========================================================================
# 5. Gradient flow
# ===========================================================================

class TestBaseConceptLayerGradients:
    def test_gradient_flows_to_input(self):
        layer = _SimpleLayer(out_concepts=3, in_concepts=4)
        x = torch.randn(2, 4, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_gradient_flows_to_weights(self):
        layer = _SimpleLayer(out_concepts=3, in_concepts=4)
        x = torch.randn(2, 4)
        loss = layer(x).sum()
        loss.backward()
        assert layer.linear.weight.grad is not None


# ===========================================================================
# 6. Encoder-style subclass (embedding → concept)
# ===========================================================================

class TestEncoderSubclass:
    def test_embedding_encoder_forward(self):
        class EmbEncoder(BaseConceptLayer):
            def __init__(self, in_emb, out_c):
                super().__init__(out_concepts=out_c, in_embeddings=in_emb)
                self.net = nn.Sequential(nn.Linear(in_emb, 64), nn.ReLU(), nn.Linear(64, out_c))

            def forward(self, embeddings):
                return self.net(embeddings)

        enc = EmbEncoder(in_emb=128, out_c=10)
        out = enc(torch.randn(4, 128))
        assert out.shape == (4, 10)
        assert enc.in_embeddings == 128
        assert enc.out_concepts == 10

    def test_encoder_has_no_in_concepts(self):
        class EmbEncoder(BaseConceptLayer):
            def __init__(self, in_emb, out_c):
                super().__init__(out_concepts=out_c, in_embeddings=in_emb)
            def forward(self, x):
                return x

        enc = EmbEncoder(in_emb=16, out_c=5)
        assert enc.in_concepts is None


# ===========================================================================
# 7. Predictor-style subclass (concept → label)
# ===========================================================================

class TestPredictorSubclass:
    def test_predictor_forward(self):
        class ConceptPredictor(BaseConceptLayer):
            def __init__(self, in_c, out_c):
                super().__init__(out_concepts=out_c, in_concepts=in_c)
                self.linear = nn.Linear(in_c, out_c)

            def forward(self, concepts):
                return self.linear(concepts)

        pred = ConceptPredictor(in_c=10, out_c=3)
        out = pred(torch.randn(4, 10))
        assert out.shape == (4, 3)

    def test_gradient_pipeline(self):
        class Enc(BaseConceptLayer):
            def __init__(self):
                super().__init__(out_concepts=10, in_embeddings=20)
                self.l = nn.Linear(20, 10)
            def forward(self, x):
                return self.l(x)

        class Pred(BaseConceptLayer):
            def __init__(self):
                super().__init__(out_concepts=5, in_concepts=10)
                self.l = nn.Linear(10, 5)
            def forward(self, c):
                return self.l(c)

        enc, pred = Enc(), Pred()
        x = torch.randn(2, 20, requires_grad=True)
        c = enc(x)
        y = pred(c)
        y.sum().backward()
        assert x.grad is not None
        assert enc.l.weight.grad is not None
        assert pred.l.weight.grad is not None
