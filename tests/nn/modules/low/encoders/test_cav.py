"""Tests for CAVEmbeddingToConcept."""
import pytest
import torch

from torch_concepts.nn import CAVEmbeddingToConcept
from torch_concepts.nn.modules.low.encoders.cav import (
    CAVEmbeddingToConcept as CAVEmbeddingToConceptDeep,
)


def _make_layer_and_data(d=16, n=2000, seed=0):
    """Layer + data with a planted concept direction (first basis vector)."""
    torch.manual_seed(seed)
    v_true = torch.zeros(d)
    v_true[0] = 1.0
    x = torch.randn(n, d)
    c = (x @ v_true > 0).float().unsqueeze(1)
    layer = CAVEmbeddingToConcept(in_embeddings=d, out_concepts=1)
    return layer, x, c, v_true


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestCAVConstruction:
    def test_import_paths_agree(self):
        assert CAVEmbeddingToConcept is CAVEmbeddingToConceptDeep

    def test_buffer_shapes(self):
        layer = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=3)
        assert layer.cavs.shape == (3, 16)
        assert layer.bias.shape == (3,)
        assert torch.equal(layer.cavs, torch.zeros(3, 16))
        assert not layer.fitted

    def test_no_trainable_parameters(self):
        layer = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=3)
        assert len(list(layer.parameters())) == 0

    def test_buffers_in_state_dict(self):
        layer = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=3)
        assert set(layer.state_dict().keys()) == {"cavs", "bias", "fitted"}

    def test_forward_before_fit_raises(self):
        layer = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=3)
        with pytest.raises(RuntimeError, match="not been fitted"):
            layer(torch.randn(4, 16))


# ===========================================================================
# 2. Fitting / CAV recovery
# ===========================================================================

class TestCAVFit:
    def test_recovers_planted_direction(self):
        layer, x, c, v_true = _make_layer_and_data()
        acc = layer.fit(x, c)
        cos = torch.nn.functional.cosine_similarity(
            layer.cavs[0], v_true, dim=0
        )
        # sign matters: the CAV must point towards the concept
        assert cos > 0.95
        assert acc.shape == (1,)
        assert acc[0] > 0.95
        assert layer.fitted

    def test_cavs_have_unit_norm(self):
        torch.manual_seed(0)
        layer = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=3)
        x = torch.randn(512, 16)
        c = (x[:, :3] > 0).float()
        layer.fit(x, c)
        assert torch.allclose(
            layer.cavs.norm(dim=1), torch.ones(3), atol=1e-6
        )

    def test_sample_count_mismatch_raises(self):
        layer = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=1)
        with pytest.raises(ValueError, match="number of samples"):
            layer.fit(torch.randn(8, 16), torch.zeros(9, 1))

    def test_recovers_shifted_boundary_bias(self):
        """A boundary NOT through the origin pins the bias: for labels
        x[0] > 0.7 the signed distance is x[0] - 0.7, so bias ~ -0.7."""
        torch.manual_seed(0)
        d = 16
        x = torch.randn(4000, d)
        c = (x[:, :1] > 0.7).float()
        layer = CAVEmbeddingToConcept(in_embeddings=d, out_concepts=1)
        layer.fit(x, c)
        assert abs(layer.bias[0].item() + 0.7) < 0.1
        assert ((layer(x) > 0).float() == c).float().mean() > 0.95

    def test_fit_kwargs_forwarded(self):
        from sklearn.exceptions import ConvergenceWarning
        layer = CAVEmbeddingToConcept(
            in_embeddings=16, out_concepts=1, max_iter=7, C=0.5
        )
        assert layer.fit_kwargs == {"max_iter": 7, "C": 0.5}
        # max_iter=1 must actually reach sklearn: it cannot converge
        layer1 = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=1,
                                       max_iter=1)
        _, x, c, _ = _make_layer_and_data()
        with pytest.warns(ConvergenceWarning):
            layer1.fit(x, c)

    def test_integer_coded_categorical_labels_rejected(self):
        """dataset.concepts stores categoricals as integer class indices;
        fitting on them must fail loudly, not fit a silent multinomial."""
        layer = CAVEmbeddingToConcept(in_embeddings=8, out_concepts=1)
        x = torch.randn(64, 8)
        c = torch.randint(0, 3, (64, 1)).float()  # {0, 1, 2}
        with pytest.raises(ValueError, match="one-hot"):
            layer.fit(x, c)

    def test_fit_wrong_feature_width_raises(self):
        """A width mismatch must not silently regroup features."""
        layer = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=2)
        with pytest.raises(ValueError, match="features"):
            layer.fit(torch.randn(100, 8), torch.zeros(50, 2))

    def test_fit_bfloat16_embeddings(self):
        layer, x, c, v_true = _make_layer_and_data()
        layer.fit(x.bfloat16(), c)
        cos = torch.nn.functional.cosine_similarity(
            layer.cavs[0], v_true, dim=0
        )
        assert cos > 0.9

    def test_constant_concept_column_names_the_culprit(self):
        layer = CAVEmbeddingToConcept(in_embeddings=8, out_concepts=2)
        x = torch.randn(32, 8)
        c = torch.cat([(x[:, :1] > 0).float(), torch.zeros(32, 1)], dim=1)
        with pytest.raises(ValueError, match="concept column 1"):
            layer.fit(x, c)

    def test_categorical_concept_as_one_hot_states(self):
        """A categorical concept is k one-vs-rest binaries: constructing
        with a nested Annotations fits one CAV per state column."""
        from torch_concepts import Annotations
        torch.manual_seed(0)
        d = 16
        ann = Annotations(labels=["color", "fine"], cardinalities=[3, 1])
        layer = CAVEmbeddingToConcept(in_embeddings=d, out_concepts=ann)
        x = torch.randn(600, d)
        color = torch.argmax(x[:, :3], dim=1)  # 3-state categorical
        labels = torch.cat([
            torch.nn.functional.one_hot(color, 3).float(),
            (x[:, 3:4] > 0).float(),
        ], dim=1)
        acc = layer.fit(x, labels)
        assert layer.cavs.shape == (4, d)  # one CAV per state + one binary
        assert (acc > 0.8).all()
        assert layer(x).shape == (600, 4)


# ===========================================================================
# 3. Forward pass
# ===========================================================================

class TestCAVForward:
    def test_output_shape(self):
        layer, x, c, _ = _make_layer_and_data()
        layer.fit(x, c)
        assert layer(x).shape == (x.shape[0], 1)

    def test_leading_dims_supported(self):
        layer, x, c, _ = _make_layer_and_data(d=16)
        layer.fit(x, c)
        assert layer(torch.randn(4, 5, 16)).shape == (4, 5, 1)

    def test_sign_predicts_concept(self):
        layer, x, c, _ = _make_layer_and_data()
        layer.fit(x, c)
        pred = (layer(x) > 0).float()
        assert (pred == c).float().mean() > 0.95

    def test_deterministic(self):
        layer, x, c, _ = _make_layer_and_data()
        layer.fit(x, c)
        assert torch.equal(layer(x), layer(x))

    def test_gradient_wrt_input_is_unit_cav(self):
        layer, x, c, _ = _make_layer_and_data()
        layer.fit(x, c)
        inp = torch.randn(8, 16, requires_grad=True)
        layer(inp).sum().backward()
        assert torch.allclose(inp.grad, layer.cavs[0].expand(8, 16))

    def test_output_dtype_follows_input(self):
        layer, x, c, _ = _make_layer_and_data()
        layer.fit(x, c)
        assert layer(x.half()).dtype == torch.half

    def test_state_dict_round_trip(self):
        layer, x, c, _ = _make_layer_and_data()
        layer.fit(x, c)
        restored = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=1)
        restored.load_state_dict(layer.state_dict())
        assert restored.fitted
        assert torch.equal(restored(x), layer(x))
