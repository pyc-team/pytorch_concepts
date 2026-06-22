"""
Comprehensive tests for torch_concepts.nn.modules.low.dense_layers

Tests activation utilities and dense layer implementations:
- get_layer_activation function
- Dense layer
- MLP (Multi-Layer Perceptron)
- ResidualMLP
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.dense_layers import (
    get_layer_activation,
    Dense,
    MLP,
    ResidualMLP,
    SelectorEmbeddingEncoder,
)


class TestGetLayerActivation(unittest.TestCase):
    """Test activation layer retrieval."""

    def test_relu_activation(self):
        """Test ReLU activation."""
        act_class = get_layer_activation('relu')
        self.assertEqual(act_class, nn.ReLU)
        act = act_class()
        self.assertIsInstance(act, nn.ReLU)

    def test_sigmoid_activation(self):
        """Test sigmoid activation."""
        act_class = get_layer_activation('sigmoid')
        self.assertEqual(act_class, nn.Sigmoid)

    def test_tanh_activation(self):
        """Test tanh activation."""
        act_class = get_layer_activation('tanh')
        self.assertEqual(act_class, nn.Tanh)

    def test_case_insensitive(self):
        """Test case insensitivity."""
        act_class_lower = get_layer_activation('relu')
        act_class_upper = get_layer_activation('RELU')
        act_class_mixed = get_layer_activation('ReLu')

        self.assertEqual(act_class_lower, act_class_upper)
        self.assertEqual(act_class_lower, act_class_mixed)

    def test_none_returns_identity(self):
        """Test that None returns Identity."""
        act_class = get_layer_activation(None)
        self.assertEqual(act_class, nn.Identity)

    def test_linear_returns_identity(self):
        """Test that 'linear' returns Identity."""
        act_class = get_layer_activation('linear')
        self.assertEqual(act_class, nn.Identity)

    def test_invalid_activation(self):
        """Test invalid activation name."""
        with self.assertRaises(ValueError):
            get_layer_activation('invalid_activation')

    def test_all_supported_activations(self):
        """Test all supported activation functions."""
        activations = [
            'elu', 'leaky_relu', 'prelu', 'relu', 'rrelu', 'selu',
            'celu', 'gelu', 'glu', 'mish', 'sigmoid', 'softplus',
            'tanh', 'silu', 'swish', 'linear'
        ]

        for act_name in activations:
            act_class = get_layer_activation(act_name)
            self.assertTrue(issubclass(act_class, nn.Module))


class TestDense(unittest.TestCase):
    """Test Dense layer."""

    def test_initialization(self):
        """Test Dense layer initialization."""
        layer = Dense(in_features=10, out_features=5)
        self.assertEqual(layer.affinity.in_features, 10)
        self.assertEqual(layer.affinity.out_features, 5)

    def test_forward(self):
        """Test forward pass."""
        layer = Dense(in_features=10, out_features=5)
        x = torch.randn(2, 10)
        output = layer(x)
        self.assertEqual(output.shape, (2, 5))

    def test_with_dropout(self):
        """Test with dropout."""
        layer = Dense(in_features=10, out_features=5, dropout=0.5)
        layer.train()
        x = torch.randn(100, 10)
        output = layer(x)
        self.assertEqual(output.shape, (100, 5))

    def test_without_bias(self):
        """Test without bias."""
        layer = Dense(in_features=10, out_features=5, bias=False)
        self.assertIsNone(layer.affinity.bias)

    def test_different_activations(self):
        """Test with different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'linear']
        for act in activations:
            layer = Dense(in_features=10, out_features=5, activation=act)
            x = torch.randn(2, 10)
            output = layer(x)
            self.assertEqual(output.shape, (2, 5))

    def test_reset_parameters(self):
        """Test parameter reset."""
        layer = Dense(in_features=10, out_features=5)
        old_weight = layer.affinity.weight.clone()
        layer.reset_parameters()
        self.assertFalse(torch.allclose(old_weight, layer.affinity.weight))

    def test_gradient_flow(self):
        """Test gradient flow."""
        layer = Dense(in_features=10, out_features=5)
        x = torch.randn(2, 10, requires_grad=True)
        output = layer(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.affinity.weight.grad)


class TestMLP(unittest.TestCase):
    """Test MLP (Multi-Layer Perceptron)."""

    def test_initialization(self):
        """Test MLP initialization."""
        mlp = MLP(input_size=10, hidden_size=64, n_layers=2)
        self.assertIsNotNone(mlp.mlp)
        self.assertEqual(len(mlp.mlp), 2)

    def test_forward_without_readout(self):
        """Test forward without readout."""
        mlp = MLP(input_size=10, hidden_size=64, n_layers=2)
        x = torch.randn(2, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_forward_with_readout(self):
        """Test forward with readout."""
        mlp = MLP(input_size=10, hidden_size=64, output_size=5, n_layers=2)
        x = torch.randn(2, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 5))

    def test_single_layer(self):
        """Test with single layer."""
        mlp = MLP(input_size=10, hidden_size=64, n_layers=1)
        x = torch.randn(2, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_deep_network(self):
        """Test deep network."""
        mlp = MLP(input_size=10, hidden_size=64, n_layers=5)
        x = torch.randn(2, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_with_dropout(self):
        """Test with dropout."""
        mlp = MLP(input_size=10, hidden_size=64, n_layers=2, dropout=0.5)
        mlp.train()
        x = torch.randn(100, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (100, 64))

    def test_different_activation(self):
        """Test with different activation."""
        mlp = MLP(input_size=10, hidden_size=64, n_layers=2, activation='tanh')
        x = torch.randn(2, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_reset_parameters(self):
        """Test parameter reset."""
        mlp = MLP(input_size=10, hidden_size=64, output_size=5, n_layers=2)
        old_weight = list(mlp.mlp[0].affinity.weight.clone() for _ in range(1))
        mlp.reset_parameters()
        # Parameters should be different after reset
        new_weight = mlp.mlp[0].affinity.weight
        self.assertFalse(torch.allclose(old_weight[0], new_weight))

    def test_gradient_flow(self):
        """Test gradient flow."""
        mlp = MLP(input_size=10, hidden_size=64, output_size=5, n_layers=2)
        x = torch.randn(2, 10, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)


class TestResidualMLP(unittest.TestCase):
    """Test ResidualMLP."""

    def test_initialization(self):
        """Test ResidualMLP initialization."""
        mlp = ResidualMLP(input_size=64, hidden_size=64, n_layers=2)
        self.assertEqual(len(mlp.layers), 2)
        self.assertEqual(len(mlp.skip_connections), 2)

    def test_forward_without_readout(self):
        """Test forward without readout."""
        mlp = ResidualMLP(input_size=64, hidden_size=64, n_layers=2)
        x = torch.randn(2, 64)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_forward_with_readout(self):
        """Test forward with readout."""
        mlp = ResidualMLP(input_size=64, hidden_size=64, output_size=5, n_layers=2)
        x = torch.randn(2, 64)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 5))

    def test_input_projection(self):
        """Test with input size different from hidden size."""
        mlp = ResidualMLP(input_size=10, hidden_size=64, n_layers=2)
        x = torch.randn(2, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_parametrized_skip(self):
        """Test with parametrized skip connections."""
        mlp = ResidualMLP(input_size=64, hidden_size=64, n_layers=2, parametrized_skip=True)
        x = torch.randn(2, 64)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_with_dropout(self):
        """Test with dropout."""
        mlp = ResidualMLP(input_size=64, hidden_size=64, n_layers=2, dropout=0.5)
        mlp.train()
        x = torch.randn(100, 64)
        output = mlp(x)
        self.assertEqual(output.shape, (100, 64))

    def test_different_activation(self):
        """Test with different activation."""
        mlp = ResidualMLP(input_size=64, hidden_size=64, n_layers=2, activation='tanh')
        x = torch.randn(2, 64)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 64))

    def test_residual_connections(self):
        """Test that residual connections work."""
        # Create a very deep network
        mlp = ResidualMLP(input_size=64, hidden_size=64, n_layers=10)
        x = torch.randn(2, 64)
        output = mlp(x)

        # Should not explode or vanish due to residuals
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_gradient_flow(self):
        """Test gradient flow through residual connections."""
        mlp = ResidualMLP(input_size=64, hidden_size=64, output_size=5, n_layers=3)
        x = torch.randn(2, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        # Gradients should not vanish
        self.assertTrue((x.grad.abs() > 1e-10).any())


class TestSelectorEmbeddingEncoder(unittest.TestCase):
    """Test SelectorEmbeddingEncoder."""

    def test_initialization(self):
        """Test initialization stores correct attributes."""
        sel = SelectorEmbeddingEncoder(
            in_features=64,
            out_features=32,
            n_embeddings=5,
            memory_size=10,
        )
        self.assertEqual(sel.out_features, 32)
        self.assertEqual(sel.memory_size, 10)
        self.assertEqual(sel.memory.num_embeddings, 5)
        self.assertEqual(sel.memory.embedding_dim, 10 * 32)

    def test_forward_soft_shape(self):
        """Test soft-selection output shape."""
        sel = SelectorEmbeddingEncoder(
            in_features=64, out_features=32, n_embeddings=5, memory_size=10
        )
        x = torch.randn(4, 64)
        out = sel(x, sampling=False)
        self.assertEqual(out.shape, (4, 5, 32))

    def test_forward_hard_shape(self):
        """Test hard (Gumbel-softmax) selection output shape."""
        sel = SelectorEmbeddingEncoder(
            in_features=64, out_features=32, n_embeddings=5, memory_size=10
        )
        x = torch.randn(4, 64)
        out = sel(x, sampling=True)
        self.assertEqual(out.shape, (4, 5, 32))

    def test_temperature_effect(self):
        """Lower temperature should produce harder distributions."""
        sel_hot = SelectorEmbeddingEncoder(
            in_features=16, out_features=8, n_embeddings=3, memory_size=4, temperature=10.0
        )
        sel_cold = SelectorEmbeddingEncoder(
            in_features=16, out_features=8, n_embeddings=3, memory_size=4, temperature=0.1
        )
        x = torch.randn(2, 16)
        # Both should run without error
        self.assertEqual(sel_hot(x).shape, (2, 3, 8))
        self.assertEqual(sel_cold(x).shape, (2, 3, 8))

    def test_gradient_flow(self):
        """Test gradient flow through soft selection."""
        sel = SelectorEmbeddingEncoder(
            in_features=16, out_features=8, n_embeddings=3, memory_size=4
        )
        x = torch.randn(2, 16, requires_grad=True)
        out = sel(x, sampling=False)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(sel.memory.weight.grad)

    def test_single_embedding(self):
        """Test with n_embeddings=1."""
        sel = SelectorEmbeddingEncoder(
            in_features=8, out_features=4, n_embeddings=1, memory_size=3
        )
        x = torch.randn(2, 8)
        self.assertEqual(sel(x).shape, (2, 1, 4))


class TestLayerComparison(unittest.TestCase):
    """Test comparisons between different layer types."""

    def test_mlp_vs_residual_mlp(self):
        """Compare MLP with ResidualMLP."""
        torch.manual_seed(42)

        mlp = MLP(input_size=64, hidden_size=64, output_size=5, n_layers=3)
        res_mlp = ResidualMLP(input_size=64, hidden_size=64, output_size=5, n_layers=3)

        x = torch.randn(2, 64)

        output_mlp = mlp(x)
        output_res = res_mlp(x)

        # Outputs should be different due to residual connections
        self.assertEqual(output_mlp.shape, output_res.shape)


if __name__ == '__main__':
    unittest.main()



# ===========================================================================
# Sequential extra tests
# ===========================================================================
import sys
sys.path.insert(0, '/Users/gdefelice/Project_local/PyC/forks/pytorch_concepts')
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.sequential import Sequential
from torch_concepts.annotations import AxisAnnotation
from torch_concepts.tensor import AnnotatedTensor


class TestSequential:
    def test_single_input_passes_through_chain(self):
        seq = Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        x = torch.randn(2, 4)
        out = seq(x)
        assert out.shape == (2, 3)

    def test_empty_container_returns_input(self):
        seq = Sequential()
        x = torch.randn(2, 4)
        out = seq(x)
        assert out is x

    def test_empty_container_multi_input_returns_none(self):
        seq = Sequential()
        out = seq(torch.randn(2, 4), torch.randn(2, 3))
        assert out is None

    def test_multi_input_first_layer(self):
        class CatLayer(nn.Module):
            def forward(self, a, b):
                return torch.cat([a, b], dim=1)
        seq = Sequential(CatLayer(), nn.Linear(7, 3))
        a, b = torch.randn(2, 4), torch.randn(2, 3)
        out = seq(a, b)
        assert out.shape == (2, 3)

    def test_annotate_with_stored_annotation(self):
        ann = AxisAnnotation(labels=['a', 'b', 'c'])
        seq = Sequential(nn.Linear(4, 3), out_concepts=ann)
        x = torch.randn(2, 3)
        result = seq.annotate(x)
        assert isinstance(result, AnnotatedTensor)

    def test_annotate_with_explicit_annotation(self):
        seq = Sequential(nn.Linear(4, 3))
        ann = AxisAnnotation(labels=['x', 'y', 'z'])
        x = torch.randn(2, 3)
        result = seq.annotate(x, out_concepts=ann)
        assert isinstance(result, AnnotatedTensor)

    def test_annotate_without_annotation_returns_tensor(self):
        seq = Sequential(nn.Linear(4, 3))
        x = torch.randn(2, 3)
        result = seq.annotate(x)
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, AnnotatedTensor)
