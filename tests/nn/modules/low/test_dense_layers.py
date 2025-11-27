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
        layer = Dense(input_size=10, output_size=5)
        self.assertEqual(layer.affinity.in_features, 10)
        self.assertEqual(layer.affinity.out_features, 5)

    def test_forward(self):
        """Test forward pass."""
        layer = Dense(input_size=10, output_size=5)
        x = torch.randn(2, 10)
        output = layer(x)
        self.assertEqual(output.shape, (2, 5))

    def test_with_dropout(self):
        """Test with dropout."""
        layer = Dense(input_size=10, output_size=5, dropout=0.5)
        layer.train()  # Enable dropout
        x = torch.randn(100, 10)
        output = layer(x)
        self.assertEqual(output.shape, (100, 5))

    def test_without_bias(self):
        """Test without bias."""
        layer = Dense(input_size=10, output_size=5, bias=False)
        self.assertIsNone(layer.affinity.bias)

    def test_different_activations(self):
        """Test with different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'linear']

        for act in activations:
            layer = Dense(input_size=10, output_size=5, activation=act)
            x = torch.randn(2, 10)
            output = layer(x)
            self.assertEqual(output.shape, (2, 5))

    def test_reset_parameters(self):
        """Test parameter reset."""
        layer = Dense(input_size=10, output_size=5)
        old_weight = layer.affinity.weight.clone()
        layer.reset_parameters()
        # Parameters should be different after reset
        self.assertFalse(torch.allclose(old_weight, layer.affinity.weight))

    def test_gradient_flow(self):
        """Test gradient flow."""
        layer = Dense(input_size=10, output_size=5)
        x = torch.randn(2, 10, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

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

