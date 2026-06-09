"""
Comprehensive tests for torch_concepts.nn.modules.low.base

Tests base classes for concept layers:
- BaseConceptLayer
- BaseEncoder
- BasePredictor
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.base.layer import (
    BaseConceptLayer,
    BaseEncoder,
    BasePredictor,
)


class TestBaseConceptLayer(unittest.TestCase):
    """Test BaseConceptLayer abstract class."""

    def test_initialization(self):
        """Test initialization with various feature dimensions."""
        class ConcreteLayer(BaseConceptLayer):
            def forward(self, x):
                return x

        layer = ConcreteLayer(
            out_concepts=5,
            in_concepts=10,
            in_embeddings=8,
        )

        self.assertEqual(layer.out_concepts, 5)
        self.assertEqual(layer.in_concepts, 10)
        self.assertEqual(layer.in_embeddings, 8)

    def test_initialization_minimal(self):
        """Test initialization with only required arguments."""
        class ConcreteLayer(BaseConceptLayer):
            def forward(self, x):
                return x

        layer = ConcreteLayer(out_concepts=5)

        self.assertEqual(layer.out_concepts, 5)
        self.assertIsNone(layer.in_concepts)
        self.assertIsNone(layer.in_embeddings)

    def test_abstract_forward(self):
        """Test that forward must be implemented."""
        # BaseConceptLayer itself should raise NotImplementedError
        layer = BaseConceptLayer(out_concepts=5)

        with self.assertRaises(NotImplementedError):
            layer(torch.randn(2, 5))

    def test_subclass_implementation(self):
        """Test proper subclass implementation."""
        class MyLayer(BaseConceptLayer):
            def __init__(self, out_concepts, in_concepts):
                super().__init__(
                    out_concepts=out_concepts,
                    in_concepts=in_concepts
                )
                self.linear = nn.Linear(in_concepts, out_concepts)

            def forward(self, endogenous):
                return torch.sigmoid(self.linear(endogenous))

        layer = MyLayer(out_concepts=5, in_concepts=10)
        x = torch.randn(2, 10)
        output = layer(x)

        self.assertEqual(output.shape, (2, 5))
        self.assertTrue((output >= 0).all() and (output <= 1).all())


class TestBaseEncoder(unittest.TestCase):
    """Test BaseEncoder abstract class."""

    def test_initialization(self):
        """Test encoder initialization."""
        class ConcreteEncoder(BaseEncoder):
            def forward(self, x):
                return x

        encoder = ConcreteEncoder(
            out_concepts=10,
            in_embeddings=784,
        )

        self.assertEqual(encoder.out_concepts, 10)
        self.assertEqual(encoder.in_embeddings, 784)
        self.assertIsNone(encoder.in_concepts)

    def test_no_endogenous_input(self):
        """Test that encoders don't accept endogenous."""
        class ConcreteEncoder(BaseEncoder):
            def forward(self, x):
                return x

        encoder = ConcreteEncoder(out_concepts=10, in_embeddings=784)

        # in_concepts should always be None for encoders
        self.assertIsNone(encoder.in_concepts)

    def test_encoder_implementation(self):
        """Test concrete encoder implementation."""
        class MyEncoder(BaseEncoder):
            def __init__(self, out_concepts, in_embeddings):
                super().__init__(
                    out_concepts=out_concepts,
                    in_embeddings=in_embeddings,
                )
                self.net = nn.Sequential(
                    nn.Linear(in_embeddings, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_concepts)
                )

            def forward(self, embeddings):
                return self.net(embeddings)

        encoder = MyEncoder(out_concepts=10, in_embeddings=784)
        x = torch.randn(4, 784)
        concepts = encoder(x)

        self.assertEqual(concepts.shape, (4, 10))

    def test_with_combined_features(self):
        """Test encoder that combines concept and embedding inputs."""
        class CombinedEncoder(BaseEncoder):
            def __init__(self, out_concepts, in_embeddings, in_concepts):
                super().__init__(
                    out_concepts=out_concepts,
                    in_embeddings=in_embeddings,
                )
                self.in_concepts = in_concepts
                total_features = in_embeddings + in_concepts
                self.net = nn.Linear(total_features, out_concepts)

            def forward(self, embeddings, concepts):
                combined = torch.cat([embeddings, concepts], dim=-1)
                return self.net(combined)

        encoder = CombinedEncoder(out_concepts=5, in_embeddings=10, in_concepts=3)
        embeddings = torch.randn(2, 10)
        concepts = torch.randn(2, 3)
        output = encoder(embeddings, concepts)

        self.assertEqual(output.shape, (2, 5))


class TestBasePredictor(unittest.TestCase):
    """Test BasePredictor abstract class."""

    def test_initialization(self):
        """Test predictor initialization."""
        class ConcretePredictor(BasePredictor):
            def forward(self, x):
                return x

        predictor = ConcretePredictor(
            out_concepts=3,
            in_concepts=10
        )

        self.assertEqual(predictor.out_concepts, 3)
        self.assertEqual(predictor.in_concepts, 10)

    def test_predictor_implementation(self):
        """Test concrete predictor implementation."""
        class MyPredictor(BasePredictor):
            def __init__(self, out_concepts, in_concepts):
                super().__init__(
                    out_concepts=out_concepts,
                    in_concepts=in_concepts,
                )
                self.linear = nn.Linear(in_concepts, out_concepts)

            def forward(self, concepts):
                return self.linear(concepts)

        predictor = MyPredictor(out_concepts=3, in_concepts=10)
        concepts = torch.randn(4, 10)
        tasks = predictor(concepts)

        self.assertEqual(tasks.shape, (4, 3))

    def test_with_embedding_features(self):
        """Test predictor with embedding features."""
        class PredictorWithEmbedding(BasePredictor):
            def __init__(self, out_concepts, in_concepts, in_latent):
                super().__init__(
                    out_concepts=out_concepts,
                    in_concepts=in_concepts,
                    in_latent=in_latent
                )
                total_features = in_concepts + in_latent
                self.linear = nn.Linear(total_features, out_concepts)

            def forward(self, concepts, latent):                # concepts are already activated (probabilities)
                combined = torch.cat([concepts, latent], dim=-1)
                return self.linear(combined)

        predictor = PredictorWithEmbedding(
            out_concepts=3,
            in_concepts=10,
            in_latent=8
        )

        concepts = torch.randn(2, 10)
        latent = torch.randn(2, 8)
        output = predictor(concepts, latent)

        self.assertEqual(output.shape, (2, 3))

    def test_numerical_stability(self):
        """Test that predictor handles extreme inputs."""
        class SimplePredictor(BasePredictor):
            def __init__(self, out_concepts, in_concepts):
                super().__init__(
                    out_concepts=out_concepts,
                    in_concepts=in_concepts,
                )
                self.linear = nn.Linear(in_concepts, out_concepts)

            def forward(self, concepts):
                return self.linear(concepts)

        predictor = SimplePredictor(out_concepts=3, in_concepts=5)

        # Test with probability-like inputs
        endogenous = torch.tensor([[0.0, 0.0, 0.5, 1.0, 1.0]])
        output = predictor(endogenous)

        # Output should be finite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestLayerIntegration(unittest.TestCase):
    """Test integration between different base classes."""

    def test_encoder_to_predictor_pipeline(self):
        """Test encoder followed by predictor."""
        class SimpleEncoder(BaseEncoder):
            def __init__(self, out_concepts, in_latent):
                super().__init__(out_concepts, in_latent)
                self.linear = nn.Linear(in_latent, out_concepts)

            def forward(self, x):
                return self.linear(x)

        class SimplePredictor(BasePredictor):
            def __init__(self, out_concepts, in_concepts):
                super().__init__(out_concepts, in_concepts)
                self.linear = nn.Linear(in_concepts, out_concepts)

            def forward(self, concepts):
                return self.linear(concepts)

        # Create pipeline
        encoder = SimpleEncoder(out_concepts=10, in_latent=784)
        predictor = SimplePredictor(out_concepts=5, in_concepts=10)

        # Test pipeline
        x = torch.randn(2, 784)
        concepts = encoder(x)
        predictions = predictor(concepts)

        self.assertEqual(concepts.shape, (2, 10))
        self.assertEqual(predictions.shape, (2, 5))

    def test_gradient_flow_through_pipeline(self):
        """Test gradient flow through encoder-predictor pipeline."""
        class SimpleEncoder(BaseEncoder):
            def __init__(self, out_concepts, in_latent):
                super().__init__(out_concepts, in_latent)
                self.linear = nn.Linear(in_latent, out_concepts)

            def forward(self, x):
                return self.linear(x)

        class SimplePredictor(BasePredictor):
            def __init__(self, out_concepts, in_concepts):
                super().__init__(out_concepts, in_concepts)
                self.linear = nn.Linear(in_concepts, out_concepts)

            def forward(self, concepts):
                return self.linear(concepts)

        encoder = SimpleEncoder(out_concepts=10, in_latent=20)
        predictor = SimplePredictor(out_concepts=5, in_concepts=10)

        x = torch.randn(2, 20, requires_grad=True)
        concepts = encoder(x)
        predictions = predictor(concepts)
        loss = predictions.sum()
        loss.backward()

        # Gradients should flow to input
        self.assertIsNotNone(x.grad)
        # Gradients should exist for both modules
        self.assertIsNotNone(encoder.linear.weight.grad)
        self.assertIsNotNone(predictor.linear.weight.grad)


if __name__ == '__main__':
    unittest.main()

