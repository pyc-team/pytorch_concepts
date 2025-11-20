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
        # Create a concrete subclass
        class ConcreteLayer(BaseConceptLayer):
            def forward(self, x):
                return x

        layer = ConcreteLayer(
            out_features=5,
            in_features_logits=10,
            in_features_embedding=8,
            in_features_exogenous=2
        )

        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.in_features_logits, 10)
        self.assertEqual(layer.in_features_embedding, 8)
        self.assertEqual(layer.in_features_exogenous, 2)

    def test_initialization_minimal(self):
        """Test initialization with only required arguments."""
        class ConcreteLayer(BaseConceptLayer):
            def forward(self, x):
                return x

        layer = ConcreteLayer(out_features=5)

        self.assertEqual(layer.out_features, 5)
        self.assertIsNone(layer.in_features_logits)
        self.assertIsNone(layer.in_features_embedding)
        self.assertIsNone(layer.in_features_exogenous)

    def test_abstract_forward(self):
        """Test that forward must be implemented."""
        # BaseConceptLayer itself should raise NotImplementedError
        layer = BaseConceptLayer(out_features=5)

        with self.assertRaises(NotImplementedError):
            layer(torch.randn(2, 5))

    def test_subclass_implementation(self):
        """Test proper subclass implementation."""
        class MyLayer(BaseConceptLayer):
            def __init__(self, out_features, in_features_logits):
                super().__init__(
                    out_features=out_features,
                    in_features_logits=in_features_logits
                )
                self.linear = nn.Linear(in_features_logits, out_features)

            def forward(self, logits):
                return torch.sigmoid(self.linear(logits))

        layer = MyLayer(out_features=5, in_features_logits=10)
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
            out_features=10,
            in_features_embedding=784
        )

        self.assertEqual(encoder.out_features, 10)
        self.assertEqual(encoder.in_features_embedding, 784)
        self.assertIsNone(encoder.in_features_logits)  # Encoders don't use logits

    def test_no_logits_input(self):
        """Test that encoders don't accept logits."""
        class ConcreteEncoder(BaseEncoder):
            def forward(self, x):
                return x

        encoder = ConcreteEncoder(
            out_features=10,
            in_features_embedding=784
        )

        # in_features_logits should always be None for encoders
        self.assertIsNone(encoder.in_features_logits)

    def test_encoder_implementation(self):
        """Test concrete encoder implementation."""
        class MyEncoder(BaseEncoder):
            def __init__(self, out_features, in_features_embedding):
                super().__init__(
                    out_features=out_features,
                    in_features_embedding=in_features_embedding
                )
                self.net = nn.Sequential(
                    nn.Linear(in_features_embedding, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_features)
                )

            def forward(self, embedding):
                return self.net(embedding)

        encoder = MyEncoder(out_features=10, in_features_embedding=784)
        x = torch.randn(4, 784)
        concepts = encoder(x)

        self.assertEqual(concepts.shape, (4, 10))

    def test_with_exogenous_features(self):
        """Test encoder with exogenous features."""
        class EncoderWithExogenous(BaseEncoder):
            def __init__(self, out_features, in_features_embedding, in_features_exogenous):
                super().__init__(
                    out_features=out_features,
                    in_features_embedding=in_features_embedding,
                    in_features_exogenous=in_features_exogenous
                )
                total_features = in_features_embedding + in_features_exogenous
                self.net = nn.Linear(total_features, out_features)

            def forward(self, embedding, exogenous):
                combined = torch.cat([embedding, exogenous], dim=-1)
                return self.net(combined)

        encoder = EncoderWithExogenous(
            out_features=5,
            in_features_embedding=10,
            in_features_exogenous=3
        )

        embedding = torch.randn(2, 10)
        exogenous = torch.randn(2, 3)
        output = encoder(embedding, exogenous)

        self.assertEqual(output.shape, (2, 5))


class TestBasePredictor(unittest.TestCase):
    """Test BasePredictor abstract class."""

    def test_initialization(self):
        """Test predictor initialization."""
        class ConcretePredictor(BasePredictor):
            def forward(self, x):
                return x

        predictor = ConcretePredictor(
            out_features=3,
            in_features_logits=10
        )

        self.assertEqual(predictor.out_features, 3)
        self.assertEqual(predictor.in_features_logits, 10)
        self.assertIsNotNone(predictor.in_activation)

    def test_default_activation(self):
        """Test default sigmoid activation."""
        class ConcretePredictor(BasePredictor):
            def forward(self, x):
                return x

        predictor = ConcretePredictor(
            out_features=3,
            in_features_logits=10
        )

        # Default should be sigmoid
        self.assertEqual(predictor.in_activation, torch.sigmoid)

    def test_custom_activation(self):
        """Test custom activation function."""
        class ConcretePredictor(BasePredictor):
            def forward(self, x):
                return x

        predictor = ConcretePredictor(
            out_features=3,
            in_features_logits=10,
            in_activation=torch.tanh
        )

        self.assertEqual(predictor.in_activation, torch.tanh)

    def test_predictor_implementation(self):
        """Test concrete predictor implementation."""
        class MyPredictor(BasePredictor):
            def __init__(self, out_features, in_features_logits):
                super().__init__(
                    out_features=out_features,
                    in_features_logits=in_features_logits,
                    in_activation=torch.sigmoid
                )
                self.linear = nn.Linear(in_features_logits, out_features)

            def forward(self, logits):
                # Apply activation to input logits
                probs = self.in_activation(logits)
                # Predict next concepts
                return self.linear(probs)

        predictor = MyPredictor(out_features=3, in_features_logits=10)
        concept_logits = torch.randn(4, 10)
        task_logits = predictor(concept_logits)

        self.assertEqual(task_logits.shape, (4, 3))

    def test_with_embedding_features(self):
        """Test predictor with embedding features."""
        class PredictorWithEmbedding(BasePredictor):
            def __init__(self, out_features, in_features_logits, in_features_embedding):
                super().__init__(
                    out_features=out_features,
                    in_features_logits=in_features_logits,
                    in_features_embedding=in_features_embedding
                )
                total_features = in_features_logits + in_features_embedding
                self.linear = nn.Linear(total_features, out_features)

            def forward(self, logits, embedding):
                probs = self.in_activation(logits)
                combined = torch.cat([probs, embedding], dim=-1)
                return self.linear(combined)

        predictor = PredictorWithEmbedding(
            out_features=3,
            in_features_logits=10,
            in_features_embedding=8
        )

        logits = torch.randn(2, 10)
        embedding = torch.randn(2, 8)
        output = predictor(logits, embedding)

        self.assertEqual(output.shape, (2, 3))

    def test_activation_application(self):
        """Test that activation is properly applied."""
        class SimplePredictor(BasePredictor):
            def __init__(self, out_features, in_features_logits):
                super().__init__(
                    out_features=out_features,
                    in_features_logits=in_features_logits,
                    in_activation=torch.sigmoid
                )
                self.linear = nn.Linear(in_features_logits, out_features)

            def forward(self, logits):
                activated = self.in_activation(logits)
                return self.linear(activated)

        predictor = SimplePredictor(out_features=3, in_features_logits=5)

        # Test with extreme logits
        logits = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])
        output = predictor(logits)

        # Output should be finite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestLayerIntegration(unittest.TestCase):
    """Test integration between different base classes."""

    def test_encoder_to_predictor_pipeline(self):
        """Test encoder followed by predictor."""
        class SimpleEncoder(BaseEncoder):
            def __init__(self, out_features, in_features_embedding):
                super().__init__(out_features, in_features_embedding)
                self.linear = nn.Linear(in_features_embedding, out_features)

            def forward(self, x):
                return self.linear(x)

        class SimplePredictor(BasePredictor):
            def __init__(self, out_features, in_features_logits):
                super().__init__(out_features, in_features_logits)
                self.linear = nn.Linear(in_features_logits, out_features)

            def forward(self, logits):
                probs = self.in_activation(logits)
                return self.linear(probs)

        # Create pipeline
        encoder = SimpleEncoder(out_features=10, in_features_embedding=784)
        predictor = SimplePredictor(out_features=5, in_features_logits=10)

        # Test pipeline
        x = torch.randn(2, 784)
        concepts = encoder(x)
        predictions = predictor(concepts)

        self.assertEqual(concepts.shape, (2, 10))
        self.assertEqual(predictions.shape, (2, 5))

    def test_gradient_flow_through_pipeline(self):
        """Test gradient flow through encoder-predictor pipeline."""
        class SimpleEncoder(BaseEncoder):
            def __init__(self, out_features, in_features_embedding):
                super().__init__(out_features, in_features_embedding)
                self.linear = nn.Linear(in_features_embedding, out_features)

            def forward(self, x):
                return self.linear(x)

        class SimplePredictor(BasePredictor):
            def __init__(self, out_features, in_features_logits):
                super().__init__(out_features, in_features_logits)
                self.linear = nn.Linear(in_features_logits, out_features)

            def forward(self, logits):
                probs = self.in_activation(logits)
                return self.linear(probs)

        encoder = SimpleEncoder(out_features=10, in_features_embedding=20)
        predictor = SimplePredictor(out_features=5, in_features_logits=10)

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

