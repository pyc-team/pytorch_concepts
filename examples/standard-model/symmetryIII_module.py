import torch
import torch.nn.functional as F
from typing import Union, Optional
from sklearn.metrics import accuracy_score

from torch_concepts.nn import CumulativeWeightsToConcept, PrototypeConceptEmbeddingToConcept


class InstanceBasedArchitecture(torch.nn.Module):
    """
    Complete instance-based architecture combining encoder and predictor.

    This is a convenience wrapper that combines CumulativeWeightsToConcept
    and PrototypeConceptEmbeddingToConcept into a single module.

    Args:
        proto_samples: Tensor of shape [max_prototypes, num_concepts, n_features] - prototype feature vectors.
        proto_scores: Tensor of shape [max_prototypes, num_concepts] - scores for sorting prototypes.
        rank_dim: Dimension of the low-rank embedding for concepts.
        temperature: Temperature for backward pass.
        temp_forward: Temperature for forward pass.
        use_straight_through: Use straight-through estimator.
        learnable_prototypes: Whether prototypes should be learnable parameters.

    Example:
        >>> proto_samples = torch.randn(10, 100, 50)
        >>> proto_scores = torch.randn(10, 100)
        >>> model = InstanceBasedArchitecture(proto_samples, proto_scores, rank_dim=32)
        >>> x = torch.randn(32, 50)
        >>> output = model(x)  # [32, 100]
    """
    def __init__(
        self,
        proto_samples: torch.Tensor,
        proto_scores: torch.Tensor,
        rank_dim: int = 32,
        temperature: float = 1.0,
        temp_forward: Optional[float] = None,
        use_straight_through: bool = True,
        learnable_prototypes: bool = False
    ):
        super().__init__()

        max_prototypes, num_concepts, n_features = proto_samples.shape

        self.encoder = CumulativeWeightsToConcept(
            out_concepts=num_concepts,
            max_prototypes=max_prototypes,
            rank_dim=rank_dim
        )

        self.predictor = PrototypeConceptEmbeddingToConcept(
            proto_samples=proto_samples,
            proto_scores=proto_scores,
            out_concepts=num_concepts,
            learnable_prototypes=learnable_prototypes,
            temperature=temperature,
            temp_forward=temp_forward,
            use_straight_through=use_straight_through
        )

        self.num_concepts = num_concepts
        self.max_prototypes = max_prototypes
        self.n_features = n_features

        # Expose prototypes for compatibility
        self.prototypes = self.predictor.prototypes
        self.embedding = self.encoder.embedding
        self.projection = self.encoder.projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape [batch, n_features] - input features.

        Returns:
            torch.Tensor: Tensor of shape [batch, num_concepts] - concept predictions.
        """
        # Generate concept weights
        concepts = self.encoder()  # [1, num_concepts, max_prototypes]

        # Aggregate with embeddings
        output = self.predictor(concepts, x)  # [batch, num_concepts]

        return output


def main():
    import time
    torch.manual_seed(42)

    print("="*70)
    print(f"Train")
    print("="*70)
    print()
    n_features = 50
    batch_size = 512
    rank_dim = 10
    max_prototypes = 10
    num_concepts = 1000
    proto_samples = torch.randn(max_prototypes, num_concepts, n_features)
    proto_scores = torch.randn(max_prototypes, num_concepts)
    x_train = torch.randn(batch_size, n_features)
    y_train = ((x_train[:, 2] + x_train[:, 3])>0).float().unsqueeze(1)

    cum_mlp = InstanceBasedArchitecture(proto_samples, proto_scores, rank_dim=rank_dim)
    predictor = torch.nn.Linear(num_concepts, 1)
    model = torch.nn.Sequential(cum_mlp, predictor)

    # Forward + backward
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x_train)
        loss = F.binary_cross_entropy_with_logits(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            task_accuracy = accuracy_score(y_train, output.detach().cpu().numpy()>0)
            print(f"Epoch: {epoch}, loss: {loss}, acc.: {task_accuracy}")

    print()
    print("="*70)
    print("InstanceBasedArchitecture Performance Benchmark")
    print("="*70)
    print()

    # Test configurations
    n_features = 10
    batch_size = 32
    rank_dim = 32
    prototype_configs = [10, 100]
    concept_configs = [10, 100, 1000, 10000]

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dimension: {n_features}")
    print(f"  Rank dimension: {rank_dim}")
    print()

    # Results table
    print(f"{'Prototypes':<12} {'Concepts':<10} {'Forward (ms)':<15} {'Memory (MB)':<15} {'Throughput':<15}")
    print("-" * 70)

    for max_prototypes in prototype_configs:
        for num_concepts in concept_configs:
            # Create prototype samples and scores
            proto_samples = torch.randn(max_prototypes, num_concepts, n_features)
            proto_scores = torch.randn(max_prototypes, num_concepts)

            # Create model
            model = InstanceBasedArchitecture(proto_samples, proto_scores, rank_dim=rank_dim)
            model.eval()

            # Create test input
            x = torch.randn(batch_size, n_features)

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)

            # Benchmark forward pass
            num_runs = 10

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(x)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            # Calculate metrics
            avg_time_ms = (end_time - start_time) / num_runs * 1000

            # Memory estimate (rough)
            param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # MB
            proto_memory = model.prototypes.numel() * 4 / (1024 * 1024)  # MB
            total_memory = param_memory + proto_memory

            # Throughput (samples per second)
            throughput = batch_size * num_runs / (end_time - start_time)

            print(f"{max_prototypes:<12} {num_concepts:<10} {avg_time_ms:<15.3f} "
                  f"{total_memory:<15.2f} {throughput:<15.0f}")

    print()
    print("="*70)
    print("Detailed Test: 10 prototypes, 1k concepts")
    print("="*70)

    max_prototypes = 10
    num_concepts = 1000

    proto_samples = torch.randn(max_prototypes, num_concepts, n_features)
    proto_scores = torch.randn(max_prototypes, num_concepts)

    model = InstanceBasedArchitecture(proto_samples, proto_scores, rank_dim=rank_dim)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Embedding: {model.embedding.weight.numel():,}")
    print(f"  - Projection weight: {model.projection.weight.numel():,}")
    print(f"  - Projection bias: {model.projection.bias.numel():,}")
    print(f"  Prototype storage: {model.prototypes.numel():,} elements")
    print(f"  Prototype memory: {model.prototypes.numel() * 4 / (1024**2):.2f} MB")
    print()

    # Test with different batch sizes
    print("Batch Size Scaling:")
    print(f"{'Batch Size':<12} {'Forward (ms)':<15} {'Per Sample (ms)':<15}")
    print("-" * 45)

    for bs in [1, 8, 32, 64]:
        x_test = torch.randn(bs, n_features)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x_test)

        # Benchmark
        num_runs = 5
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x_test)
        end_time = time.time()

        avg_time_ms = (end_time - start_time) / num_runs * 1000
        per_sample_ms = avg_time_ms / bs

        print(f"{bs:<12} {avg_time_ms:<15.3f} {per_sample_ms:<15.4f}")

    print()
    print("="*70)
    print("Gradient Computation Test")
    print("="*70)

    # Test gradient computation overhead
    x_train = torch.randn(batch_size, n_features)
    y_train = torch.randn(batch_size, num_concepts)

    # Forward only
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            output = model(x_train)
    forward_time = (time.time() - start_time) / 10 * 1000

    # Forward + backward
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x_train)
        loss = F.mse_loss(output, y_train)
        loss.backward()
        optimizer.step()
    train_time = (time.time() - start_time) / 10 * 1000

    print(f"\nWith {num_concepts:,} concepts, {max_prototypes} prototypes:")
    print(f"  Forward pass:          {forward_time:.2f} ms")
    print(f"  Forward + Backward:    {train_time:.2f} ms")
    print(f"  Backward overhead:     {train_time - forward_time:.2f} ms ({(train_time/forward_time - 1)*100:.1f}% increase)")

    print()
    print("="*70)
    print("Monotonicity Verification (1k concepts sample)")
    print("="*70)

    # Check monotonicity for a few concepts
    with torch.no_grad():
        for c in range(min(3, num_concepts)):
            proto_inputs = model.prototypes[c]  # [max_prototypes, n_features]
            proto_outputs = model(proto_inputs)[:, c]  # Predictions for concept c

            diffs = torch.diff(proto_outputs)
            is_monotonic = torch.all(diffs >= -1e-6).item()

            print(f"\nConcept {c}:")
            print(f"  Monotonic: {is_monotonic}")
            print(f"  Min diff: {diffs.min().item():.6f}")
            print(f"  Max diff: {diffs.max().item():.6f}")
            print(f"  Mean diff: {diffs.mean().item():.6f}")


if __name__ == "__main__":
    main()
