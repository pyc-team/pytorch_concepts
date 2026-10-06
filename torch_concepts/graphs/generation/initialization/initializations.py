
def entropy_initialization(data: Any) -> Callable[[Any], None]:
    """Return an entropy-based initialization bound to training concept data."""
    @torch.no_grad()
    def initialize(generator: Any) -> None:
        values = data.concepts if hasattr(data, "concepts") else data
        values = values.tensor if hasattr(values, "tensor") else values
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError(
                "Entropy initialization data must be a 2D tensor or expose "
                "a 2D `concepts` tensor."
            )
        if values.shape[1] != generator.n_concepts:
            raise ValueError(
                "Entropy initialization data must have one column per graph node."
            )
        values = values.detach().cpu().numpy()
        adjacency = np.zeros((generator.n_concepts, generator.n_concepts))
        entropies = [
            _entropy(values[:, index:index + 1])
            for index in range(generator.n_concepts)
        ]
        for source in range(generator.n_concepts):
            for target in range(generator.n_concepts):
                if source != target:
                    joint = _entropy(values[:, [source, target]])
                    adjacency[source, target] = 1 - (joint - entropies[target])
        # Upstream computes entropy in NumPy float64, then normalizes in float32.
        adjacency = torch.tensor(adjacency, dtype=torch.float32)
        if generator.no_out_task and generator.task_indices:
            adjacency[generator.task_indices, :] = 0
        mean = adjacency.mean()
        if mean != 0:
            adjacency = adjacency / mean
        adjacency.clamp_(0, 0.99)
        adjacency = adjacency.to(generator.fc1.weight)
        generator.fc1.weight.copy_(adjacency)

    return initialize


def fixed_dagma_initialization(adjacency: Any) -> Callable[[Any], None]:
    """Return an initializer that seeds DAGMA-CGM from a fixed adjacency.

    Use this with :class:`GraphGeneratorLearnable` when the learnable DAGMA-CGM
    parameters should start from an externally computed graph. It is not a
    fixed graph generator: training may still change the graph afterwards.
    """
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32).clone()

    @torch.no_grad()
    def initialize(generator: Any) -> None:
        if adjacency.shape != generator.fc1.weight.shape:
            raise ValueError(
                "Fixed DAGMA initialization adjacency must match "
                f"fc1 weight shape {tuple(generator.fc1.weight.shape)}."
            )
        generator.fc1.weight.copy_(adjacency.to(generator.fc1.weight))
        generator.fc1.weight.requires_grad_(False)
        generator.edge_matrix.zero_()

    return initialize


@torch.no_grad()
def random_initialization(generator: Any) -> None:
    """Reset DAGMA-CGM weights and clear explicit edge logits."""
    generator.fc1.reset_parameters()
    generator.edge_matrix.zero_()


def _entropy(values: np.ndarray) -> np.float64:
    """Compute empirical entropy for one or more discrete columns."""
    _, counts = np.unique(values, axis=0, return_counts=True)
    probabilities = counts / len(values)
    return np.sum(-probabilities * np.log2(probabilities))

