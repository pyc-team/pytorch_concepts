from torch_concepts.data import ToyDataset, CompletenessDataset


def main():
    """Test loading toy datasets with the current API."""
    # Test XOR dataset
    data = ToyDataset('xor', n_gen=1000, seed=42)
    labels = data.annotations.get_axis_annotation(1).labels
    print(f"XOR: input={data.input_data.shape}, concepts={data.concepts.shape}, labels={labels}")

    # Test Trigonometry dataset
    data = ToyDataset('trigonometry', n_gen=1000, seed=42)
    labels = data.annotations.get_axis_annotation(1).labels
    print(f"Trigonometry: input={data.input_data.shape}, concepts={data.concepts.shape}, labels={labels}")

    # Test Dot dataset
    data = ToyDataset('dot', n_gen=1000, seed=42)
    labels = data.annotations.get_axis_annotation(1).labels
    print(f"Dot: input={data.input_data.shape}, concepts={data.concepts.shape}, labels={labels}")

    # Test Checkmark dataset (has graph)
    data = ToyDataset('checkmark', n_gen=1000, seed=42)
    labels = data.annotations.get_axis_annotation(1).labels
    print(f"Checkmark: input={data.input_data.shape}, concepts={data.concepts.shape}, labels={labels}, graph={data.graph}")

    # Test CompletenessDataset
    data = CompletenessDataset(name='test1', n_gen=1000, n_concepts=3, n_hidden_concepts=0, n_tasks=2, seed=42)
    labels = data.annotations.get_axis_annotation(1).labels
    print(f"Completeness (no hidden): input={data.input_data.shape}, concepts={data.concepts.shape}, labels={labels}")

    # Test CompletenessDataset with hidden concepts
    data = CompletenessDataset(name='test2', n_gen=1000, n_concepts=3, n_hidden_concepts=1, n_tasks=2, seed=42)
    labels = data.annotations.get_axis_annotation(1).labels
    print(f"Completeness (1 hidden): input={data.input_data.shape}, concepts={data.concepts.shape}, labels={labels}")

    print("\nAll toy dataset examples passed!")


if __name__ == "__main__":
    main()
