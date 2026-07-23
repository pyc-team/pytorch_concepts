"""
Example: Loading ToyDAGDataset (Categorical/Discrete Variables)

This example demonstrates how to use ToyDAGDataset with discrete variables
and conditional probability tables (CPTs) to generate synthetic data from
a Directed Acyclic Graph (DAG).
"""

from torch_concepts.data import ToyDAGDataset, ToyDAGDataModule


def main():
    """Test ToyDAGDataset with categorical variables."""
    print("=" * 70)
    print("ToyDAG Dataset Example: Car Starting Scenario (Categorical)")
    print("=" * 70)
    print("\nDAG Structure:")
    print("  engine (binary) → car_start (binary)")
    print("  wheels (binary) → car_start (binary)")
    print()
    
    # Define the DAG structure
    variables = ['engine', 'wheels', 'car_start']
    dag = [
        ('engine', 'car_start'),
        ('wheels', 'car_start'),
    ]
    
    # Define conditional probabilities using explicit parent states
    conditional_probs = {
        'engine': [0.9, 0.1],  # P(engine=0)=0.9, P(engine=1)=0.1
        'wheels': [0.95, 0.05],  # P(wheels=0)=0.95, P(wheels=1)=0.05
        'car_start': {
            # Format: "parent1=value1,parent2=value2": [P(child=0), P(child=1)]
            "engine=0,wheels=0": [0.95, 0.05],  # Both fail → car unlikely starts
            "engine=0,wheels=1": [0.90, 0.10],  # Only engine fails
            "engine=1,wheels=0": [0.85, 0.15],  # Only wheels fail
            "engine=1,wheels=1": [0.10, 0.90],  # Both work → car likely starts
        }
    }
    
    # Test dataset loading
    print("1. Creating Dataset...")
    dataset = ToyDAGDataset(
        variables=variables,
        dag=dag,
        conditional_probs=conditional_probs,
        root="./data/categorical_toy_dag",
        seed=42,
        n_gen=2000,
        autoencoder_kwargs={
            'latent_dim': 16,
            'epochs': 300,
            'batch_size': 256,
        }
    )
    
    print(f"Dataset: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of features: {dataset.n_features}")
    print(f"Number of concepts: {dataset.n_concepts}")
    print(f"Concept names: {dataset.concept_names}")
    print(f"Cardinalities: {dataset.cardinalities}")
    
    # Test sample access
    print("\n2. Testing Sample Access...")
    sample = dataset[0]
    print(f"Sample structure: {sample.keys()}")
    print(f"Input shape: {sample['inputs']['x'].shape}")
    print(f"Concepts shape: {sample['concepts']['c'].shape}")
    print(f"Concepts values: {sample['concepts']['c']}")
    print(f"Graph (adjacency matrix):\n{sample['graph']}")
    
    # Test datamodule
    print("\n3. Testing DataModule...")
    datamodule = ToyDAGDataModule(
        variables=variables,
        dag=dag,
        conditional_probs=conditional_probs,
        root="./data/categorical_toy_dag",
        seed=42,
        batch_size=128,
        val_size=0.15,
        test_size=0.15,
        n_gen=2000,
        autoencoder_kwargs={
            'latent_dim': 16,
            'epochs': 300,
            'batch_size': 256,
        }
    )
    
    datamodule.setup('fit')
    print(f"\nDataModule: {datamodule}")
    print(f"Train size: {datamodule.train_len}")
    print(f"Val size: {datamodule.val_len}")
    print(f"Test size: {datamodule.test_len}")
    
    # Test dataloader
    print("\n4. Testing Dataloader...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch structure: {batch.keys()}")
    print(f"Batch input shape: {batch['inputs']['x'].shape}")
    print(f"Batch concepts shape: {batch['concepts']['c'].shape}")
    print(f"Batch graph shape: {batch['graph'].shape}")
    
    # Show some statistics
    print("\n5. Concept Statistics...")
    concepts = dataset.concepts.numpy()
    engine = concepts[:, 0]
    wheels = concepts[:, 1]
    car_start = concepts[:, 2]
    
    print(f"Engine failure rate: {engine.mean():.3f} (expected ~0.10)")
    print(f"Wheels failure rate: {wheels.mean():.3f} (expected ~0.05)")
    print(f"Car start success rate: {car_start.mean():.3f}")
    
    print("\n" + "=" * 70)
    print("Categorical ToyDAG Example Complete! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
