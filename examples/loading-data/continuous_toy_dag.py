"""
Example: Loading ToyFunctionDAGDataset (Continuous Variables)

This example demonstrates how to use ToyFunctionDAGDataset with continuous
variables and mathematical functions (using sympy expressions) to generate
synthetic data from a Directed Acyclic Graph (DAG).
"""

from torch_concepts.data.datasets import ToyFunctionDAGDataset
from torch_concepts.data.datamodules import ToyFunctionDAGDataModule
import numpy as np


def main():
    """Test ToyFunctionDAGDataset with continuous variables."""
    print("=" * 70)
    print("ToyFunctionDAG Dataset Example: Mathematical Functions (Continuous)")
    print("=" * 70)
    print("\nDAG Structure:")
    print("  x1 (root) → x2 = x1²")
    print("  x1 (root) → x3 = sin(x1)")
    print("  x2, x3 → x4 = x2 + 0.5*x3")
    print()
    
    # Define the DAG structure
    variables = ['x1', 'x2', 'x3', 'x4']
    dag = [
        ('x1', 'x2'),  # x1 → x2
        ('x1', 'x3'),  # x1 → x3
        ('x2', 'x4'),  # x2 → x4
        ('x3', 'x4'),  # x3 → x4
    ]
    
    # Define functions using sympy expressions
    node_functions = {
        'x2': 'x1**2',           # Quadratic relationship
        'x3': 'sin(x1)',         # Trigonometric relationship
        'x4': 'x2 + 0.5*x3',     # Linear combination
    }
    
    # Test dataset loading
    print("1. Creating Dataset...")
    dataset = ToyFunctionDAGDataset(
        variables=variables,
        dag=dag,
        node_functions=node_functions,
        variable_type='continuous',
        source_mean=0.0,
        source_std=1.0,
        gamma=0.1,  # Additive Gaussian noise std
        theta=0.05,  # Embedding-level noise std
        root="./data/continuous_toy_dag",
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
    print(f"Variable type: continuous")
    
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
    datamodule = ToyFunctionDAGDataModule(
        variables=variables,
        dag=dag,
        node_functions=node_functions,
        variable_type='continuous',
        source_mean=0.0,
        source_std=1.0,
        gamma=0.1,
        theta=0.05,
        root="./data/continuous_toy_dag",
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
    
    # Verify functional relationships
    print("\n5. Verifying Functional Relationships...")
    concepts = dataset.concepts.numpy()
    x1 = concepts[:, 0]
    x2 = concepts[:, 1]
    x3 = concepts[:, 2]
    x4 = concepts[:, 3]
    
    # Expected values (without noise)
    x2_expected = x1**2
    x3_expected = np.sin(x1)
    x4_expected = x2 + 0.5 * x3
    
    # Compute correlations (should be high despite noise)
    corr_x2 = np.corrcoef(x2, x2_expected)[0, 1]
    corr_x3 = np.corrcoef(x3, x3_expected)[0, 1]
    corr_x4 = np.corrcoef(x4, x4_expected)[0, 1]
    
    print(f"Correlation: x2 = x1²        → r = {corr_x2:.3f}")
    print(f"Correlation: x3 = sin(x1)    → r = {corr_x3:.3f}")
    print(f"Correlation: x4 = x2+0.5*x3  → r = {corr_x4:.3f}")
    
    print("\n6. Variable Statistics...")
    print(f"x1: mean={x1.mean():.3f}, std={x1.std():.3f}")
    print(f"x2: mean={x2.mean():.3f}, std={x2.std():.3f}")
    print(f"x3: mean={x3.mean():.3f}, std={x3.std():.3f}")
    print(f"x4: mean={x4.mean():.3f}, std={x4.std():.3f}")
    
    print("\n" + "=" * 70)
    print("Continuous ToyFunctionDAG Example Complete! ✓")
    print("=" * 70)
    print("\nSupported Mathematical Functions:")
    print("  - Basic: +, -, *, /, **")
    print("  - Trigonometric: sin, cos, tan, asin, acos, atan")
    print("  - Exponential: exp, log, log10, sqrt")
    print("  - Other: abs, sign")
    print("  - Logical (binary): & (AND), | (OR), ^ (XOR), ~ (NOT)")


if __name__ == "__main__":
    main()
