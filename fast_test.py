"""
Visual testing script for the new datasets.

Generates each dataset with a small number of samples, prints shapes and
metadata, plots sample images with their concept/task values, and verifies
that __getitem__ and DataLoader work correctly.

Usage:
    python test_new_datasets_visual.py [--output_dir OUTPUT_DIR]
"""

import os
import argparse
import tempfile
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def test_pendulum(output_dir, temp_dir):
    """Test PendulumDataset: generate, load, plot batch."""
    from torch_concepts.data import PendulumDataset

    print("\n" + "=" * 60)
    print("TESTING: PendulumDataset")
    print("=" * 60)

    dataset = PendulumDataset(
        root=os.path.join(temp_dir, 'pendulum'),
        n_theta=5,
        n_phi=10,
        seed=42,
    )

    print(f"  Dataset:        {dataset}")
    print(f"  Num samples:    {len(dataset)}")
    print(f"  Concept names:  {dataset.concept_names}")
    print(f"  Num concepts:   {dataset.n_concepts}")
    print(f"  Has concepts:   {dataset.has_concepts}")

    # Test __getitem__
    sample = dataset[0]
    x = sample['inputs']['x']
    c = sample['concepts']['c']
    print(f"  Sample x shape: {x.shape}")
    print(f"  Sample c shape: {c.shape}")
    print(f"  Sample c values: {c.tolist()}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    print(f"  Batch x shape:  {batch['inputs']['x'].shape}")
    print(f"  Batch c shape:  {batch['concepts']['c'].shape}")

    # Plot all images in the batch
    batch_x = batch['inputs']['x']
    batch_c = batch['concepts']['c']
    n = batch_x.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    for i, ax in enumerate(axes):
        img = batch_x[i].permute(1, 2, 0).numpy()
        concepts = batch_c[i]
        ax.imshow(img)
        ax.set_title(f"theta={concepts[0]:.2f}\nphi={concepts[1]:.2f}\nx={concepts[2]:.2f}", fontsize=8)
        ax.axis('off')
    plt.suptitle("PendulumDataset Batch")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pendulum_samples.png'), dpi=150)
    plt.close()
    print(f"  Saved plot to {output_dir}/pendulum_samples.png")
    print("  PASSED")


def test_mnist_arithmetic(output_dir, temp_dir):
    """Test MNISTArithmeticDataset: generate, load, plot batch."""
    from torch_concepts.data import MNISTArithmeticDataset

    print("\n" + "=" * 60)
    print("TESTING: MNISTArithmeticDataset")
    print("=" * 60)

    dataset = MNISTArithmeticDataset(
        root=os.path.join(temp_dir, 'mnist_arithmetic'),
        num_train_samples=50,
        num_test_samples=20,
        img_size=64,
        seed=42,
    )

    print(f"  Dataset:        {dataset}")
    print(f"  Num samples:    {len(dataset)}")
    print(f"  Concept names:  {dataset.concept_names}")
    print(f"  Num concepts:   {dataset.n_concepts}")
    print(f"  Has concepts:   {dataset.has_concepts}")

    # Test __getitem__
    sample = dataset[0]
    x = sample['inputs']['x']
    c = sample['concepts']['c']
    print(f"  Sample x shape: {x.shape}")
    print(f"  Sample c shape: {c.shape}")
    print(f"  Sample c values: {c.tolist()}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    print(f"  Batch x shape:  {batch['inputs']['x'].shape}")
    print(f"  Batch c shape:  {batch['concepts']['c'].shape}")

    # Plot all images in the batch
    batch_x = batch['inputs']['x']
    batch_c = batch['concepts']['c']
    n = batch_x.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    for i, ax in enumerate(axes):
        img = batch_x[i].permute(1, 2, 0).numpy()
        concepts = batch_c[i]
        ax.imshow(img)
        ax.set_title(
            f"d1={concepts[0]:.0f}, d2={concepts[1]:.0f}\nresult={concepts[2]:.2f}",
            fontsize=8,
        )
        ax.axis('off')
    plt.suptitle("MNISTArithmeticDataset Batch")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mnist_arithmetic_samples.png'), dpi=150)
    plt.close()
    print(f"  Saved plot to {output_dir}/mnist_arithmetic_samples.png")
    print("  PASSED")


def test_dsprites_regression(output_dir, temp_dir):
    """Test DSpritesRegressionDataset: load, compute targets, plot batch."""
    from torch_concepts.data import DSpritesRegressionDataset

    print("\n" + "=" * 60)
    print("TESTING: DSpritesRegressionDataset")
    print("=" * 60)

    formulas = {
        'square': 'x_position + y_position',
        'circle': 'x_position * y_position',
        'heart': 'x_position - y_position',
    }
    dataset = DSpritesRegressionDataset(
        root=os.path.join(temp_dir, 'dsprites_regression'),
        concept_subset=['x_position', 'y_position'],
        formulas=formulas,
        seed=42,
    )

    print(f"  Dataset:        {dataset}")
    print(f"  Num samples:    {len(dataset)}")
    print(f"  Concept names:  {dataset.concept_names}")
    print(f"  Num concepts:   {dataset.n_concepts}")
    print(f"  Has concepts:   {dataset.has_concepts}")

    # Test __getitem__
    sample = dataset[0]
    x = sample['inputs']['x']
    c = sample['concepts']['c']
    print(f"  Sample x shape: {x.shape}")
    print(f"  Sample c shape: {c.shape}")
    print(f"  Sample c values: {c.tolist()}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    print(f"  Batch x shape:  {batch['inputs']['x'].shape}")
    print(f"  Batch c shape:  {batch['concepts']['c'].shape}")

    # Plot all images in the batch
    batch_x = batch['inputs']['x']
    batch_c = batch['concepts']['c']
    labels = dataset.concept_names
    n = batch_x.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    for i, ax in enumerate(axes):
        img = batch_x[i].squeeze(0).numpy()
        concepts = batch_c[i]
        ax.imshow(img, cmap='gray')
        title = "\n".join(f"{labels[j]}={concepts[j]:.3f}" for j in range(len(labels)))
        ax.set_title(title, fontsize=6)
        ax.axis('off')
    plt.suptitle("DSpritesRegressionDataset Batch")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dsprites_regression_samples.png'), dpi=150)
    plt.close()
    print(f"  Saved plot to {output_dir}/dsprites_regression_samples.png")
    print("  PASSED")


def main():
    parser = argparse.ArgumentParser(description="Visual test for new datasets")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_dataset_outputs',
        help='Directory to save sample plots',
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = os.path.join(os.getcwd(), 'temp_dataset')
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temp data directory: {temp_dir}")
    print(f"Output directory: {output_dir}")

    passed = 0
    failed = 0
    errors = []

    for test_fn in [test_pendulum, test_mnist_arithmetic, test_dsprites_regression]:
        try:
            test_fn(output_dir, temp_dir)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        for name, err in errors:
            print(f"  FAILED: {name} - {err}")
    print("=" * 60)

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
