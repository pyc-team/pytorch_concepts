"""
DataModule integration test for the new datasets.

Instantiates each DataModule with small parameters, calls setup(),
verifies train/val/test splits, and checks that __getitem__ and
DataLoader batches produce the expected structure and shapes.

Usage:
    python test_new_datamodules.py
"""

import os
import tempfile
import shutil
import torch
from torch.utils.data import DataLoader


def check_sample(sample, dataset_name):
    """Verify a single sample has the expected structure."""
    assert 'inputs' in sample, f"[{dataset_name}] Missing 'inputs' key"
    assert 'concepts' in sample, f"[{dataset_name}] Missing 'concepts' key"
    assert 'x' in sample['inputs'], f"[{dataset_name}] Missing 'x' in inputs"
    assert 'c' in sample['concepts'], f"[{dataset_name}] Missing 'c' in concepts"
    assert isinstance(sample['inputs']['x'], torch.Tensor), f"[{dataset_name}] x is not a Tensor"
    assert isinstance(sample['concepts']['c'], torch.Tensor), f"[{dataset_name}] c is not a Tensor"


def check_batch(batch, dataset_name, batch_size):
    """Verify a batch has the expected structure."""
    assert batch['inputs']['x'].ndim >= 2, f"[{dataset_name}] batch x should be at least 2D"
    assert batch['concepts']['c'].ndim == 2, f"[{dataset_name}] batch c should be 2D"
    assert batch['inputs']['x'].shape[0] <= batch_size, f"[{dataset_name}] batch x has wrong batch dim"
    assert batch['concepts']['c'].shape[0] <= batch_size, f"[{dataset_name}] batch c has wrong batch dim"


# def test_pendulum_datamodule(temp_dir):
#     """Test PendulumDataModule: setup, splits, getitem, dataloader."""
#     from torch_concepts.data.datamodules import PendulumDataModule

#     print("\n" + "=" * 60)
#     print("TESTING: PendulumDataModule")
#     print("=" * 60)

#     dm = PendulumDataModule(
#         root=os.path.join(temp_dir, 'pendulum'),
#         n_theta=5,
#         n_phi=10,
#         seed=42,
#         val_size=0.1,
#         test_size=0.2,
#         batch_size=4,
#         precompute_embs=False,
#     )

#     dm.setup(verbose=False)

#     print(f"  Dataset len:    {len(dm.dataset)}")
#     print(f"  Concept names:  {dm.concept_names}")
#     print(f"  n_concepts:     {dm.n_concepts}")
#     print(f"  Train size:     {dm.train_len}")
#     print(f"  Val size:       {dm.val_len}")
#     print(f"  Test size:      {dm.test_len}")

#     assert dm.train_len is not None and dm.train_len > 0, "Train set is empty"
#     assert dm.val_len is not None and dm.val_len > 0, "Val set is empty"
#     assert dm.test_len is not None and dm.test_len > 0, "Test set is empty"
#     assert dm.train_len + dm.val_len + dm.test_len == len(dm.dataset), "Split sizes don't sum to dataset length"

#     # Test getitem on each split
#     for split_name, subset in [('train', dm.trainset), ('val', dm.valset), ('test', dm.testset)]:
#         sample = subset[0]
#         check_sample(sample, f"Pendulum/{split_name}")
#         print(f"  {split_name} sample x: {sample['inputs']['x'].shape}, c: {sample['concepts']['c'].shape}")

#     # Test dataloaders
#     for split_name, loader_fn in [('train', dm.train_dataloader), ('val', dm.val_dataloader), ('test', dm.test_dataloader)]:
#         loader = loader_fn()
#         batch = next(iter(loader))
#         check_batch(batch, f"Pendulum/{split_name}", dm.batch_size)
#         print(f"  {split_name} batch x: {batch['inputs']['x'].shape}, c: {batch['concepts']['c'].shape}")

#     print("  PASSED")


def test_mnist_arithmetic_datamodule(temp_dir):
    """Test MNISTArithmeticDataModule: setup, splits, getitem, dataloader."""
    from torch_concepts.data.datamodules import MNISTArithmeticDataModule

    print("\n" + "=" * 60)
    print("TESTING: MNISTArithmeticDataModule")
    print("=" * 60)

    dm = MNISTArithmeticDataModule(
        root=os.path.join(temp_dir, 'mnist_arith'),
        num_train_samples=30,
        num_test_samples=10,
        val_size=0.2,
        img_size=64,
        seed=42,
        batch_size=4,
        precompute_embs=False,
    )

    dm.setup(verbose=False)

    print(f"  Dataset len:    {len(dm.dataset)}")
    print(f"  Concept names:  {dm.concept_names}")
    print(f"  n_concepts:     {dm.n_concepts}")
    print(f"  Train size:     {dm.train_len}")
    print(f"  Val size:       {dm.val_len}")
    print(f"  Test size:      {dm.test_len}")

    assert dm.train_len is not None and dm.train_len > 0, "Train set is empty"
    assert dm.val_len is not None and dm.val_len > 0, "Val set is empty"
    assert dm.test_len is not None and dm.test_len > 0, "Test set is empty"
    assert dm.train_len + dm.val_len + dm.test_len == len(dm.dataset), "Split sizes don't sum to dataset length"

    # Verify NativeSplitter preserved MNIST train/test boundary
    n_train_pool = 30
    n_test = 10
    n_val = int(n_train_pool * 0.2)
    assert dm.test_len == n_test, f"Test size should be {n_test}, got {dm.test_len}"
    assert dm.val_len == n_val, f"Val size should be {n_val}, got {dm.val_len}"
    assert dm.train_len == n_train_pool - n_val, f"Train size should be {n_train_pool - n_val}, got {dm.train_len}"
    print(f"  NativeSplitter sizes verified: train={dm.train_len}, val={dm.val_len}, test={dm.test_len}")

    # Test getitem on each split
    for split_name, subset in [('train', dm.trainset), ('val', dm.valset), ('test', dm.testset)]:
        sample = subset[0]
        check_sample(sample, f"MNISTArithmetic/{split_name}")
        print(f"  {split_name} sample x: {sample['inputs']['x'].shape}, c: {sample['concepts']['c'].shape}")

    # Test dataloaders
    for split_name, loader_fn in [('train', dm.train_dataloader), ('val', dm.val_dataloader), ('test', dm.test_dataloader)]:
        loader = loader_fn()
        batch = next(iter(loader))
        check_batch(batch, f"MNISTArithmetic/{split_name}", dm.batch_size)
        print(f"  {split_name} batch x: {batch['inputs']['x'].shape}, c: {batch['concepts']['c'].shape}")

    print("  PASSED")


# def test_dsprites_regression_datamodule(temp_dir):
#     """Test DSpritesRegressionDataModule: setup, splits, getitem, dataloader."""
#     from torch_concepts.data.datamodules import DSpritesRegressionDataModule

#     print("\n" + "=" * 60)
#     print("TESTING: DSpritesRegressionDataModule")
#     print("=" * 60)

#     dm = DSpritesRegressionDataModule(
#         root=os.path.join(temp_dir, 'dsprites_reg'),
#         concepts=['value_x_position', 'value_y_position'],
#         formulas={
#             'square': 'value_x_position + value_y_position',
#             'circle': 'value_x_position * value_y_position',
#             'heart': 'value_x_position - value_y_position',
#         },
#         num_samples=50,
#         seed=42,
#         val_size=0.1,
#         test_size=0.2,
#         batch_size=4,
#         precompute_embs=False,
#     )

#     dm.setup(verbose=False)

#     print(f"  Dataset len:    {len(dm.dataset)}")
#     print(f"  Concept names:  {dm.concept_names}")
#     print(f"  n_concepts:     {dm.n_concepts}")
#     print(f"  Train size:     {dm.train_len}")
#     print(f"  Val size:       {dm.val_len}")
#     print(f"  Test size:      {dm.test_len}")

#     assert dm.train_len is not None and dm.train_len > 0, "Train set is empty"
#     assert dm.val_len is not None and dm.val_len > 0, "Val set is empty"
#     assert dm.test_len is not None and dm.test_len > 0, "Test set is empty"
#     assert dm.train_len + dm.val_len + dm.test_len == len(dm.dataset), "Split sizes don't sum to dataset length"

#     # Test getitem on each split
#     for split_name, subset in [('train', dm.trainset), ('val', dm.valset), ('test', dm.testset)]:
#         sample = subset[0]
#         check_sample(sample, f"DSpritesRegression/{split_name}")
#         print(f"  {split_name} sample x: {sample['inputs']['x'].shape}, c: {sample['concepts']['c'].shape}")

#     # Test dataloaders
#     for split_name, loader_fn in [('train', dm.train_dataloader), ('val', dm.val_dataloader), ('test', dm.test_dataloader)]:
#         loader = loader_fn()
#         batch = next(iter(loader))
#         check_batch(batch, f"DSpritesRegression/{split_name}", dm.batch_size)
#         print(f"  {split_name} batch x: {batch['inputs']['x'].shape}, c: {batch['concepts']['c'].shape}")

#     print("  PASSED")


def main():
    temp_dir = os.path.join(os.getcwd(), 'temp_datamodule_test')
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temp data directory: {temp_dir}")

    passed = 0
    failed = 0
    errors = []

    for test_fn in [test_mnist_arithmetic_datamodule]: #[test_pendulum_datamodule, test_mnist_arithmetic_datamodule, test_dsprites_regression_datamodule]:
        try:
            test_fn(temp_dir)
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

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
