"""
DataModule integration test for AWA2 and CEBaB datasets.

Creates mock AWA2 data synthetically (no download required) and downloads
CEBaB from the HuggingFace Hub (skipped automatically if the required
packages are not installed).

Usage
-----
    python datamodule_test.py
"""

import os
import shutil
import traceback
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers for building synthetic AWA2 raw data
# ---------------------------------------------------------------------------

def create_mock_awa2_data(root: str, n_images_per_class: int = 2):
    """Populate *root* with minimal synthetic AwA2 raw files."""
    from PIL import Image
    from torch_concepts.data.datasets.awa2 import CLASS_NAMES

    rng = np.random.default_rng(42)
    img_dir = os.path.join(root, "JPEGImages")

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(img_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n_images_per_class):
            arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img.save(os.path.join(class_dir, f"{i:04d}.jpg"))

    # 50 classes × 85 binary attributes
    pred_mat = rng.integers(0, 2, (50, 85), dtype=int)
    with open(os.path.join(root, "predicate-matrix-binary.txt"), "w") as fh:
        for row in pred_mat:
            fh.write(" ".join(str(v) for v in row) + "\n")

    # classes.txt: 1-based index + tab + class_name
    with open(os.path.join(root, "classes.txt"), "w") as fh:
        for i, name in enumerate(CLASS_NAMES):
            fh.write(f"{i + 1}\t{name}\n")


# ---------------------------------------------------------------------------
# Per-sample / per-batch validators
# ---------------------------------------------------------------------------

def check_sample_image(sample: dict, tag: str):
    assert "inputs" in sample,   f"[{tag}] missing 'inputs' key"
    assert "concepts" in sample, f"[{tag}] missing 'concepts' key"
    assert "x" in sample["inputs"],  f"[{tag}] missing 'x' in inputs"
    assert "c" in sample["concepts"], f"[{tag}] missing 'c' in concepts"
    assert isinstance(sample["inputs"]["x"],   torch.Tensor), f"[{tag}] x is not a Tensor"
    assert isinstance(sample["concepts"]["c"], torch.Tensor), f"[{tag}] c is not a Tensor"


def check_batch_image(batch: dict, tag: str, batch_size: int):
    x = batch["inputs"]["x"]
    c = batch["concepts"]["c"]
    assert x.ndim >= 2,              f"[{tag}] batch x should be at least 2-D"
    assert c.ndim == 2,              f"[{tag}] batch c should be 2-D"
    assert x.shape[0] <= batch_size, f"[{tag}] unexpected batch dim in x"
    assert c.shape[0] <= batch_size, f"[{tag}] unexpected batch dim in c"


def check_sample_nlp(sample: dict, tag: str):
    assert "inputs"   in sample,              f"[{tag}] missing 'inputs' key"
    assert "concepts" in sample,              f"[{tag}] missing 'concepts' key"
    assert "input_ids" in sample["inputs"],   f"[{tag}] missing 'input_ids' in inputs"
    assert "c" in sample["concepts"],          f"[{tag}] missing 'c' in concepts"
    assert isinstance(sample["inputs"]["input_ids"],   torch.Tensor), f"[{tag}] input_ids is not a Tensor"
    assert isinstance(sample["concepts"]["c"], torch.Tensor),         f"[{tag}] c is not a Tensor"


def check_batch_nlp(batch: dict, tag: str, batch_size: int):
    ids = batch["inputs"]["input_ids"]
    c   = batch["concepts"]["c"]
    assert ids.ndim == 2,              f"[{tag}] batch input_ids should be 2-D"
    assert c.ndim   == 2,              f"[{tag}] batch c should be 2-D"
    assert ids.shape[0] <= batch_size, f"[{tag}] unexpected batch dim in input_ids"
    assert c.shape[0]   <= batch_size, f"[{tag}] unexpected batch dim in c"


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_awa2_datamodule(temp_dir: str):
    """Test AWA2DataModule with synthetic data — no download required."""
    from torch_concepts.data import AWA2DataModule

    print("\n" + "=" * 60)
    print("TESTING: AWA2DataModule")
    print("=" * 60)

    root = os.path.join(temp_dir, "awa2")
    print("  Creating mock AWA2 raw data ...")
    create_mock_awa2_data(root, n_images_per_class=2)

    dm = AWA2DataModule(
        root=root,
        image_size=32,
        seed=42,
        train_size=0.6,
        val_size=0.2,
        batch_size=4,
        backbone=None,
        precompute_embs=False,
    )

    dm.setup(verbose=False)

    print(f"  Dataset len:  {len(dm.dataset)}")
    print(f"  Concept names ({dm.n_concepts}): {dm.concept_names[:6]} ...")
    print(f"  Train / Val / Test: {dm.train_len} / {dm.val_len} / {dm.test_len}")

    assert dm.train_len > 0, "Train set is empty"
    assert dm.val_len   > 0, "Val set is empty"
    assert dm.test_len  > 0, "Test set is empty"
    assert dm.train_len + dm.val_len + dm.test_len == len(dm.dataset), \
        "Split sizes do not sum to dataset length"

    # __getitem__ check on every split
    for split_name, subset in [
        ("train", dm.trainset),
        ("val",   dm.valset),
        ("test",  dm.testset),
    ]:
        sample = subset[0]
        check_sample_image(sample, f"AWA2/{split_name}")
        print(
            f"  {split_name:5s} sample  x:{sample['inputs']['x'].shape}  "
            f"c:{sample['concepts']['c'].shape}"
        )

    # DataLoader batch check on every split
    for split_name, loader_fn in [
        ("train", dm.train_dataloader),
        ("val",   dm.val_dataloader),
        ("test",  dm.test_dataloader),
    ]:
        batch = next(iter(loader_fn()))
        check_batch_image(batch, f"AWA2/{split_name}", dm.batch_size)
        print(
            f"  {split_name:5s} batch   x:{batch['inputs']['x'].shape}  "
            f"c:{batch['concepts']['c'].shape}"
        )

    # Concept vector checks
    sample = dm.trainset[0]
    c = sample["concepts"]["c"]
    assert c.shape[0] == dm.n_concepts, \
        f"Expected {dm.n_concepts} concepts, got {c.shape[0]}"
    # Binary attributes (columns 0-84) should be 0 or 1
    assert c[:85].max() <= 1 and c[:85].min() >= 0, \
        "Binary attributes should be in {0, 1}"
    # Class index (last column) should be in [0, 49]
    assert 0 <= c[-1] <= 49, f"Class index out of range: {c[-1]}"

    print("  PASSED")


def test_cebab_datamodule(temp_dir: str):
    """Test CEBaBDataModule — skipped if datasets/transformers are not installed."""
    try:
        import datasets     # noqa: F401
        import transformers # noqa: F401
    except ImportError as exc:
        print(f"\n  SKIPPED: CEBaBDataModule ({exc})")
        return

    from torch_concepts.data import CEBaBDataModule
    from torch_concepts.data.datasets.cebab import ASPECT_NAMES, TASK_NAME  # module-level constants, not re-exported

    print("\n" + "=" * 60)
    print("TESTING: CEBaBDataModule")
    print("=" * 60)

    dm = CEBaBDataModule(
        root=os.path.join(temp_dir, "cebab"),
        pre_trained_model_name="bert-base-uncased",
        max_length=128,          # shorter sequences to speed up tokenisation
        batch_size=4,
        backbone=None,
        precompute_embs=False,
    )

    dm.setup(verbose=False)

    print(f"  Dataset len:  {len(dm.dataset)}")
    print(f"  Concept names ({dm.n_concepts}): {dm.concept_names}")
    print(f"  Train / Val / Test: {dm.train_len} / {dm.val_len} / {dm.test_len}")

    assert dm.train_len > 0, "Train set is empty"
    assert dm.val_len   > 0, "Val set is empty"
    assert dm.test_len  > 0, "Test set is empty"
    assert dm.train_len + dm.val_len + dm.test_len == len(dm.dataset), \
        "Split sizes do not sum to dataset length"

    expected_concepts = ASPECT_NAMES + [TASK_NAME]
    assert dm.n_concepts == len(expected_concepts), \
        f"Expected {len(expected_concepts)} concepts, got {dm.n_concepts}"

    # __getitem__ check on every split
    for split_name, subset in [
        ("train", dm.trainset),
        ("val",   dm.valset),
        ("test",  dm.testset),
    ]:
        sample = subset[0]
        check_sample_nlp(sample, f"CEBaB/{split_name}")
        print(
            f"  {split_name:5s} sample  "
            f"input_ids:{sample['inputs']['input_ids'].shape}  "
            f"c:{sample['concepts']['c'].shape}"
        )

    # DataLoader batch check on every split
    for split_name, loader_fn in [
        ("train", dm.train_dataloader),
        ("val",   dm.val_dataloader),
        ("test",  dm.test_dataloader),
    ]:
        batch = next(iter(loader_fn()))
        check_batch_nlp(batch, f"CEBaB/{split_name}", dm.batch_size)
        print(
            f"  {split_name:5s} batch   "
            f"input_ids:{batch['inputs']['input_ids'].shape}  "
            f"c:{batch['concepts']['c'].shape}"
        )

    # Concept value range checks
    sample = dm.trainset[0]
    c = sample["concepts"]["c"]
    # Aspect columns (0-3): values in {0, 1, 2}
    for i in range(4):
        assert c[i].item() in {0.0, 1.0, 2.0}, \
            f"Aspect concept {i} has unexpected value {c[i].item()}"
    # Review majority (column 4): values in {0, 1, 2, 3, 4}
    assert 0 <= c[4].item() <= 4, \
        f"Review majority out of range: {c[4].item()}"

    # Tokeniser output sanity
    ids = sample["inputs"]["input_ids"]
    assert ids.shape[0] == 128, f"Expected max_length=128, got {ids.shape[0]}"
    mask = dm.dataset._attention_mask[0]
    assert mask.shape == ids.shape, "attention_mask shape mismatch"

    print("  PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    temp_dir = os.path.join(os.getcwd(), "temp_datamodule_test")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temp data directory: {temp_dir}")

    passed = 0
    failed = 0
    errors = []

    for test_fn in [test_awa2_datamodule]: #[test_awa2_datamodule, test_cebab_datamodule]:
        try:
            test_fn(temp_dir)
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((test_fn.__name__, str(exc)))
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        for name, err in errors:
            print(f"  FAILED: {name} — {err}")
    print("=" * 60)

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
