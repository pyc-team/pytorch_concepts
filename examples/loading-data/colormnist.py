"""
Color-MNIST DataModule: unbiased vs. biased (Data Interface)
============================================================

This example demonstrates how to:
1. Load Color-MNIST through PyC's datamodule utilities
2. Build the *unbiased* version, where the colour of a digit is drawn at random
3. Build a *biased* version, where the colour is determined by the digit
4. Read off the digit/colour statistics that tell the two apart

Key Components:
- ColorMNISTDataModule: PyC datamodule wrapping MNIST digits tinted red/green
- `coloring`: how a colour is assigned -- 'random', or a {colour: digits} map
- `coloring_test`: the colouring of the test population, which is what turns a
  bias into a train/test *distribution shift*

Dataset: MNIST tinted red or green, with 3 concepts (digit, parity, color)
Statistic: P(red | digit < 5) per split -- 0.5 under no bias, 1.0 under the
shortcut, and 0.0 on a split whose colouring was reversed
"""
from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule

# Low digits red, high digits green -- and the reverse, to shift the test split.
LOW_IS_RED = {'red': range(5), 'green': range(5, 10)}
LOW_IS_GREEN = {'red': range(5, 10), 'green': range(5)}


def red_given_low_digit(dm, split):
    """P(colour == red | digit < 5) over one split.

    0.5 means the colour says nothing about the digit; 1.0 (or 0.0) means it
    gives the digit away, which is the shortcut a concept-based model can learn
    instead of reading the shape.
    """
    concepts = dm.dataset.concepts[getattr(dm, f"{split}set").indices]
    digit = concepts['digit'].flatten()
    color = concepts['color'].flatten()
    return (color[digit < 5] == 0).float().mean().item()


def report(dm, title):
    dm.setup()
    stats = " | ".join(
        f"{split}: n={len(getattr(dm, f'{split}set')):>5} "
        f"P(red|digit<5)={red_given_low_digit(dm, split):.2f}"
        for split in ('train', 'val', 'test')
    )
    print(f"   {title}\n     {stats}")


def main():
    seed_everything(42)

    # =========================================================================
    # Unbiased Color-MNIST
    # =========================================================================
    print("\n1. Unbiased dataset (colour drawn at random)...")

    # MNIST is downloaded on first use. The train and test images are colourized
    # separately and appended, so the default split keeps MNIST's own boundary.
    dm = ColorMNISTDataModule(root='./data/mnist', batch_size=512)
    print(f"   Dataset size: {dm.n_samples} samples")
    print(f"   Image shape: {dm.n_features}")
    print(f"   Concepts: {dm.concept_names} (cardinalities {dm.annotations.cardinalities})")
    report(dm, "colour is independent of the digit:")

    # =========================================================================
    # Biased Color-MNIST
    # =========================================================================
    print("\n2. Biased dataset (colour determined by the digit)...")

    # One `coloring` for every split: the shortcut is present everywhere, so a
    # model exploiting it still scores well at test time.
    dm = ColorMNISTDataModule(root='./data/mnist', batch_size=512,
                              coloring=LOW_IS_RED)
    report(dm, "shortcut in every split:")

    # `coloring_test` reverses it on the test population only -- now the
    # shortcut is actively misleading, and relying on it costs accuracy.
    dm = ColorMNISTDataModule(root='./data/mnist', batch_size=512,
                              coloring=LOW_IS_RED, coloring_test=LOW_IS_GREEN)
    report(dm, "shortcut reversed at test time (distribution shift):")

    # =========================================================================
    # Batches
    # =========================================================================
    print("\n3. Reading a batch...")

    batch = next(iter(dm.train_dataloader()))
    print(f"   inputs:   {tuple(batch['inputs']['x'].shape)}")
    print(f"   concepts: {tuple(batch['concepts']['c'].shape)} "
          f"labels={batch['concepts']['c'].annotation.labels}")


if __name__ == "__main__":
    main()
