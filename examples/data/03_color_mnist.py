"""
Color-MNIST: Unbiased and Biased
================================

Color-MNIST tints MNIST digits red or green, with three concepts: ``digit``
(10 classes), ``parity`` (1 if the digit is even) and ``color`` (red or green).
How the color is assigned decides what a model can learn from it:

- ``coloring="random"`` (the default): the color says nothing about the digit;
- ``coloring={color: digits}``: the color is determined by the digit, a shortcut
  a model can learn instead of reading the shape;
- ``coloring_test``: a different coloring for the test split, which turns the
  shortcut into a distribution shift.

The statistic ``P(red | digit < 5)`` tells them apart: 0.5 without a shortcut,
1.0 with it, and 0.0 where the test coloring reverses it. MNIST (~60 MB) is
downloaded on first run.
"""

# %%
from torch_concepts import seed_everything
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.env import DATA_ROOT

seed_everything(42)

# low digits red, high digits green, and the reverse
LOW_IS_RED = {"red": range(5), "green": range(5, 10)}
LOW_IS_GREEN = {"red": range(5, 10), "green": range(5)}


def report(title, datamodule):
    print(title)
    datamodule.setup()
    for split in ("train", "val", "test"):
        indices = getattr(datamodule, f"{split}set").indices
        concepts = datamodule.dataset.concepts[indices]
        low = concepts["digit"].flatten() < 5
        red = concepts["color"].flatten() == 0
        p_red = red[low].float().mean()
        print(f"  {split:<5} {low.numel():>6} images, P(red | digit < 5) = {p_red:.2f}")


# %%
# Unbiased
# --------
datamodule = ColorMNISTDataModule(root=str(DATA_ROOT / "colormnist"), batch_size=512)
print(f"images:        {datamodule.n_features}")
print(f"concepts:      {datamodule.concept_names}")
print(f"cardinalities: {datamodule.annotations.cardinalities}")
report("unbiased:", datamodule)

# %%
# A shortcut, everywhere
# ----------------------
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    batch_size=512,
    coloring=LOW_IS_RED,
)
report("shortcut in every split:", datamodule)

# %%
# A shortcut reversed at test time
# --------------------------------
datamodule = ColorMNISTDataModule(
    root=str(DATA_ROOT / "colormnist"),
    batch_size=512,
    coloring=LOW_IS_RED,
    coloring_test=LOW_IS_GREEN,
)
report("shortcut reversed at test time:", datamodule)

# %%
# A batch
# -------
batch = next(iter(datamodule.train_dataloader()))
x, c = batch["inputs"]["x"], batch["concepts"]["c"]
print(f"batch inputs:   {tuple(x.shape)}")
print(f"batch concepts: {tuple(c.shape)} {c.annotations.labels}")
