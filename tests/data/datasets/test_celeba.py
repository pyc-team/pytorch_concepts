"""Tests for CelebADataset image loading (torch_concepts.data.datasets.celeba).

These build a bare dataset instance and a tiny in-memory zip so the zip-reading
path in ``__getitem__`` is exercised offline, without a CelebA download.
"""
import io
import zipfile

import numpy as np
import torch
from PIL import Image

from torch_concepts.data.datasets.celeba import CelebADataset


def _make_zip(root, filename="x.png"):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buf, format="PNG")
    with zipfile.ZipFile(raw / "img_align_celeba.zip", "w") as zf:
        zf.writestr(f"img_align_celeba/{filename}", buf.getvalue())


def _bare_dataset(root, filename="x.png"):
    # Bypass the heavy __init__ (download/build): we only test __getitem__.
    ds = CelebADataset.__new__(CelebADataset)
    ds.root = str(root)
    ds._zip = None
    ds.embs_precomputed = False
    ds.input_data = [filename]
    ds.concepts = torch.zeros(1, 2)
    return ds


def test_reads_image_from_zip(tmp_path):
    _make_zip(tmp_path)
    ds = _bare_dataset(tmp_path)
    sample = ds[0]
    assert sample["inputs"]["x"].shape == (3, 4, 4)


def test_zip_entry_handle_is_closed(tmp_path, monkeypatch):
    """The per-item zip entry handle is closed after __getitem__ (no FD leak)."""
    _make_zip(tmp_path)
    ds = _bare_dataset(tmp_path)

    opened = []
    orig_open = zipfile.ZipFile.open

    def tracking_open(self, name, *args, **kwargs):
        fh = orig_open(self, name, *args, **kwargs)
        opened.append(fh)
        return fh

    monkeypatch.setattr(zipfile.ZipFile, "open", tracking_open)

    ds[0]

    assert opened, "expected the zip entry to be opened"
    assert all(fh.closed for fh in opened)


def _raw_celeba(root, attrs, n=12):
    """A minimal but complete CelebA raw tree: attributes, splits, and images."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    names = [f"{i:06d}.jpg" for i in range(1, n + 1)]

    # list_attr_celeba.txt is "<count>\n<header>\n<rows>", values in {-1, 1}.
    (raw / "list_attr_celeba.txt").write_text(
        f"{n}\n{' '.join(attrs)}\n"
        + "".join(
            f"{name} " + " ".join("1" if (i + j) % 2 else "-1" for j in range(len(attrs))) + "\n"
            for i, name in enumerate(names)
        )
    )
    # list_eval_partition.txt has no header: 0=train, 1=valid, 2=test.
    (raw / "list_eval_partition.txt").write_text(
        "".join(f"{name} {0 if i < n - 4 else (1 if i < n - 2 else 2)}\n"
                for i, name in enumerate(names))
    )
    buf = io.BytesIO()
    Image.fromarray(np.zeros((218, 178, 3), dtype=np.uint8)).save(buf, format="JPEG")
    with zipfile.ZipFile(raw / "img_align_celeba.zip", "w") as zf:
        for name in names:
            zf.writestr(f"img_align_celeba/{name}", buf.getvalue())
    return names


def test_concept_subset_with_dataframe_concepts(tmp_path):
    """A concept subset must survive CelebA's DataFrame-backed concepts.

    CelebA is the only dataset whose concepts arrive as a ``pd.DataFrame``, and
    ``set_concepts`` selects columns by name — so a reduced annotation whose
    ``labels`` were a *tuple* used to be read as one MultiIndex key and raise
    ``KeyError``. Regression guard for ``_maybe_reduce_annotations``.
    """
    from torch_concepts.data.datamodules.celeba import CelebADataModule
    from torch_concepts.data.splitters.native import NativeSplitter

    attrs = ["Smiling", "Male", "Eyeglasses", "Blond_Hair", "Young", "Bald"]
    subset = attrs[:3]
    _raw_celeba(tmp_path, attrs)

    dm = CelebADataModule(
        root=str(tmp_path), splitter=NativeSplitter(), batch_size=2,
        concept_subset=subset, image_size=64,
    )
    dm.setup("fit")

    assert list(dm.annotations.labels) == subset
    assert dm.n_features == (3, 64, 64)
    batch = next(iter(dm.train_dataloader()))
    assert batch["inputs"]["x"].shape[1:] == (3, 64, 64)
    assert batch["concepts"]["c"].shape[1] == len(subset)
