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
