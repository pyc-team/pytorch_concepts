"""
Example: CLIPAnnotator on CUB images

This example downloads CUB-200-2011 if needed, loads real CUB images, and
annotates a small subset with label-free CLIP concept similarity scores.

Install the optional CLIP dependency before running:

    pip install open-clip-torch

Run from the repository root with:

    python examples/utilization/4_label_free/1_clip_concept_annotator.py

The first run downloads the official CUB_200_2011.tgz archive into --data-root.
"""

import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset

from torch_concepts.data.annotators import CLIPAnnotator
from torch_concepts.data.datasets.cub import CONCEPT_SEMANTICS, SELECTED_CONCEPTS


CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_MD5 = "97eceeb196236b17998738112f37df78"


class CUBImageDataset(Dataset):
    def __init__(self, cub_root):
        self.cub_root = Path(cub_root)
        self.images_root = self.cub_root / "images"
        self.image_paths = self._read_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return {"x": Image.open(self.image_paths[index]).convert("RGB")}

    def _read_image_paths(self):
        images_file = self.cub_root / "images.txt"
        if not images_file.is_file():
            raise FileNotFoundError(f"Expected CUB metadata file at {images_file}")

        image_paths = []
        with images_file.open() as file:
            for line in file:
                _, relative_path = line.strip().split(maxsplit=1)
                image_paths.append(self.images_root / relative_path)
        return image_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CUB_DATA_ROOT", "./data"),
        help="Directory used to store/download CUB. Defaults to CUB_DATA_ROOT or ./data.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of CUB images to annotate.",
    )
    parser.add_argument(
        "--num-concepts",
        type=int,
        default=12,
        help="Number of selected CUB attributes to use as text concepts.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading CUB when it is missing.",
    )
    return parser.parse_args()


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CUB-200-2011 to {destination}...")
    urllib.request.urlretrieve(url, destination)


def check_md5(path, expected_md5):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_md5 = digest.hexdigest()
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}"
        )


def safe_extract(tar, destination):
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise RuntimeError(f"Refusing to extract unsafe path: {member.name}")
    tar.extractall(destination)


def prepare_cub(data_root, download=True):
    data_root = Path(data_root).expanduser()
    cub_root = data_root / "CUB_200_2011"
    if (cub_root / "images.txt").is_file() and (cub_root / "images").is_dir():
        return cub_root

    if not download:
        raise FileNotFoundError(
            f"CUB was not found at {cub_root}. Re-run without --no-download to fetch it."
        )

    archive_path = data_root / "CUB_200_2011.tgz"
    if not archive_path.is_file():
        download_file(CUB_URL, archive_path)

    print("Verifying CUB archive checksum...")
    check_md5(archive_path, CUB_MD5)

    print(f"Extracting {archive_path} into {data_root}...")
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        safe_extract(tar, data_root)

    if not (cub_root / "images.txt").is_file():
        raise RuntimeError(f"CUB extraction did not create the expected directory: {cub_root}")

    return cub_root


def clean_cub_concept_name(name):
    return name.replace("::", " ").replace("_", " ")


def main():
    args = parse_args()
    cub_root = prepare_cub(args.data_root, download=not args.no_download)
    dataset = Subset(CUBImageDataset(cub_root), range(args.num_samples))

    concepts = [
        clean_cub_concept_name(name)
        for name in np.array(CONCEPT_SEMANTICS)[
            SELECTED_CONCEPTS[: args.num_concepts]
        ]
    ]

    annotator = CLIPAnnotator(
        model_name="ViT-B-32",
        pretrained="openai",
        prompt_template="a photo of {}",
        output="similarity",
    )

    concept_dataset = annotator.annotate(
        dataset=dataset,
        concepts=concepts,
        name="cub_clip_concepts",
    )

    print(concept_dataset)
    print("Concept names:", concept_dataset.concept_names)
    print("Similarity scores shape:", tuple(concept_dataset.concepts.shape))
    print(concept_dataset.concepts)


if __name__ == "__main__":
    main()
