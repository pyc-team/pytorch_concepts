"""HAM10000 skin-lesion dataset."""
import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from PIL import Image

from torch_concepts import Annotations
from torch_concepts.data.base import ConceptDataset
from torch_concepts.data.io import download_url, download_url_wget, extract_zip


logger = logging.getLogger(__name__)

DATAVERSE_URL = "https://dataverse.harvard.edu"
PERSISTENT_ID = "doi:10.7910/DVN/DBW86T"
DIAGNOSIS_STATES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
RAW_FILENAMES = (
    "HAM10000_images_part_1.zip",
    "HAM10000_images_part_2.zip",
    "HAM10000_metadata.csv",
)


class HAM10000Dataset(ConceptDataset):
    """HAM10000 dermoscopic skin-lesion dataset.

    HAM10000 contains 10,015 dermoscopic images in seven diagnostic categories.
    This implementation exposes ``diagnosis``, ``age``, ``sex``, and
    ``localization`` as native concepts; task selection is deliberately left to
    the user or model. HAM10000 has no native fine-grained visual concept
    annotations. Missing metadata is preserved rather than imputed, while
    ``image_id``, ``lesion_id``, and ``dx_type`` remain available in
    :attr:`metadata`.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        image_size: int = 224,
        concept_subset: Optional[List[str]] = None,
    ):
        if root is None:
            root = os.path.join(os.getcwd(), "data", "ham10000")
        self.root = root
        self.image_size = int(image_size)
        self.metadata: pd.DataFrame

        image_paths, concepts, annotations, graph = self.load()
        super().__init__(
            input_data=image_paths,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            reorder_by_type=False,
            name="HAM10000Dataset",
        )

    @property
    def raw_filenames(self) -> List[str]:
        return [os.path.join("raw", filename) for filename in RAW_FILENAMES]

    @property
    def processed_filenames(self) -> List[str]:
        return [
            "image_paths.txt",
            "concepts.h5",
            "annotations.pt",
            "metadata.h5",
        ]

    def download(self) -> None:
        """Discover and download the required files from Harvard Dataverse."""
        query = urlencode({"persistentId": PERSISTENT_ID})
        api_url = f"{DATAVERSE_URL}/api/datasets/:persistentId/?{query}"
        with urlopen(api_url) as response:
            dataset = json.load(response)

        files = dataset["data"]["latestVersion"]["files"]
        file_ids = {}
        for file_info in files:
            data_file = file_info["dataFile"]
            filename = data_file["filename"]
            if filename in RAW_FILENAMES:
                if filename in file_ids:
                    raise RuntimeError(f"Dataverse returned multiple files named {filename!r}.")
                file_ids[filename] = data_file["id"]

        missing = set(RAW_FILENAMES) - set(file_ids)
        if missing:
            raise RuntimeError(
                "The HAM10000 Dataverse release is missing required files: "
                f"{sorted(missing)}."
            )

        raw_dir = os.path.join(self.root_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        for filename in RAW_FILENAMES:
            url = (
                f"{DATAVERSE_URL}/api/access/datafile/{file_ids[filename]}?"
                "format=original"
            )
            destination = os.path.join(raw_dir, filename)
            if filename.endswith(".zip"):
                download_url_wget(url, destination)
            else:
                download_url(url, raw_dir, filename=filename)

    def build(self) -> None:
        """Extract image archives and cache paths, concepts, and metadata."""
        self.maybe_download()

        raw_dir = os.path.join(self.root_dir, "raw")
        images_dir = os.path.join(self.root_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for filename in RAW_FILENAMES[:2]:
            extract_zip(os.path.join(raw_dir, filename), images_dir)

        metadata = pd.read_csv(os.path.join(raw_dir, RAW_FILENAMES[2]))
        required_columns = {
            "image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization"
        }
        missing_columns = required_columns - set(metadata.columns)
        if missing_columns:
            raise ValueError(f"HAM10000 metadata is missing columns: {sorted(missing_columns)}.")

        image_map = self._build_image_map(images_dir)
        image_ids = metadata["image_id"].astype(str)
        if image_ids.duplicated().any():
            raise ValueError("HAM10000 metadata contains duplicate image_id values.")
        unresolved = sorted(set(image_ids) - set(image_map))
        if unresolved:
            raise ValueError(
                f"{len(unresolved)} HAM10000 metadata rows do not resolve to an image "
                f"(for example: {unresolved[:3]})."
            )

        concepts, annotations = self._make_concepts(metadata)
        image_paths = [
            os.path.relpath(image_map[image_id], self.root_dir)
            for image_id in image_ids
        ]

        with open(self.processed_paths[0], "w") as file:
            file.write("\n".join(image_paths))
        concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        torch.save(annotations, self.processed_paths[2])
        metadata.to_hdf(self.processed_paths[3], key="metadata", mode="w")

    @staticmethod
    def _build_image_map(images_dir: str) -> dict[str, str]:
        image_paths: dict[str, List[str]] = {}
        for path in Path(images_dir).rglob("*.jpg"):
            image_paths.setdefault(path.stem, []).append(str(path))

        duplicates = sorted(image_id for image_id, paths in image_paths.items() if len(paths) != 1)
        if duplicates:
            raise ValueError(
                "HAM10000 image archives contain duplicate image IDs "
                f"(for example: {duplicates[:3]})."
            )
        return {image_id: paths[0] for image_id, paths in image_paths.items()}

    @staticmethod
    def _make_concepts(metadata: pd.DataFrame) -> tuple[pd.DataFrame, Annotations]:
        diagnoses = set(metadata["dx"].dropna())
        unexpected_diagnoses = diagnoses - set(DIAGNOSIS_STATES)
        if unexpected_diagnoses:
            raise ValueError(f"Unknown HAM10000 diagnoses: {sorted(unexpected_diagnoses)}.")

        sex_states = sorted(metadata["sex"].dropna().unique().tolist())
        localization_states = sorted(metadata["localization"].dropna().unique().tolist())
        sex_codes = {value: index for index, value in enumerate(sex_states)}
        localization_codes = {value: index for index, value in enumerate(localization_states)}
        diagnosis_codes = {value: index for index, value in enumerate(DIAGNOSIS_STATES)}

        concepts = pd.DataFrame(
            {
                "diagnosis": metadata["dx"].map(diagnosis_codes),
                "age": pd.to_numeric(metadata["age"], errors="raise"),
                "sex": metadata["sex"].map(sex_codes),
                "localization": metadata["localization"].map(localization_codes),
            }
        ).astype(np.float32)
        annotations = Annotations(
            labels=["diagnosis", "age", "sex", "localization"],
            states=[list(DIAGNOSIS_STATES), ["0"], sex_states, localization_states],
            cardinalities=[len(DIAGNOSIS_STATES), 1, len(sex_states), len(localization_states)],
            types=["categorical", "continuous", "categorical", "categorical"],
        )
        return concepts, annotations

    def load_raw(self):
        self.maybe_build()
        with open(self.processed_paths[0]) as file:
            image_paths = file.read().splitlines()
        concepts = pd.read_hdf(self.processed_paths[1], key="concepts")
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        self.metadata = pd.read_hdf(self.processed_paths[3], key="metadata")
        return image_paths, concepts, annotations, None

    def load(self):
        return self.load_raw()

    def __getitem__(self, item: int) -> dict:
        sample = super().__getitem__(item)
        if self.embs_precomputed:
            return sample

        image_path = os.path.join(self.root_dir, self.input_data[item])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        sample["inputs"]["x"] = x
        return sample

    def _subset_rows(self, indices) -> None:
        row_indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        super()._subset_rows(row_indices)
        self.metadata = self.metadata.iloc[row_indices].reset_index(drop=True)

    @property
    def n_samples(self) -> int:
        return len(self.input_data)

    @property
    def n_features(self) -> tuple:
        return tuple(self[0]["inputs"]["x"].shape)

    @property
    def shape(self) -> tuple:
        return (self.n_samples, *self.n_features)
