"""SIIM-ACR Pneumothorax Segmentation dataset."""
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from torch_concepts import Annotations
from torch_concepts.data.base import ConceptDataset

KAGGLE_DATASET = "jesperdramsch/siim-acr-pneumothorax-segmentation-data"
KAGGLE_COMPETITION = "siim-acr-pneumothorax-segmentation"
ANNOTATION_FILENAME = "stage_2_train.csv"


def _import_kagglehub():
    """Import KaggleHub only when a download is requested."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "PneumothoraxDataset download requires `kagglehub`. Install it with: "
            "pip install kagglehub"
        ) from exc
    return kagglehub


def _import_pydicom():
    """Import pydicom only when DICOM metadata or pixels are needed."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "PneumothoraxDataset requires `pydicom` for SIIM-ACR DICOM images. "
            "Install it with: pip install pydicom"
        ) from exc
    return pydicom


class PneumothoraxDataset(ConceptDataset):
    """SIIM-ACR chest radiographs with Pneumothorax labels.

    The official SIIM-ACR Pneumothorax Segmentation competition data contains
    DICOM chest radiographs and run-length encoded segmentation masks. This
    dataset exposes whether a radiograph contains a pneumothorax as one binary
    native concept. Image, patient, study, and series identifiers, together
    with available DICOM demographic and acquisition fields, are retained in
    :attr:`metadata` rather than treated as concepts.

    The original competition image endpoint has been retired. By default,
    ``download()`` retrieves the stage-1 training and test images from a
    community DICOM re-host, then obtains ``stage_2_train.csv`` from the
    competition. Together these form the labeled stage-2 training set.
    Kaggle credentials and acceptance of the competition rules are required.
    Matching DICOM files and ``stage_2_train.csv`` may also be supplied
    manually below ``root/raw``.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored or downloaded. Default:
        ``./data/pneumothorax``.
    image_size : int, optional
        Side length (px) to resize images to. Default: 224.
    concept_subset : list of str, optional
        Subset of native concepts to retain. Default: ``None``.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        image_size: int = 224,
        concept_subset: Optional[List[str]] = None,
    ):
        if root is None:
            root = os.path.join(os.getcwd(), "data", "pneumothorax")
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
            name="PneumothoraxDataset",
        )

    @property
    def raw_filenames(self) -> List[str]:
        """Return the path to the Stage 2 annotations."""
        return [os.path.join("raw", ANNOTATION_FILENAME)]

    @property
    def processed_filenames(self) -> List[str]:
        return [
            "image_paths.txt",
            "concepts.h5",
            "annotations.pt",
            "metadata.h5",
        ]

    def _annotation_path(self) -> Optional[Path]:
        raw_dir = Path(self.root_dir) / "raw"
        direct_path = raw_dir / ANNOTATION_FILENAME
        if direct_path.is_file():
            return direct_path
        matches = list(raw_dir.rglob(ANNOTATION_FILENAME))
        if len(matches) > 1:
            raise ValueError(
                f"Found multiple Stage 2 annotation files: {matches[:3]}."
            )
        if matches:
            return matches[0]
        return None

    def _dicom_paths(self) -> List[Path]:
        raw_dir = Path(self.root_dir) / "raw"
        train_dir = raw_dir / "dicom-images-train"
        if train_dir.is_dir():
            paths = list(train_dir.rglob("*.dcm"))
            paths.extend((raw_dir / "dicom-images-test").rglob("*.dcm"))
            return sorted(paths)
        return sorted(raw_dir.rglob("*.dcm"))

    def _raw_is_complete(self) -> bool:
        return self._annotation_path() is not None and bool(self._dicom_paths())

    def maybe_download(self) -> None:
        """Download only when no usable manually-provided raw data exists."""
        if not self._raw_is_complete():
            self.download()

    def download(self) -> None:
        """Download the maintained SIIM-ACR DICOM re-host through KaggleHub.

        The original competition's Cloud Healthcare image endpoint has been
        retired. This uses a public Kaggle dataset that re-hosts the DICOM
        images. The competition supplies the stage-2 annotations for both the
        original training and test images. Files are downloaded directly into
        ``raw`` so PyC's processed cache remains self-contained.
        """
        kagglehub = _import_kagglehub()
        raw_dir = Path(self.root_dir) / "raw"
        try:
            # Check competition access before downloading the larger image archive.
            if self._annotation_path() is None:
                kagglehub.competition_download(
                    KAGGLE_COMPETITION,
                    path=ANNOTATION_FILENAME,
                    output_dir=str(raw_dir),
                )
            kagglehub.dataset_download(
                KAGGLE_DATASET,
                output_dir=str(raw_dir),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not download SIIM-ACR images or stage-2 annotations. "
                "Configure Kaggle credentials and accept the competition rules at "
                "https://www.kaggle.com/competitions/"
                "siim-acr-pneumothorax-segmentation/rules, then retry; "
                "alternatively place matching DICOM files and stage_2_train.csv "
                f"under {Path(self.root_dir) / 'raw'}."
            ) from exc

        if not self._raw_is_complete():
            raise RuntimeError(
                "The downloaded SIIM-ACR files do not contain both a DICOM image tree "
                "and stage_2_train.csv."
            )

    def build(self) -> None:
        """Cache image paths, native concepts, and DICOM metadata."""
        self.maybe_download()
        annotation_path = self._annotation_path()
        if annotation_path is None:
            raise RuntimeError("No SIIM-ACR annotation CSV was found after download.")

        annotations = pd.read_csv(annotation_path)
        annotations.columns = annotations.columns.str.strip()
        required_columns = {"ImageId", "EncodedPixels"}
        missing_columns = required_columns - set(annotations.columns)
        if missing_columns:
            raise ValueError(
                "Pneumothorax annotation CSV is missing columns: "
                f"{sorted(missing_columns)}."
            )
        # Convert each image's mask annotations into one presence label.
        labels = self._pneumothorax_labels(annotations)

        # Check that every labeled image has a corresponding DICOM file.
        image_map = self._build_image_map(self._dicom_paths())
        missing_images = sorted(set(labels) - set(image_map))
        if missing_images:
            raise ValueError(
                f"{len(missing_images)} annotated SIIM-ACR images are missing DICOMs "
                f"(for example: {missing_images[:3]}). Provide all images for the "
                "selected annotation release."
            )
        image_ids = sorted(labels)
        if not image_ids:
            raise ValueError("No annotated Pneumothorax image IDs resolve to DICOM files.")

        # Keep DICOM identifiers and patient information separate from concepts.
        metadata = self._read_metadata(image_ids, image_map)
        concepts = pd.DataFrame(
            {"pneumothorax": [labels[image_id] for image_id in image_ids]},
            dtype=np.float32,
        )
        native_annotations = Annotations(
            labels=["pneumothorax"],
            states=[["pneumothorax"]],
            cardinalities=[1],
            types=["binary"],
        )
        image_paths = [
            os.path.relpath(image_map[image_id], self.root_dir) for image_id in image_ids
        ]

        # Save the prepared data so future loads do not reread every DICOM header.
        os.makedirs(self.root_dir, exist_ok=True)
        with open(self.processed_paths[0], "w") as file:
            file.write("\n".join(image_paths))
        concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        torch.save(native_annotations, self.processed_paths[2])
        metadata.to_hdf(self.processed_paths[3], key="metadata", mode="w")

    @staticmethod
    def _pneumothorax_labels(annotations: pd.DataFrame) -> dict[str, int]:
        labels: dict[str, int] = {}
        for image_id, encoded_pixels in annotations.groupby("ImageId")["EncodedPixels"]:
            masks = encoded_pixels.dropna().astype(str).str.strip()
            has_pneumothorax = False
            for mask in masks:
                if mask and mask != "-1":
                    has_pneumothorax = True
                    break
            labels[str(image_id)] = int(has_pneumothorax)
        return labels

    @staticmethod
    def _build_image_map(image_paths: List[Path]) -> dict[str, str]:
        image_map: dict[str, str] = {}
        duplicates = []
        for path in image_paths:
            image_id = path.stem
            if image_id in image_map:
                duplicates.append(image_id)
            else:
                image_map[image_id] = str(path)
        if duplicates:
            raise ValueError(
                "SIIM-ACR DICOM files contain duplicate image IDs "
                f"(for example: {sorted(duplicates)[:3]})."
            )
        return image_map

    @staticmethod
    def _read_metadata(image_ids: List[str], image_map: dict[str, str]) -> pd.DataFrame:
        pydicom = _import_pydicom()
        records = []
        for image_id in image_ids:
            dicom = pydicom.dcmread(
                image_map[image_id],
                stop_before_pixels=True,
            )
            patient_id = str(getattr(dicom, "PatientID", "") or image_id)
            records.append(
                {
                    "image_id": image_id,
                    "patient_id": patient_id,
                    "study_id": str(getattr(dicom, "StudyInstanceUID", "") or ""),
                    "series_id": str(getattr(dicom, "SeriesInstanceUID", "") or ""),
                    "patient_age": str(getattr(dicom, "PatientAge", "") or ""),
                    "patient_sex": str(getattr(dicom, "PatientSex", "") or ""),
                    "view_position": str(getattr(dicom, "ViewPosition", "") or ""),
                }
            )
        return pd.DataFrame.from_records(records)

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

        pydicom = _import_pydicom()
        dicom = pydicom.dcmread(os.path.join(self.root_dir, self.input_data[item]))
        pixels = np.asarray(dicom.pixel_array, dtype=np.float32)
        slope = float(getattr(dicom, "RescaleSlope", 1))
        intercept = float(getattr(dicom, "RescaleIntercept", 0))
        pixels = pixels * slope + intercept
        minimum = float(pixels.min())
        maximum = float(pixels.max())
        if maximum > minimum:
            pixels = (pixels - minimum) / (maximum - minimum)
        else:
            pixels = np.zeros_like(pixels)
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 1 - pixels
        image = Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        sample["inputs"]["x"] = image_tensor.float() / 255.0
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
