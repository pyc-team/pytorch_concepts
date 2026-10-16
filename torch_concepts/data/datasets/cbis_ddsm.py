"""Native CBIS-DDSM abnormality/view observations and distributed DICOM images."""
import os
import shutil
import zipfile
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch_concepts import Annotations
from torch_concepts.data.base import ConceptDataset


TCIA_URL = "https://www.cancerimagingarchive.net/collection/cbis-ddsm/"
CSV_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/"
NBIA_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
IMAGE_COLUMNS = {
    "full": "image_file_path",
    "cropped": "cropped_image_file_path",
    "mask": "roi_mask_file_path",
}
CATEGORICAL_STATES = {
    "breast_density": ("1", "2", "3", "4", "0"),
    "assessment": ("0", "1", "2", "3", "4", "5"),
    "subtlety": ("0", "1", "2", "3", "4", "5"),
    "pathology": ("BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"),
}
# Atomic vocabularies verified against all four official CSVs and TFDS's
# cbis_ddsm_{mass_shapes,mass_margins,calc_types,calc_distributions}.txt.
# N/A and both legacy LUCENT_CENTER spellings remain distinct source values.
DESCRIPTOR_STATES = {
    "mass_shape": (
        "N/A", "ARCHITECTURAL_DISTORTION", "ASYMMETRIC_BREAST_TISSUE",
        "FOCAL_ASYMMETRIC_DENSITY", "IRREGULAR", "LOBULATED", "LYMPH_NODE",
        "OVAL", "ROUND",
    ),
    "mass_margin": (
        "N/A", "CIRCUMSCRIBED", "ILL_DEFINED", "MICROLOBULATED", "OBSCURED",
        "SPICULATED",
    ),
    "calcification_type": (
        "N/A", "AMORPHOUS", "COARSE", "DYSTROPHIC", "EGGSHELL",
        "FINE_LINEAR_BRANCHING", "LARGE_RODLIKE", "LUCENT_CENTER",
        "LUCENT_CENTERED", "MILK_OF_CALCIUM", "PLEOMORPHIC", "PUNCTATE",
        "ROUND_AND_REGULAR", "SKIN", "VASCULAR",
    ),
    "calcification_distribution": (
        "N/A", "CLUSTERED", "DIFFUSELY_SCATTERED", "LINEAR", "REGIONAL",
        "SEGMENTAL",
    ),
}
DESCRIPTOR_COLUMNS = {
    "mass": {"mass_shape": "mass_shape", "mass_margin": "mass_margins"},
    "calcification": {
        "calcification_type": "calc_type",
        "calcification_distribution": "calc_distribution",
    },
}
IDENTITY_COLUMNS = (
    "patient_id", "left_or_right_breast", "image_view", "abnormality_id",
)


class CBISDDSMDataset(ConceptDataset):
    """CBIS-DDSM, a curated subset of scanned-film DDSM mammography.

    One sample is one official CSV observation: an abnormality on a breast/view.
    Multiple abnormalities can share a full mammogram; no rows are deduplicated.
    The selected CSV determines train/test membership. Mass and calcification
    have separate descriptor axes, alongside categorical density, assessment,
    subtlety, and the three original pathology states.

    Missing CSVs and image series are downloaded from TCIA automatically. If
    acquisition fails or downloaded instances cannot be matched unambiguously,
    an error asks for manual acquisition using the NBIA Data Retriever. Put
    CSVs in ``root/raw`` and DICOMs below ``root``, keeping the source
    ``StudyUID/SeriesUID/filename.dcm`` suffixes. Prefix directories may differ.

    Pixels are float32 tensors of shape ``(1, H, W)`` at native resolution and
    stored dynamic range. MONOCHROME1 is linearly inverted using the declared
    BitsStored range, not observed image extrema. No modality/VOI LUT, resizing,
    normalization, or channel replication is applied. ``transform`` receives
    this tensor and can supply any downstream preprocessing.

    Raw CSV cell strings (including blanks and literal N/A) are retained in
    ``metadata`` with normalized column names, source row, split, sample ID,
    and the selected resolved image path. Missing concepts use NaN. Explicit
    descriptor N/A gets its own indicator; medical descriptor indicators are
    unknown (NaN) for N/A, rather than asserted absent.

    See :doc:`/guides/cbis_ddsm` for acquisition, exact state ordering, known
    source ambiguities, and the source metadata audit.

    Args:
        root: Dataset directory. Defaults to ``data/cbis_ddsm``.
        abnormality_type: ``"mass"`` or ``"calcification"``.
        split: Official ``"train"`` or ``"test"`` CSV membership.
        image_type: Distributed ``"full"``, ``"cropped"``, or ``"mask"``.
        transform: Optional callable applied to decoded image tensors.
        concept_subset: Concept labels or PyC annotation group names.

    References:
        Lee et al., Scientific Data 4, 170177 (2017),
        https://doi.org/10.1038/sdata.2017.177.
        TCIA release: https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        abnormality_type: str = "mass",
        split: str = "train",
        image_type: str = "cropped",
        transform: Optional[Callable] = None,
        concept_subset: Optional[List[str]] = None,
    ):
        if abnormality_type not in DESCRIPTOR_COLUMNS:
            raise ValueError("abnormality_type must be 'mass' or 'calcification'.")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'.")
        if image_type not in IMAGE_COLUMNS:
            raise ValueError("image_type must be 'full', 'cropped', or 'mask'.")
        self.root = os.fspath(root) if root is not None else os.path.join(
            os.getcwd(), "data", "cbis_ddsm"
        )
        self.abnormality_type = abnormality_type
        self.split = split
        self.image_type = image_type
        self.transform = transform
        self.metadata: pd.DataFrame

        image_paths, concepts, annotations, graph = self.load()
        if concept_subset is not None:
            concept_subset = annotations.resolve(concept_subset)[1].labels
        super().__init__(
            input_data=image_paths,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            reorder_by_type=False,
            name="CBISDDSMDataset",
        )

    @property
    def raw_filenames(self) -> List[str]:
        prefix = "mass" if self.abnormality_type == "mass" else "calc"
        return [f"raw/{prefix}_case_description_{self.split}_set.csv"]

    @property
    def processed_filenames(self) -> List[str]:
        directory = f"processed/{self.abnormality_type}/{self.split}/{self.image_type}"
        return [f"{directory}/{name}" for name in (
            "concepts.h5", "annotations.pt", "metadata.h5",
        )]

    def maybe_download(self) -> None:
        """Download only when the selected CSV or referenced images are missing."""
        if not Path(self.raw_paths[0]).is_file():
            self.download()
            return
        metadata = self._read_metadata(self.raw_paths[0], self.abnormality_type)
        paths = self._build_path_index(self.root_dir)
        for reference in metadata[IMAGE_COLUMNS[self.image_type]]:
            try:
                self._resolve_path(reference, paths)
            except FileNotFoundError:
                self.download()
                return

    def download(self) -> None:
        """Fetch missing selected sources, falling back to manual instructions."""
        csv_path = Path(self.raw_paths[0])
        try:
            if not csv_path.is_file():
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=csv_path.parent) as directory:
                    temporary = Path(directory) / csv_path.name
                    self._download_file(CSV_URL + csv_path.name, temporary)
                    self._read_metadata(str(temporary), self.abnormality_type)
                    os.replace(temporary, csv_path)
        except (OSError, HTTPException, ValueError) as exc:
            raise self._manual_download_error(exc) from exc
        metadata = self._read_metadata(str(csv_path), self.abnormality_type)
        paths = self._build_path_index(self.root_dir)
        missing = {}
        for reference in dict.fromkeys(metadata[IMAGE_COLUMNS[self.image_type]]):
            try:
                self._resolve_path(reference, paths)
            except FileNotFoundError:
                suffix = PurePosixPath(reference.strip().replace("\\", "/")).parts[-3:]
                if len(suffix) != 3 or any(part in ("..", ".") for part in suffix):
                    raise ValueError(f"Invalid source DICOM reference: {reference!r}.")
                missing.setdefault(suffix[:2], []).append(suffix[2])
        for (study, series), filenames in tqdm(
            missing.items(), desc="Downloading CBIS-DDSM series", disable=not missing,
        ):
            try:
                self._download_series(study, series, filenames)
            except (OSError, HTTPException, ValueError, zipfile.BadZipFile, EOFError) as exc:
                raise self._manual_download_error(exc) from exc

    def _manual_download_error(self, error: Exception) -> FileNotFoundError:
        return FileNotFoundError(
            f"Automatic CBIS-DDSM download failed: {error} "
            f"Please download {self.abnormality_type} {self.split} "
            f"{self.image_type} images and {Path(self.raw_paths[0]).name} "
            f"manually from {TCIA_URL} using the NBIA Data Retriever. "
            f"Place the CSV in {Path(self.root_dir) / 'raw'} and DICOMs below "
            f"{self.root_dir}, preserving StudyUID/SeriesUID/filename.dcm "
            "suffixes matching the CSV. Completed files are kept; rerun after downloading."
        )

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        """Stream to a temporary file so interrupted transfers are not reused."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as directory:
            temporary = Path(directory) / destination.name
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
                expected = response.headers.get("Content-Length")
            if expected is not None and temporary.stat().st_size != int(expected):
                raise OSError(f"Incomplete download from {url}.")
            os.replace(temporary, destination)

    def _download_series(self, study: str, series: str, filenames: List[str]) -> None:
        import pydicom

        query = urlencode({"SeriesInstanceUID": series, "NewFileNames": "No"})
        destination = Path(self.root_dir) / "images" / study / series
        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination) as directory:
            archive_path = Path(directory) / "series.zip"
            self._download_file(f"{NBIA_URL}?{query}", archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                members = [member for member in archive.infolist() if member.filename.endswith(".dcm")]
                selected = []
                for filename in dict.fromkeys(filenames):
                    matches = [m for m in members if PurePosixPath(m.filename).name == filename]
                    if not matches and len(members) == 1 and len(set(filenames)) == 1:
                        matches = members
                    if len(matches) != 1:
                        raise ValueError(
                            f"Cannot uniquely map CSV instance {study}/{series}/{filename} "
                            f"to API filenames {[m.filename for m in members]}. "
                            "Crop/mask identity cannot be inferred from archive order."
                        )
                    member = matches[0]
                    with archive.open(member) as source:
                        try:
                            header = pydicom.dcmread(source, stop_before_pixels=True)
                        except pydicom.errors.InvalidDicomError as exc:
                            raise ValueError(f"Invalid downloaded DICOM: {member.filename}.") from exc
                    if (header.get("StudyInstanceUID"), header.get("SeriesInstanceUID")) != (study, series):
                        raise ValueError(f"Downloaded DICOM UIDs disagree with {study}/{series}.")
                    temporary = Path(directory) / filename
                    with archive.open(member) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    selected.append((temporary, destination / filename))
                for temporary, target in selected:
                    os.replace(temporary, target)
                if "LICENSE" in archive.namelist():
                    license_path = Path(self.root_dir) / "raw" / "TCIA_LICENSE"
                    license_path.write_bytes(archive.read("LICENSE"))

    @staticmethod
    def _read_metadata(csv_path: str, abnormality_type: str) -> pd.DataFrame:
        metadata = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        metadata.columns = [
            "_".join(column.strip().lower().split()) for column in metadata.columns
        ]
        if metadata.columns.duplicated().any():
            raise ValueError(f"Duplicate normalized columns in {csv_path}.")
        required = {
            *IDENTITY_COLUMNS, *CATEGORICAL_STATES, *IMAGE_COLUMNS.values(),
            *DESCRIPTOR_COLUMNS[abnormality_type].values(), "abnormality_type",
        }
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"CBIS-DDSM metadata is missing columns: {sorted(missing)}.")
        identifiers = metadata[list(IDENTITY_COLUMNS)].apply(lambda s: s.str.strip())
        if identifiers.eq("").any().any():
            raise ValueError(f"Blank CBIS-DDSM sample identifiers in {csv_path}.")
        if not metadata["abnormality_type"].str.strip().eq(abnormality_type).all():
            raise ValueError(f"Unexpected abnormality_type values in {csv_path}.")
        return metadata

    @staticmethod
    def _build_path_index(root: str) -> dict[tuple, List[str]]:
        paths = {}
        for path in sorted(Path(root).rglob("*.dcm")):
            suffix = path.parts[-3:]
            paths.setdefault(suffix, []).append(os.path.relpath(path, root))
        return paths

    @staticmethod
    def _resolve_path(reference: str, paths: dict, context: str = "") -> str:
        suffix = PurePosixPath(reference.strip().replace("\\", "/")).parts[-3:]
        matches = paths.get(suffix, []) if len(suffix) == 3 else []
        if not matches:
            raise FileNotFoundError(
                f"No local DICOM matches {reference!r} {context}. "
                "Download the requested image product below root and preserve "
                "the CSV StudyUID/SeriesUID/filename.dcm suffix."
            )
        if len(matches) != 1:
            raise ValueError(
                f"Ambiguous DICOM reference {reference!r} {context}: "
                f"{len(matches)} matches, including {matches[:3]}. "
                "Keep exactly one local file for this source suffix."
            )
        return matches[0]

    @staticmethod
    def _make_concepts(
        metadata: pd.DataFrame, abnormality_type: str,
    ) -> tuple[pd.DataFrame, Annotations]:
        values = {}
        states = []
        types = []
        groups = {}
        for label, vocabulary in CATEGORICAL_STATES.items():
            source = metadata[label].str.strip()
            unknown = sorted(set(source) - set(vocabulary) - {""})
            if unknown:
                raise ValueError(f"Unknown CBIS-DDSM {label} values: {unknown}.")
            values[label] = source.map({state: i for i, state in enumerate(vocabulary)})
            states.append(list(vocabulary))
            types.append("categorical")
        groups["density"] = ["breast_density"]

        for family, column in DESCRIPTOR_COLUMNS[abnormality_type].items():
            vocabulary = DESCRIPTOR_STATES[family]
            rows = []
            for raw in metadata[column]:
                source = raw.strip()
                tokens = source.split("-") if source else []
                unknown = sorted(set(tokens) - set(vocabulary))
                if unknown or ("N/A" in tokens and len(tokens) != 1):
                    raise ValueError(f"Unknown or mixed CBIS-DDSM {column}: {raw!r}.")
                if not tokens:
                    rows.append([np.nan] * len(vocabulary))
                elif tokens == ["N/A"]:
                    rows.append([1.0] + [np.nan] * (len(vocabulary) - 1))
                else:
                    rows.append([float(state in tokens) for state in vocabulary])
            encoded = np.asarray(rows, dtype=np.float32).reshape(-1, len(vocabulary))
            members = []
            for i, state in enumerate(vocabulary):
                label = f"{family}.{state}"
                members.append(label)
                values[label] = encoded[:, i]
                states.append(["present"])
                types.append("binary")
            groups[family] = members

        labels = list(values)
        groups["radiological"] = [label for label in labels if label != "pathology"]
        annotations = Annotations(
            labels=labels, states=states, types=types, groups=groups,
        )
        return pd.DataFrame(values).astype(np.float32), annotations

    def build(self) -> None:
        """Cache only source metadata, resolved paths, and concept annotations."""
        self.maybe_download()
        metadata = self._read_metadata(self.raw_paths[0], self.abnormality_type)
        metadata["split"] = self.split
        metadata["source_csv"] = os.path.basename(self.raw_paths[0])
        metadata["source_row"] = np.arange(len(metadata))
        identity = metadata[list(IDENTITY_COLUMNS)].apply(lambda s: s.str.strip())
        occurrences = identity.groupby(list(IDENTITY_COLUMNS), sort=False).cumcount()
        metadata["sample_id"] = [
            "/".join((self.abnormality_type, self.split, *row, str(occurrence)))
            for row, occurrence in zip(identity.itertuples(index=False, name=None), occurrences)
        ]
        paths = self._build_path_index(self.root_dir)
        column = IMAGE_COLUMNS[self.image_type]
        metadata["resolved_image_path"] = [
            self._resolve_path(
                reference, paths, f"in {self.raw_paths[0]}, observation {i}, {column}"
            )
            for i, reference in enumerate(metadata[column])
        ]
        concepts, annotations = self._make_concepts(metadata, self.abnormality_type)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        concepts.to_hdf(self.processed_paths[0], key="concepts", mode="w")
        torch.save(annotations, self.processed_paths[1])
        metadata.to_hdf(self.processed_paths[2], key="metadata", mode="w")

    def load_raw(self):
        self.maybe_build()
        concepts = pd.read_hdf(self.processed_paths[0], key="concepts")
        annotations = torch.load(self.processed_paths[1], weights_only=False)
        self.metadata = pd.read_hdf(self.processed_paths[2], key="metadata")
        paths = self.metadata["resolved_image_path"].tolist()
        missing = [path for path in paths if not os.path.isfile(os.path.join(self.root_dir, path))]
        if missing:
            raise FileNotFoundError(
                f"Cached CBIS-DDSM DICOM paths no longer exist: {missing[:3]}. "
                "Restore the files or remove this selection's processed directory to rebuild."
            )
        return paths, concepts, annotations, None

    def __getitem__(self, item: int) -> dict:
        sample = super().__getitem__(item)
        if self.embs_precomputed:
            return sample
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("CBISDDSMDataset requires pydicom: pip install pydicom") from exc
        path = os.path.join(self.root_dir, self.input_data[item])
        dicom = pydicom.dcmread(path)
        if dicom.PhotometricInterpretation not in ("MONOCHROME1", "MONOCHROME2"):
            raise ValueError(f"Expected monochrome CBIS-DDSM DICOM: {path}.")
        pixels = dicom.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError(f"Expected a single grayscale frame: {path}.")
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            minimum = -(2 ** (dicom.BitsStored - 1)) if dicom.PixelRepresentation else 0
            maximum = minimum + 2 ** dicom.BitsStored - 1
            pixels = minimum + maximum - pixels
        x = torch.from_numpy(pixels).unsqueeze(0)
        sample["inputs"]["x"] = self.transform(x) if self.transform is not None else x
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
