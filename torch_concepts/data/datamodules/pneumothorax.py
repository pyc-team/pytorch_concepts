"""DataModule for SIIM-ACR Pneumothorax images."""
import numpy as np

from ..base.datamodule import ConceptDataModule
from ..base.splitter import Splitter
from ..datasets.pneumothorax import PneumothoraxDataset
from ..splitters import FixedIndicesSplitter
from ..utils import resolve_size


def _patient_grouped_split(patient_ids, val_size, test_size, seed):
    """Assign whole patients to one split, preventing patient leakage."""
    patients = np.unique(patient_ids)
    np.random.default_rng(seed).shuffle(patients)
    n_test = resolve_size(test_size, len(patients))
    n_val = resolve_size(val_size, len(patients))

    if n_test + n_val > len(patients):
        raise ValueError("Validation and test sizes exceed the number of patients.")

    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test:n_test + n_val])
    train_indices = []
    val_indices = []
    test_indices = []

    for index, patient_id in enumerate(patient_ids):
        if patient_id in test_patients:
            test_indices.append(index)
        elif patient_id in val_patients:
            val_indices.append(index)
        else:
            train_indices.append(index)

    return train_indices, val_indices, test_indices


class PneumothoraxDataModule(ConceptDataModule):
    """DataModule for SIIM-ACR Pneumothorax chest radiographs.

    The competition test images have no labels, so SIIM-ACR does not provide
    an official labeled train/validation/test split. The default split groups
    all images from a DICOM ``PatientID`` into the same partition. If a source
    file omits that field, the dataset falls back to grouping that image by its
    own ID. In that case, patient separation cannot be guaranteed.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored or downloaded. Default:
        ``None`` (auto-creates ``./data/pneumothorax``).
    image_size : int, optional
        Side length (px) to resize images to. Default: 224.
    splitter : Splitter, optional
        Splitting strategy. Default: ``None``, which creates a patient-grouped
        split sized by ``val_size`` and ``test_size``.
    val_size : int or float, optional
        Validation size as a fraction or patient count. Default: 0.1.
    test_size : int or float, optional
        Test size as a fraction or patient count. Default: 0.2.
    batch_size : int, optional
        Number of samples per batch. Default: 512.
    concept_subset : list of str, optional
        Subset of concepts to retain. Default: ``None`` (``pneumothorax``).
    workers : int, optional
        Number of data-loading worker processes. Default: 0.
    seed : int, optional
        Seed for patient grouping and ``max_samples`` subsampling. Default:
        ``None``.

    Examples
    --------
    >>> from torch_concepts.data import PneumothoraxDataModule
    >>> dm = PneumothoraxDataModule(root="./data/pneumothorax", batch_size=8, seed=42)
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        root: str = None,
        image_size: int = 224,
        splitter: Splitter = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        concept_subset: list | None = None,
        workers: int = 0,
        seed: int | None = None,
        **kwargs,
    ):
        dataset = PneumothoraxDataset(
            root=root,
            image_size=image_size,
            concept_subset=concept_subset,
        )
        if splitter is None:
            train_indices, val_indices, test_indices = _patient_grouped_split(
                dataset.metadata["patient_id"].to_numpy(),
                val_size=val_size,
                test_size=test_size,
                seed=seed,
            )
            splitter = FixedIndicesSplitter(
                train_indices, val_indices, test_indices
            )
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            workers=workers,
            splitter=splitter,
            seed=seed,
            **kwargs,
        )
