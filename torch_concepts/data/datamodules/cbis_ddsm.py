"""Lightning data loading for the official CBIS-DDSM splits."""

from copy import copy
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ...tensor import AnnotatedTensor
from ..base.datamodule import ConceptDataModule
from ..datasets.cbis_ddsm import CBISDDSMDataset
from ..splitters import FixedIndicesSplitter
from ..utils import resolve_size


class CBISDDSMDataModule(ConceptDataModule):
    """Batch CBIS-DDSM images while preserving official train/test membership.

    Both official CSVs are loaded for the selected abnormality and image
    product. Validation patients are sampled only from the official training
    CSV; every observation of a patient stays in the same train/validation
    split. The official test CSV is used in full. Set ``val_size=0`` to train
    on the complete official training split. Batches include the final partial
    batch and follow :meth:`ConceptDataset.collate`'s dictionary format.

    Args:
        root: Dataset directory, as for :class:`CBISDDSMDataset`.
        abnormality_type: ``"mass"`` or ``"calcification"``.
        image_type: ``"full"``, ``"cropped"``, or ``"mask"``.
        transform: Image transform for validation/test and, by default,
            training. Native images have variable sizes; supply resizing or
            padding before batching multiple images.
        train_transform: Optional replacement transform for training images.
            It receives the native decoded tensor, like ``transform``.
        concept_subset: Concept labels or annotation group names to retain.
        val_size: Fraction or number of training patients held out for
            validation. Defaults to 0.1.
        batch_size: Number of images per batch. Defaults to 32.
        workers: Number of data-loading processes. Defaults to 0.
        seed: Seed for the patient split and optional subsampling.
        **kwargs: Additional :class:`ConceptDataModule` arguments, including
            ``pin_memory``, ``max_samples``, ``scalers``, or a custom
            ``splitter``. Custom splitters override the official membership;
            ``max_samples`` subsamples across all splits.

    Native annotations, including the three pathology states and unknown
    descriptor values, are unchanged. Task-specific label projections belong
    in the model or example.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        abnormality_type: str = "mass",
        image_type: str = "cropped",
        transform: Optional[Callable] = None,
        train_transform: Optional[Callable] = None,
        concept_subset: Optional[list[str]] = None,
        val_size: int | float = 0.1,
        batch_size: int = 32,
        workers: int = 0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        train, test = [CBISDDSMDataset(
            root=root, abnormality_type=abnormality_type, split=split,
            image_type=image_type, transform=transform,
            concept_subset=concept_subset,
        ) for split in ("train", "test")]

        # Keep the dataset's DICOM decoder and row-aligned metadata. Only paths
        # and annotations are joined; image pixels remain on disk.
        dataset = copy(train)
        dataset.input_data = train.input_data + test.input_data
        dataset.metadata = pd.concat([train.metadata, test.metadata], ignore_index=True)
        dataset.set_concepts(AnnotatedTensor(
            torch.cat([train.native_concepts.tensor, test.native_concepts.tensor]),
            train.native_concepts.annotations, axis=1,
        ))

        splitter = kwargs.pop("splitter", None)
        if splitter is None:
            patient_ids = train.metadata["patient_id"].str.strip().to_numpy()
            patients = np.unique(patient_ids)
            np.random.default_rng(seed).shuffle(patients)
            n_val = resolve_size(val_size, len(patients))
            if n_val >= len(patients):
                raise ValueError("val_size must leave at least one training patient.")
            validation = np.isin(patient_ids, patients[:n_val])
            splitter = FixedIndicesSplitter(
                train_idxs=np.flatnonzero(~validation).tolist(),
                val_idxs=np.flatnonzero(validation).tolist(),
                test_idxs=range(len(train), len(dataset)),
            )

        self.train_transform = train_transform
        super().__init__(
            dataset=dataset, splitter=splitter, batch_size=batch_size,
            workers=workers, seed=seed, **kwargs,
        )

    def setup(self, stage=None) -> None:
        super().setup(stage)
        if self.train_transform is not None and self.trainset is not None:
            # A separate view keeps augmentation out of validation/test and
            # still shares any precomputed embeddings and concept sources.
            training = copy(self.dataset)
            training.transform = self.train_transform
            self.trainset = Subset(training, self.trainset.indices)

    def get_dataloader(self, split=None, shuffle=False, batch_size=None):
        """Create a loader without dropping observations from any split."""
        if split not in (None, "train", "val", "test"):
            raise ValueError("split must be 'train', 'val', 'test', or None.")
        dataset = self.dataset if split is None else getattr(self, f"{split}set")
        if dataset is None:
            return None
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=shuffle,
            num_workers=self.workers, pin_memory=self.pin_memory,
            collate_fn=self.dataset.collate,
        )

    def val_dataloader(self, shuffle=False, batch_size=None):
        """Return an empty loader list when validation is disabled."""
        loader = self.get_dataloader("val", shuffle, batch_size)
        return loader if loader is not None else []
