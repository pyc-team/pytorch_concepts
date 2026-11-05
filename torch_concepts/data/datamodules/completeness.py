from ..datasets.toy import CompletenessDataset

from ..base.datamodule import ConceptDataModule


class CompletenessDataModule(ConceptDataModule):
    """DataModule for :class:`CompletenessDataset`.

    Args:
        seed: Random seed for the train/val/test split.
        generation_seed: Random seed for data generation.
        root: Unused, the data is generated in memory (conceptarium sets it
            for every dataset).
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        max_samples: Optional subsample size, as in ``ConceptDataModule``.
        scalers: Optional scalers, as in ``ConceptDataModule``.
        workers: Number of workers for dataloaders.
        **dataset_kwargs: Arguments of :class:`CompletenessDataset`.
    """

    def __init__(
        self,
        seed: int = 42,
        generation_seed: int = 42,
        root: str | None = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        max_samples: int | None = None,
        scalers: dict | None = None,
        workers: int = 0,
        **dataset_kwargs,
    ):
        super().__init__(
            dataset=CompletenessDataset(seed=generation_seed, **dataset_kwargs),
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            max_samples=max_samples,
            scalers=scalers,
            workers=workers,
            seed=seed,
        )
