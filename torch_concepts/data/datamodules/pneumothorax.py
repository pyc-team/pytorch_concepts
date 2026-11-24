"""DataModule for SIIM-ACR Pneumothorax images."""
from ..base.datamodule import ConceptDataModule
from ..base.splitter import Splitter
from ..datasets.pneumothorax import PneumothoraxDataset


class PneumothoraxDataModule(ConceptDataModule):
    """DataModule for SIIM-ACR Pneumothorax chest radiographs.

    The labeled Stage 2 training set contains both Stage 1 training and test
    images. It has no official labeled train/validation/test split, so the
    default is a random image split. The DICOM patient IDs are unique per image,
    so they cannot be used to group images by patient.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored or downloaded. Default:
        ``None`` (auto-creates ``./data/pneumothorax``).
    image_size : int, optional
        Side length (px) to resize images to. Default: 224.
    splitter : Splitter, optional
        Splitting strategy. Default: ``None``, which creates a random image-level
        split sized by ``val_size`` and ``test_size``.
    val_size : int or float, optional
        Validation size as a fraction or image count. Default: 0.1.
    test_size : int or float, optional
        Test size as a fraction or image count. Default: 0.2.
    batch_size : int, optional
        Number of samples per batch. Default: 512.
    concept_subset : list of str, optional
        Subset of concepts to retain. Default: ``None`` (``pneumothorax``).
    workers : int, optional
        Number of data-loading worker processes. Default: 0.
    seed : int, optional
        Seed for the random split and ``max_samples`` subsampling. Default:
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
