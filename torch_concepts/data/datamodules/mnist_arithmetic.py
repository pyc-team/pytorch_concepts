from typing import Tuple

from ..datasets import MNISTArithmeticDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..base.splitter import Splitter
from ..splitters import NativeSplitter


class MNISTArithmeticDataModule(ConceptDataModule):
    """DataModule for MNIST Arithmetic dataset with concept-based learning support.

    Handles data loading, splitting, and batching for the MNIST arithmetic
    composite image dataset. Training/validation composites use MNIST train
    digits while test composites use MNIST test digits, preventing digit-level
    leakage. Supports precomputing backbone embeddings.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored or will be generated.
        Default: None (auto-creates ``./data/mnist_arithmetic``).
    mnist_root : str, optional
        Root directory for MNIST download. Default: None (``./data``).
    num_train_samples : int, optional
        Number of composite samples from MNIST train. Default: 10000
    num_test_samples : int, optional
        Number of composite samples from MNIST test. Default: 2000
    val_size : float, optional
        Fraction of MNIST-train composites for validation. Default: 0.1
    img_size : int, optional
        Output image size (square). Default: 224
    operators : tuple of str, optional
        Arithmetic operators to sample from. Default: ('+', '-', 'x', '/')
    seed : int, optional
        Random seed for reproducibility. Default: 42
    splitter : Splitter, optional
        Splitting strategy. Default: NativeSplitter() (uses the native
        train/val/test mapping built from MNIST splits).
    batch_size : int, optional
        Number of samples per batch. Default: 512
    backbone : BackboneType, optional
        Backbone model for feature extraction. Default: None
    precompute_embs : bool, optional
        Whether to precompute and cache backbone embeddings. Default: True
    force_recompute : bool, optional
        If True, recompute embeddings even if cached. Default: False
    concept_subset : list of str, optional
        Subset of concept names to use. Default: None
    workers : int, optional
        Number of data loading workers. Default: 0

    Examples
    --------
    >>> from torch_concepts.data.datamodules import MNISTArithmeticDataModule
    >>>
    >>> dm = MNISTArithmeticDataModule(
    ...     num_train_samples=1000,
    ...     num_test_samples=200,
    ...     img_size=64,
    ...     batch_size=32, seed=42,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    See Also
    --------
    MNISTArithmeticDataset : The underlying dataset class
    ConceptDataModule : Parent class with common datamodule functionality
    """

    def __init__(
        self,
        root: str = None,
        mnist_root: str = None,
        num_train_samples: int = 10000,
        num_test_samples: int = 2000,
        val_size: float = 0.1,
        img_size: int = 224,
        operators: Tuple[str, ...] = ('+', '-', 'x', '/'),
        seed: int = 42,
        splitter: Splitter = NativeSplitter(),
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = True,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        dataset = MNISTArithmeticDataset(
            root=root,
            mnist_root=mnist_root,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            val_size=val_size,
            img_size=img_size,
            operators=operators,
            seed=seed,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
        )

        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=0.0,  # test size handled natively
            batch_size=batch_size,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
            splitter=splitter,
        )
