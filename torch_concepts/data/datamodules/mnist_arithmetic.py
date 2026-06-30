from typing import Tuple

from ..datasets.mnist_arithmetic import MNISTArithmeticDataset

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
    num_train_samples : int, optional
        Number of composite samples from MNIST train. Default: 10000
    num_test_samples : int, optional
        Number of composite samples from MNIST test. Default: 2000
    val_size : float, optional
        Fraction of MNIST-train composites for validation. Default: 0.1
    img_size : int, optional
        Output image size (square). Default: 224
    seed : int, optional
        Random seed for the train/val/test split. Default: 42
    generation_seed : int, optional
        Random seed for data generation. Default: 42
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
    workers : int, optional
        Number of data loading workers. Default: 0

    Examples
    --------
    >>> from torch_concepts.data import MNISTArithmeticDataModule
    >>>
    >>> dm = MNISTArithmeticDataModule(
    ...     num_train_samples=1000,
    ...     num_test_samples=200,
    ...     img_size=224,
    ...     batch_size=32, seed=42,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        root: str = None,
        num_train_samples: int = 10000,
        num_test_samples: int = 2000,
        val_size: float = 0.1,
        img_size: int = 224,
        seed: int = 42,
        generation_seed: int = 42,
        splitter: Splitter = NativeSplitter(),
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = True,
        force_recompute: bool = False,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        dataset = MNISTArithmeticDataset(
            root=root,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            val_size=val_size,
            img_size=img_size,
            seed=generation_seed,
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
            seed=seed,
        )
