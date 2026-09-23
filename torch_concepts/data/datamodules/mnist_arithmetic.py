from typing import List, Optional, Sequence

from ..datasets.mnist_arithmetic import MNISTArithmeticDataset, DEFAULT_OPERATORS

from ..base.datamodule import ConceptDataModule
from ..base.splitter import Splitter
from ..splitters import NativeSplitter


class MNISTArithmeticDataModule(ConceptDataModule):
    """DataModule for MNIST Arithmetic dataset with concept-based learning support.

    Handles data loading, splitting, and batching for the MNIST arithmetic
    composite image dataset. Training/validation composites use MNIST train
    digits while test composites use MNIST test digits, preventing digit-level
    leakage.

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
    test_size : int or float, optional
        Ignored by the default splitter, which uses MNIST's own test split;
        it only applies when a size-based ``splitter`` is passed instead.
        Default: 0.2
    img_size : int, optional
        Output image size (square). Default: 224
    seed : int, optional
        Random seed for the train/val/test split. Default: 42
    generation_seed : int, optional
        Random seed for data generation. Default: 42
    splitter : Splitter, optional
        Splitting strategy. Default: NativeSplitter() (uses the native
        train/val/test mapping built from MNIST splits).
    operators : sequence of str, optional
        Operator symbols composites are drawn from.
        Default: ``('+', '-', 'x', '/')``.
    batch_size : int, optional
        Number of samples per batch. Default: 512
    concept_subset : list of str, optional
        Subset of concept names to use. Default: None
    workers : int, optional
        Number of data loading workers. Default: 0
    **kwargs
        Forwarded to :class:`ConceptDataModule` (e.g. ``max_samples``,
        ``scalers``, ``pin_memory``).

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
        test_size: int | float = 0.2,
        img_size: int = 224,
        seed: int = 42,
        generation_seed: int = 42,
        splitter: Splitter = NativeSplitter(),
        operators: Sequence[str] = DEFAULT_OPERATORS,
        batch_size: int = 512,
        concept_subset: Optional[List[str]] = None,
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
            operators=operators,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
        )

        super().__init__(
            dataset=dataset,
            val_size=val_size,
            # Declared rather than hard-coded: the default splitter takes the
            # test split from MNIST's own test digits and ignores this, but a
            # caller (or a config) may still pass it for a size-based splitter.
            test_size=test_size,
            batch_size=batch_size,
            workers=workers,
            splitter=splitter,
            seed=seed,
            **kwargs,
        )
