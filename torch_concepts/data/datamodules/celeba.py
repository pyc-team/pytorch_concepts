from ..datasets import CelebADataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType
from ..splitters import StandardSplitter, RandomSplitter


class CelebADataModule(ConceptDataModule):
    """DataModule for CelebA dataset.

    Handles data loading, splitting, and batching for CelebA dataset
    with support for concept-based learning.
    
    Args:
        seed: Random seed for reproducibility.
        name: Dataset identifier (default: 'celeba').
        split: Dataset split to use ('train', 'valid', or 'test').
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        ftune_size: Fine-tuning set size (fraction or absolute count).
        ftune_val_size: Fine-tuning validation set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        download: Whether to download the dataset if not present.
        task_label: List of attributes to use as task labels.
        concept_subset: Subset of concepts to use. If None, uses all concepts.
        label_descriptions: Dictionary mapping concept names to descriptions.
        backbone: Model backbone to use (if applicable).
        workers: Number of workers for dataloaders.
        DATA_ROOT: Root directory for data storage.
    """
    
    def __init__(
        self,
        seed: int, # seed for reproducibility
        name: str, # dataset identifier
        root: str, # root directory for dataset
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        ftune_size: int | float = 0.0,
        ftune_val_size: int | float = 0.0,
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = True,
        force_recompute: bool = False,
        task_label: list | None = None,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        splitter: str = "standard",
        workers: int = 0,
        DATA_ROOT = None,
        **kwargs
    ):
        
        dataset = CelebADataset(
            name=name,
            root=root,
            transform=None,
            task_label=task_label,
            class_attributes=task_label,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions
        )

        # check configura
        if splitter== "standard":
            splitter = StandardSplitter()
        else: 
            splitter = RandomSplitter(
                val_size=val_size,
                test_size=test_size
            )
        
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
            splitter=splitter
        )
