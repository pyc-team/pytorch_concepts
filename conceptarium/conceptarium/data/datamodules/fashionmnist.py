from env import CACHE
import torch
from typing import Union
from torchvision.transforms import Compose

from torch_concepts.data import FashionMNISTDataset

from ..base.datamodule import ConceptDataModule
from ..splitters.coloring import ColoringSplitter
from ...typing import BackboneType


class FashionMNISTDataModule(ConceptDataModule):
    """DataModule for the FashionMNIST dataset.

    Handles data loading, splitting, and batching for the FashionMNIST dataset
    with support for concept-based learning.
    
    Args:
        seed: Random seed for data generation and splitting.
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        ftune_size: Fine-tuning set size (fraction or absolute count).
        ftune_val_size: Fine-tuning validation set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        concept_subset: Subset of concepts to use. If None, uses all concepts.
        label_descriptions: Dictionary mapping concept names to descriptions.
        backbone: Model backbone to use (if applicable).
        workers: Number of workers for dataloaders.
    """
    
    def __init__(
        self,
        seed, # seed for data generation
        transform: Union[Compose, torch.nn.Module] = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        ftune_size: int | float = 0.0,
        ftune_val_size: int | float = 0.0,
        batch_size: int = 512,
        task_type: str = 'classification',
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        coloring: dict | None = None
    ):

        # add to coloring the field "percentages" according to the split, to generate data accordingly
        coloring['training_percentage'] = 1.0 -  test_size - ftune_size - ftune_val_size
        coloring['test_percentage'] = test_size + ftune_size + ftune_val_size

        dataset = FashionMNISTDataset(root=str(CACHE / "fashionmnist"),
                       seed=seed,
                       concept_subset=concept_subset,
                       label_descriptions=label_descriptions,
                       task_type=task_type,
                       transform=transform,
                       coloring=coloring
                       )

        splitter = ColoringSplitter(root=str(CACHE / "fashionmnist"),
                                    seed=seed,
                                    val_size=val_size,
                                    test_size=test_size,
                                    ftune_size=ftune_size,
                                    ftune_val_size=ftune_val_size
                                    )
        
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            ftune_size=ftune_size,
            ftune_val_size=ftune_val_size,
            batch_size=batch_size,
            task_type=task_type,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
            splitter=splitter
        )
