import os
import torch
from typing import Union
from torchvision.transforms import Compose

from ..datasets import ColorMNISTDataset

from ..base.datamodule import ConceptDataModule
from ..splitters.coloring import ColoringSplitter
from ...typing import BackboneType


class ColorMNISTDataModule(ConceptDataModule):
    """DataModule for the ColorMNIST dataset.

    Handles data loading, splitting, and batching for the ColorMNIST dataset
    with support for concept-based learning.
    
    Args:
        seed: Random seed for data generation and splitting.
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        concept_subset: Subset of concepts to use. If None, uses all concepts.
        label_descriptions: Dictionary mapping concept names to descriptions.
        backbone: Model backbone to use (if applicable).
        workers: Number of workers for dataloaders.
    """
    
    def __init__(
        self,
        seed, # seed for data generation
        root: str,
        transform: Union[Compose, torch.nn.Module] = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        task_type: str = 'classification',
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        coloring: dict | None = None,
    ):

        # add to coloring the field "percentages" according to the split, to generate data accordingly
        coloring['training_percentage'] = 1.0 - test_size
        coloring['test_percentage'] = test_size

        dataset = ColorMNISTDataset(
            root=root,
            seed=seed,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            task_type=task_type,
            transform=transform,
            coloring=coloring
        )
    
        splitter = ColoringSplitter(
            root=root,
            seed=seed,
            val_size=val_size,
            test_size=test_size
        )
        
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            task_type=task_type,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
            splitter=splitter
        )
