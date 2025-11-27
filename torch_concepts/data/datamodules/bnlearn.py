import os

from ..datasets import BnLearnDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType


class BnLearnDataModule(ConceptDataModule):
    """DataModule for all Bayesian Network datasets.

    Handles data loading, splitting, and batching for all Bayesian Network datasets
    with support for concept-based learning.
    
    Args:
        seed: Random seed for data generation and splitting.
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        n_samples: Total number of samples to generate.
        autoencoder_kwargs: Configuration for autoencoder-based feature extraction.
        concept_subset: Subset of concepts to use. If None, uses all concepts.
        label_descriptions: Dictionary mapping concept names to descriptions.
        backbone: Model backbone to use (if applicable).
        workers: Number of workers for dataloaders.
    """
    
    def __init__(
        self,
        seed: int, # seed for data generation
        name: str, # name of the bnlearn DAG
        root: str,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        n_gen: int = 10000,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        autoencoder_kwargs: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        dataset = BnLearnDataset(
            name=name,
            root=root,
            seed=seed,
            n_gen=n_gen,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            autoencoder_kwargs=autoencoder_kwargs
        )
        
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers
        )
