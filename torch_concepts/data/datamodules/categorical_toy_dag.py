"""
DataModule for ToyDAG synthetic datasets.
"""
import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from ..datasets.categorical_toy_dag import ToyDAGDataset
from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType


class ToyDAGDataModule(ConceptDataModule):
    """DataModule for ToyDAG synthetic datasets.
    
    Handles data loading, splitting, and batching for DAG-based synthetic datasets
    with support for concept-based learning.
    
    This datamodule wraps the ToyDAGDataset and provides standard train/val/test splits
    along with optional backbone feature extraction and embedding caching.
    
    Args:
        variables: List of all variable names in the DAG.
        cardinalities: Dictionary mapping variable names to their cardinality.
        dag: List of edges representing the DAG structure as (parent, child) tuples.
        conditional_probs: Dictionary mapping variables to their conditional probability tables.
        seed: Random seed for data generation and splitting.
        root: Root directory to store/load the dataset.
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        backbone: Model backbone to use (if applicable).
        precompute_embs: Whether to precompute embeddings from backbone.
        force_recompute: Force recomputation of cached embeddings.
        n_gen: Total number of samples to generate.
        target_variable: Name of the target variable (optional).
        latent_variables: List of latent variable names.
        concept_subset: Subset of concepts to use.
        label_descriptions: Dictionary mapping concept names to descriptions.
        autoencoder_kwargs: Configuration for autoencoder-based feature extraction.
        workers: Number of workers for dataloaders.
    """
    
    def __init__(
        self,
        variables: List[str],
        cardinalities: Dict[str, int],
        dag: List[Tuple[str, str]],
        conditional_probs: Optional[Dict[Union[Tuple[str, str], Tuple[str]], Union[np.ndarray, list]]] = None,
        seed: int = 42,
        root: str = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        n_gen: int = 10000,
        target_variable: Optional[str] = None,
        latent_variables: Optional[List[str]] = None,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        autoencoder_kwargs: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        # Create dataset
        dataset = ToyDAGDataset(
            variables=variables,
            cardinalities=cardinalities,
            dag=dag,
            conditional_probs=conditional_probs,
            root=root,
            seed=seed,
            n_gen=n_gen,
            target_variable=target_variable,
            latent_variables=latent_variables,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            autoencoder_kwargs=autoencoder_kwargs,
        )
        
        # Initialize parent class with default behavior
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
        )
