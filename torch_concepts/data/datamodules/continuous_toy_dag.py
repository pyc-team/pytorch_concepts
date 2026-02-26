"""
DataModule for ToyFunctionDAG datasets.
"""
from typing import Dict, List, Tuple, Optional

from ..datasets import ToyFunctionDAGDataset
from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType

class ToyFunctionDAGDataModule(ConceptDataModule):
    """DataModule for function-based DAG synthetic datasets.
    
    Handles data loading, splitting, and batching for DAG-based synthetic datasets
    where child nodes are computed from parents using sympy expressions.
    
    Args:
        variables: List of all variable names in the DAG.
        dag: List of edges representing the DAG structure as (parent, child) tuples.
        node_functions: Dictionary mapping node names to sympy expression strings.
        variable_type: 'binary' or 'continuous'.
        cardinalities: For binary variables, dictionary mapping names to cardinalities.
        source_mean: For continuous, mean for sampling root nodes.
        source_std: For continuous, std for sampling root nodes.
        gamma: Noise parameter (continuous: additive, binary: swap probability).
        theta: Embedding noise multiplier.
        seed: Random seed for data generation and splitting.
        root: Root directory to store/load the dataset.
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        backbone: Model backbone to use (if applicable).
        precompute_embs: Whether to precompute embeddings from backbone.
        force_recompute: Force recomputation of cached embeddings.
        n_gen: Total number of samples to generate.
        latent_variables: List of latent variable names.
        concept_subset: Subset of concepts to use.
        label_descriptions: Dictionary mapping concept names to descriptions.
        autoencoder_kwargs: Configuration for autoencoder-based feature extraction.
        workers: Number of workers for dataloaders.
    """
    
    def __init__(
        self,
        variables: List[str],
        dag: List[Tuple[str, str]],
        node_functions: Dict[str, str],
        variable_type: str = 'continuous',
        cardinalities: Optional[Dict[str, int]] = None,
        source_mean: float = 0.0,
        source_std: float = 1.0,
        gamma: float = 0.0,
        theta: float = 0.0,
        seed: int = 42,
        root: str = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        n_gen: int = 10000,
        latent_variables: Optional[List[str]] = None,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        autoencoder_kwargs: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        # Create dataset
        dataset = ToyFunctionDAGDataset(
            variables=variables,
            dag=dag,
            node_functions=node_functions,
            variable_type=variable_type,
            cardinalities=cardinalities,
            source_mean=source_mean,
            source_std=source_std,
            gamma=gamma,
            theta=theta,
            root=root,
            seed=seed,
            n_gen=n_gen,
            latent_variables=latent_variables,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            autoencoder_kwargs=autoencoder_kwargs,
        )
        
        # Initialize parent class
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
