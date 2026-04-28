import os

from ..datasets.toy import CompletenessDataset

from ..base.datamodule import ConceptDataModule
from ...typing import BackboneType


class CompletenessDataModule(ConceptDataModule):
    """
    """
    
    def __init__(
        self,
        name: str, # name of the bnlearn DAG
        root: str,
        seed: int = 42, # seed for data generation
        p: int = 2,  # dimensionality of each view
        n_views: int = 10,  # number of views
        n_concepts: int = 2,  # number of concepts
        n_hidden_concepts: int = 0,  # number of hidden concepts
        n_tasks: int = 1,  # number of tasks
        n_gen: int = 10000,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        **kwargs
    ):
        dataset = CompletenessDataset(
            name=name,
            root=root,
            seed=seed,
            p=p,
            n_views=n_views,
            n_concepts=n_concepts,
            n_hidden_concepts=n_hidden_concepts,
            n_tasks=n_tasks,
            n_gen=n_gen,
            concept_subset=concept_subset
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
