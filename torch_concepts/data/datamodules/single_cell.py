"""DataModule for synthetic single-cell gene counts."""

from ..base.datamodule import ConceptDataModule
from ..datasets.single_cell import SingleCellDataset


class SingleCellDataModule(ConceptDataModule):
    """DataModule for synthetic single-cell datasets.

    Handles data loading, splitting, and batching for the synthetic single-cell
    dataset. Global ``model_parameters`` and realized ``cell_background`` are
    saved together with ``generation_config`` in one processed file. With
    ``max_samples``, the in-memory concepts and cell background follow the
    selected dataset rows.
    Only binary indicators are concepts; celltype, tissue and batch belong to
    the fixed cell background.
    Paper: Andersson, A., et al. (2026). Single-Cell Concept Bottleneck Generative Models for Interpretable and Controllable Cellular Editing.
    In ICLR 2026 Workshop on Machine Learning for Genomics Explorations. https://arxiv.org/pdf/2606.07760.

    Args:
        seed: Random seed for the train/val/test split.
        root: Directory for processed dataset files. If None, uses
            './data/single_cell'.
        generation_seed: Random seed for data generation.
        val_size: Validation set size (fraction or absolute count).
        test_size: Test set size (fraction or absolute count).
        batch_size: Batch size for dataloaders.
        concept_subset: Subset of concepts to use. If None, uses all concepts.
        label_descriptions: Dictionary mapping concept names to descriptions.
        generation_config: Partial counts and distribution parameters grouped
            by quantity: cells, genes, batch, celltype, tissue, original_concept,
            baseline, library, noise.
            Omitted parameters use the defaults documented in
            :class:`~torch_concepts.data.datasets.single_cell.SingleCellDataset`.
            For example, {"cells": {"n": 1000}, "batch": {"n": 2, "std": 0.15}}
            generates 1000 cells with two batches and stronger batch effects.
        workers: Number of workers for dataloaders.
        exog_noise: Optional Poisson quantiles forwarded to SingleCellDataset.
        splitter: Custom train/val/test splitting strategy. If None, uses
            RandomSplitter.
        **kwargs: Additional arguments for ConceptDataModule, such as scalers
            and max_samples.
    """

    def __init__(
        self,
        seed: int = 42,
        root: str | None = None,
        generation_seed: int = 42,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        splitter=None,
        generation_config: dict | None = None,
        exog_noise=None,
        **kwargs,
    ):
        dataset = SingleCellDataset(
            root=root,
            seed=generation_seed,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            generation_config=generation_config,
            exog_noise=exog_noise,
        )
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            workers=workers,
            splitter=splitter,
            seed=seed,
            **kwargs,
        )
