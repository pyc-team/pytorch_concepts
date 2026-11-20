"""DataModule for synthetic single-cell gene expression."""

from ..base.datamodule import ConceptDataModule
from ..datasets.single_cell import SingleCellDataset


class SingleCellDataModule(ConceptDataModule):
    """DataModule for synthetic single-cell gene expression from scCBGM.

    Generates gene counts with binary concept labels and prepares train,
    validation and test dataloaders.
    See :class:`SingleCellDataset` for the dataset description and concept editing
    and *scCBGM: Interpretable Single-Cell Counterfactual Editing* for more
    details: https://arxiv.org/abs/2606.07760.

    Parameters
    ----------
    seed : int
        Random seed for data splitting.
    root : str, optional
        Directory for generated data. Defaults to ./data/single_cell.
        A saved dataset directory is reloaded when generation settings are omitted.
    generation_seed : int, optional
        Random seed for data generation, independent of the split seed.
        Defaults to 42 when generating data.
    val_size, test_size : int or float
        Number or fraction of cells reserved for validation and testing.
    batch_size : int
        Number of cells per batch.
    concept_subset : list of str, optional
        Which factors to return as concept labels, e.g. ["concept_0", "concept_2"].
        Defaults to all concepts. Does not change the generated gene expression.
    label_descriptions : dict, optional
        Mapping from concept names to descriptions.
    workers : int
        Number of dataloader worker processes.
    splitter : Splitter, optional
        Train/val/test splitting strategy. Defaults to RandomSplitter.
    generation_config : dict, optional
        Settings for the generation described in :class:`SingleCellDataset`,
        in a flat dictionary, e.g. {"n_cells": 1000, "n_concepts": 5}.
        Set ``n_cells`` and ``n_genes`` here to control the generated dimensions.
        Omitted settings keep their defaults.
    exog_noise : array-like, optional
        Values controlling count sampling; see :class:`SingleCellDataset`.
    normalize : bool
        Scale each nonempty cell to ``target_sum`` total counts, then apply
        log1p. Defaults to True, as in the original scCBGM DataModule.
        Set False to use raw counts. Saved counts are unchanged.
    target_sum : float
        Positive, finite total count per cell before log1p. Defaults to 10,000.
        10,000 is a conventional scaling factor in single-cell RNA-seq data
        normalization.
        Passed to the dataset; generated variants retain both normalization
        settings. Ignored when ``normalize=False``.
    **kwargs
        Additional arguments for ConceptDataModule, such as scalers and
        pin_memory. ``max_samples`` must be None; set the number of generated
        cells with ``generation_config["n_cells"]``.

    Examples
    --------
    >>> dm = SingleCellDataModule(
    ...     generation_config={"n_cells": 1000, "n_genes": 100}, batch_size=64,
    ...     target_sum=20_000,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        seed: int = 42,
        root: str | None = None,
        generation_seed: int | None = None,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        batch_size: int = 512,
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        workers: int = 0,
        splitter=None,
        generation_config: dict | None = None,
        exog_noise=None,
        normalize: bool = True,
        target_sum: float = 10_000,
        **kwargs,
    ):
        if kwargs.pop("max_samples", None) is not None:
            raise ValueError(
                'SingleCellDataModule does not support max_samples. '
                'Set generation_config["n_cells"] to control the number of generated cells.'
            )
        dataset = SingleCellDataset(
            root=root,
            seed=generation_seed,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            generation_config=generation_config,
            exog_noise=exog_noise,
            normalize=normalize,
            target_sum=target_sum,
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
