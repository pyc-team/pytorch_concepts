"""Standard splitting and loading for UniRef50 + Swiss-Prot."""

from ..base.datamodule import ConceptDataModule
from ..datasets.uniref_swissprot import UniRefSwissProtDataset


class UniRefSwissProtDataModule(ConceptDataModule):
    """ConceptDataModule using disk-backed protein samples.

    Uses the standard library splitter and DataLoaders. Sequences and concepts
    stay on disk; split indices and shuffle permutations use RAM.

    Args:
        root: Directory for source files and SQLite cache.
        seed: Seed for the train/validation/test partition.
        max_length: Sequence truncation length, default 512.
        max_sequences: Records read per source; None uses all records.
        val_size: Validation fraction or count, default 0.05.
        test_size: Test fraction or count, default 0.05.
        batch_size: Samples per batch, default 64.
        concept_subset: Optional concept names to retain.
        workers: Loader processes, default 0.
        splitter: Optional library splitter; defaults to RandomSplitter.
        include_uniref: False reads only Swiss-Prot.
        one_hot_encoding: True uses one binary indicator per category found in
            the data; False uses six class indices (UNKNOWN-only groups are
            constant binary columns). Both modes include 14 continuous properties.
        **kwargs: Forwarded to ConceptDataModule, including pin_memory,
            max_samples and scalers, with their existing behavior.
    """

    def __init__(
        self, root=None, seed=42, max_length=512, max_sequences=None,
        val_size=0.05, test_size=0.05, batch_size=64, concept_subset=None,
        workers=0, splitter=None, include_uniref=True, one_hot_encoding=True, **kwargs,
    ):
        dataset = UniRefSwissProtDataset(
            root=root, max_length=max_length, max_sequences=max_sequences,
            concept_subset=concept_subset, include_uniref=include_uniref,
            one_hot_encoding=one_hot_encoding,
        )
        super().__init__(
            dataset=dataset, val_size=val_size, test_size=test_size,
            batch_size=batch_size, workers=workers, splitter=splitter, seed=seed, **kwargs,
        )
