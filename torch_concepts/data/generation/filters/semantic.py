"""Semantic concept vocabulary filtering."""
from typing import Callable, Sequence

import torch
from torch.nn import functional as F

from torch_concepts import Annotations
from ..base.filter_generator import FilterGenerator


class SemanticConceptFilter(FilterGenerator):
    """Filter concept vocabularies by class and pairwise text similarity.

    Removes long names, class aliases, and semantically similar concepts.
    By default, preserves the exact LF-CBM reference rules for compatibility.

    Supply ``clip_text_encoder(texts)`` using CLIP ViT-B/16 and
    ``sentence_encoder(texts)`` using all-mpnet-base-v2 to reproduce the authors'
    conceptset_utils.py. Both return sample-by-feature tensors/arrays; CLIP
    features are normalized here. MPNet output is used as returned, matching
    the reference. No models or optional text dependencies load implicitly.

    Concepts are sorted alphabetically, as in the reference class filter.
    Similarities combine (MPNet + 3 * CLIP) / 4. Pairwise pruning keeps the
    concept with greater summed similarity, with ties favoring the earlier one.
    For numerical reference compatibility, the class comparison also preserves
    its ``class_index != concept_index`` condition, despite the unrelated axes.
    Set ``reference_class_filter=False`` to compare every class/concept pair,
    including matching indices. The default index exception is a reproduction
    quirk, not a general semantic filtering rule. Output order is alphabetical
    in both modes. This filter is optional for published, already-filtered lists.
    """

    def __init__(
        self, class_names: Sequence[str], clip_text_encoder: Callable,
        sentence_encoder: Callable, *, max_length: int = 30,
        class_cutoff: float = .85, similarity_cutoff: float = .9,
        reference_class_filter: bool = True,
    ):
        self.class_names = list(class_names)
        self.clip_text_encoder = clip_text_encoder
        self.sentence_encoder = sentence_encoder
        self.max_length = max_length
        self.class_cutoff = class_cutoff
        self.similarity_cutoff = similarity_cutoff
        self.reference_class_filter = reference_class_filter

    def _similarities(self, left, right):
        with torch.no_grad():
            a = F.normalize(torch.as_tensor(self.clip_text_encoder(left)).float().cpu(), dim=1)
            b = F.normalize(torch.as_tensor(self.clip_text_encoder(right)).float().cpu(), dim=1)
            c = torch.as_tensor(self.sentence_encoder(left)).float().cpu()
            d = torch.as_tensor(self.sentence_encoder(right)).float().cpu()
        return (c @ d.T + 3 * (a @ b.T)) / 4

    def filter(self, concepts: Annotations) -> Annotations:
        aliases = set()
        for name in self.class_names:
            aliases.update(prefix + name for prefix in ("", "a ", "A ", "an ", "An ", "the ", "The "))
            aliases.update((name.upper(), name[:1].upper() + name[1:]))
        labels = sorted({name for name in concepts.labels if len(name) <= self.max_length and name not in aliases})
        if not labels:
            raise ValueError("No concepts survived the length and class-alias filters.")
        similarity = self._similarities(self.class_names, labels)
        keep = [
            name for j, name in enumerate(labels)
            if not any(similarity[i, j] >= self.class_cutoff
                       and (not self.reference_class_filter or i != j)
                       for i in range(len(self.class_names)))
        ]
        if not keep:
            raise ValueError("No concepts survived the class-similarity filter.")
        similarity = self._similarities(keep, keep)
        generality = similarity.sum(1)
        removed = set()
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                if i not in removed and j not in removed and similarity[i, j] >= self.similarity_cutoff:
                    removed.add(i if generality[i] < generality[j] else j)
        return concepts.subset([name for i, name in enumerate(keep) if i not in removed])
