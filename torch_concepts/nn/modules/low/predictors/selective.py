"""Linear predictor that reads only the concepts it is built with, by name."""
from typing import Union

import torch

from torch_concepts import Annotations, AnnotatedTensor
from ..base.layer import BaseConceptLayer


class SelectiveLinearConceptToConcept(BaseConceptLayer):
    """Linear predictor over a selection of concepts, read by name.

    Built with the annotations of the concepts it may use, it takes a whole concept
    tensor (e.g. the full bottleneck, as an :class:`AnnotatedTensor`) and uses only
    their columns, in their declared order. A concept it does not declare has exactly
    zero effect on its output, with no mask or index bookkeeping. The selection can
    come from any attribute of the annotations: names (e.g. a task's parents in a
    concept graph), types or cardinalities.

    The input must describe its concepts as the layer does: a concept with the same
    name but another type or cardinality (e.g. a binary concept now continuous, or a
    categorical concept with other states) raises, rather than being read as if it
    were the same concept. Columns in another order are realigned by name.

    Args:
        in_concepts: Annotations of the concepts the layer reads.
        out_concepts: Number or annotations of the output concepts.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations, AnnotatedTensor
        >>> from torch_concepts.nn import SelectiveLinearConceptToConcept
        >>> bottleneck = AnnotatedTensor(torch.rand(8, 3), Annotations(labels=['a', 'b', 'c']))
        >>> layer = SelectiveLinearConceptToConcept(in_concepts=Annotations(labels=['c', 'a']), out_concepts=1)
        >>> layer(bottleneck).shape  # reads only 'c' and 'a'
        torch.Size([8, 1])
    """

    def __init__(
        self, 
        in_concepts: Annotations, 
        out_concepts: Union[int, Annotations]
    ):
        assert isinstance(in_concepts, Annotations), \
            f"in_concepts must be an Annotations, got {type(in_concepts).__name__}"
        super().__init__(in_concepts=in_concepts, out_concepts=out_concepts)

        self.linear = torch.nn.Linear(self.in_concepts_shape, self.out_concepts_shape)
        self._checked = None  # the last input annotation found to match: later batches skip the check

    def forward(self, concepts: AnnotatedTensor) -> torch.Tensor:
        assert isinstance(concepts, AnnotatedTensor), \
            f"concepts must be an AnnotatedTensor, got {type(concepts).__name__}"
        selected = concepts[self.in_concepts]
        if selected.annotations is not self._checked:
            for field in ("types", "cardinalities"):
                got, expected = getattr(selected.annotations, field), getattr(self.in_concepts, field)
                if list(got) != list(expected):
                    raise ValueError(f"{type(self).__name__}: the input's {field} of {self.in_concepts.labels} "
                                     f"are {list(got)}, but the layer was built for {list(expected)}.")
            self._checked = selected.annotations
        return self.linear(selected.tensor)
