"""
Annotated sequential container for concept-based pipelines.
"""
from typing import Optional, Union

import torch

from torch_concepts.annotations import AxisAnnotation
from torch_concepts.tensor import AnnotatedTensor


class Sequential(torch.nn.Sequential):
    """``nn.Sequential`` whose **first** module may take multiple inputs.

    Standard ``nn.Sequential`` threads one tensor through the chain, so its first
    layer cannot be a multi-input PyC layer such as
    :class:`~torch_concepts.nn.MixConceptEmbeddingToConcept` (``forward(concepts,
    embeddings)``). This subclass forwards **all** of its inputs to the first
    module, then threads that module's single output through the rest — while a
    single-tensor ``seq(x)`` still behaves exactly like ``nn.Sequential``.

    If ``out_concepts`` (an :class:`~torch_concepts.AxisAnnotation`) is set,
    :meth:`annotate` wraps an output in an
    :class:`~torch_concepts.tensor.AnnotatedTensor` to label its columns.
    """

    def __init__(self, *args, out_concepts: Optional[AxisAnnotation] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_concepts = out_concepts

    def forward(self, *args, **kwargs):
        it = iter(self)
        try:
            output = next(it)(*args, **kwargs)  # first layer takes all inputs
        except StopIteration:  # empty container: mirror nn.Sequential's identity
            return args[0] if len(args) == 1 and not kwargs else None
        for module in it:
            output = module(output)  # the rest are single-tensor
        return output
    
    def annotate(self, x, out_concepts: AxisAnnotation = None) -> AnnotatedTensor:
        if out_concepts is None:
            if isinstance(self.out_concepts, AxisAnnotation):
                out_concepts = self.out_concepts
            else:
                return x
        return AnnotatedTensor(x, out_concepts)