"""
Annotated sequential container for concept-based pipelines.
"""
from typing import Optional, Union

import torch

from torch_concepts.annotations import AxisAnnotation
from torch_concepts.tensor import AnnotatedTensor


class ConceptSequential(torch.nn.Sequential):
    """
    A :class:`torch.nn.Sequential` that optionally annotates its output.

    Drop-in replacement for ``torch.nn.Sequential``: accepts the same
    constructor arguments (positional modules or an ``OrderedDict``), plus
    an ``out_annotation`` keyword argument.  When ``out_annotation`` is
    provided, the tensor returned by ``forward`` is wrapped in an
    :class:`~torch_concepts.tensor.AnnotatedTensor` so that downstream code
    can refer to output columns by name.

    Args:
        *args: Modules passed directly to :class:`torch.nn.Sequential`.
        out_annotation: Optional :class:`~torch_concepts.AxisAnnotation`
            describing axis 1 of the output tensor.  When ``None`` (default)
            ``forward`` returns a plain :class:`torch.Tensor`.

    Example:
        >>> import torch
        >>> from torch_concepts import AxisAnnotation
        >>> from torch_concepts.nn.modules.low.sequential import ConceptSequential
        >>>
        >>> ann = AxisAnnotation(labels=["cat", "dog", "bird"])
        >>> pipeline = ConceptSequential(
        ...     torch.nn.Linear(8, 3),
        ...     torch.nn.Sigmoid(),
        ...     out_concepts=ann,
        ... )
        >>> out = pipeline(torch.rand(4, 8))
        >>> out.out_concepts.labels   # ['cat', 'dog', 'bird']
        >>> out.tensor.shape        # torch.Size([4, 3])
        >>>
        >>> # Without annotation — behaves exactly like nn.Sequential
        >>> plain = ConceptSequential(torch.nn.Linear(8, 3))
        >>> isinstance(plain(torch.rand(4, 8)), torch.Tensor)   # True
    """

    def __init__(self, *args, out_concepts: Optional[AxisAnnotation] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_concepts = out_concepts

    def annotate(self, x, out_concepts: AxisAnnotation = None) -> AnnotatedTensor:
        if out_concepts is None:
            if isinstance(self.out_concepts, AxisAnnotation):
                out_concepts = self.out_concepts
            else:
                return x
        return AnnotatedTensor(x, out_concepts)
