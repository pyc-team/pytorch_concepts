"""A ``Sequential`` container whose first module may take multiple inputs."""

import torch


class Sequential(torch.nn.Sequential):
    r"""``nn.Sequential`` whose **first** module may take more than one input.

    Plain ``torch.nn.Sequential`` threads a single tensor through every child,
    so it cannot host a PyC layer like
    :class:`~torch_concepts.nn.MixConceptEmbeddingToConcept` or
    :class:`~torch_concepts.nn.HyperlinearConceptEmbeddingToConcept` whose
    ``forward(concepts, embeddings)`` takes two inputs.

    This subclass simply forwards **all** of its inputs — positional or keyword —
    to the *first* module, then threads that module's single output through the
    remaining (plain, single-tensor) modules. That one rule makes it a superset
    of ``nn.Sequential``:

    - ``net(x)`` behaves exactly like ``nn.Sequential`` (single tensor);
    - ``net(concepts=c, embeddings=e)`` feeds a multi-input PyC first layer.

    As a :class:`~torch_concepts.nn.ParametricCPD` parametrization it adapts
    automatically: the factor reads its input signature from the first layer
    (see ``_module_input_names``), so a PyC first layer makes the whole chain a
    PyC layer and a standard first layer makes it a standard one.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import (
        ...     Sequential, HyperlinearConceptEmbeddingToConcept)
        >>> net = Sequential(
        ...     HyperlinearConceptEmbeddingToConcept(in_concepts=4, in_embeddings=8),
        ...     torch.nn.Sigmoid(),
        ... )
        >>> out = net(concepts=torch.randn(2, 4), embeddings=torch.randn(2, 3, 8))
        >>> out.shape
        torch.Size([2, 3])
        >>> # ...and still a drop-in single-input Sequential:
        >>> Sequential(torch.nn.Linear(5, 3), torch.nn.ReLU())(torch.randn(2, 5)).shape
        torch.Size([2, 3])
    """

    def forward(self, *args, **kwargs):
        it = iter(self)
        try:
            output = next(it)(*args, **kwargs)  # first layer takes all inputs
        except StopIteration:  # empty container: mirror nn.Sequential's identity
            return args[0] if len(args) == 1 and not kwargs else None
        for module in it:
            output = module(output)  # the rest are single-tensor
        return output
