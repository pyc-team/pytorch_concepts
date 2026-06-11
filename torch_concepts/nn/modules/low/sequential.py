"""A ``Sequential`` container whose first module may take multiple inputs."""

import torch


class Sequential(torch.nn.Sequential):
    r"""``nn.Sequential`` whose **first** module may take ``concepts`` and/or
    ``embeddings`` instead of a single tensor.

    Standard ``nn.Sequential`` threads one tensor through every child, so it
    cannot host a PyC layer like
    :class:`~torch_concepts.nn.MixConceptEmbeddingToConcept` or
    :class:`~torch_concepts.nn.HyperlinearConceptEmbeddingToConcept` whose
    ``forward(concepts, embeddings)`` takes two inputs. This subclass forwards
    all of its inputs to the first module, then threads that module's single
    output through the remaining (plain, single-tensor) modules exactly like
    ``nn.Sequential``.

    It exposes ``concepts``/``embeddings`` explicitly so that
    :class:`~torch_concepts.nn.ParametricFactor` recognises it as a PyC layer
    (see ``_PYC_PARAM_SETS``) and calls it as
    ``module(concepts=..., embeddings=...)``. Whichever input the aggregator
    does not supply is dropped before the first module is called (so a first
    layer that takes only one of them works), and a single positional tensor is
    also accepted, keeping it a drop-in ``nn.Sequential``.

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
    """

    def forward(self, *args, concepts=None, embeddings=None, **kwargs):
        # Re-assemble only the PyC inputs that were actually supplied; the names
        # exist in the signature so the factor recognises this as a PyC layer,
        # but an absent one must not be forwarded (the first layer may take only
        # one of them).
        named = {}
        if concepts is not None:
            named["concepts"] = concepts
        if embeddings is not None:
            named["embeddings"] = embeddings
        modules = list(self)
        output = modules[0](*args, **named, **kwargs)
        for module in modules[1:]:
            output = module(output)
        return output
