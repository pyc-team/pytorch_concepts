"""The standard activation for a distribution parameter.

A :class:`~torch_concepts.nn.modules.mid.factors.cpd.ParametricCPD` applies no
activation: every parametrization module must already emit a value in its
parameter's natural domain. That leaves the caller having to know which squashing
function each parameter needs — sigmoid for a ``Bernoulli``'s ``probs``, a
*per-member* softmax for a categorical's, softplus for a ``Normal``'s ``scale``,
a Cholesky assembly for a ``MultivariateNormal``'s ``scale_tril``, and nothing at
all for ``logits`` or ``loc``.

:class:`DefaultActivation` looks that answer up in the family's
:class:`~torch_concepts.nn.modules.mid.distributions.DistributionSpec` instead, so
a head is written as a raw layer composed with the variable it parametrizes::

    ParametricCPD(concepts, parametrization=Sequential(
        MLP(64, 128, concepts.size), DefaultActivation(concepts, 'probs')))

Adding a family therefore stays a single registry entry: declare its
``param_activations`` and every head built this way follows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .distributions import spec_for
from .variable import Variable


class DefaultActivation(nn.Module):
    """Map a raw network output into a distribution parameter's domain.

    The activation is resolved once, at construction, from the family's
    :attr:`~torch_concepts.nn.modules.mid.distributions.DistributionSpec.param_activations`
    and held as a child module — so it is part of ``state_dict()`` and follows
    ``.to(device)`` like any other layer.

    Parameters
    ----------
    variable : Variable
        The variable whose parameter is being produced.
    param : str
        The distribution parameter this activation produces (``'probs'``,
        ``'scale'``, ``'loc'``, ``'logits'``, ``'scale_tril'``, ``'value'``).
        Must be a parameter the variable's family accepts.

    Raises
    ------
    ValueError
        If ``param`` is not a parameter of the variable's distribution.

    Examples
    --------
    A Bernoulli head, in a plain ``nn.Sequential``:

    >>> import torch
    >>> from torch.distributions import Bernoulli, OneHotCategorical
    >>> from torch_concepts.nn import ConceptVariable, DefaultActivation
    >>> c = ConceptVariable('c', distribution=Bernoulli, size=3)
    >>> head = torch.nn.Sequential(torch.nn.Linear(4, c.size),
    ...                            DefaultActivation(c, 'probs'))
    >>> probs = head(torch.randn(8, 4))
    >>> bool(((probs >= 0) & (probs <= 1)).all())
    True

    A categorical plate of 2 members with 3 states each: the 6 columns normalise
    per member, not across the whole row.

    >>> plate = ConceptVariable('p', members=['a', 'b'],
    ...                         distribution=OneHotCategorical, size=3)
    >>> probs = DefaultActivation(plate, 'probs')(torch.randn(8, plate.size))
    >>> probs.shape
    torch.Size([8, 6])
    >>> torch.allclose(probs.reshape(8, 2, 3).sum(-1), torch.ones(8, 2))
    True

    ``logits`` are unconstrained, so the same call is a no-op:

    >>> raw = torch.randn(8, 3)
    >>> torch.equal(DefaultActivation(c, 'logits')(raw), raw)
    True

    See Also
    --------
    torch_concepts.nn.TrilActivation : the ``scale_tril`` activation this composes.
    """

    def __init__(self, variable: Variable, param: str) -> None:
        super().__init__()
        spec = spec_for(variable.distribution)
        if param not in spec.param_sizes:
            raise ValueError(
                f"DefaultActivation: {variable.distribution.__name__} has no parameter "
                f"{param!r}. Its parameters are {sorted(spec.param_sizes)}."
            )
        self.param = param
        self.distribution = variable.distribution
        factory = spec.param_activations.get(param)
        # A family that declares no activation for this parameter leaves it
        # unconstrained: the raw output is already valid.
        self.activation: nn.Module = (
            factory(variable.size, variable.member_size)
            if factory is not None
            else nn.Identity()
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.activation(raw)

    def extra_repr(self) -> str:
        return f"param={self.param!r}, distribution={self.distribution.__name__}"
