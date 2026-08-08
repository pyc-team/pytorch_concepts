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
a head is written as a raw layer composed with the name of what it produces::

    Sequential(MLP(64, 128, n), DefaultActivation('probs', Bernoulli))

Adding a family therefore stays a single registry entry: declare its
``param_activations`` and every head built this way follows.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .distributions import spec_for


class DefaultActivation(nn.Module):
    """Map a raw network output into a distribution parameter's domain.

    The activation is resolved once, at construction, from the family's
    :attr:`~torch_concepts.nn.modules.mid.distributions.DistributionSpec.param_activations`
    and held as a child module — so it is part of ``state_dict()`` and follows
    ``.to(device)`` like any other layer. A parameter the family leaves
    unconstrained (``logits``, ``loc``, a ``Delta``'s ``value``) resolves to
    ``nn.Identity``, which keeps the class usable unconditionally: a caller whose
    variable may be of any type does not have to branch.

    Parameters
    ----------
    param : str
        The distribution parameter this activation produces (``'probs'``,
        ``'scale'``, ``'loc'``, ``'logits'``, ``'scale_tril'``, ``'value'``).
        Must be a parameter ``distribution`` actually accepts.
    distribution : type
        The distribution family. Required, and not inferable: ``'probs'`` needs a
        sigmoid for a ``Bernoulli`` but a softmax for a ``OneHotCategorical``.
    size : int, optional
        Total event width of the variable. Only consulted by the parameters whose
        activation depends on it — required for ``scale_tril`` (the Cholesky
        factor's side length), and used with ``member_size`` to normalise a
        categorical plate per member.
    member_size : int, optional
        A plate's per-member width, i.e. a categorical member's class count.
        Defaults to treating the whole event as one member, which is the right
        thing for a lone variable.

    Raises
    ------
    ValueError
        If ``param`` is not a parameter of ``distribution``, if ``distribution``
        is not a registered family, or if the parameter needs a ``size`` that was
        not given.

    Examples
    --------
    A Bernoulli head, in a plain ``nn.Sequential``:

    >>> import torch
    >>> from torch.distributions import Bernoulli, OneHotCategorical
    >>> from torch_concepts.nn import DefaultActivation
    >>> head = torch.nn.Sequential(torch.nn.Linear(4, 3),
    ...                            DefaultActivation('probs', Bernoulli))
    >>> probs = head(torch.randn(8, 4))
    >>> bool(((probs >= 0) & (probs <= 1)).all())
    True

    A categorical plate of 2 members with 3 states each: the 6 columns normalise
    per member, not across the whole row.

    >>> act = DefaultActivation('probs', OneHotCategorical, size=6, member_size=3)
    >>> probs = act(torch.randn(8, 6))
    >>> probs.shape
    torch.Size([8, 6])
    >>> torch.allclose(probs.reshape(8, 2, 3).sum(-1), torch.ones(8, 2))
    True

    ``logits`` are unconstrained, so the same call is a no-op:

    >>> raw = torch.randn(8, 3)
    >>> torch.equal(DefaultActivation('logits', Bernoulli)(raw), raw)
    True

    See Also
    --------
    torch_concepts.nn.TrilActivation : the ``scale_tril`` activation this composes.
    """

    def __init__(
        self,
        param: str,
        distribution: type,
        size: Optional[int] = None,
        member_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        spec = spec_for(distribution, f"DefaultActivation({param!r})")
        if param not in spec.param_sizes:
            raise ValueError(
                f"DefaultActivation: {distribution.__name__} has no parameter "
                f"{param!r}. Its parameters are {sorted(spec.param_sizes)}."
            )
        self.param = param
        self.distribution = distribution
        factory = spec.param_activations.get(param)
        # A family that declares no activation for this parameter leaves it
        # unconstrained: the raw output is already valid.
        self.activation: nn.Module = (
            factory(size=size, member_size=member_size)
            if factory is not None
            else nn.Identity()
        )

    @classmethod
    def for_variable(cls, variable, param: str) -> "DefaultActivation":
        """Resolve from a :class:`~torch_concepts.nn.Variable`.

        The variable carries everything the constructor may need — the family,
        the event ``size`` and, for a plate, the per-member width — so this is
        the form to use wherever a variable is in scope (it is what the
        high-level models call).

        Parameters
        ----------
        variable : Variable
            The variable whose parameter is being produced.
        param : str
            The parameter name, as in :class:`DefaultActivation`.
        """
        return cls(
            param,
            variable.distribution,
            size=variable.size,
            member_size=variable.member_size,
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.activation(raw)

    def extra_repr(self) -> str:
        return (
            f"param={self.param!r}, "
            f"distribution={getattr(self.distribution, '__name__', self.distribution)}"
        )
