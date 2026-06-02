"""Amortised variational guides for ``BayesianNetwork`` latents.

Guides live alongside the model CPDs as submodules of the
:class:`BayesianNetwork` itself, so ``pgm.parameters()`` collects both the
prior parameters and the variational posterior parameters and a single
checkpoint round-trips the full model + guide.
"""
from __future__ import annotations

from typing import Dict, Optional, Type

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule

from .variable import Variable, param_dim
from .cpd import _split_raw_params, _activate_raw_param
from .samplers import build_relaxed_distribution


def _default_mlp(parent_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(parent_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim),
    )


class _BaseGuide(PyroModule):
    """Common scaffolding: a 2-layer MLP (hidden=64) over the concatenated
    conditioning input, producing ``param_dim(distribution, size)`` outputs."""

    def __init__(self, variable: Variable, parent_dim: int):
        super().__init__()
        self.variable = variable
        self.parent_dim = parent_dim
        self.net = _default_mlp(
            parent_dim, param_dim(variable.distribution, variable.size)
        )


class STBernoulliGuide(_BaseGuide):
    """Amortised variational guide for ``Bernoulli`` latents.

    Reads the concatenated conditioning input, predicts per-sample
    probabilities through a 2-layer MLP (hidden=64) with sigmoid output, and
    emits a ``pyro.sample`` site drawn from
    ``RelaxedBernoulliStraightThrough(temperature, logits)``.
    """

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if self.variable.size == 1:
            raw = raw.squeeze(-1)
        probs = torch.sigmoid(raw)
        logits = torch.log(probs.clamp(min=1e-8)) - torch.log(
            (1 - probs).clamp(min=1e-8)
        )
        q = dist.RelaxedBernoulliStraightThrough(temperature=temperature, logits=logits)
        if self.variable.size > 1:
            q = q.to_event(1)
        return pyro.sample(self.variable.name, q)


class STOneHotGuide(_BaseGuide):
    """Amortised variational guide for ``OneHotCategorical`` latents using a
    ``RelaxedOneHotCategoricalStraightThrough`` distribution; same MLP
    architecture as :class:`STBernoulliGuide`."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        probs = torch.softmax(raw, dim=-1)
        logits = torch.log(probs.clamp(min=1e-8))
        q = dist.RelaxedOneHotCategoricalStraightThrough(
            temperature=temperature, logits=logits
        )
        return pyro.sample(self.variable.name, q)


class NormalGuide(_BaseGuide):
    """Amortised variational guide for ``Normal`` latents using a
    reparameterised ``Normal(loc, scale)`` distribution; same MLP architecture
    as :class:`STBernoulliGuide`, with Softplus on the scale."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        s = self.variable.size
        loc, scale = raw[..., :s], raw[..., s:]
        if s == 1:
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
        scale = torch.nn.functional.softplus(scale) + 1e-4
        q = dist.Normal(loc=loc, scale=scale)
        if s > 1:
            q = q.to_event(1)
        return pyro.sample(self.variable.name, q)


class MVNGuide(_BaseGuide):
    """Amortised variational guide for ``MultivariateNormal`` latents using a
    reparameterised ``MultivariateNormal(loc, scale_tril)`` distribution; same
    MLP architecture as :class:`STBernoulliGuide`, with Softplus on the
    diagonal of the lower-triangular Cholesky factor."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        s = self.variable.size
        loc = raw[..., :s]
        tril_flat = raw[..., s:]
        tril = torch.zeros(*raw.shape[:-1], s, s, device=raw.device, dtype=raw.dtype)
        idx = torch.tril_indices(s, s)
        tril[..., idx[0], idx[1]] = tril_flat
        diag_idx = torch.arange(s)
        tril[..., diag_idx, diag_idx] = (
            torch.nn.functional.softplus(tril[..., diag_idx, diag_idx]) + 1e-4
        )
        q = dist.MultivariateNormal(loc=loc, scale_tril=tril)
        return pyro.sample(self.variable.name, q)


DEFAULT_GUIDES: Dict[Type[dist.Distribution], Type[_BaseGuide]] = {
    dist.Bernoulli: STBernoulliGuide,
    dist.OneHotCategorical: STOneHotGuide,
    dist.Normal: NormalGuide,
    dist.MultivariateNormal: MVNGuide,
}


class CustomGuide(PyroModule):
    """Wraps a user-supplied ``nn.Module`` as an amortised variational guide.

    The wrapped module plays the same role as the ``parametrization`` of a
    ``ParametricCPD``: it maps the conditioning input ``x`` (the concatenated
    observed values, shape ``(*batch, parent_dim)``) to a raw parameter tensor
    of *any* shape.  The guide flattens the feature dimensions, splits the
    result into the named parameters of ``variable.distribution``, applies the
    appropriate bounding activations, builds the relaxed surrogate
    distribution, and emits ``pyro.sample(variable.name, q)``.

    For unconditional guides (``parent_dim == 0``) the module is called with
    no arguments, exactly like a root ``ParametricCPD``.

    Parameters
    ----------
    variable
        The latent ``Variable`` this guide targets.
    parent_dim
        Total feature size of the conditioning input (sum of parent sizes).
        ``0`` for an unconditional guide.
    parametrization
        An ``nn.Module`` whose ``forward`` signature is
        ``forward(x: Tensor) -> Tensor`` for conditional guides or
        ``forward() -> Tensor`` for unconditional ones.  The output may be
        any shape; feature dimensions are flattened automatically before
        the parameter split.
    """

    def __init__(
        self,
        variable: Variable,
        parent_dim: int,
        parametrization: nn.Module,
    ):
        super().__init__()
        if not isinstance(parametrization, nn.Module):
            raise TypeError(
                f"ParametricGuide({variable.name!r}): `parametrization` must "
                f"be an nn.Module, got {type(parametrization).__name__}."
            )
        self.variable = variable
        self.parent_dim = parent_dim
        self.parametrization = parametrization

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """Run the guide for one batch.

        Parameters
        ----------
        x
            Conditioning input, shape ``(*batch, parent_dim)`` or
            ``(*batch, 0)`` for unconditional guides.
        temperature
            Relaxation temperature for discrete distributions.
        """
        if x.shape[-1] == 0:
            # Unconditional: call module with no arguments (root-like).
            raw = self.parametrization()
            raw = raw.flatten()
        else:
            raw = self.parametrization(x)
            # Flatten feature dims; keep all batch dims intact.
            batch_ndim = x.dim() - 1
            raw = raw.reshape(*raw.shape[:batch_ndim], -1)

        params = _split_raw_params(self.variable, raw)
        q = build_relaxed_distribution(self.variable, params, temperature)
        return pyro.sample(self.variable.name, q)
