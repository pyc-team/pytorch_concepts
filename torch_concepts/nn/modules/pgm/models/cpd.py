"""ParametricCPD — Conditional distribution parameterised by a neural network."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import pyro.distributions as dist

from .factor import ParametricFactor
from .variable import Variable, param_dim


def _infer_out_features(module: nn.Module) -> Optional[int]:
    """Given a PyTorch module, infer its output feature dimension. 
    Returns an int if successful, else None.
    """
    if isinstance(module, nn.Identity):
        return None  # depth-dependent
    if isinstance(module, nn.Linear):
        return module.out_features
    if isinstance(module, nn.Sequential):
        for m in reversed(list(module.children())):
            d = _infer_out_features(m)
            if d is not None:
                return d
        return None
    if hasattr(module, "out_features"):
        try:
            return int(module.out_features)
        except Exception:
            return None
    return None


class ParametricCPD(ParametricFactor):
    """Conditional distribution parameterised by a neural network
    :math:`p(c_i \\mid \\mathrm{PA}(c_i))`.

    The wrapped ``nn.Module`` maps the concatenation of parent values to the
    raw parameter tensor of ``variable.distribution``; bounding activations
    (sigmoid / softmax / softplus / lower-triangular) are applied automatically.
    Roots take their tensor value as direct input and emit it via
    ``pyro.deterministic`` (pass ``nn.Identity()`` for an untransformed root).

    Passing a list of ``Variable`` instances returns a list of independent CPDs
    sharing the same parent list; ``parametrization`` may be a single module
    (deep-copied per CPD) or a per-CPD list of modules.
    """

    def __new__(
        cls,
        variable: Union[Variable, List[Variable]],
        parametrization: Union[nn.Module, List[nn.Module]],
        parents: Optional[List[Variable]] = None,
    ):
        # Single-Variable path: defer to normal __init__.
        if isinstance(variable, Variable):
            if isinstance(parametrization, list):
                raise ValueError(
                    "ParametricCPD: when `variable` is a single Variable, "
                    "`parametrization` must be a single nn.Module, not a list."
                )
            return super().__new__(cls)

        # List path: broadcast.
        if not isinstance(variable, list) or not all(
            isinstance(v, Variable) for v in variable
        ):
            raise TypeError(
                "ParametricCPD: `variable` must be a Variable or a list of "
                f"Variables, got {type(variable).__name__}."
            )

        n = len(variable)
        if isinstance(parametrization, list):
            if len(parametrization) != n:
                raise ValueError(
                    f"ParametricCPD: `parametrization` list length "
                    f"{len(parametrization)} does not match `variable` length {n}."
                )
            modules = list(parametrization)
        else:
            # Single module broadcast: deep-copy once per Variable so each CPD
            # owns an independent parameter set.
            modules = [copy.deepcopy(parametrization) for _ in range(n)]

        return [
            cls(
                v,
                modules[i],
                parents=list(parents) if parents is not None else None,
            )
            for i, v in enumerate(variable)
        ]

    def __init__(
        self,
        variable: Variable,
        parametrization: nn.Module,
        parents: Optional[List[Variable]] = None,
    ):
        super().__init__()
        # When __new__ returned a list, __init__ is also invoked once per
        # element with a singular Variable, so the list-path is a no-op here.
        if not isinstance(variable, Variable):
            return  # pragma: no cover
        if parametrization is None or isinstance(parametrization, list) or not isinstance(parametrization, nn.Module):
            raise ValueError(
                f"ParametricCPD({variable.name!r}): `parametrization` must "
                "be a single nn.Module; pass nn.Identity() for an untransformed root."
            )
        if parents is not None:
            for p in parents:
                if not isinstance(p, Variable):
                    raise TypeError(
                        f"ParametricCPD({variable.name!r}): every parent "
                        f"must be a Variable, got {type(p).__name__}."
                    )

        self.variable: Variable = variable
        self.parametrization: nn.Module = parametrization
        self.parents: List[Variable] = list(parents) if parents else []

        # Check whether the parametrization's output dim matches the variable's distributional parameter count.
        if self.parents and variable.distribution is not None:
            need = param_dim(variable.distribution, variable.size)
            got = _infer_out_features(self.parametrization)
            if got is not None and got != need:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): parametrization "
                    f"output dim is {got} but param_dim("
                    f"{variable.distribution.__name__}, {variable.size}) = {need}. "
                    "Adjust the network's last layer."
                )

    @property
    def is_root(self) -> bool:
        return len(self.parents) == 0

    # ------------------------------------------------------------------ split
    def _split_params(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map the raw network output to a named-parameter dict for the variable's
        distribution, applying the appropriate bounding activation
        (sigmoid for ``Bernoulli``, softmax for categorical families,
        softplus on the scale for ``Normal``/``MultivariateNormal``, identity
        for ``Delta``)."""
        v = self.variable
        D = v.distribution
        s = v.size
        if D is dist.Bernoulli:
            probs = torch.sigmoid(raw if s > 1 else raw.squeeze(-1))
            return {"probs": probs}
        if D in (dist.Categorical, dist.OneHotCategorical):
            return {"probs": torch.softmax(raw, dim=-1)}
        if D is dist.Normal:
            loc, scale = raw[..., :s], raw[..., s:]
            if s == 1:
                loc = loc.squeeze(-1)
                scale = scale.squeeze(-1)
            return {"loc": loc, "scale": torch.nn.functional.softplus(scale) + 1e-6}
        if D is dist.MultivariateNormal:
            loc = raw[..., :s]
            tril_flat = raw[..., s:]
            tril = torch.zeros(*raw.shape[:-1], s, s, device=raw.device, dtype=raw.dtype)
            idx = torch.tril_indices(s, s)
            tril[..., idx[0], idx[1]] = tril_flat
            diag_idx = torch.arange(s)
            tril[..., diag_idx, diag_idx] = (
                torch.nn.functional.softplus(tril[..., diag_idx, diag_idx]) + 1e-6
            )
            return {"loc": loc, "scale_tril": tril}
        if D is dist.Delta:
            # Sampling is identity; no bounding activation.
            return {"v": raw}
        raise ValueError(f"Unsupported distribution {D!r}")

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        evidence_value: Optional[torch.Tensor] = None,
        parent_values: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Compute the CPD output.

        For roots, returns ``parametrization(evidence_value)`` as a raw tensor.
        For non-roots, ``parent_values`` is a ``Dict[str, Tensor]`` keyed by
        parent name; values are gathered in ``self.parents`` order, concatenated
        along the last dim, passed through ``parametrization``, and returned as
        a dict of named distribution parameters (e.g. ``{'probs': ...}`` or
        ``{'loc': ..., 'scale': ...}``).
        """
        if self.is_root:
            assert evidence_value is not None
            return self.parametrization(evidence_value)

        assert parent_values is not None, (
            f"ParametricCPD({self.variable.name!r}): non-root forward requires "
            "parent_values."
        )
        expected = {p.name for p in self.parents}
        got = set(parent_values.keys())
        if expected != got:
            missing = expected - got
            extra = got - expected
            raise ValueError(
                f"ParametricCPD({self.variable.name!r}): parent_values keys "
                f"do not match parents. expected={sorted(expected)}, "
                f"got={sorted(got)}, missing={sorted(missing)}, extra={sorted(extra)}."
            )

        ordered: List[torch.Tensor] = [parent_values[p.name] for p in self.parents]

        # Concatenate along feature axis. Each parent value has shape
        # (..., parent.size); Bernoulli observed values may be scalar so we
        # unsqueeze if needed.
        normed: List[torch.Tensor] = []
        for p_val, p_var in zip(ordered, self.parents):
            if p_val.dim() < 1 or (p_var.size > 1 and p_val.shape[-1] != p_var.size):
                if p_var.size == 1 and (p_val.dim() == 0 or p_val.shape[-1] != 1):
                    p_val = p_val.unsqueeze(-1)
            elif p_var.size == 1 and p_val.shape[-1] != 1:
                p_val = p_val.unsqueeze(-1)
            normed.append(p_val.float() if not p_val.is_floating_point() else p_val)
        cat = torch.cat(normed, dim=-1)
        raw = self.parametrization(cat)
        return self._split_params(raw)
