"""ParametricCPD — neural-network parameterised conditional distribution."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import pyro.distributions as dist

from .factor import ParametricFactor
from .variable import Variable, param_dim


def _infer_out_features(module: nn.Module) -> Optional[int]:
    """Best-effort static introspection of a module's output feature dim."""
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
    """Directed factor :math:`p(c_i \\mid \\mathrm{PA}(c_i))` (§2.2)."""

    def __new__(
        cls,
        concepts: Union[str, List[str]],
        parametrization: Union[nn.Module, List[nn.Module]],
        parents: Optional[List[Union[Variable, str]]] = None,
    ):
        # Single-name path.
        if isinstance(concepts, str):
            if isinstance(parametrization, list):
                raise ValueError(
                    "ParametricCPD: when `concepts` is a string, "
                    "`parametrization` must be a single nn.Module, not a list."
                )
            return super().__new__(cls)

        if not isinstance(concepts, list) or not all(
            isinstance(n, str) for n in concepts
        ):
            raise TypeError(
                "ParametricCPD: `concepts` must be a string or a list of "
                f"strings, got {type(concepts).__name__}."
            )

        n = len(concepts)
        # List-of-modules: assign each to its concept (no deepcopy).
        if isinstance(parametrization, list):
            if len(parametrization) != n:
                raise ValueError(
                    f"ParametricCPD: `parametrization` list length "
                    f"{len(parametrization)} does not match `concepts` length {n}."
                )
            modules = list(parametrization)
        else:
            # Single module broadcast: deep-copy once per concept.
            modules = [copy.deepcopy(parametrization) for _ in range(n)]

        return [
            cls(
                name,
                modules[i],
                parents=list(parents) if parents is not None else None,
            )
            for i, name in enumerate(concepts)
        ]

    def __init__(
        self,
        concepts: Union[str, List[str]],
        parametrization: Union[nn.Module, List[nn.Module]],
        parents: Optional[List[Union[Variable, str]]] = None,
    ):
        super().__init__()
        # When __new__ returned a list, __init__ is invoked once per element
        # with the singular form already.
        if not isinstance(concepts, str):
            return  # pragma: no cover
        if parametrization is None or isinstance(parametrization, list):
            raise ValueError(
                f"ParametricCPD({concepts!r}): `parametrization` must be a "
                "single nn.Module; pass nn.Identity() for an untransformed root."
            )
        self.concept: str = concepts
        self.parametrization: nn.Module = parametrization
        # Parents may be Variable instances or names (resolved later).
        self._raw_parents: List[Union[Variable, str]] = (
            list(parents) if parents else []
        )
        self.parents: List[Variable] = []   # filled by ProbabilisticModel._bind
        self._variable: Optional[Variable] = None  # filled by _bind

    @property
    def is_root(self) -> bool:
        return len(self._raw_parents) == 0

    # ------------------------------------------------------------------ bind
    def _bind(self, variable: Variable, parents: List[Variable]) -> None:
        """Resolve parent strings to Variables; validate output dim statically."""
        self._variable = variable
        self.parents = parents
        if not self.is_root:
            in_dim = sum(p.size for p in parents)
            need = param_dim(variable.distribution, variable.size)
            got = _infer_out_features(self.parametrization)
            if got is not None and got != need:
                raise ValueError(
                    f"ParametricCPD({variable.concept!r}): parametrization "
                    f"output dim is {got} but param_dim({variable.distribution.__name__ if variable.distribution else None}, "
                    f"{variable.size}) = {need}. Adjust the network's last layer."
                )
            # in_dim is informational; we can't reliably check Identity-input
            # CPDs here, but we record it for reference.
            self._expected_in_dim = in_dim
        else:
            self._expected_in_dim = None

    # ------------------------------------------------------------------ split
    def _split_params(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split network output into named distribution parameters with bounding (§3.3)."""
        v = self._variable
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
        parent_values: Optional[List[torch.Tensor]] = None,
    ):
        """Root branch: returns ``parametrization(evidence_value)`` raw.
        Non-root: concatenates parents along the last dim and returns a
        dict of named distribution parameters.
        """
        if self.is_root:
            assert evidence_value is not None
            return self.parametrization(evidence_value)
        assert parent_values is not None
        # Concatenate along feature axis. Each parent value has shape
        # (..., parent.size); Bernoulli observed values may be scalar so we
        # unsqueeze if needed.
        normed: List[torch.Tensor] = []
        for p_val, p_var in zip(parent_values, self.parents):
            if p_val.dim() < 1 or (p_var.size > 1 and p_val.shape[-1] != p_var.size):
                # Try to broadcast a scalar/last-dim-missing parent up.
                if p_var.size == 1 and (p_val.dim() == 0 or p_val.shape[-1] != 1):
                    p_val = p_val.unsqueeze(-1)
            elif p_var.size == 1 and p_val.shape[-1] != 1:
                p_val = p_val.unsqueeze(-1)
            normed.append(p_val.float() if not p_val.is_floating_point() else p_val)
        cat = torch.cat(normed, dim=-1)
        raw = self.parametrization(cat)
        return self._split_params(raw)
