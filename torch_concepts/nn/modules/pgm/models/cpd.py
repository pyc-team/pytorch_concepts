"""ParametricCPD — Conditional distribution parameterised by a neural network."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import pyro.distributions as dist

from .factor import ParametricFactor
from .variable import Variable, param_dim

# Expected parameter names per distribution family, used to validate and
# dispatch per-parameter module dicts.
_DIST_PARAM_NAMES: Dict[type, List[str]] = {
    dist.Bernoulli: ["probs"],
    dist.Categorical: ["probs"],
    dist.OneHotCategorical: ["probs"],
    dist.Normal: ["loc", "scale"],
    dist.MultivariateNormal: ["loc", "scale_tril"],
    dist.Delta: ["v"],
}


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

    The wrapped ``nn.Module`` maps parent values, or no arguments for roots,
    to the raw parameter tensor of ``variable.distribution``; bounding
    activations are applied automatically.

    Passing a list of ``Variable`` instances returns a list of independent CPDs
    sharing the same parent list; ``parametrization`` may be a single module
    (deep-copied per CPD) or a per-CPD list of modules.
    """

    def __new__(
        cls,
        variable: Union[Variable, List[Variable]],
        parametrization: Optional[Union[nn.Module, List[nn.Module]]] = None,
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
        elif parametrization is not None:
            # Single module broadcast: deep-copy once per Variable so each CPD
            # owns an independent parameter set.
            modules = [copy.deepcopy(parametrization) for _ in range(n)]
        else:
            modules = [None] * n

        return [
            cls(v, modules[i], parents=parents)
            for i, v in enumerate(variable)
        ]

    def __init__(
        self,
        variable: Variable,
        parametrization: Optional[Union[nn.Module, Dict[str, nn.Module]]] = None,
        parents: Optional[List[Variable]] = None,
    ):
        super().__init__()
        # When __new__ returned a list, __init__ is also invoked once per
        # element with a singular Variable, so the list-path is a no-op here.
        if not isinstance(variable, Variable):
            return  # pragma: no cover
        if isinstance(parametrization, list) or (
            parametrization is not None
            and not isinstance(parametrization, (nn.Module, dict))
        ):
            raise ValueError(
                f"ParametricCPD({variable.name!r}): `parametrization` must "
                "be an nn.Module, a dict mapping parameter names to nn.Modules, or None."
            )
        if isinstance(parametrization, dict):
            expected_keys = set(_DIST_PARAM_NAMES.get(variable.distribution, []))
            got_keys = set(parametrization.keys())
            if not got_keys:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): `parametrization` dict "
                    "must not be empty."
                )
            missing = expected_keys - got_keys
            if missing:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): `parametrization` dict is "
                    f"missing keys {sorted(missing)} required for "
                    f"{variable.distribution.__name__}."
                )
            extra = got_keys - expected_keys
            if extra:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): `parametrization` dict has "
                    f"unexpected keys {sorted(extra)} for "
                    f"{variable.distribution.__name__}. "
                    f"Expected: {sorted(expected_keys)}."
                )
            for pname, mod in parametrization.items():
                if not isinstance(mod, nn.Module):
                    raise TypeError(
                        f"ParametricCPD({variable.name!r}): "
                        f"parametrization[{pname!r}] must be an nn.Module, "
                        f"got {type(mod).__name__}."
                    )
            parametrization = nn.ModuleDict(parametrization)
        if parents is not None:
            for p in parents:
                if not isinstance(p, Variable):
                    raise TypeError(
                        f"ParametricCPD({variable.name!r}): every parent "
                        f"must be a Variable, got {type(p).__name__}."
                    )

        if parametrization is None:
            if parents:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): `parametrization` is "
                    "required for non-root CPDs."
                )

        self.variable: Variable = variable
        self.parametrization: Optional[nn.Module] = parametrization
        self.parents: List[Variable] = list(parents) if parents else []

        if self.parents:
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
        """Split the raw network output into per-parameter chunks, then activate
        each chunk via ``_activate_param``.  Activation logic lives only there."""
        D = self.variable.distribution
        s = self.variable.size
        if D is dist.Bernoulli:
            chunks = {"probs": raw}
        elif D in (dist.Categorical, dist.OneHotCategorical):
            chunks = {"probs": raw}
        elif D is dist.Normal:
            chunks = {"loc": raw[..., :s], "scale": raw[..., s:]}
        elif D is dist.MultivariateNormal:
            chunks = {"loc": raw[..., :s], "scale_tril": raw[..., s:]}
        elif D is dist.Delta:
            chunks = {"v": raw}
        else:
            raise ValueError(f"Unsupported distribution {D!r}")
        return {k: self._activate_param(k, v) for k, v in chunks.items()}

    # ------------------------------------------------------ per-param activate
    def _activate_param(self, param_name: str, raw: torch.Tensor) -> torch.Tensor:
        """Apply the bounding activation for a single named distribution parameter.

        Used by the per-parameter-module path when ``parametrization`` is an
        ``nn.ModuleDict``.  Each sub-module produces the raw (pre-activation)
        tensor for its parameter; this method maps it to the bounded value.
        """
        D = self.variable.distribution
        s = self.variable.size
        if D is dist.Bernoulli:
            if param_name == "probs":
                return torch.sigmoid(raw if s > 1 else raw.squeeze(-1))
        elif D in (dist.Categorical, dist.OneHotCategorical):
            if param_name == "probs":
                return torch.softmax(raw, dim=-1)
        elif D is dist.Normal:
            if param_name == "loc":
                return raw.squeeze(-1) if s == 1 else raw
            if param_name == "scale":
                raw_ = raw.squeeze(-1) if s == 1 else raw
                return torch.nn.functional.softplus(raw_) + 1e-6
        elif D is dist.MultivariateNormal:
            if param_name == "loc":
                return raw
            if param_name == "scale_tril":
                tril = torch.zeros(
                    *raw.shape[:-1], s, s, device=raw.device, dtype=raw.dtype
                )
                idx = torch.tril_indices(s, s)
                tril[..., idx[0], idx[1]] = raw
                diag_idx = torch.arange(s)
                tril[..., diag_idx, diag_idx] = (
                    torch.nn.functional.softplus(tril[..., diag_idx, diag_idx]) + 1e-6
                )
                return tril
        elif D is dist.Delta:
            if param_name == "v":
                return raw
        raise ValueError(
            f"ParametricCPD._activate_param: unknown parameter {param_name!r} "
            f"for distribution {D!r}."
        )

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        parent_values: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Compute the CPD output as a dict of distribution parameters.

        Root parametrizations are called with no arguments and the raw output
        is split into the family's named parameter dict by ``_split_params``.

        Non-root CPDs gather parent tensors from ``parent_values`` in
        ``self.parents`` order, concatenate along the last dim, push through
        ``parametrization``, and split into the family's parameter dict.
        """
        if self.is_root:
            if parent_values is not None and len(parent_values) > 0:
                raise ValueError(
                    f"ParametricCPD({self.variable.name!r}): root "
                    "prior CPD received non-empty parent_values; pass None or {}."
                )
            if isinstance(self.parametrization, nn.ModuleDict):
                raw_params = {pname: mod() for pname, mod in self.parametrization.items()}
                return {k: self._activate_param(k, v) for k, v in raw_params.items()}
            raw = self.parametrization()
            return self._split_params(raw)

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
        if isinstance(self.parametrization, nn.ModuleDict):
            raw_params = {pname: mod(cat) for pname, mod in self.parametrization.items()}
            return {k: self._activate_param(k, v) for k, v in raw_params.items()}
        raw = self.parametrization(cat)
        return self._split_params(raw)
