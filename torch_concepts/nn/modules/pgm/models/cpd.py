"""ParametricCPD — Conditional distribution parameterised by a neural network."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import pyro.distributions as dist

from .factor import ParametricFactor
from .variable import Variable, PARAM_DIM

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


def _split_raw_params(variable: "Variable", raw: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split and activate a flat raw tensor into the named parameter dict for
    ``variable``'s distribution family.

    ``raw`` must already be flat in its feature dimension, i.e. shape
    ``(*batch, flat_size)`` for non-root CPDs or ``(flat_size,)`` for root
    CPDs.  Call :func:`_activate_raw_param` on each chunk.
    """
    D = variable.distribution
    s = variable.size
    if D is dist.Bernoulli:
        chunks: Dict[str, torch.Tensor] = {"probs": raw}
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
    return {k: _activate_raw_param(variable, k, v) for k, v in chunks.items()}


def _activate_raw_param(
    variable: "Variable", param_name: str, raw: torch.Tensor
) -> torch.Tensor:
    """Apply the bounding activation for a single named distribution parameter.

    ``raw`` must already be flat in its feature dimension.
    """
    D = variable.distribution
    s = variable.size
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
            return torch.nn.functional.softplus(raw_) + 1e-4
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
                torch.nn.functional.softplus(tril[..., diag_idx, diag_idx]) + 1e-4
            )
            return tril
    elif D is dist.Delta:
        if param_name == "v":
            return raw
    raise ValueError(
        f"_activate_raw_param: unknown parameter {param_name!r} "
        f"for distribution {D!r}."
    )



class _ParameterModule(nn.Module):
    """Thin ``nn.Module`` wrapper around a single ``nn.Parameter``.

    Its ``forward()`` takes no arguments and returns the stored parameter
    tensor.  Used internally by :class:`ParametricCPD` when the caller
    supplies an ``nn.Parameter`` directly as the ``parametrization`` of a
    root (no-parent) CPD.
    """

    def __init__(self, param: nn.Parameter) -> None:
        super().__init__()
        self.param = param

    def forward(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        return self.param


class ParametricCPD(ParametricFactor):
    """Conditional distribution parameterised by a neural network
    :math:`p(c_i \\mid \\mathrm{PA}(c_i))`.

    The wrapped ``nn.Module`` maps parent values, or no arguments for roots,
    to the raw parameter tensor of ``variable.distribution``; bounding
    activations are applied automatically.

    For root (no-parent) CPDs the ``parametrization`` argument also accepts a
    bare ``nn.Parameter``, which is automatically wrapped in a thin module
    whose ``forward()`` returns it.  This lets you write::

        prior = ParametricCPD(
            variable=z,
            parametrization=nn.Parameter(torch.zeros(z.size)),
        )

    Passing a list of ``Variable`` instances returns a list of independent CPDs
    sharing the same parent list; ``parametrization`` may be a single module
    (deep-copied per CPD) or a per-CPD list of modules.
    """

    def __new__(
        cls,
        variable: Union[Variable, List[Variable]],
        parametrization: Optional[Union[nn.Parameter, nn.Module, List[nn.Module]]] = None,
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
        parametrization: Optional[Union[nn.Parameter, nn.Module, Dict[str, Union[nn.Module, nn.Parameter]]]] = None,
        parents: Optional[List[Variable]] = None,
    ):
        super().__init__()
        # When __new__ returned a list, __init__ is also invoked once per
        # element with a singular Variable, so the list-path is a no-op here.
        if not isinstance(variable, Variable):
            return  # pragma: no cover
        if isinstance(parametrization, list) or (
            parametrization is not None
            and not isinstance(parametrization, (nn.Parameter, nn.Module, dict))
        ):
            raise ValueError(
                f"ParametricCPD({variable.name!r}): `parametrization` must "
                "be an nn.Parameter, an nn.Module, a dict mapping parameter names "
                "to nn.Modules, or None."
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
                if not isinstance(mod, (nn.Module, nn.Parameter)):
                    raise TypeError(
                        f"ParametricCPD({variable.name!r}): "
                        f"parametrization[{pname!r}] must be an nn.Module or "
                        f"nn.Parameter, got {type(mod).__name__}."
                    )
            parametrization = nn.ModuleDict({
                pname: _ParameterModule(mod) if isinstance(mod, nn.Parameter) else mod
                for pname, mod in parametrization.items()
            })
        if isinstance(parametrization, nn.Parameter):
            if parents:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): nn.Parameter parametrization "
                    "is only valid for root (no-parent) CPDs."
                )
            need = PARAM_DIM[variable.distribution](variable.size)
            got_n = parametrization.numel()
            if got_n != need:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): nn.Parameter has {got_n} "
                    f"element(s) but PARAM_DIM[{variable.distribution.__name__}]({variable.size})"
                    f" = {need}. "
                    f"Use nn.Parameter(torch.zeros({need}))."
                )
            parametrization = _ParameterModule(parametrization)
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

        # Store the target variable whose conditional distribution this CPD models.
        self.variable: Variable = variable
        # Store the neural-network (or parameter) module that outputs raw distribution params.
        self.parametrization: Optional[nn.Module] = parametrization
        # Store the ordered list of parent variables whose values are the CPD inputs.
        self.parents: List[Variable] = list(parents) if parents else []

    @property
    def is_root(self) -> bool:
        return len(self.parents) == 0

    # ------------------------------------------------------------------ split
    def _split_params(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Delegate to the module-level :func:`_split_raw_params`."""
        return _split_raw_params(self.variable, raw)

    # ------------------------------------------------------ per-param activate
    def _activate_param(self, param_name: str, raw: torch.Tensor) -> torch.Tensor:
        """Delegate to the module-level :func:`_activate_raw_param`."""
        return _activate_raw_param(self.variable, param_name, raw)

    # ---------------------------------------------------------------- validate
    def _validate_parent_values(
        self,
        parent_values: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        """Validate that ``parent_values`` keys exactly match ``self.parents``.

        Raises ``ValueError`` if any expected parent is missing or an unexpected
        key is supplied.  Call this during debugging by uncommenting the line in
        :meth:`forward`.
        """
        expected = {p.name for p in self.parents}
        got = set(parent_values.keys()) if parent_values else set()
        if expected != got:
            missing = expected - got
            extra = got - expected
            raise ValueError(
                f"ParametricCPD({self.variable.name!r}): parent_values keys "
                f"do not match parents. expected={sorted(expected)}, "
                f"got={sorted(got)}, missing={sorted(missing)}, extra={sorted(extra)}."
            )

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        parent_values: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Compute the CPD output as a dict of distribution parameters.

        Root CPDs are called with no parent values: the parametrization module
        is called with no arguments and its raw scalar output is split into the
        named parameter dict for ``self.variable.distribution``.

        Non-root CPDs receive a ``parent_values`` dict mapping each parent name
        to its tensor value (shape ``(*batch, parent.size)``). These tensors are
        concatenated along the last feature axis, pushed through the
        parametrization network, and then split+activated into the distribution
        parameter dict.

        Returns a ``Dict[str, Tensor]`` ready to pass to the distribution
        constructor (e.g. ``{"probs": ...}`` for Bernoulli,
        ``{"loc": ..., "scale": ...}`` for Normal).
        """
        if self.is_root:
            # Root CPD: no parents expected — raise if any are accidentally passed.
            if parent_values is not None and len(parent_values) > 0:
                raise ValueError(
                    f"ParametricCPD({self.variable.name!r}): CPD with no parents "
                    "received non-empty parent_values; pass None or {}."
                )
            if isinstance(self.parametrization, nn.ModuleDict):
                # Per-parameter modules: call each module independently, then activate.
                raw_params = {pname: mod() for pname, mod in self.parametrization.items()}
                return {k: self._activate_param(k, v.flatten()) for k, v in raw_params.items()}
            # Single-module path: module returns a flat raw tensor, then split+activate.
            raw = self.parametrization()
            return self._split_params(raw.flatten())

        assert parent_values is not None, (
            f"ParametricCPD({self.variable.name!r}): non-root forward requires "
            "parent_values."
        )
        # Uncomment the line below during debugging to validate parent keys:
        # self._validate_parent_values(parent_values)

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
        # ``batch_ndim`` is the number of leading batch dimensions in ``cat``
        # (everything except the last feature axis).  We flatten the network
        # output from that point onward so that _split_params always receives
        # a flat feature vector, regardless of the internal output shape.
        batch_ndim = cat.dim() - 1
        if isinstance(self.parametrization, nn.ModuleDict):
            raw_params = {pname: mod(cat) for pname, mod in self.parametrization.items()}
            # Only flatten feature dims if the module output actually has batch
            # dims (e.g. nn.Linear). _ParameterModule returns a bare parameter
            # with no batch axis, so leave it unchanged.
            return {
                k: self._activate_param(
                    k, v.reshape(*v.shape[:batch_ndim], -1) if v.dim() > batch_ndim else v
                )
                for k, v in raw_params.items()
            }
        raw = self.parametrization(cat)
        return self._split_params(raw.reshape(*raw.shape[:batch_ndim], -1))
