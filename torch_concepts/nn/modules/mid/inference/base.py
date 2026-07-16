"""Backend-agnostic scaffolding for inference engines."""
from __future__ import annotations

import inspect
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..models.probabilistic_model import ProbabilisticModel
from ...outputs import InferenceOutput


class BaseInference(nn.Module):
    """Abstract base class for all inference engines.

    The engine *wraps* a :class:`ProbabilisticModel` by holding a reference
    to it (``self.pgm = pgm``). 

    Backend-specific subclasses (e.g., ``PyroBaseInference`` or ``TorchBaseInference``)
    layer engine-specific machinery on top.

    ``query`` and ``evidence`` are attribute-style containers keyed by PGM
    variable name. ``__call__`` delegates to :meth:`query`.
    """
    
    name: str = "BaseInference"

    def __init__(self, pgm: ProbabilisticModel):
        super().__init__()
        # NOTE: nn.Module.__setattr__ auto-registers ``pgm`` as a submodule, so
        # the engine shares parameters with the original PGM (no copy).
        self.pgm = pgm

        # Factor-based (not variable-keyed): a ProbabilisticModel keys factors by
        # factor name, and undirected potentials have no ``is_root``. Only root CPDs
        # whose parametrization needs inputs must receive constant evidence every call.
        roots_needing_input: List[str] = [
            f.name
            for f in pgm.factors.values()
            if getattr(f, "is_root", False)
            and any(
                len(inspect.signature(mod.forward).parameters) > 0
                for mod in f.parametrization.values()
            )
        ]
        if roots_needing_input:
            warnings.warn(
                "\033[33m"
                f"{self.name}: the following root variables have a parametrization "
                f"that requires input arguments: {roots_needing_input}. "
                "These must be supplied as constant evidence on every query call."
                "\033[0m",
                UserWarning,
                stacklevel=2,
            )

    def _require_directed(self) -> None:
        """Guard for engines that need a topological order.

        The directed engines traverse ``pgm.levels`` / ``pgm.sorted_variables``,
        which only a :class:`BayesianNetwork` provides. Passing a general
        (undirected or mixed) :class:`ProbabilisticModel` raises a clear error
        pointing at :class:`BeliefPropagation`.
        """
        from ..models.bayesian_network import BayesianNetwork

        if not isinstance(self.pgm, BayesianNetwork):
            raise TypeError(
                f"{self.name} requires a directed BayesianNetwork (an acyclic, "
                "all-CPD model with a topological order); got a general "
                f"{type(self.pgm).__name__}. Use BeliefPropagation for undirected "
                "or mixed (chain) graphs."
            )

    def _validate_containers(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
    ) -> None:
        """Check that:
         - query and evidence keys are valid variable names,
         - all values are tensors, and
         - batch sizes match.
        """

        all_names = getattr(self.pgm, "queryable_names", None)
        if all_names is None:
            all_names = {v.name for v in self.pgm.variables.values()}
        unknown_q = set(query.keys()) - all_names
        if unknown_q:
            raise ValueError(f"{self.name}: unknown query names {sorted(unknown_q)}.")
        unknown_e = set(evidence.keys()) - all_names
        if unknown_e:
            raise ValueError(f"{self.name}: unknown evidence names {sorted(unknown_e)}.")

        for name, val in evidence.items():
            if not isinstance(val, torch.Tensor):
                raise ValueError(
                    f"{self.name}: evidence[{name!r}] must be a Tensor, "
                    f"got {type(val).__name__}."
                )

        if not query and not evidence:
            raise ValueError("nothing to do")

        all_tensors = {name: val for name, val in query.items() if val is not None}
        all_tensors.update(evidence)
        batch_sizes = {name: t.shape[0] for name, t in all_tensors.items()}
        if len(set(batch_sizes.values())) > 1:
            shapes = {name: tuple(t.shape) for name, t in all_tensors.items()}
            raise ValueError(f"{self.name}: mismatched batch sizes {shapes}.")

    @staticmethod
    def _normalize_query(
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Normalize query input to a dict mapping variable names to optional tensors."""
        if isinstance(query, list):
            return {name: None for name in query}
        return query

    # Uniform plate routing — the only plate awareness outside the models. Each
    # call is an identity/no-op for ordinary variables, so engines run the same
    # line for plate and non-plate names (never behind an is_plate branch).
    def _split_evidence(
        self, evidence: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """Split evidence into whole-variable entries and per-owner member entries.

        Returns ``(whole, member_evidence)`` where ``whole`` keeps every key that
        names a variable, and ``member_evidence[owner_name][member_name]`` collects
        keys that name plate members. O(#evidence), never O(#members).
        """
        whole: Dict[str, torch.Tensor] = {}
        member_evidence: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, value in evidence.items():
            var = self.pgm.resolve(name)
            if name == var.name:
                whole[name] = value
            else:
                member_evidence.setdefault(var.name, {})[name] = value
        return whole, member_evidence

    def _expose_params(self, params, query_names):
        """Add sliced entries for queried member names (whole-variable entries
        already present in ``params`` are the source; a view, no copy).

        Slicing is a pure function of the variable's own column layout
        (``Variable.select``), so this works for directed and undirected models
        alike. ``resolve`` is hoisted out of the loop: it goes through
        ``nn.Module.__getattr__``, which dominates the per-name cost when many
        members are queried at once.
        """
        resolve = self.pgm.resolve
        for name in query_names:
            var = resolve(name)
            owner = var.name
            if name != owner and owner in params:
                params[name] = var.select(params[owner], name)
        return params

    def _expose_values(self, values, query_names):
        """Value-side twin of :meth:`_expose_params` (``select_value`` instead
        of ``select``)."""
        resolve = self.pgm.resolve
        for name in query_names:
            var = resolve(name)
            owner = var.name
            if name != owner and owner in values:
                values[name] = var.select_value(values[owner], name)
        return values

    def __call__(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        return self.query(query=query, evidence=evidence)

    def _format_repr(self, **fields) -> str:
        """Render ``EngineName(param=value, ...)`` for the given inference
        parameters.

        The wrapped :class:`ProbabilisticModel` is intentionally excluded — only
        the engine's own configuration is shown. ``nn.Module`` values are
        rendered by their class name and (non-string) callables by their
        ``__name__`` so the PGM is never recursively printed.
        """
        items = []
        for key, val in fields.items():
            if isinstance(val, nn.Module):
                rendered = type(val).__name__
            elif callable(val) and not isinstance(val, str):
                rendered = getattr(val, "__name__", repr(val))
            else:
                rendered = repr(val)
            items.append(f"{key}={rendered}")
        return f"{type(self).__name__}({', '.join(items)})"

    def __repr__(self) -> str:
        # Concrete engines override this to surface their own parameters; the
        # base fallback shows just the engine name (no parameters, no PGM).
        return self._format_repr()
