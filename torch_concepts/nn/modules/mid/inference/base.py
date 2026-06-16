"""Backend-agnostic scaffolding for inference engines."""
from __future__ import annotations

import inspect
import warnings
from typing import Dict, List, Optional, Union

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

        roots_needing_input: List[str] = [
            v.name
            for v in pgm.variables.values()
            if pgm.factors[v.name].is_root
            and any(
                len(inspect.signature(mod.forward).parameters) > 0
                for mod in pgm.factors[v.name].parametrization.values()
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

    def __call__(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        return self.query(query=query, evidence=evidence)
