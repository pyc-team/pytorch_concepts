import contextlib
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple, Union, Callable, Optional, List, Sequence

import fnmatch
import torch
import torch.nn as nn
import torch.distributions as D

from torch_concepts import ConceptTensor
from torch_concepts.nn import BaseConceptLayer


def _get_parent_and_key(root: nn.ModuleDict, path: str) -> Tuple[nn.ModuleDict, str]:
    """
    Resolve a dotted path like 'encoder_layer.scorer' to the parent ModuleDict
    and the final key to replace. Traversal logic:
      - If the current object is a ModuleDict, use segment as key in it.
      - Else, if it has `.intervenable_modules`, descend into that ModuleDict.
      - Otherwise, fail.
    """
    parts = path.split(".")
    cur = root
    parent = None
    key = None

    for i, seg in enumerate(parts):
        if isinstance(cur, nn.ModuleDict):
            if seg not in cur:
                raise KeyError(f"ModuleDict has no key '{seg}' in path '{path}'")
            parent = cur
            key = seg
            cur = cur[seg]
        elif hasattr(cur, "intervenable_modules"):
            cur = getattr(cur, "intervenable_modules")
            if not isinstance(cur, nn.ModuleDict):
                raise TypeError(f"intervenable_modules must be a ModuleDict at segment '{seg}' in '{path}'")
            # re-try this same path segment against the intervenable dict
            if seg not in cur:
                raise KeyError(f"intervenable_modules has no key '{seg}' in path '{path}'")
            parent = cur
            key = seg
            cur = cur[seg]
        else:
            raise TypeError(f"Cannot descend into '{seg}' in '{path}': "
                            f"neither ModuleDict nor has intervenable_modules")

    if parent is None or key is None:
        raise ValueError(f"Invalid path '{path}'")
    return parent, key


class BaseInference(torch.nn.Module):
    """
    BaseInference is an abstract class for inference modules.
    """
    def __init__(self, model: torch.nn.Module):
        super(BaseInference, self).__init__()
        self.model = model

    def forward(self,
                x: torch.Tensor,
                *args,
                **kwargs) -> ConceptTensor:
        return self.query(x, *args, **kwargs)

    @abstractmethod
    def query(self,
              x: torch.Tensor,
              c: torch.Tensor,
              *args,
              **kwargs) -> ConceptTensor:
        """
        Query model to get concepts.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor, optional): Concept tensor for interventions. Defaults to None.

        Returns:
            ConceptTensor: Queried concepts.
        """
        raise NotImplementedError


class BaseIntervention(BaseInference, ABC):
    """
    Returns {path: replacement_module}. For each path we compute the
    target feature shape (from the parent model or layer) and pass it
    into `query(..., target_shape=...)`.
    """
    def __init__(self, module_dict: nn.ModuleDict):
        super().__init__(model=module_dict)

    def _feature_shape_for(self, parent_model: nn.Module, layer: nn.Module) -> Sequence[int]:
        """
        Decide the feature shape (no batch) for the replacement output.
        Priority:
        - If parent model exposes .out_shapes['concept_probs'] -> use that tuple
        - Else if layer has .out_features (int) -> (out_features,)
        - Else raise (cannot infer)
        """
        if hasattr(parent_model, "out_shapes"):
            out_shapes = getattr(parent_model, "out_shapes")
            if isinstance(out_shapes, dict) and "concept_probs" in out_shapes:
                shp = out_shapes["concept_probs"]
                if isinstance(shp, (list, tuple)):
                    return tuple(shp)
        if hasattr(layer, "out_features"):
            return (int(getattr(layer, "out_features")),)
        raise RuntimeError(
            "Cannot infer target feature shape: neither parent.out_shapes['concept_probs'] "
            "nor layer.out_features is available."
        )

    def forward(self, keys: List[str], *args, **kwargs) -> Dict[str, nn.Module]:
        repl = {}
        for path in keys:
            parent, key = _get_parent_and_key(self.model, path)
            layer = parent[key]
            # parent model is the top-level module addressed by the first segment
            top_key = path.split('.')[0]
            parent_model = self.model[top_key] if top_key in self.model else layer
            target_shape = self._feature_shape_for(parent_model, layer)
            # pass the computed feature shape into the specific intervention
            repl[path] = self.query(layer, *args, target_shape=target_shape, **kwargs)
        return repl
