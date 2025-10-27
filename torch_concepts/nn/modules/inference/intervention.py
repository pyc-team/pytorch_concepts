import contextlib
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple, Union, Callable, Optional

import fnmatch
import torch
import torch.nn as nn
import torch.distributions as D

from ...base.inference import BaseIntervention


class ConstantTensorIntervention(BaseIntervention):
    """do(X = c): always return the provided tensor."""
    def __init__(self, module_dict: nn.ModuleDict, value: torch.Tensor):
        super().__init__(module_dict)
        self.value = value.detach()

    def query(self, layer: nn.Module, *_, **__) -> nn.Module:
        m = nn.Identity()
        def fwd(*args, **kwargs):
            # follow caller device/dtype if there is a tensor input
            dev = args[0].device if args and isinstance(args[0], torch.Tensor) else self.value.device
            return self.value.to(device=dev)
        m.forward = fwd
        return m


class ConstantLikeIntervention(BaseIntervention):
    """Return a tensor like the first input, filled with `fill`."""
    def __init__(self, module_dict: nn.ModuleDict, fill: float = 0.0):
        super().__init__(module_dict)
        self.fill = float(fill)

    def query(self, layer: nn.Module, *_, **__) -> nn.Module:
        m = nn.Identity()
        m.forward = lambda x, *a, **k: torch.full_like(x, self.fill)
        return m


class DistributionIntervention(BaseIntervention):
    """Sample from a torch.distributions distribution; shape matches first input."""
    def __init__(self, module_dict: nn.ModuleDict, dist: D.Distribution, rsample: bool = False):
        super().__init__(module_dict)
        self.dist, self.rsample = dist, bool(rsample)

    def query(self, layer: nn.Module, *_, **__) -> nn.Module:
        m = nn.Identity()

        def fwd(x, *a, **k):
            size = x.shape
            # safer: handle both rsample and sample without assuming generator kwarg exists
            if self.rsample and hasattr(self.dist, "rsample"):
                return self.dist.rsample(size)
            return self.dist.sample(size)

        m.forward = fwd
        return m


@contextlib.contextmanager
def intervene_in_dict(modules: nn.ModuleDict, replacements: dict):
    """
    Temporarily replace entries in a ModuleDict.
    Example:
        iv = ConstantLikeIntervention(model.blocks, fill=0.0)
        with intervene_in_dict(model.blocks, {"dropout": iv("dropout")}):
            y = model(x)
    """
    originals = {}
    try:
        for k, new_mod in replacements.items():
            if k not in modules:
                raise KeyError(f"ModuleDict has no key '{k}'")
            originals[k] = modules[k]
            new_mod.train(originals[k].training)
            modules[k] = new_mod
        yield modules
    finally:
        for k, old in originals.items():
            modules[k] = old
