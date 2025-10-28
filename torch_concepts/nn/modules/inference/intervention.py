import contextlib
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple, Union, Callable, Optional, List

import fnmatch
import torch
import torch.nn as nn
import torch.distributions as D

from ...base.inference import BaseIntervention, _get_parent_and_key


def _get_parent_key_owner(root: nn.ModuleDict, path: str) -> Tuple[nn.ModuleDict, str, Optional[nn.Module]]:
    """
    Resolve dotted path to (parent_dict, key, owner_module_or_None).

    - Walks through root ModuleDict.
    - When entering a module that has .intervenable_modules (a ModuleDict),
      we descend into that dict and remember its owner (the module that
      exposes it). If we replace an entry in that dict, we also replace
      the owner’s attribute that references the same module object.
    """
    parts = path.split(".")
    cur = root
    parent = None
    key = None
    owner = None  # module that holds .intervenable_modules we’re in

    for seg in parts:
        if isinstance(cur, nn.ModuleDict):
            if seg not in cur:
                raise KeyError(f"ModuleDict has no key '{seg}' in path '{path}'")
            parent, key, cur = cur, seg, cur[seg]
            # if the next hop exposes intervenables, remember it as a potential owner
            if hasattr(cur, "intervenable_modules") and isinstance(cur.intervenable_modules, nn.ModuleDict):
                owner = cur
        elif hasattr(cur, "intervenable_modules") and isinstance(cur.intervenable_modules, nn.ModuleDict):
            # we are inside an owner’s intervenable space
            md = cur.intervenable_modules
            if seg not in md:
                raise KeyError(f"intervenable_modules has no key '{seg}' in path '{path}'")
            parent, key, cur = md, seg, md[seg]
            owner = cur if hasattr(cur, "intervenable_modules") else owner
        else:
            raise TypeError(f"Cannot descend into '{seg}' in '{path}'")
    if parent is None or key is None:
        raise ValueError(f"Invalid path '{path}'")
    # If parent is an intervenable_modules dict, owner is the module that exposes it.
    # If parent is a top-level ModuleDict, owner stays None.
    return parent, key, owner


class ConstantTensorIntervention(BaseIntervention):
    def __init__(self, module_dict: nn.ModuleDict, value: torch.Tensor):
        super().__init__(module_dict)
        self.value = value.detach()
    def query(self, layer: nn.Module, *_, target_shape=None, **__) -> nn.Module:
        # Constant stays constant; we only align device/dtype at call time.
        m = nn.Identity()
        def fwd(*args, **kwargs):
            dev = args[0].device if args and isinstance(args[0], torch.Tensor) else self.value.device
            return self.value.to(dev)
        m.forward = fwd
        return m


class ConstantLikeIntervention(BaseIntervention):
    def __init__(self, module_dict: nn.ModuleDict, fill: float = 0.0):
        super().__init__(module_dict); self.fill = float(fill)
    def query(self, layer: nn.Module, *_, target_shape, **__) -> nn.Module:
        m = nn.Identity()
        def fwd(x, *a, **k):
            batch = x.shape[0]
            shp = (batch, *tuple(target_shape))
            return x.new_full(shp, self.fill)
        m.forward = fwd
        return m


class DistributionIntervention(BaseIntervention):
    def __init__(self, module_dict: nn.ModuleDict, dist: D.Distribution, rsample: bool = False):
        super().__init__(module_dict); self.dist, self.rsample = dist, bool(rsample)
    def query(self, layer: nn.Module, *_, target_shape, **__) -> nn.Module:
        m = nn.Identity()
        def fwd(x, *a, **k):
            batch = x.shape[0]
            shp = (batch, *tuple(target_shape))
            if self.rsample and hasattr(self.dist, "rsample"):
                return self.dist.rsample(shp)
            return self.dist.sample(shp)
        m.forward = fwd
        return m


@contextlib.contextmanager
def intervene_in_dict(root: nn.ModuleDict, replacements: Dict[str, nn.Module]):
    """
    Temporarily replace leaf modules addressed by dotted paths, and
    also swap the owner’s attribute that points to the same module object.
    """
    # records: (parent_dict, key, old_module, owner_module_or_None, owner_attr_name_or_None)
    originals = []
    try:
        for path, new_mod in replacements.items():
            parent, key, owner = _get_parent_key_owner(root, path)
            old = parent[key]
            new_mod.train(old.training)

            # Replace entry in the parent dict (top-level or intervenable_modules)
            parent[key] = new_mod

            # If we're in an intervenable dict, also replace the owner’s attribute
            owner_attr = None
            if owner is not None:
                for name, sub in owner._modules.items():
                    if sub is old:
                        owner._modules[name] = new_mod
                        owner_attr = name
                        break

            originals.append((parent, key, old, owner, owner_attr))
        yield root
    finally:
        for parent, key, old, owner, owner_attr in reversed(originals):
            parent[key] = old
            if owner is not None and owner_attr is not None:
                owner._modules[owner_attr] = old
