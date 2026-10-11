import torch

from ...base.intervention import ModuleInterventionStrategy


class PositiveWeightsIntervention(ModuleInterventionStrategy):
    """
    Intervention that evaluates the wrapped module with its parameters clipped
    to be non-negative (ReLU), leaving the module itself untouched.
    """

    def __init__(self):
        super().__init__()

    def transform(self, module, *args, **kwargs):
        """``module`` evaluated with ReLU-ed parameters. Nothing is modified or
        copied, and gradients still reach the original parameters."""
        params = {name: torch.relu(p) for name, p in module.named_parameters()}
        return lambda *a, **k: torch.func.functional_call(module, params, a, k)
