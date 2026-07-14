import torch
from typing import Optional

from ...base.intervention import BaseInterventionPolicy


class GradientPolicy(BaseInterventionPolicy):
    """
    Gradient-based intervention policy.

    Scores concepts by the magnitude of the gradient of a downstream output
    with respect to each concept. Concepts with larger gradient magnitude have
    higher influence on the downstream task and are prioritised for intervention.

    Requires ``concept_grads`` to be provided via a ``build_context`` callable
    on the :class:`InterventionModule`. Falls back to uniform (zero) scores if
    no gradients are available.

    Example::

        def build_context(module, predictions, *args, **kwargs):
            with torch.enable_grad():
                pred = predictions.detach().requires_grad_(True)
                task_out = module.task_head(pred)
                grads = torch.autograd.grad(task_out.sum(), pred)[0]
            return {"concept_grads": grads.detach()}

        intervention_module = InterventionModule(
            concept_encoder, strategy, GradientPolicy(),
            build_context=build_context,
            extra_modules={"task_head": task_head},
            quantile=0.5,
        )
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        concepts: torch.Tensor,
        *args,
        concept_grads: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute intervention scores based on gradient magnitude.

        Args:
            concepts: Input concepts of shape ``(batch_size, n_concepts)``.
            concept_grads: Gradient of a downstream output w.r.t. each concept,
                same shape as ``concepts``. Supplied automatically when a
                ``build_context`` function is attached to the
                :class:`InterventionModule`.

        Returns:
            torch.Tensor: Gradient magnitude scores (``|concept_grads|``), or
            zeros of the same shape if ``concept_grads`` is not available.
        """
        if concept_grads is not None:
            return concept_grads.abs()
        return torch.zeros_like(concepts)

