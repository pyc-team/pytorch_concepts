import torch

from ...base.intervention import ConceptInterventionStrategy


class DoIntervention(ConceptInterventionStrategy):
    """
    Intervention that sets the intervened outputs to constant values, do(C=c).

    Args:
        constants: A scalar, one value per output (shape ``[F]``), or any shape
            that broadcasts to the layer output ``[..., F]``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import DoIntervention
        >>>
        >>> strategy = DoIntervention(torch.tensor([0.0, 1.0]))
        >>> strategy(torch.randn(3, 2)).tolist()
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    """

    def __init__(self, constants: torch.Tensor | float):
        super().__init__()
        const = constants if torch.is_tensor(constants) else torch.tensor(constants)
        self.register_buffer("constants", const)

    def forward(self, x, *args, **kwargs):
        v = self.constants
        try:
            v = torch.broadcast_to(v, x.shape)
        except RuntimeError as e:
            raise ValueError(
                f"constants of shape {tuple(self.constants.shape)} cannot be "
                f"broadcast to concept tensor shape {tuple(x.shape)} "
                f"(expects scalar, [F], or any shape broadcastable against "
                f"[..., F])"
            ) from e

        return v.to(dtype=x.dtype, device=x.device)
