"""
Memory selector module for memory selection.

This module provides a memory-based selector that learns to attend over
a memory bank of concept exogenous.
"""
import numpy as np
import torch
import torch.nn.functional as F


from ..base.layer import BaseEncoder


class SelectorLatentToExogenous(BaseEncoder):
    """
    Memory-based selector for exogenous variables with attention mechanism.

    This module maintains a learnable memory bank of exogenous and uses an
    attention mechanism to select relevant exogenous based on input. It
    supports both soft (weighted) and hard (Gumbel-softmax) selection.

    Attributes:
        temperature (float): Temperature for softmax/Gumbel-softmax.
        memory_size (int): Number of memory slots per concept.
        out_exogenous (int): Dimension of each memory exogenous.
        memory (nn.Embedding): Learnable memory bank.
        selector (nn.Sequential): Attention network for memory selection.

    Args:
        in_latent: Number of input latent features.
        memory_size: Number of memory slots per concept.
        out_exogenous: Dimension of each memory exogenous.
        out_concepts: Number of output concept representations.
        temperature: Temperature parameter for selection (default: 1.0).
        *args: Additional arguments for the linear layer.
        **kwargs: Additional keyword arguments for the linear layer.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import SelectorLatentToExogenous
        >>>
        >>> # Create memory selector
        >>> selector = SelectorLatentToExogenous(
        ...     in_latent=64,
        ...     memory_size=10,
        ...     out_exogenous=32,
        ...     out_concepts=5,
        ...     temperature=0.5
        ... )
        >>>
        >>> # Forward pass with soft selection
        >>> latent = torch.randn(4, 64)  # batch_size=4
        >>> selected = selector(latent, sampling=False)
        >>> print(selected.shape)
        torch.Size([4, 5, 32])
        >>>
        >>> # Forward pass with hard selection (Gumbel-softmax)
        >>> selected_hard = selector(latent, sampling=True)
        >>> print(selected_hard.shape)
        torch.Size([4, 5, 32])

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024. https://arxiv.org/abs/2407.15527
    """
    def __init__(
        self,
        in_latent: int,
        memory_size: int,
        out_exogenous: int,
        out_concepts: int,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize the memory selector.

        Args:
            in_latent: Number of input latent features.
            memory_size: Number of memory slots per concept.
            out_exogenous: Dimension of each memory exogenous.
            out_concepts: Number of output concepts.
            temperature: Temperature for selection (default: 1.0).
            *args: Additional arguments for the linear layer.
            **kwargs: Additional keyword arguments for the linear layer.
        """
        super().__init__(
            in_latent=in_latent,
            out_concepts=out_concepts,
        )
        self.temperature = temperature
        self.memory_size = memory_size
        self.out_exogenous = out_exogenous
        self._selector_out_shape = (out_concepts, memory_size)
        self._selector_out_dim = np.prod(self._selector_out_shape).item()

        # init memory of exogenous [out_concepts, memory_size * out_exogenous]
        self.memory = torch.nn.Embedding(out_concepts, memory_size * out_exogenous)

        # init selector [B, out_concepts]
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(in_latent, out_exogenous),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                out_exogenous,
                self._selector_out_dim,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self._selector_out_shape),
        )

    def forward(
        self,
        latent: torch.Tensor,
        sampling: bool = False,
    ) -> torch.Tensor:
        """
        Select memory exogenous based on latent input.

        Computes attention weights over memory slots and returns a weighted
        combination of memory exogenous. Can use soft attention or hard
        selection via Gumbel-softmax.

        Args:
            latent: Input latent of shape (batch_size, in_latent).
            sampling: If True, use Gumbel-softmax for hard selection;
                     if False, use soft attention (default: False).

        Returns:
            torch.Tensor: Selected exogenous of shape
                         (batch_size, out_concepts, out_exogenous).
        """
        memory = self.memory.weight.view(-1, self.memory_size, self.out_exogenous)
        mixing_coeff = self.selector(latent)
        if sampling:
            mixing_probs = F.gumbel_softmax(mixing_coeff, dim=1, tau=self.temperature, hard=True)
        else:
            mixing_probs = torch.softmax(mixing_coeff / self.temperature, dim=1)

        exogenous = torch.einsum("btm,tme->bte", mixing_probs, memory) # [Batch x Task x Memory] x [Task x Memory x Emb] -> [Batch x Task x Emb]
        return exogenous
