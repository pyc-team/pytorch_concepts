"""
Memory selector module for memory selection.

This module provides a memory-based selector that learns to attend over
a memory bank of concept exogenous.
"""
import numpy as np
import torch
import torch.nn.functional as F


from ..base.layer import BaseEncoder


class SelectorZU(BaseEncoder):
    """
    Memory-based selector for concept exogenous with attention mechanism.

    This module maintains a learnable memory bank of exogenous and uses an
    attention mechanism to select relevant exogenous based on input. It
    supports both soft (weighted) and hard (Gumbel-softmax) selection.

    Attributes:
        temperature (float): Temperature for softmax/Gumbel-softmax.
        memory_size (int): Number of memory slots per concept.
        exogenous_size (int): Dimension of each memory exogenous.
        memory (nn.Embedding): Learnable memory bank.
        selector (nn.Sequential): Attention network for memory selection.

    Args:
        in_features: Number of input latent features.
        memory_size: Number of memory slots per concept.
        exogenous_size: Dimension of each memory exogenous.
        out_features: Number of output concepts.
        temperature: Temperature parameter for selection (default: 1.0).
        *args: Additional arguments for the linear layer.
        **kwargs: Additional keyword arguments for the linear layer.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import SelectorZU
        >>>
        >>> # Create memory selector
        >>> selector = SelectorZU(
        ...     in_features=64,
        ...     memory_size=10,
        ...     exogenous_size=32,
        ...     out_features=5,
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
        in_features: int,
        memory_size : int,
        exogenous_size: int,
        out_features: int,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize the memory selector.

        Args:
            in_features: Number of input latent features.
            memory_size: Number of memory slots per concept.
            exogenous_size: Dimension of each memory exogenous.
            out_features: Number of output concepts.
            temperature: Temperature for selection (default: 1.0).
            *args: Additional arguments for the linear layer.
            **kwargs: Additional keyword arguments for the linear layer.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
        )
        self.temperature = temperature
        self.memory_size = memory_size
        self.exogenous_size = exogenous_size
        self._annotation_out_features = out_features
        self._exogenous_out_features = memory_size * exogenous_size
        self._selector_out_shape = (self._annotation_out_features, memory_size)
        self._selector_out_features = np.prod(self._selector_out_shape).item()

        # init memory of exogenous [out_features, memory_size * exogenous_size]
        self.memory = torch.nn.Embedding(self._annotation_out_features, self._exogenous_out_features)

        # init selector [B, out_features]
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(in_features, exogenous_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                exogenous_size,
                self._selector_out_features,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self._selector_out_shape),
        )

    def forward(
        self,
        input: torch.Tensor = None,
        sampling: bool = False,
    ) -> torch.Tensor:
        """
        Select memory exogenous based on input input.

        Computes attention weights over memory slots and returns a weighted
        combination of memory exogenous. Can use soft attention or hard
        selection via Gumbel-softmax.

        Args:
            input: Input latent of shape (batch_size, in_features).
            sampling: If True, use Gumbel-softmax for hard selection;
                     if False, use soft attention (default: False).

        Returns:
            torch.Tensor: Selected exogenous of shape
                         (batch_size, out_features, exogenous_size).
        """
        memory = self.memory.weight.view(-1, self.memory_size, self.exogenous_size)
        mixing_coeff = self.selector(input)
        if sampling:
            mixing_probs = F.gumbel_softmax(mixing_coeff, dim=1, tau=self.temperature, hard=True)
        else:
            mixing_probs = torch.softmax(mixing_coeff / self.temperature, dim=1)

        exogenous = torch.einsum("btm,tme->bte", mixing_probs, memory) # [Batch x Task x Memory] x [Task x Memory x Emb] -> [Batch x Task x Emb]
        return exogenous
