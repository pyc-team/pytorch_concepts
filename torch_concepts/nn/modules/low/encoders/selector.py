"""
Memory selector module for memory selection.

This module provides a memory-based selector that learns to attend over
a memory bank of concept embeddings.
"""
import numpy as np
import torch
import torch.nn.functional as F


from ..base.layer import BaseEncoder


class MemorySelector(BaseEncoder):
    """
    Memory-based selector for concept embeddings with attention mechanism.

    This module maintains a learnable memory bank of embeddings and uses an
    attention mechanism to select relevant embeddings based on input. It
    supports both soft (weighted) and hard (Gumbel-softmax) selection.

    Attributes:
        temperature (float): Temperature for softmax/Gumbel-softmax.
        memory_size (int): Number of memory slots per concept.
        embedding_size (int): Dimension of each memory embedding.
        memory (nn.Embedding): Learnable memory bank.
        selector (nn.Sequential): Attention network for memory selection.

    Args:
        in_features_embedding: Number of input embedding features.
        memory_size: Number of memory slots per concept.
        embedding_size: Dimension of each memory embedding.
        out_features: Number of output concepts.
        temperature: Temperature parameter for selection (default: 1.0).
        *args: Additional arguments for the linear layer.
        **kwargs: Additional keyword arguments for the linear layer.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MemorySelector
        >>>
        >>> # Create memory selector
        >>> selector = MemorySelector(
        ...     in_features_embedding=64,
        ...     memory_size=10,
        ...     embedding_size=32,
        ...     out_features=5,
        ...     temperature=0.5
        ... )
        >>>
        >>> # Forward pass with soft selection
        >>> embeddings = torch.randn(4, 64)  # batch_size=4
        >>> selected = selector(embeddings, sampling=False)
        >>> print(selected.shape)
        torch.Size([4, 5, 32])
        >>>
        >>> # Forward pass with hard selection (Gumbel-softmax)
        >>> selected_hard = selector(embeddings, sampling=True)
        >>> print(selected_hard.shape)
        torch.Size([4, 5, 32])

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024. https://arxiv.org/abs/2407.15527
    """
    def __init__(
        self,
        in_features_embedding: int,
        memory_size : int,
        embedding_size: int,
        out_features: int,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize the memory selector.

        Args:
            in_features_embedding: Number of input embedding features.
            memory_size: Number of memory slots per concept.
            embedding_size: Dimension of each memory embedding.
            out_features: Number of output concepts.
            temperature: Temperature for selection (default: 1.0).
            *args: Additional arguments for the linear layer.
            **kwargs: Additional keyword arguments for the linear layer.
        """
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_features=out_features,
        )
        self.temperature = temperature
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self._annotation_out_features = out_features
        self._embedding_out_features = memory_size * embedding_size
        self._selector_out_shape = (self._annotation_out_features, memory_size)
        self._selector_out_features = np.prod(self._selector_out_shape).item()

        # init memory of embeddings [out_features, memory_size * embedding_size]
        self.memory = torch.nn.Embedding(self._annotation_out_features, self._embedding_out_features)

        # init selector [B, out_features]
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(in_features_embedding, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                embedding_size,
                self._selector_out_features,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self._selector_out_shape),
        )

    def forward(
        self,
        embedding: torch.Tensor = None,
        sampling: bool = False,
    ) -> torch.Tensor:
        """
        Select memory embeddings based on input embeddings.

        Computes attention weights over memory slots and returns a weighted
        combination of memory embeddings. Can use soft attention or hard
        selection via Gumbel-softmax.

        Args:
            embedding: Input embeddings of shape (batch_size, in_features_embedding).
            sampling: If True, use Gumbel-softmax for hard selection;
                     if False, use soft attention (default: False).

        Returns:
            torch.Tensor: Selected embeddings of shape
                         (batch_size, out_features, embedding_size).
        """
        memory = self.memory.weight.view(-1, self.memory_size, self.embedding_size)
        logits = self.selector(embedding)
        if sampling:
            probs = F.gumbel_softmax(logits, dim=1, tau=self.temperature, hard=True)
        else:
            probs = torch.softmax(logits / self.temperature, dim=1)

        exogenous = torch.einsum("btm,tme->bte", probs, memory) # [Batch x Task x Memory] x [Task x Memory x Emb] -> [Batch x Task x Emb]
        return exogenous
