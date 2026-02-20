"""
Linear encoder modules for concept prediction from latent features.

These modules provide encoder layers that transform latent or exogenous
variables into concept representations.
"""
import torch

from ..base.layer import BaseEncoder

# NOTE: The LinearLatentToConcept is equivalent to a 
# torch.nn.Linear layer and may not be necessary as a separate class.
class LinearLatentToConcept(BaseEncoder):
    """
    Encoder that predicts concept representations from latent features.

    This encoder transforms latent features into concept representations 
    (logits or features) using a linear layer. It's typically used as the 
    first layer in concept bottleneck models to extract concepts from 
    neural network input.

    Attributes:
        in_latent (int): Number of input latent features.
        out_concepts (int): Number of output concept representations.
        encoder (nn.Sequential): The encoding network.

    Args:
        in_latent: Number of input latent features.
        out_concepts: Number of output concept representations.
        *args: Additional arguments for torch.nn.Linear.
        **kwargs: Additional keyword arguments for torch.nn.Linear.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearLatentToConcept
        >>>
        >>> # Create encoder
        >>> encoder = LinearLatentToConcept(
        ...     in_latent=128,
        ...     out_concepts=10
        ... )
        >>>
        >>> # Forward pass with latent from a neural network
        >>> latent = torch.randn(4, 128)  # batch_size=4, latent_dim=128
        >>> concepts = encoder(latent)
        >>> print(concepts.shape)
        torch.Size([4, 10])
        >>>
        >>> # Apply sigmoid to get probabilities
        >>> concept_probs = torch.sigmoid(concepts)
        >>> print(concept_probs.shape)
        torch.Size([4, 10])

    References:
        Koh et al. "Concept Bottleneck Models", ICML 2020.
        https://arxiv.org/pdf/2007.04612
    """
    def __init__(
        self,
        in_latent: int,
        out_concepts: int,
        *args,
        **kwargs,
    ):
        """
        Initialize the latent encoder.

        Args:
            in_latent: Number of input latent features.
            out_concepts: Number of output concept representations.
            *args: Additional arguments for torch.nn.Linear.
            **kwargs: Additional keyword arguments for torch.nn.Linear.
        """
        super().__init__(
            in_latent=in_latent,
            out_concepts=out_concepts,
        )
        self.encoder = torch.nn.Linear(
            in_latent,
            out_concepts,
            *args,
            **kwargs,
        )

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode latent into concept representations.

        Args:
            latent: Input latent of shape (batch_size, in_latent).

        Returns:
            torch.Tensor: Concept representations of shape (batch_size, out_concepts).
        """
        return self.encoder(latent)


class LinearExogenousToConcept(BaseEncoder):
    """
    Encoder that extracts concepts from exogenous variables.

    This encoder processes exogenous latent variables to produce
    concept representations. It requires at least one exogenous variable per concept.

    Attributes:
        in_exogenous (int): Number of exogenous input features.
        encoder (nn.Sequential): The encoding network.

    Args:
        in_exogenous: Number of exogenous input features.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearExogenousToConcept
        >>>
        >>> # Create encoder of 5 exogenous vars (one per concept)
        >>> encoder = LinearExogenousToConcept(
        ...     in_exogenous=5,
        ... )
        >>>
        >>> # Forward pass with exogenous variables
        >>> # Expected input shape: (batch, out_concepts, in_exogenous)
        >>> exog_vars = torch.randn(4, 3, 5)  # batch=4, concepts=3, exog_features=5
        >>> concepts = encoder(exog_vars)
        >>> print(concepts.shape)
        torch.Size([4, 3])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    def __init__(
        self,
        in_exogenous: int,
    ):
        """
        Initialize the exogenous encoder.

        Args:
            in_exogenous: Number of exogenous input features.
        """
        super().__init__(
            in_exogenous=in_exogenous,
            out_concepts=-1,
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_exogenous,
                1
            ),
            torch.nn.Flatten(),
        )

    def forward(
        self,
        exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode exogenous variables into concept representations.

        Args:
            exogenous: Exogenous variables of shape
                (batch_size, concepts, in_exogenous).

        Returns:
            torch.Tensor: Concept representations of shape 
                (batch_size, concepts).
        """
        return self.encoder(exogenous)
