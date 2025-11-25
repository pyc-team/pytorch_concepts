"""Base model class for concept-based neural networks.

This module defines the abstract BaseModel class that serves as the foundation
for all concept-based models in the library. It handles backbone integration,
encoder setup, and provides hooks for data preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Mapping, Dict
import torch
import torch.nn as nn

from ...low.dense_layers import MLP
from .....typing import BackboneType

class BaseModel(nn.Module, ABC):
    """Abstract base class for concept-based models.

    Provides common functionality for models that use backbones for feature extraction, 
    and encoders for latent representations. All concrete model implementations 
    should inherit from this class.

    Args:
        input_size (int): Dimensionality of input features (after backbone, if used).
        backbone (BackboneType, optional): Feature extraction backbone (e.g., ResNet,
            ViT). Can be a nn.Module or callable. If None, assumes latent representations
            are pre-computed. Defaults to None.
        latent_encoder_kwargs (Dict, optional): Arguments for MLP latent encoder
            (e.g., {'hidden_size': 128, 'n_layers': 2}). If None, uses Identity.
            Defaults to None.

    Attributes:
        annotations (Annotations): Annotated concept variables with distribution info.
        backbone (BackboneType): Feature extraction module (None if precomputed).
        latent_encoder_out_features (int): Output dimensionality of latent encoder.
    """

    def __init__(
        self,
        input_size: int,
        backbone: Optional[BackboneType] = None,
        latent_encoder: Optional[nn.Module] = None,
        latent_encoder_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._backbone = backbone

        if latent_encoder is not None:
            self._latent_encoder = latent_encoder(input_size,
                                    **(latent_encoder_kwargs or {}))
        elif latent_encoder_kwargs is not None:
            # assume an MLP encoder if latent_encoder_kwargs provided but no latent_encoder
            self._latent_encoder = MLP(input_size=input_size,
                                **latent_encoder_kwargs)
        else:
            self._latent_encoder = nn.Identity()

        self.latent_size = latent_encoder_kwargs.get('hidden_size') if latent_encoder_kwargs else input_size

    def __repr__(self):
        backbone_name = self.backbone.__class__.__name__ if self.backbone is not None else "None"
        latent_encoder_name = self._latent_encoder.__class__.__name__ if self._latent_encoder is not None else "None"
        return f"{self.__class__.__name__}(backbone={backbone_name}, latent_encoder={latent_encoder_name})"

    @property
    def backbone(self) -> BackboneType:
        """The backbone feature extractor.

        Returns:
            BackboneType: Backbone module or callable.
        """
        return self._backbone

    @property
    def latent_encoder(self) -> nn.Module:
        """The encoder mapping backbone output to input(s).

        Returns:
            nn.Module: Latent encoder network.
        """
        return self._latent_encoder

    # TODO: add decoder?
    # @property
    # def encoder(self) -> nn.Module:
    #     """The decoder mapping back to the input space.

    #     Returns:
    #         nn.Module: Decoder network.
    #     """
    #     return self._encoder

    @abstractmethod
    def filter_output_for_loss(self, forward_out, target):
        """Filter model outputs before passing to loss function.

        Override this method in your model to customize what outputs are passed to the loss.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in loss computation or viceversa.

        Args:
            forward_out: Model output (typically concept predictions).
            target: Ground truth concepts.
        Returns:
            dict: Filtered outputs for loss computation.
        """
        pass

    @abstractmethod
    def filter_output_for_metric(self, forward_out, target):
        """Filter model outputs before passing to metric computation.

        Override this method in your model to customize what outputs are passed to the metrics.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in metric computation or viceversa.

        Args:
            forward_out: Model output (typically concept predictions).
            target: Ground truth concepts.
        Returns:
            dict: Filtered outputs for metric computation.
        """
        pass

    # ------------------------------------------------------------------
    # Features extraction helpers
    # ------------------------------------------------------------------

    def maybe_apply_backbone(
        self,
        x: torch.Tensor,
        backbone_args: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the backbone to ``x`` unless features are pre-computed.

        Args:
            x (torch.Tensor): Raw input tensor or already computed embeddings.
            backbone_kwargs (Any): Extra keyword arguments forwarded to the
                backbone callable when it is invoked.

        Returns:
            torch.Tensor: Feature embeddings.

        Raises:
            TypeError: If backbone is not None and not callable.
        """

        if self.backbone is None:
            return x

        if not callable(self.backbone):
            raise TypeError(
                "The provided backbone is not callable. Received "
                f"instance of type {type(self.backbone).__name__}."
            )

        return self.backbone(x, **backbone_args if backbone_args else {})


    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    
    def filter_output_for_loss(self, out_concepts):
        """Filter model outputs before passing to loss function.

        Override this method to customize what outputs are passed to the loss.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in loss computation or viceversa.

        Args:
            out_concepts: Model output (typically concept predictions).

        Returns:
            Filtered output passed to loss function. By default, returns
            out_concepts unchanged.

        Example:
            >>> def filter_output_for_loss(self, out):
            ...     # Only use concept predictions, ignore attention weights
            ...     return out['concepts']
        """
        return out_concepts
    
    def filter_output_for_metric(self, out_concepts):
        """Filter model outputs before passing to metrics.

        Override this method to customize what outputs are passed to metrics.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in metric computation or viceversa.

        Args:
            out_concepts: Model output (typically concept predictions).

        Returns:
            Filtered output passed to metrics. By default, returns
            out_concepts unchanged.
        """
        return out_concepts
