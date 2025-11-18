"""Base model class for concept-based neural networks.

This module defines the abstract BaseModel class that serves as the foundation
for all concept-based models in the library. It handles backbone integration,
encoder setup, annotation management, and provides hooks for data preprocessing.
"""

from abc import ABC
from typing import Any, Optional, Tuple, Mapping, Dict
import torch
import torch.nn as nn

from torch_concepts import Annotations
from torch_concepts.nn import BaseInference

from ...nn.dense_layers import MLP
from ...typing import BackboneType
from ...utils import add_distribution_to_annotations

class BaseModel(nn.Module, ABC):
    """Abstract base class for concept-based models.
    
    Provides common functionality for models that use concept annotations,
    backbones for feature extraction, and encoders for latent representations.
    All concrete model implementations should inherit from this class.
    
    Args:
        annotations (Annotations): Concept annotations defining variables and 
            their properties (names, types, cardinalities).
        variable_distributions (Mapping): Dictionary mapping variable names to 
            their distribution types (e.g., {'age': 'categorical', 'score': 'continuous'}).
        input_size (int): Dimensionality of input features (after backbone, if used).
        embs_precomputed (bool, optional): Whether embeddings are pre-computed 
            (skips backbone). Defaults to False.
        backbone (BackboneType, optional): Feature extraction backbone (e.g., ResNet, 
            ViT). Can be a nn.Module or callable. Defaults to None.
        encoder_kwargs (Dict, optional): Arguments for MLP encoder 
            (e.g., {'hidden_size': 128, 'n_layers': 2}). If None, uses Identity. 
            Defaults to None.
            
    Attributes:
        annotations (Annotations): Annotated concept variables with distribution info.
        embs_precomputed (bool): Whether to skip backbone processing.
        backbone (BackboneType): Feature extraction module.
        encoder_out_features (int): Output dimensionality of encoder.
    """

    def __init__(
        self,
        annotations: Annotations,
        variable_distributions: Mapping,
        input_size: int,
        embs_precomputed: bool = False,
        backbone: BackboneType = None,
        encoder_kwargs: Dict = None,
    ) -> None:
        super().__init__()

        # Add distribution information to annotations metadata
        annotations = add_distribution_to_annotations(
            annotations, variable_distributions
        )
        # store annotations, these will be used outside the model to track metrics and loss
        # if you extend these annotations, keep in mind that
        # the annotations used for metrics and loss computation should remain consistent
        # you can use the 'preprocess_batch' method to adapt data to your model
        self.annotations = annotations

        self.embs_precomputed = embs_precomputed
        self.backbone = backbone

        if encoder_kwargs is not None:
            self._encoder = MLP(input_size=input_size,
                               **encoder_kwargs)
        else:
            self._encoder = nn.Identity()

        self.encoder_out_features = encoder_kwargs.get('hidden_size') if encoder_kwargs else input_size

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        backbone_repr = (
            self.backbone.__class__.__name__
            if isinstance(self.backbone, nn.Module)
            else type(self.backbone).__name__
            if self.backbone is not None
            else "None"
        )
        return (
            f"{cls_name}(backbone={backbone_repr})"
        )

    @property
    def encoder(self) -> nn.Module:
        """The encoder mapping backbone output to latent code(s).
        
        Returns:
            nn.Module: Encoder network (MLP or Identity).
        """
        return self._encoder

    # TODO: add decoder?
    # @property
    # @abstractmethod
    # def decoder(self) -> nn.Module:
    #     """The decoder mapping concepts and derivatives to an output."""
    #     pass

    def forward(self,
                x: torch.Tensor,
                backbone_kwargs: Optional[Mapping[str, Any]] = None,
                *args,
                **kwargs):
        """Forward pass through backbone and encoder.
        
        Args:
            x (torch.Tensor): Input tensor. Raw data if backbone is used,
                or pre-computed embeddings if embs_precomputed=True.
            backbone_kwargs (Mapping[str, Any], optional): Additional arguments
                passed to the backbone (e.g., {'return_features': True}).
                
        Returns:
            torch.Tensor: Encoded representations.
            
        Note:
            Subclasses typically override this to add concept prediction layers.
        """
        features = self.maybe_apply_backbone(x, backbone_kwargs)
        out = self.encoder(features)
        return out


    # ------------------------------------------------------------------
    # Embeddings extraction helpers
    # ------------------------------------------------------------------

    def maybe_apply_backbone(
        self,
        x: torch.Tensor,
        backbone_kwargs: Any,
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

        if self.embs_precomputed or self.backbone is None:
            return x

        if not callable(self.backbone):
            raise TypeError(
                "The provided backbone is not callable. Received "
                f"instance of type {type(self.backbone).__name__}."
            )

        return self.backbone(x, **backbone_kwargs)


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
    

    # ------------------------------------------------------------------
    # Model-specific data processing
    # ------------------------------------------------------------------

    def preprocess_batch(
        self,
        inputs: torch.Tensor,
        concepts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model-specific preprocessing of a batch.

        Override this to apply transformations before forward pass. Useful for:
        - Data augmentation
        - Normalization specific to your model
        - Handling missing values
        - Converting data formats

        Args:
            inputs (torch.Tensor): Raw input tensor.
            concepts (torch.Tensor): Ground-truth concepts tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - preprocessed_inputs: Preprocessed input tensor.
                - preprocessed_concepts: Preprocessed concepts tensor.
                
        Example:
            >>> def preprocess_batch(self, inputs, concepts):
            ...     # Add noise augmentation
            ...     inputs = inputs + 0.01 * torch.randn_like(inputs)
            ...     return inputs, concepts
        """
        return inputs, concepts


    # ------------------------------------------------------------------
    # Inference configuration
    # ------------------------------------------------------------------
    def set_inference(self, inference: BaseInference) -> None:
        """Set the inference strategy for the model.
        
        Args:
            inference (BaseInference): Instantiated inference object 
                (e.g., MaximumLikelihood, MaximumAPosteriori).
        """
        self.inference = inference

    def set_and_instantiate_inference(self, inference: BaseInference) -> None:
        """Set and instantiate inference strategy using model's PGM.
        
        Args:
            inference (BaseInference): Uninstantiated inference class that 
                will be instantiated with pgm=self.pgm.
                
        Note:
            Requires the model to have a 'pgm' attribute (probabilistic 
            graphical model).
        """
        self.inference = inference(pgm=self.pgm)
