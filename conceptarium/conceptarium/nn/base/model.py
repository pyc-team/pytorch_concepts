from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Mapping, Dict
import torch
import torch.nn as nn

from torch_concepts import Annotations
from torch_concepts.nn import BaseInference

from ...nn.dense_layers import MLP
from ...typing import BackboneType
from ...utils import add_distribution_to_annotations

class BaseModel(nn.Module, ABC):

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
        """The encoder mapping backbone output to latent code(s)."""
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
        """"""
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

        Parameters
        ----------
        x: Raw input tensor or already computed embeddings.
        **backbone_kwargs: Extra keyword arguments forwarded to the backbone callable when
            it is invoked.
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
        return out_concepts
    
    def filter_output_for_metric(self, out_concepts):
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

        Parameters
        ----------
        inputs: Raw input tensor.
        concepts: Ground-truth concepts tensor.

        Returns
        -------
        preprocessed_inputs: Preprocessed input tensor.
        preprocessed_concepts: Preprocessed concepts tensor.
        """
        return inputs, concepts


    # ------------------------------------------------------------------
    # Inference configuration
    # ------------------------------------------------------------------
    def set_inference(self, inference: BaseInference) -> None:
        self.inference = inference

    def set_and_instantiate_inference(self, inference: BaseInference) -> None:
        self.inference = inference(probabilistic_model=self.probabilistic_model)
