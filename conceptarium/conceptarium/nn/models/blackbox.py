import torch
from typing import Any, Optional, Dict, Mapping

from torch_concepts import Annotations, Variable
from torch_concepts.nn import Factor, ProbEncoderFromEmb, ProbabilisticGraphicalModel

from ..dense_layers import MLP
from ..base.model import BaseModel


class BlackBox(BaseModel):
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Mapping,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        **kwargs
    ) -> None:
        super().__init__(
            annotations=annotations,
            variable_distributions=variable_distributions,
            # encoder params
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder_kwargs=encoder_kwargs,
        )

        # Variable and Factor for the latent code ('self.emb') 
        # are initialized in the BaseModel
        
        # variables initialization
        concept_names = self.annotations.get_axis_labels(1)
        concepts = Variable(concept_names,
                            parents=['emb'], # all concepts have the same parent='emb'
                            distribution=[annotations[1].metadata[c]['distribution'] for c in concept_names],
                            size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in concept_names])
        
        # layers initialization
        concept_encoders = Factor(concept_names, 
                                  module_class=[ProbEncoderFromEmb(in_features_embedding=self.emb.size, 
                                                                   out_features=c.size) for c in concepts])
        

        # PGM Initialization
        self.pgm = ProbabilisticGraphicalModel(
            variables=[self.emb, *concepts],
            factors=[self.emb_factor, *concept_encoders]
        )

    def filter_output_for_loss(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out
    





class BlackBox_torch(BaseModel):
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Mapping,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        **kwargs
    ) -> None:
        super().__init__(
            annotations=annotations,
            variable_distributions=variable_distributions,
            # encoder params
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder_kwargs=encoder_kwargs,
        )

        self.concept_annotations = annotations.get_axis_annotation(1)
        self.mlp = MLP(input_size=input_size,
                       output_size=sum(self.concept_annotations.cardinalities),
                       **encoder_kwargs
                       )
        

    def forward(self,
                x: torch.Tensor,
                backbone_kwargs: Optional[Mapping[str, Any]] = None,
                *args,
                **kwargs):
        features = self.maybe_apply_backbone(x, backbone_kwargs)
        logits = self.mlp(features)
        return logits

    def filter_output_for_loss(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out