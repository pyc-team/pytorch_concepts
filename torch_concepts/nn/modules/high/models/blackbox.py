import torch
from torch import nn
from typing import Any, List, Optional, Dict, Mapping

from .....annotations import Annotations
from ....modules.mid.models.variable import Variable
from .....distributions.delta import Delta
from ....modules.mid.models.factor import Factor
from ....modules.low.encoders.linear import ProbEncoderFromEmb
from ....modules.mid.models.probabilistic_model import ProbabilisticModel
from ....modules.low.base.inference import BaseInference

from ...low.dense_layers import MLP
from ..base.model import BaseModel



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
        
    def filter_output_for_loss(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out
    
    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
                *args,
                backbone_kwargs: Optional[Mapping[str, Any]] = None,
                **kwargs
            ) -> torch.Tensor:
        features = self.maybe_apply_backbone(x, backbone_kwargs)
        logits = self.mlp(features)
        return logits



class BlackBox(BaseModel):
    def __init__(
        self,
        input_size: int,
        inference: BaseInference,
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

        # init variable for the latent embedding from the encoder
        embedding = Variable("embedding", parents=[], distribution=Delta, size=self.encoder_out_features)
        embedding_factor = Factor("embedding", module_class=nn.Identity())
        
        # variables initialization
        concept_names = self.annotations.get_axis_labels(1)
        concepts = Variable(concept_names,
                            parents=['embedding'], # all concepts have the same parent='embedding'
                            distribution=[annotations[1].metadata[c]['distribution'] for c in concept_names],
                            size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in concept_names])
        
        # layers initialization
        concept_encoders = Factor(concept_names, 
                                  module_class=[ProbEncoderFromEmb(in_features_embedding=embedding.size, 
                                                                   out_features=c.size) for c in concepts])
        
        # ProbabilisticModel Initialization
        self.probabilistic_model = ProbabilisticModel(
            variables=[embedding, *concepts],
            factors=[embedding_factor, *concept_encoders]
        )

        self.inference = inference(self.probabilistic_model)

    def filter_output_for_loss(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out
    
    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
                *args,
                backbone_kwargs: Optional[Mapping[str, Any]] = None,
                **kwargs
            ) -> torch.Tensor:
        
        # (b, input_size) -> (b, backbone_out_features)
        features = self.maybe_apply_backbone(x, backbone_kwargs)

        # (b, backbone_out_features) -> (b, encoder_out_features)
        features = self.encoder(features)

        # inference
        # get logits for the query concepts
        # (b, encoder_out_features) -> (b, sum(concept_cardinalities))
        out = self.inference.query(query, evidence={'embedding': features})
        return out