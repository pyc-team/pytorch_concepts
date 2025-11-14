from typing import Any, Dict, List, Optional, Union, Mapping
from torch import nn
import torch

from torch_concepts import Annotations, Variable
from torch_concepts.distributions import Delta
from torch_concepts.nn import BipartiteModel, ProbEncoderFromEmb, ProbPredictor, ProbabilisticGraphicalModel, \
                              Factor, Propagator, BaseInference

from ..base.model import BaseModel

class CBM(BaseModel):
    """High-level implementation of Concept Bottleneck Model (CBM) \
        using BipartiteModel."""
    def __init__(
        self,
        task_names: Union[List[str], str, List[int]],
        inference: BaseInference,
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
        
        model = BipartiteModel(task_names=task_names,
                               input_size=self.encoder_out_features,
                               annotations=annotations,
                               encoder=Propagator(ProbEncoderFromEmb),
                               predictor=Propagator(ProbPredictor))
        self.pgm = model.pgm

        self.inference = inference(self.pgm)

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





class CBM_factors(BaseModel):
    """Mid-level implementation of Concept Bottleneck Model (CBM) \
        using Variables, Factors and ProbabilisticGraphicalModel."""
    def __init__(
        self,
        task_names: Union[List[str], str, List[int]],
        inference: BaseInference,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Mapping,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        **kwargs
    ) -> None:
        # Initialize the BaseModel
        # this will setup the encoder (torch) layers and the annotations metadata
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
        concept_names = [c for c in annotations.get_axis_annotation(1).labels if c not in task_names]
        concepts = Variable(concept_names,
                            parents=['embedding'], # all concepts have the same parent='embedding'
                            distribution=[annotations[1].metadata[c]['distribution'] for c in concept_names],
                            size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in concept_names])
        
        tasks = Variable(task_names,
                         parents=concept_names, # all tasks have the same parents='concepts'
                         distribution=[annotations[1].metadata[c]['distribution'] for c in task_names],
                         size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in task_names])

        # layers initialization
        concept_encoders = Factor(concept_names, 
                                  module_class=[ProbEncoderFromEmb(in_features_embedding=embedding.size, 
                                                                   out_features=c.size) for c in concepts])
        
        task_predictors = Factor(task_names, 
                                 module_class=[ProbPredictor(in_features_logits=sum([c.size for c in concepts]), 
                                                             out_features=t.size) for t in tasks])

        # PGM Initialization
        self.pgm = ProbabilisticGraphicalModel(
            variables=[embedding, *concepts, *tasks],
            factors=[embedding_factor, *concept_encoders, *task_predictors]
        )

        self.inference = inference(self.pgm)

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