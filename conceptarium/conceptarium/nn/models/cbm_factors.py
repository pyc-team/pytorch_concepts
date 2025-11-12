from typing import Dict, List, Optional, Union, Mapping

from torch_concepts import Annotations, Variable
from torch_concepts.nn import ProbEncoderFromEmb, ProbPredictor, ProbabilisticGraphicalModel, Factor

from ..base.model import BaseModel

class CBM(BaseModel):
    def __init__(
        self,
        task_names: Union[List[str], str, List[int]],
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
        concept_names = [c for c in annotations.get_axis_annotation(1).labels if c not in task_names]
        concepts = Variable(concept_names,
                            parents=['emb'], # all concepts have the same parent='emb'
                            distribution=[annotations[1].metadata[c]['distribution'] for c in concept_names],
                            size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in concept_names])
        
        tasks = Variable(task_names,
                         parents=concept_names, # all tasks have the same parents='concepts'
                         distribution=[annotations[1].metadata[c]['distribution'] for c in task_names],
                         size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in task_names])

        # layers initialization
        encoder_in_size = self.emb.size
        concept_encoders = Factor(concept_names, 
                                  module_class=[ProbEncoderFromEmb(in_features_embedding=encoder_in_size, 
                                                                   out_features=c.size) for c in concepts])
        
        predictor_in_size = sum([c.size for c in concepts])
        task_predictors = Factor(task_names, 
                                 module_class=[ProbPredictor(in_features_logits=predictor_in_size, 
                                                             out_features=t.size) for t in tasks])

        # PGM Initialization
        self.pgm = ProbabilisticGraphicalModel(
            variables=[self.emb, *concepts, *tasks],
            factors=[self.emb_factor, *concept_encoders, *task_predictors]
        )

    def filter_output_for_loss(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out