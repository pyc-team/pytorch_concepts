from typing import Dict, List, Optional, Union, Tuple, Mapping
from torch import Tensor

from torch_concepts import Annotations
from torch_concepts.nn import BipartiteModel, ProbEncoderFromEmb, ProbPredictor, Propagator

from conceptarium.nn.base.model import BaseModel


class CBM(BaseModel):
    def __init__(
        self,
        task_names: Union[List[str], List[int]],
        input_size: int,
        concept_annotations: Annotations,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        **kwargs
    ) -> None:
        super().__init__(
            concept_annotations=concept_annotations,
            # encoder params
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder_kwargs=encoder_kwargs,
        )

        concept_encoder = Propagator(ProbEncoderFromEmb)
        concept_predictor = Propagator(ProbPredictor)
        
        self.model = BipartiteModel(task_names=task_names,
                                    encoder=concept_encoder,
                                    predictor=concept_predictor,
                                    annotations=concept_annotations,
                                    predictor_in_embedding=0,
                                    predictor_in_exogenous=0,
                                    has_self_exogenous=False,
                                    has_parent_exogenous=False,
                                    input_size=self.encoder_out_features)

    def filter_output_for_loss(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out