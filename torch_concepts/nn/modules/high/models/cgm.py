from typing import Dict, List, Optional, Union, Tuple, Mapping
from torch import Tensor

from .....data.annotations import Annotations
from ....modules.mid.constructors.graph import GraphModel as LearnedGraphModel
from ....modules.low.encoders.exogenous import ExogEncoder
from ....modules.low.encoders.linear import ProbEncoderFromExog
from ....modules.low.predictors.embedding import MixProbExogPredictor
from ....modules.propagator import LazyConstructor
from ....modules.low.graph.wanda import WANDAGraphLearner as COSMOGraphLearner

from ..base.model import BaseModel


class CGM(BaseModel):
    def __init__(
        self,
        input_size: int,
        concept_annotations: Annotations,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        exog_encoder_embedding_size: int = 16,
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

        exogenous_encoder = LazyConstructor(ExogEncoder,
                                       embedding_size=exog_encoder_embedding_size*2)
        
        concept_encoder = LazyConstructor(ProbEncoderFromExog)

        concept_predictor = LazyConstructor(MixProbExogPredictor)

        self.model = LearnedGraphModel(model_graph=COSMOGraphLearner,
                                       exogenous=exogenous_encoder,
                                       encoder=concept_encoder,
                                       predictor=concept_predictor,
                                       annotations=concept_annotations,
                                       predictor_in_embedding=0,
                                       predictor_in_exogenous=exog_encoder_embedding_size,
                                       has_self_exogenous=False,
                                       has_parent_exogenous=True,
                                       input_size=self.encoder_out_features)

    def filter_output_for_loss(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        # forward_out: logits
        # return: logits
        return forward_out