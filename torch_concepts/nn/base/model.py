import numpy as np
import torch

from torch_concepts import AnnotatedAdjacencyMatrix, Annotations, nn
from typing import Union, List

from ..modules.propagator import Propagator
from .graph import BaseGraphLearner


class BaseModel(torch.nn.Module):
    """
    BaseModel is an abstract class for all Model modules.
    """

    def __init__(self,
                 input_size: int,
                 annotations: Annotations,
                 encoder: Propagator,  # layer for root concepts
                 predictor: Propagator,
                 model_graph: Union[AnnotatedAdjacencyMatrix, BaseGraphLearner]
                 ):
        super(BaseModel, self).__init__()
        self.emb_size = input_size
        self.concept_names = annotations.get_axis_labels(axis=1)
        self._encoder_builder = encoder
        self._predictor_builder = predictor
        self.annotations = annotations

        # instantiate model graph
        self.model_graph = model_graph

        # # set self.tensor_mode to 'nested' if there are concepts with cardinality > 1
        # if any(v['cardinality'] > 1 for v in self.concept_metadata.values()):
        #     self.tensor_mode = 'nested'
        # else:
        #     self.tensor_mode = 'tensor'
        self.tensor_mode = 'tensor' # TODO: fixme

    def _init_encoder(self, layer: Propagator, concept_names: List[str], in_features_embedding=None, in_features_exogenous=None) -> torch.nn.Module:
        output_annotations = self.annotations.select(axis=1, keep_labels=concept_names)
        propagator = layer.build(
            in_features_embedding=in_features_embedding,
            in_features_logits=None,
            in_features_exogenous=in_features_exogenous,
            out_annotations=output_annotations,
        )
        return propagator

    def _init_predictors(self, 
                         layer: Propagator, 
                         concept_names: List[str],
                         parent_names: str = None) -> torch.nn.Module:
        if parent_names:
            _parent_names = parent_names

        propagators = torch.nn.ModuleDict()
        for c_name in concept_names:
            output_annotations = self.annotations.select(axis=1, keep_labels=[c_name])

            if parent_names is None:
                _parent_names = self.model_graph.get_predecessors(c_name)

            in_features_embedding = 0
            in_features_logits = 0
            in_features_exogenous = 0
            if self.has_exogenous:
                in_features_exogenous = self.predictor_in_exogenous

            for p in _parent_names:
                in_features_embedding += self.predictor_in_embedding
                in_features_logits += self.predictor_in_logits
                in_features_exogenous += self.predictor_in_exogenous

            if parent_names is None:
                for name, m in propagators.items():
                    c = None
                    if name in _parent_names:
                        c = m.out_features
                    if c is not None:
                        in_features_logits += self.predictor_in_logits

            in_features_embedding = None if in_features_embedding == 0 else in_features_embedding
            in_features_logits = None if in_features_logits == 0 else in_features_logits
            in_features_exogenous = None if in_features_exogenous == 0 else in_features_exogenous

            propagators[c_name] = layer.build(
                in_features_embedding=in_features_embedding,
                in_features_logits=in_features_logits,
                in_features_exogenous=in_features_exogenous,
                out_annotations=output_annotations,
            )

        return propagators
    
    def to_concept(self, i: int) -> str:
        return self.concept_names[i]

    def to_index(self, c: str) -> int:
        return self.concept_names.index(c)
