import numpy as np
import torch

from torch_concepts import AnnotatedAdjacencyMatrix, Annotations, nn
from typing import Union, List

from ..modules.encoders.embedding import ProbEmbEncoder
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

    def _init_encoder(self, layer: Propagator, input_size, concept_names: List[str]) -> torch.nn.Module:
        output_annotations = self.annotations.select(axis=1, keep_labels=concept_names)
        propagator = layer.build(input_size, output_annotations)
        out_features = {}
        for c_name in concept_names:
            output_annotations = self.annotations.select(axis=1, keep_labels=[c_name])
            out_features[c_name] = layer.build(input_size, output_annotations).out_features
        return propagator, out_features
    
    # def _init_exogenous(self, layer: Propagator, input_size, concept_names: List[str]) -> torch.nn.Module:
    #     output_annotations = self.annotations.select(axis=1, keep_labels=concept_names)
    #     propagator = layer.build(input_size, output_annotations)
    #     return propagator

    def _init_predictors(self, 
                         layer: Propagator, 
                         concept_names: List[str], 
                         out_features_roots: dict, 
                         out_features_exog_internal: dict,
                         parent_names: str = None) -> torch.nn.Module:
        if parent_names:
            _parent_names = parent_names

        propagators = torch.nn.ModuleDict()
        for c_name in concept_names:
            output_annotations = self.annotations.select(axis=1, keep_labels=c_name)

            if parent_names is None:
                _parent_names = self.model_graph.get_predecessors(c_name)

            if self.has_exogenous:
                in_features = [out_features_exog_internal[c_name]]
            else:
                in_features = []

            in_features += [out_features for c, out_features in out_features_roots.items() if c in _parent_names]

            if parent_names is None:
                for name, m in propagators.items():
                    c = None
                    if name in _parent_names:
                        c = m.out_features
                    if c is not None:
                        in_features += [c]

            propagators[c_name] = layer.build(in_features, output_annotations)

        return propagators
    
    def to_concept(self, i: int) -> str:
        return self.concept_names[i]

    def to_index(self, c: str) -> int:
        return self.concept_names.index(c)
