import numpy as np
import torch

from torch_concepts import AnnotatedAdjacencyMatrix, Annotations
from typing import Union, List

from ..modules.encoders.embedding import ProbEmbEncoderLayer
from ..modules.propagator import Propagator
from .graph import BaseGraphLearner


class BaseModel(torch.nn.Module):
    """
    BaseReasoner is an abstract class for reasoner modules.
    """

    def __init__(self,
                 input_size: int,
                 annotations: Annotations,
                 encoder: Propagator,  # layer for root concepts
                 predictor: Propagator,
                 model_graph: Union[AnnotatedAdjacencyMatrix, BaseGraphLearner],
                 include_encoders: bool,
                 include_predictors: bool,
                 ):
        super(BaseModel, self).__init__()
        self.emb_size = input_size
        concept_names = annotations.get_axis_labels(axis=1)
        self.concept_names = concept_names
        self._encoder_builder = encoder
        self._predictor_builder = predictor
        self.include_encoders = include_encoders
        self.include_predictors = include_predictors

        # handle model graph
        self.model_graph = model_graph
        if isinstance(model_graph, AnnotatedAdjacencyMatrix):
            assert model_graph.is_directed_acyclic(), "Input model graph must be a directed acyclic graph."
            assert model_graph.annotations.get_axis_labels(axis=1) == concept_names, "concept_names must match model_graph annotations."
            self.roots = model_graph.get_root_nodes()
            self.graph_order = model_graph.topological_sort()  # TODO: group by graph levels?
        else:
            # if model_graph is None, create a fully connected graph, and sparsify this during training
            self.roots = concept_names  # all concepts are roots in a fully connected graph
            self.graph_order = None

        # handle concept metadata
        self.annotations = annotations
        self.root_nodes = [r for r in self.roots]
        self.internal_nodes = [c for c in concept_names if c not in self.root_nodes]
        # # set self.tensor_mode to 'nested' if there are concepts with cardinality > 1
        # if any(v['cardinality'] > 1 for v in self.concept_metadata.values()):
        #     self.tensor_mode = 'nested'
        # else:
        #     self.tensor_mode = 'tensor'
        self.tensor_mode = 'tensor' # TODO: fixme

        # define the layers based on the model_graph structure
        if isinstance(model_graph, AnnotatedAdjacencyMatrix):
            self.encoders = self._init_encoders(encoder, concept_names=self.root_nodes)
            self.predictors = self._init_predictors(predictor, concept_names=self.internal_nodes)
        else:
            self.encoders = self._init_encoders(encoder, concept_names=self.concept_names)
            self.predictors = self._init_predictors(predictor, concept_names=self.concept_names)
            self.graph_learner = model_graph(annotations=annotations)

    def _init_encoders(self, layer: Propagator, concept_names: List[str]) -> torch.nn.Module:
        propagators = torch.nn.ModuleDict()
        for c_name in concept_names:
            output_annotations = self.annotations.select(axis=1, keep_labels=[c_name])
            propagators[c_name] = layer.build(self.emb_size, output_annotations)
        return propagators

    def _init_predictors(self, layer: Propagator, concept_names: List[str]) -> torch.nn.Module:
        propagators = torch.nn.ModuleDict()
        for c_name in concept_names:
            output_annotations = self.annotations.select(axis=1, keep_labels=c_name)
            if isinstance(self.model_graph, AnnotatedAdjacencyMatrix):
                parent_names = self.model_graph.get_predecessors(c_name)
            else:
                parent_names = self.concept_names

            in_contracts = []
            if self.include_encoders:
                in_contracts += [m.out_contract for name, m in self.encoders.items() if name in parent_names]

            if self.include_predictors:
                for name, m in propagators.items():
                    c = None
                    if name in parent_names:
                        c = m.out_contract
                    if c is not None:
                        in_contracts += [c]

            # FIXME
            # if self.residual_encoders and :
            #     c
            propagators[c_name] = layer.build(in_contracts, output_annotations)

        return propagators

    def to_concept(self, i: int) -> str:
        return self.concept_names[i]

    def to_index(self, c: str) -> int:
        return self.concept_names.index(c)
