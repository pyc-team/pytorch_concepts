from operator import itemgetter

import numpy as np
import torch

from torch_concepts import ConceptGraph, Annotations, nn
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
                 model_graph: Union[ConceptGraph, BaseGraphLearner],
                 predictor_in_embedding: int,
                 predictor_in_exogenous: int,
                 has_self_exogenous: bool = False,
                 has_parent_exogenous: bool = False,
                 exogenous: Propagator = None
                 ):
        super(BaseModel, self).__init__()
        self.emb_size = input_size
        self.concept_names = annotations.get_axis_labels(axis=1)
        self.name2id = {name: i for i, name in enumerate(self.concept_names)}
        self._encoder_builder = encoder
        self._predictor_builder = predictor
        self._exogenous_builder = exogenous
        self.annotations = annotations

        # instantiate model graph
        self.model_graph = model_graph

        # # set self.tensor_mode to 'nested' if there are concepts with cardinality > 1
        # if any(v['cardinality'] > 1 for v in self.concept_metadata.values()):
        #     self.tensor_mode = 'nested'
        # else:
        #     self.tensor_mode = 'tensor'
        self.tensor_mode = 'tensor' # TODO: fixme

        self.predictor_in_embedding = predictor_in_embedding
        self.predictor_in_exogenous = predictor_in_exogenous
        self.predictor_in_logits = 1
        self.has_self_exogenous = has_self_exogenous
        self.has_parent_exogenous = has_parent_exogenous

        self.has_exogenous = exogenous is not None

    def _init_encoder(self, layer: Propagator, concept_names: List[str], in_features_embedding=None, in_features_exogenous=None) -> torch.nn.Module:
        output_annotations = self.annotations.select(axis=1, keep_labels=concept_names)
        propagator = layer.build(
            in_features_embedding=in_features_embedding,
            in_features_logits=None,
            in_features_exogenous=in_features_exogenous,
            out_annotations=output_annotations,
        )
        return propagator

    def _make_single_fetcher(self, idx: int):
        """Return a callable that always yields a 1-tuple (outs[idx],)."""
        return lambda vals, j=idx: (vals[j],)

    def _init_fetchers(self, parent_names = None):
        """Build fetchers that read tensors by fixed concept-id."""
        if parent_names:
            self.arity = len(parent_names)
            pids = tuple(self.name2id[p] for p in parent_names)
            self.fetchers = itemgetter(*pids)
            return

        fetchers = []
        arity = []
        name2id = self.name2id  # pre-computed map name → concept-id

        cardinalities = self.annotations.get_axis_annotation(axis=1).cardinalities
        for c_name in self.internal_nodes:
            parents = self.model_graph.get_predecessors(c_name)

            pids = tuple(name2id[p] for p in parents)
            n_parents = len(pids)
            if cardinalities is not None:
                card = sum([cardinalities[p] for p in pids])
            else:
                card = n_parents
            arity.append(card)

            if n_parents == 1:
                fetchers.append(self._make_single_fetcher(pids[0]))  # 1-tuple
            else:
                fetchers.append(itemgetter(*pids))  # tuple of tensors

        self.fetchers = fetchers
        self.arity = arity
        return

    def _init_predictors(self, 
                         layer: Propagator, 
                         concept_names: List[str]) -> torch.nn.Module:
        propagators = torch.nn.ModuleDict()
        for c_id, c_name in enumerate(concept_names):
            output_annotations = self.annotations.select(axis=1, keep_labels=[c_name])

            if isinstance(self.arity, int):
                n_parents = self.arity
            else:
                n_parents = self.arity[c_id]

            in_features_logits = self.predictor_in_logits * n_parents
            in_features_embedding = self.predictor_in_embedding
            in_features_exogenous = self.predictor_in_exogenous

            # if parent_names is None:
            #     for name, m in propagators.items():
            #         c = None
            #         if name in _parent_names:
            #             c = m.out_features
            #         if c is not None:
            #             in_features_logits += self.predictor_in_logits

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
