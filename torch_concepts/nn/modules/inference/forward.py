import copy
from abc import ABC

import torch
from torch import nn

from torch_concepts import AnnotatedTensor, ConceptTensor, Annotations, AnnotatedAdjacencyMatrix
from typing import List, Union, Optional, Tuple, Mapping

from ... import GraphModel
from ...base.inference import BaseInference


class KnownGraphInference(BaseInference):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model=model)
        self.train_mode = 'joint'

    def query(self, x: torch.Tensor, *args, **kwargs) -> ConceptTensor:
        # get exogenous
        if self.model.has_exogenous:
            c_exog_roots = self.model.exogenous_roots(x)
            c_exog_internal = self.model.exogenous_internal(x)
        
        # get roots
        if self.model.has_exogenous:
            input_obj = c_exog_roots.extract_by_annotation(self.model.root_nodes)
        else:
            input_obj = x
        c_all = self.model.encoder(input_obj)

        for c_name in self.model.internal_nodes:
            propagator = self.model.predictors[c_name]
            parents = list(self.model.model_graph.get_predecessors(c_name))
            input_obj = c_all.extract_by_annotation(parents)

            if self.model.has_exogenous:
                exog = c_exog_internal.extract_by_annotation([c_name])
                c_out = propagator(input_obj, exog)
            else:
                c_out = propagator(input_obj)

            c_all = c_all.join(c_out)
        return c_all


class UnknownGraphInference(BaseInference):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model=model)
        self.train_mode = 'independent'

    def mask_concept_tensor(self, c: ConceptTensor, model_graph: AnnotatedAdjacencyMatrix, c_name: str) -> torch.Tensor:
        broadcast_shape = [1] * len(c.size())
        broadcast_shape[1] = c.size(1)
        mask = model_graph[:, self.model.to_index(c_name)].view(*broadcast_shape)  # FIXME: get_by_nodes does not work!
        return c * mask.data

    def query(self, x: torch.Tensor, c: ConceptTensor, *args, **kwargs) -> List[ConceptTensor]:
        # --- maybe from embeddings to exogenous
        if self.model.has_exogenous:
            c_exog = self.model.exogenous(x)
        
        #  get roots
        if self.model.has_exogenous:
            input_obj = c_exog.extract_by_annotation(self.model.root_nodes)
        else:
            input_obj = x
        c_encoder = self.model.encoder(input_obj)

        # --- from concepts to concepts copy
        model_graph = self.model.graph_learner()
        c_predictor = ConceptTensor(self.model.annotations)

        for c_name in self.model.annotations.get_axis_labels(axis=1):
            propagator = self.model.predictors[c_name]
            # Mask the input concept object to get only parent concepts
            c_encoder_masked = self.mask_concept_tensor(c_encoder, model_graph, c_name)
            c_masked = self.mask_concept_tensor(c, model_graph, c_name)
            if c_encoder.concept_embs is None:
                input_obj = ConceptTensor(self.model.annotations, concept_probs=c_masked)
            else:
                input_obj = ConceptTensor(self.model.annotations, concept_embs=c_encoder_masked, concept_probs=c_masked)

            if self.model.has_exogenous:
                exog = c_exog.extract_by_annotation([c_name])
                c_out = propagator(input_obj, exog)
            else:
                c_out = propagator(input_obj)

            c_predictor = c_predictor.join(c_out)
        return c_encoder, c_predictor

    def get_model_known_graph(self) -> GraphModel:
        if not hasattr(self, "graph_learner"):
            raise RuntimeError("This LearnedGraphModel was not initialised with a graph learner.")
        known_graph: AnnotatedAdjacencyMatrix = self.graph_learner()

        # Build a GraphModel using the SAME builders -> predictors get the correct in_features
        gm = GraphModel(
            input_size=self.emb_size,
            annotations=self.annotations,
            encoder=self._encoder_builder,
            predictor=self._predictor_builder,
            model_graph=known_graph,
        )

        # ---- helpers ----
        full_order = list(self.concept_names)
        cards = self.annotations.get_axis_cardinalities(axis=1)
        per_card = {lab: (cards[i] if cards is not None else 1) for i, lab in enumerate(full_order)}

        # flat offsets in the "all-concepts" layout used by the wide predictors
        offsets = {}
        cur = 0
        for lab in full_order:
            offsets[lab] = cur
            cur += per_card[lab]

        def expand_indices(labels: list[str]) -> list[int]:
            keep = []
            for lab in labels:
                base = offsets[lab]
                width = per_card[lab]
                keep.extend(range(base, base + width))
            return keep

        def first_linear(module: nn.Module) -> nn.Linear | None:
            if isinstance(module, nn.Linear):
                return module
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        return layer
            # common attribute names
            for name in ("in_proj", "fc", "proj", "input", "linear"):
                m = getattr(module, name, None)
                if isinstance(m, nn.Linear):
                    return m
            return None

        def copy_overlap_columns(old_mod: nn.Module, new_mod: nn.Module, keep_idx: list[int]) -> None:
            old_lin = first_linear(old_mod)
            new_lin = first_linear(new_mod)
            if old_lin is None or new_lin is None:
                return  # nothing generic to copy
            # sanity: output dim must match; new input dim must match keep_idx
            if old_lin.weight.size(0) != new_lin.weight.size(0):
                return
            if new_lin.weight.size(1) != len(keep_idx):
                return
            if len(keep_idx) == 0:
                # no parents -> just copy bias if present
                with torch.no_grad():
                    if new_lin.bias is not None and old_lin.bias is not None:
                        new_lin.bias.copy_(old_lin.bias)
                return
            if max(keep_idx) >= old_lin.weight.size(1):
                return
            with torch.no_grad():
                new_lin.weight.copy_(old_lin.weight[:, keep_idx])
                if new_lin.bias is not None and old_lin.bias is not None:
                    new_lin.bias.copy_(old_lin.bias)

        # ---- copy encoders exactly (roots in known graph) ----
        enc_out = nn.ModuleDict()
        for c in gm.root_nodes:
            enc_out[c] = copy.deepcopy(self.encoders[c]) if hasattr(self, "encoders") and c in self.encoders else \
            gm.encoders[c]
        gm.encoders = enc_out

        # ---- predictors: new (pruned) shapes already correct; now copy overlapping weights ----
        pred_out = nn.ModuleDict()
        for c in gm.internal_nodes:
            parents = list(known_graph.get_predecessors(c))  # labels in some order
            keep_idx = expand_indices(parents)  # flat indices into the old "all-concepts" layout

            new_pred = gm.predictors[c]  # built with correct in_features by _predictor_builder
            if hasattr(self, "predictors") and c in self.predictors:
                old_pred = self.predictors[c]
                copy_overlap_columns(old_pred, new_pred, keep_idx)
            pred_out[c] = new_pred
        gm.predictors = pred_out

        return gm