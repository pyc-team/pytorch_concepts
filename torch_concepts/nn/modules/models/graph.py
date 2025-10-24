from copy import deepcopy

import torch
from torch import nn

from torch_concepts import AnnotatedAdjacencyMatrix, Annotations
from ....nn import BaseModel, Propagator, BaseGraphLearner


class GraphModel(BaseModel):
    """
    Model using a given graph structure between concepts and tasks.
    The graph structure is provided as an adjacency matrix during initialization.
    """
    def __init__(self,
                 input_size: int,
                 annotations: Annotations,
                 encoder: Propagator,
                 predictor: Propagator,
                 model_graph: AnnotatedAdjacencyMatrix,
                 ):
        super(GraphModel, self).__init__(
            input_size=input_size,
            annotations=annotations,
            encoder=encoder,
            predictor=predictor,
            model_graph=model_graph,
            include_encoders=True,
            include_predictors=True,
        )


class LearnedGraphModel(BaseModel):
    """
    Model using a graph structure between concepts and tasks.
    The graph structure is learned during training.
    """
    def __init__(self,
                 input_size: int,
                 annotations: Annotations,
                 encoder: Propagator,
                 predictor: Propagator,
                 model_graph: BaseGraphLearner,
                 ):
        super(LearnedGraphModel, self).__init__(
            input_size=input_size,
            annotations=annotations,
            encoder=encoder,
            predictor=predictor,
            model_graph=model_graph,  # learned graph
            include_encoders=True,
            include_predictors=False,
        )

    def get_model_known_graph(self) -> GraphModel:
        """
        Convert this LearnedGraphModel into a GraphModel with a fixed, materialised graph.
        Each predictor is deep-copied and its FIRST Linear layer is physically pruned so that
        in_features equals the sum of the kept parents' cardinalities; the kept columns and
        bias are copied so behaviour matches the original when dropped inputs are zeroed.
        """
        if not hasattr(self, "graph_learner"):
            raise RuntimeError("This LearnedGraphModel was not initialised with a graph learner.")
        known_graph: AnnotatedAdjacencyMatrix = self.graph_learner()

        # Build a light GraphModel shell; we will overwrite encoders/predictors
        class _NoOpProp:
            def build(self, input_size: int, output_annotations: Annotations) -> nn.Module:
                return nn.Identity()

        gm = GraphModel(
            input_size=self.emb_size,
            annotations=self.annotations,
            encoder=_NoOpProp(),
            predictor=_NoOpProp(),
            model_graph=known_graph,
        )

        # ---------------- helpers ---------------- #
        full_order = list(self.concept_names)
        cards = self.annotations.get_axis_cardinalities(axis=1)
        per_card = {lab: (cards[i] if cards is not None else 1) for i, lab in enumerate(full_order)}

        # flat offsets in the "all-concepts" parent layout used by the wide predictors
        offsets = {}
        cur = 0
        for lab in full_order:
            offsets[lab] = cur
            cur += per_card[lab]

        def expand_indices(labels: list[str]) -> list[int]:
            """Expand parent concept labels to flat feature indices (respecting cardinalities)."""
            keep = []
            for lab in labels:
                base = offsets[lab]
                width = per_card[lab]
                keep.extend(range(base, base + width))
            return keep

        def _find_first_linear(parent: nn.Module):
            """
            Depth-first search to locate the first nn.Linear and its parent + attr key
            so we can replace it robustly (works for nested/Sequential/custom containers).
            Returns (parent_module, key, linear_module) where key is either int (Sequential)
            or str (attribute name). Returns (None, None, None) if not found.
            """
            # direct module is Linear
            if isinstance(parent, nn.Linear):
                return None, None, parent  # caller will handle root replacement

            # search named children
            for name, child in parent.named_children():
                if isinstance(child, nn.Linear):
                    return parent, name, child
                # dive deeper
                p, k, lin = _find_first_linear(child)
                if lin is not None:
                    return p if p is not None else parent, k, lin
            return None, None, None

        # FIXME: this runs but is untested
        def _prune_first_linear_inplace(module: nn.Module, keep_idx: list[int]) -> nn.Module:
            """
            Return a new module where the first nn.Linear has been replaced by a pruned Linear
            with in_features=len(keep_idx) and copied weight columns + bias.
            Works even for deeply nested predictors. If no Linear is found, returns a deepcopy.
            """
            mod = deepcopy(module)
            parent, key, lin = _find_first_linear(mod)

            if lin is None:
                # Nothing to prune generically; return a copy as-is
                return mod

            out_f, in_f = lin.weight.shape
            new_in = len(keep_idx)

            # Build pruned Linear; PyTorch supports in_features=0 (weight [out,0]) → output = bias
            new_lin = nn.Linear(new_in, out_f, bias=(lin.bias is not None),
                                dtype=lin.weight.dtype, device=lin.weight.device)
            with torch.no_grad():
                if new_in > 0:
                    # safety: ensure indices are valid
                    if keep_idx and max(keep_idx) >= in_f:
                        raise RuntimeError(f"keep_idx contains invalid column (>= {in_f})")
                    new_lin.weight.copy_(lin.weight[:, keep_idx])
                else:
                    new_lin.weight.zero_()
                if new_lin.bias is not None and lin.bias is not None:
                    new_lin.bias.copy_(lin.bias)

            # Replace lin under its parent (root if parent is None)
            if parent is None:
                # module itself is Linear
                mod = new_lin
            else:
                if isinstance(parent, nn.Sequential) and isinstance(key, str):
                    # named_children on Sequential yields string keys; convert to int index
                    idx = int(key)
                    parent[idx] = new_lin
                elif isinstance(key, int):
                    parent[key] = new_lin
                else:
                    setattr(parent, key, new_lin)

            return mod

        # ---------------- copy encoders exactly ---------------- #
        enc_out = nn.ModuleDict()
        for c_name in gm.root_nodes:
            enc_out[c_name] = deepcopy(self.encoders[c_name]) if hasattr(self,
                                                                         "encoders") and c_name in self.encoders else nn.Identity()
        gm.encoders = enc_out

        # ---------------- prune predictors to known parents ---------------- #
        pred_out = nn.ModuleDict()
        for c_name in gm.internal_nodes:
            parents = list(known_graph.get_predecessors(c_name))  # list of parent concept labels
            keep_idx = expand_indices(parents)  # flat indices in the wide parent layout

            if hasattr(self, "predictors") and c_name in self.predictors:
                old_pred = self.predictors[c_name]
                new_pred = _prune_first_linear_inplace(old_pred, keep_idx)
                pred_out[c_name] = new_pred
            else:
                # no trained predictor → minimal compatible default
                in_dim = len(keep_idx)
                out_dim = per_card[c_name]
                pred_out[c_name] = nn.Identity() if in_dim == 0 else nn.Sequential(nn.Linear(in_dim, out_dim))
        gm.predictors = pred_out

        return gm
