import inspect

import torch

from torch_concepts import ConceptGraph, Variable
from torch_concepts.nn import BaseModel
from typing import List, Tuple, Dict

from ..models.pgm import ProbabilisticGraphicalModel
from ...base.inference import BaseInference


class ForwardInference(BaseInference):
    def __init__(self, pgm: ProbabilisticGraphicalModel):
        self.pgm = pgm
        self.concept_map = {var.concepts[0]: var for var in pgm.variables}
        self.sorted_variables = self._topological_sort()

        if len(self.sorted_variables) != len(self.pgm.variables):
            raise RuntimeError("The PGM contains cycles and cannot be processed in topological order.")

    def _topological_sort(self) -> List[Variable]:
        """
        Sorts the variables topologically (parents before children).
        """
        in_degree = {var.concepts[0]: 0 for var in self.pgm.variables}
        adj = {var.concepts[0]: [] for var in self.pgm.variables}

        for var in self.pgm.variables:
            child_name = var.concepts[0]
            for parent_var in var.parents:
                parent_name = parent_var.concepts[0]
                adj[parent_name].append(child_name)
                in_degree[child_name] += 1

        # Start with nodes having zero incoming edges (root nodes)
        queue = [self.concept_map[name] for name, degree in in_degree.items() if degree == 0]
        sorted_variables = []

        while queue:
            var = queue.pop(0)
            sorted_variables.append(var)

            for neighbor_name in adj[var.concepts[0]]:
                in_degree[neighbor_name] -= 1
                if in_degree[neighbor_name] == 0:
                    queue.append(self.concept_map[neighbor_name])

        return sorted_variables

    def predict(self, external_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass prediction across the entire PGM using the topological order.

        Args:
            external_inputs: A dictionary of {root_concept_name: input_tensor} for the root variables.
                           E.g., {'emb': torch.randn(87, 10)}.

        Returns:
            A dictionary of {concept_name: predicted_feature_tensor} for all concepts.
        """

        results = {}

        # Iterate in topological order
        for var in self.sorted_variables:
            concept_name = var.concepts[0]
            factor = self.pgm.get_factor_of_variable(concept_name)

            if factor is None:
                raise RuntimeError(f"Missing factor for variable/concept: {concept_name}")

            # 1. Handle Root Nodes (no parents)
            if not var.parents:
                if concept_name not in external_inputs:
                    raise ValueError(
                        f"Root variable '{concept_name}' requires an external input tensor in the 'external_inputs' dictionary.")

                input_tensor = external_inputs[concept_name]

                # Root factors (like LinearModule) expect a single 'input' keyword argument
                output_tensor = factor.forward(input=input_tensor)

                # 2. Handle Child Nodes (has parents)
            else:
                parent_kwargs = {}
                parent_logits = []
                parent_latent = []
                for parent_var in var.parents:
                    parent_name = parent_var.concepts[0]
                    if parent_name not in results:
                        # Should not happen with correct topological sort
                        raise RuntimeError(
                            f"Parent data missing: Cannot compute {concept_name} because parent {parent_name} has not been computed yet.")

                    # Parent tensor is fed into the factor using the parent's concept name as the key
                    # parent_kwargs[parent_name] = results[parent_name]
                    if parent_var.distribution in [torch.distributions.Bernoulli, torch.distributions.Categorical]:
                        # For probabilistic parents, pass logits
                        parent_logits.append(results[parent_name])
                    else:
                        # For continuous parents, pass latent features
                        parent_latent.append(results[parent_name])

                sig = inspect.signature(factor.module_class.forward)
                params = sig.parameters
                allowed = {
                    name for name, p in params.items()
                    if name != "self" and p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                if 'input' in allowed:
                    # this is a standard torch layer: concatenate all inputs into 'x'
                    parent_kwargs['input'] = torch.cat(parent_logits + parent_latent, dim=-1)
                else:
                    # this is a PyC layer: separate logits and latent inputs
                    if 'logits' in allowed:
                        parent_kwargs['logits'] = torch.cat(parent_logits, dim=-1)
                    if 'embedding' in allowed:
                        parent_kwargs['embedding'] = torch.cat(parent_latent, dim=-1)
                    elif 'exogenous' in allowed:
                        parent_kwargs['exogenous'] = torch.cat(parent_latent, dim=1)

                # Child factors concatenate parent outputs based on the kwargs
                output_tensor = factor.forward(**parent_kwargs)

            results[concept_name] = output_tensor

        return results

    def query(self, query_concepts: List[str], evidence: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Executes a forward pass and returns only the specified concepts concatenated
        into a single tensor, in the order requested.

        Args:
            query_concepts: A list of concept names to retrieve, e.g., ["c2", "c1", "xor_class"].
            evidence: A dictionary of {root_concept_name: input_tensor} for the root variables.

        Returns:
            A single torch.Tensor containing the concatenated predictions for the
            requested concepts, ordered as requested (Batch x TotalFeatures).
        """
        # 1. Run the full forward pass to get all necessary predictions
        all_predictions = self.predict(evidence)

        # 2. Filter and concatenate results
        result_tensors = []

        for concept_name in query_concepts:
            if concept_name not in all_predictions:
                raise ValueError(
                    f"Query concept '{concept_name}' was requested but could not be computed. "
                    f"Available predictions: {list(all_predictions.keys())}"
                )
            result_tensors.append(all_predictions[concept_name])

        if not result_tensors:
            return torch.empty(0)  # Return empty tensor if query list was empty

        # 3. Concatenate tensors along the last dimension (features)
        # Check if batch sizes match before concatenation
        batch_size = result_tensors[0].shape[0]
        if any(t.shape[0] != batch_size for t in result_tensors):
            raise RuntimeError("Batch size mismatch detected in query results before concatenation.")

        # Concatenate results into the final output tensor (Batch x TotalFeatures)
        final_tensor = torch.cat(result_tensors, dim=-1)

        # 4. Perform final check for expected shape
        expected_feature_dim = sum(self.concept_map[c].out_features for c in query_concepts)
        if final_tensor.shape[1] != expected_feature_dim:
            raise RuntimeError(
                f"Concatenation error. Expected total feature dimension of {expected_feature_dim}, "
                f"but got {final_tensor.shape[1]}. Check Variable.out_features logic."
            )

        return final_tensor


class KnownGraphInference(BaseInference):
    def __init__(self):
        super().__init__()
        self.train_mode = 'joint'

    def query(self, 
              x: torch.Tensor, 
              model: BaseModel,
              *args, 
              **kwargs) -> torch.Tensor:
        # get exogenous
        num_concepts = len(model.concept_names)
        if model.has_exogenous:
            c_exog_roots = model.exogenous_roots(x)
            c_exog_internal = model.exogenous_internal(x)

            c_exog_vals = [None] * num_concepts
            chunks = torch.split_with_sizes(c_exog_roots, split_sizes=model.split_sizes_roots, dim=1)
            for cid, t in zip(model.root_nodes_idx, chunks):
                c_exog_vals[cid] = t

            chunks = torch.split_with_sizes(c_exog_internal, split_sizes=model.split_sizes_internal, dim=1)
            for cid, t in zip(model.internal_node_idx, chunks):
                c_exog_vals[cid] = t

        # get roots
        vals = [None] * num_concepts
        if model.has_exogenous:
            input_obj = c_exog_roots
        else:
            input_obj = x
        c_all = model.encoder(input_obj)
        chunks = torch.split_with_sizes(c_all, split_sizes=model.split_sizes_roots, dim=1)
        for cid, t in zip(model.root_nodes_idx, chunks):
            vals[cid] = t

        for c_id, c_name in enumerate(model.internal_nodes):
            propagator = model.predictors[c_name]
            fetcher = model.fetchers[c_id]
            input_obj = torch.cat(fetcher(vals), dim=1)

            if model.has_self_exogenous:
                exog = c_exog_vals[model.internal_node_idx[c_id]]
                c_out = propagator(input_obj, exog)
            elif model.has_parent_exogenous:
                input_exog = torch.cat(fetcher(c_exog_vals), dim=1)
                c_out = propagator(input_obj, input_exog)
            else:
                c_out = propagator(input_obj)

            cid = model.name2id[c_name]
            vals[cid] = c_out

        out = torch.cat(vals, dim=1)
        return out


class UnknownGraphInference(BaseInference):
    def __init__(self):
        super().__init__()
        self.train_mode = 'independent'

    def mask_concept_tensor(self, 
                            c: torch.Tensor, 
                            model: BaseModel,
                            model_graph: ConceptGraph, 
                            c_name: str, 
                            cardinality: List[int]) -> torch.Tensor:
        broadcast_shape = [1] * len(c.size())
        broadcast_shape[1] = c.size(1)
        mask = torch.repeat_interleave(
            model_graph[:, model.to_index(c_name)],
            torch.tensor(cardinality, device=c.device)
        ).view(*broadcast_shape)
        return c * mask.data

    def query(self, x: torch.Tensor, c: torch.Tensor, model: BaseModel, *args, **kwargs) -> Tuple[torch.Tensor]:
        # --- maybe from embeddings to exogenous
        num_concepts = len(model.concept_names)
        if model.has_exogenous:
            c_exog = model.exogenous(x)

            c_exog_vals = [None] * num_concepts
            chunks = torch.split_with_sizes(c_exog, split_sizes=model.split_sizes_roots, dim=1)
            for cid, t in zip(model.root_nodes_idx, chunks):
                c_exog_vals[cid] = t
        
        #  get roots
        if model.has_exogenous:
            input_obj = c_exog
        else:
            input_obj = x
        c_encoder = model.encoder(input_obj)

        # --- from concepts to concepts copy
        model_graph = model.graph_learner()

        vals = []
        for c_id, c_name in enumerate(model.annotations.get_axis_labels(axis=1)):
            propagator = model.predictors[c_name]
            c_masked = self.mask_concept_tensor(c, model, model_graph, c_name, model.split_sizes_roots)

            if model.has_self_exogenous:
                exog = c_exog_vals[model.internal_node_idx[c_id]]
                c_out = propagator(c_masked, exogenous=exog)
            elif model.has_parent_exogenous:
                c_exog_masked = self.mask_concept_tensor(c_exog, model, model_graph, c_name, model.split_sizes_roots)
                c_out = propagator(c_masked, c_exog_masked)
            else:
                c_out = propagator(c_masked)

            vals.append(c_out)

        c_predictor = torch.cat(vals, dim=1)
        return c_encoder, c_predictor

    # def get_model_known_graph(self) -> GraphModel:
    #     if not hasattr(self, "graph_learner"):
    #         raise RuntimeError("This LearnedGraphModel was not initialised with a graph learner.")
    #     known_graph: ConceptGraph = self.graph_learner()

    #     # Build a GraphModel using the SAME builders -> predictors get the correct in_features
    #     gm = GraphModel(
    #         input_size=self.emb_size,
    #         annotations=self.annotations,
    #         encoder=self._encoder_builder,
    #         predictor=self._predictor_builder,
    #         model_graph=known_graph,
    #     )

    #     # ---- helpers ----
    #     full_order = list(self.concept_names)
    #     cards = self.annotations.get_axis_cardinalities(axis=1)
    #     per_card = {lab: (cards[i] if cards is not None else 1) for i, lab in enumerate(full_order)}

    #     # flat offsets in the "all-concepts" layout used by the wide predictors
    #     offsets = {}
    #     cur = 0
    #     for lab in full_order:
    #         offsets[lab] = cur
    #         cur += per_card[lab]

    #     def expand_indices(labels: list[str]) -> list[int]:
    #         keep = []
    #         for lab in labels:
    #             base = offsets[lab]
    #             width = per_card[lab]
    #             keep.extend(range(base, base + width))
    #         return keep

    #     def first_linear(module: nn.Module) -> nn.Linear | None:
    #         if isinstance(module, nn.Linear):
    #             return module
    #         if isinstance(module, nn.Sequential):
    #             for layer in module:
    #                 if isinstance(layer, nn.Linear):
    #                     return layer
    #         # common attribute names
    #         for name in ("in_proj", "fc", "proj", "input", "linear"):
    #             m = getattr(module, name, None)
    #             if isinstance(m, nn.Linear):
    #                 return m
    #         return None

    #     def copy_overlap_columns(old_mod: nn.Module, new_mod: nn.Module, keep_idx: list[int]) -> None:
    #         old_lin = first_linear(old_mod)
    #         new_lin = first_linear(new_mod)
    #         if old_lin is None or new_lin is None:
    #             return  # nothing generic to copy
    #         # sanity: output dim must match; new input dim must match keep_idx
    #         if old_lin.weight.size(0) != new_lin.weight.size(0):
    #             return
    #         if new_lin.weight.size(1) != len(keep_idx):
    #             return
    #         if len(keep_idx) == 0:
    #             # no parents -> just copy bias if present
    #             with torch.no_grad():
    #                 if new_lin.bias is not None and old_lin.bias is not None:
    #                     new_lin.bias.copy_(old_lin.bias)
    #             return
    #         if max(keep_idx) >= old_lin.weight.size(1):
    #             return
    #         with torch.no_grad():
    #             new_lin.weight.copy_(old_lin.weight[:, keep_idx])
    #             if new_lin.bias is not None and old_lin.bias is not None:
    #                 new_lin.bias.copy_(old_lin.bias)

    #     # ---- copy encoders exactly (roots in known graph) ----
    #     enc_out = nn.ModuleDict()
    #     for c in gm.root_nodes:
    #         enc_out[c] = copy.deepcopy(self.encoders[c]) if hasattr(self, "encoders") and c in self.encoders else \
    #         gm.encoders[c]
    #     gm.encoders = enc_out

    #     # ---- predictors: new (pruned) shapes already correct; now copy overlapping weights ----
    #     pred_out = nn.ModuleDict()
    #     for c in gm.internal_nodes:
    #         parents = list(known_graph.get_predecessors(c))  # labels in some order
    #         keep_idx = expand_indices(parents)  # flat indices into the old "all-concepts" layout

    #         new_pred = gm.predictors[c]  # built with correct in_features by _predictor_builder
    #         if hasattr(self, "predictors") and c in self.predictors:
    #             old_pred = self.predictors[c]
    #             copy_overlap_columns(old_pred, new_pred, keep_idx)
    #         pred_out[c] = new_pred
    #     gm.predictors = pred_out

    #     return gm