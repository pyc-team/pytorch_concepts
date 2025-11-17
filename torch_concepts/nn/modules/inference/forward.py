import inspect
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.distributions import RelaxedBernoulli, Bernoulli, RelaxedOneHotCategorical

from torch_concepts import Variable
from torch_concepts.nn import BaseGraphLearner
from typing import List, Dict, Union, Tuple, Set

from .intervention import _InterventionWrapper
from ..models.pgm import ProbabilisticGraphicalModel
from ...base.inference import BaseInference


class ForwardInference(BaseInference):
    def __init__(self, pgm: ProbabilisticGraphicalModel, graph_learner: BaseGraphLearner = None, *args, **kwargs):
        super().__init__()
        self.pgm = pgm
        self.graph_learner = graph_learner
        self.concept_map = {var.concepts[0]: var for var in pgm.variables}

        # topological order + levels (list of lists of Variables)
        self.sorted_variables, self.levels = self._topological_sort()

        if graph_learner is not None:
            self.row_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.row_labels)}
            self.col_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.col_labels)}

        if len(self.sorted_variables) != len(self.pgm.variables):
            raise RuntimeError("The PGM contains cycles and cannot be processed in topological order.")

    @abstractmethod
    def get_results(self, results: torch.tensor, parent_variable: Variable):
        pass

    def _topological_sort(self):
        """
        Sort variables topologically and compute levels
        (variables that share the same topological depth).
        """
        in_degree = {var.concepts[0]: 0 for var in self.pgm.variables}
        adj = {var.concepts[0]: [] for var in self.pgm.variables}

        for var in self.pgm.variables:
            child_name = var.concepts[0]
            for parent_var in var.parents:
                parent_name = parent_var.concepts[0]
                adj[parent_name].append(child_name)
                in_degree[child_name] += 1

        # Nodes with zero inbound edges = level 0
        queue = [self.concept_map[name] for name, deg in in_degree.items() if deg == 0]

        sorted_variables = []
        levels = []

        # Track current BFS frontier
        current_level = queue.copy()
        while current_level:
            levels.append(current_level)
            next_level = []

            for var in current_level:
                sorted_variables.append(var)

                for neighbour_name in adj[var.concepts[0]]:
                    in_degree[neighbour_name] -= 1
                    if in_degree[neighbour_name] == 0:
                        next_level.append(self.concept_map[neighbour_name])

            current_level = next_level

        return sorted_variables, levels

    def _compute_single_variable(
            self,
            var: Variable,
            external_inputs: Dict[str, torch.Tensor],
            results: Dict[str, torch.Tensor],
    ) -> Tuple[str, torch.Tensor]:
        """
        Compute the output tensor for a single variable, given the current results.
        Returns (concept_name, output_tensor) without mutating `results`.
        """
        concept_name = var.concepts[0]
        factor = self.pgm.get_module_of_concept(concept_name)

        if factor is None:
            raise RuntimeError(f"Missing factor for variable/concept: {concept_name}")

        # 1. Root nodes (no parents)
        if not var.parents:
            if concept_name not in external_inputs:
                raise ValueError(f"Root variable '{concept_name}' requires an external input tensor in the 'external_inputs' dictionary.")
            input_tensor = external_inputs[concept_name]
            parent_kwargs = self.get_parent_kwargs(factor, [input_tensor], [])
            output_tensor = factor.forward(**parent_kwargs)
            output_tensor = self.get_results(output_tensor, var)

        # 2. Child nodes (has parents)
        else:
            parent_logits = []
            parent_latent = []
            for parent_var in var.parents:
                parent_name = parent_var.concepts[0]
                if parent_name not in results:
                    # Should not happen with correct topological sort
                    raise RuntimeError(f"Parent data missing: Cannot compute {concept_name} because parent {parent_name} has not been computed yet.")

                if parent_var.distribution in [Bernoulli, RelaxedBernoulli, RelaxedOneHotCategorical]:
                    # For probabilistic parents, pass logits
                    weight = 1
                    if self.graph_learner is not None:
                        weight = self.graph_learner.weighted_adj[self.row_labels2id[parent_name], self.col_labels2id[concept_name]]
                    parent_logits.append(results[parent_name] * weight)
                else:
                    # For continuous parents, pass latent features
                    parent_latent.append(results[parent_name])

            parent_kwargs = self.get_parent_kwargs(factor, parent_latent, parent_logits)
            output_tensor = factor.forward(**parent_kwargs)
            if not isinstance(factor.module_class, _InterventionWrapper):
                output_tensor = self.get_results(output_tensor, var)

        return concept_name, output_tensor

    def predict(self, external_inputs: Dict[str, torch.Tensor], debug: bool = False) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass prediction across the entire PGM using the topological level structure.

        Args:
            external_inputs: external inputs for root variables.
            debug: if True, disables parallelism and executes sequentially for easier debugging.

        Returns:
            A dictionary {concept_name: output_tensor}.
        """

        results: Dict[str, torch.Tensor] = {}

        levels = getattr(self, "levels", None)
        if levels is None:
            levels = [self.sorted_variables]

        for level in levels:

            # === DEBUG MODE: always run sequentially ===
            if debug or len(level) <= 1:
                for var in level:
                    concept_name, output_tensor = self._compute_single_variable(var, external_inputs, results)
                    results[concept_name] = output_tensor
                continue

            # === PARALLEL MODE ===
            level_outputs = []

            # GPU: parallel via CUDA streams
            if torch.cuda.is_available():
                streams = [torch.cuda.Stream(device=torch.cuda.current_device()) for _ in level]

                for var, stream in zip(level, streams):
                    with torch.cuda.stream(stream):
                        concept_name, output_tensor = self._compute_single_variable(var, external_inputs, results)
                        level_outputs.append((concept_name, output_tensor))

                torch.cuda.synchronize()

            # CPU: parallel via threads
            else:
                with ThreadPoolExecutor(max_workers=len(level)) as executor:
                    futures = [executor.submit(self._compute_single_variable, var, external_inputs, results) for var in level]
                    for fut in futures:
                        level_outputs.append(fut.result())

            # Update results
            for concept_name, output_tensor in level_outputs:
                results[concept_name] = output_tensor

        return results

    def get_parent_kwargs(self, factor,
                          parent_latent: Union[List[torch.Tensor], torch.Tensor] = None,
                          parent_logits: Union[List[torch.Tensor], torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        parent_kwargs = {}
        if isinstance(factor.module_class, _InterventionWrapper):
            forward_to_check = factor.module_class.forward_to_check
        else:
            forward_to_check = factor.module_class.forward

        sig = inspect.signature(forward_to_check)
        params = sig.parameters
        allowed = {
            name for name, p in params.items()
            if name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        if allowed not in [{'logits'}, {'logits', 'embedding'}, {'logits', 'exogenous'}, {'embedding'}, {'exogenous'}]:
             # standard torch module
            parent_kwargs[allowed.pop()] = torch.cat(parent_logits + parent_latent, dim=-1)
        else:
            # this is a PyC layer: separate logits and latent inputs
            if 'logits' in allowed:
                parent_kwargs['logits'] = torch.cat(parent_logits, dim=-1)
            if 'embedding' in allowed:
                parent_kwargs['embedding'] = torch.cat(parent_latent, dim=-1)
            elif 'exogenous' in allowed:
                parent_kwargs['exogenous'] = torch.cat(parent_latent, dim=1)

        return parent_kwargs

    def query(self, query_concepts: List[str], evidence: Dict[str, torch.Tensor], debug: bool = False) -> torch.Tensor:
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
        all_predictions = self.predict(evidence, debug=debug)

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

    @property
    def available_query_vars(self) -> Set[str]:
        """
        A tuple of all variable names available for querying.

        After calling `unrolled_pgm`, this reflects the unrolled variables;
        before that, it reflects the original PGM variables.
        """
        if hasattr(self, "_unrolled_query_vars"):
            return self._unrolled_query_vars
        return set(var.concepts[0] for var in self.pgm.variables)

    def unrolled_pgm(self) -> ProbabilisticGraphicalModel:
        """
        Build an 'unrolled' view of the PGM based on the graph_learner adjacency.

        Rules:
        - For root columns in the adjacency (no incoming edges), keep the row factor,
          drop the corresponding column factor.
        - For non-root columns, keep the column factor, drop the corresponding row factor,
          and replace usages of that row factor as a parent with the kept column factor.
        - Recursively drop any variable X if all its direct children are dropped.
        """

        if self.graph_learner is None or not hasattr(self.graph_learner, "weighted_adj"):
            raise RuntimeError("unrolled_pgm requires a graph_learner with a 'weighted_adj' attribute.")

        adj = self.graph_learner.weighted_adj
        row_labels = list(self.graph_learner.row_labels)
        col_labels = list(self.graph_learner.col_labels)

        n_rows, n_cols = adj.shape
        if n_rows != len(row_labels) or n_cols != len(col_labels):
            raise RuntimeError("Mismatch between adjacency shape and row/col labels length.")

        # --- 0) Build children map from the raw PGM (no adjacency, no renaming) ---
        # children_map[parent_name] -> set(child_name)
        children_map: Dict[str, Set[str]] = defaultdict(set)
        for var in self.pgm.variables:
            child_name = var.concepts[0]
            for parent in var.parents:
                parent_name = parent.concepts[0]
                children_map[parent_name].add(child_name)

        # All variable names in the PGM
        all_names: Set[str] = {var.concepts[0] for var in self.pgm.variables}

        # --- 1) Determine which side we keep for each row/col pair (using adjacency) ---
        # Root factor (in adjacency sense) = column with no incoming edges
        col_has_parent = (adj != 0).any(dim=0)  # bool per column

        rename_map: Dict[str, str] = {}   # old_name -> new_name
        keep_names_initial: Set[str] = set()
        drop_names: Set[str] = set()

        # For each index i, (row_labels[i], col_labels[i]) is a pair of copies
        for idx in range(min(n_rows, n_cols)):
            src = row_labels[idx]  # "row" factor
            dst = col_labels[idx]  # "column" factor

            is_root = not bool(col_has_parent[idx].item())
            if is_root:
                # Root column: keep row factor, drop its column copy
                rename_map[dst] = src
                keep_names_initial.add(src)
                drop_names.add(dst)
            else:
                # Non-root column: keep column factor, drop original row factor
                rename_map[src] = dst
                keep_names_initial.add(dst)
                drop_names.add(src)

        # Add all other variables that are not explicitly dropped
        keep_names_initial |= {name for name in all_names if name not in drop_names}

        # --- 2) GENERAL RECURSIVE PRUNING RULE ---
        # If X has children Yi and ALL Yi are in drop_names -> drop X as well.
        drop: Set[str] = set(drop_names)

        while True:
            changed = False
            for parent_name, children in children_map.items():
                if parent_name in drop:
                    continue
                if not children:
                    continue  # no children: do not auto-drop (could be sink / output)
                # Only consider children that actually exist as variables
                eff_children = {c for c in children if c in all_names}
                if not eff_children:
                    continue
                if eff_children.issubset(drop):
                    drop.add(parent_name)
                    changed = True
            if not changed:
                break

        # Final kept names: everything not in drop
        keep_names: Set[str] = {name for name in all_names if name not in drop}

        # --- 3) Rewrite parents using keep_names, rename_map, and adjacency gating ---
        for var in self.pgm.variables:
            child_name = var.concepts[0]
            new_parents: List[Variable] = []
            seen: Set[str] = set()

            for parent in var.parents:
                parent_orig = parent.concepts[0]

                # 3a) Adjacency gating: if adj defines this edge and it's zero, drop it
                keep_edge = True
                if (
                    hasattr(self, "row_labels2id")
                    and hasattr(self, "col_labels2id")
                    and parent_orig in self.row_labels2id
                    and child_name in self.col_labels2id
                ):
                    r = self.row_labels2id[parent_orig]
                    c = self.col_labels2id[child_name]
                    if adj[r, c].item() == 0:
                        keep_edge = False

                if not keep_edge:
                    continue

                # 3b) Apply renaming: map parent_orig through rename_map chain
                mapped_parent = parent_orig
                while mapped_parent in rename_map:
                    mapped_parent = rename_map[mapped_parent]

                # 3c) Drop if final parent is not kept
                if mapped_parent not in keep_names:
                    continue

                if mapped_parent in seen:
                    continue  # avoid duplicates

                new_parents.append(self.concept_map[mapped_parent])
                seen.add(mapped_parent)

            var.parents = new_parents

        # --- 4) Build final ordered list of variables (unique, no duplicates) ---
        new_variables: List[Variable] = []
        seen_var_names: Set[str] = set()

        for var in self.sorted_variables:
            name = var.concepts[0]
            if name in keep_names and name not in seen_var_names:
                new_variables.append(var)
                seen_var_names.add(name)

        # --- 5) Unique list of factors corresponding to these variables ---
        new_factors: List[object] = []
        seen_factors: Set[object] = set()

        repeats = [self.pgm.concept_to_variable[p].size for p in row_labels]
        for var in new_variables:
            factor = self.pgm.factors[var.concepts[0]]
            if factor is not None and factor not in seen_factors:
                if factor.concepts[0] in rename_map.values() and factor.concepts[0] in col_labels:
                    col_id = self.col_labels2id[factor.concepts[0]]
                    mask = adj[:, col_id] != 0
                    mask_without_self_loop = torch.cat((mask[:col_id], mask[col_id + 1:]))
                    rep = repeats[:col_id] + repeats[col_id + 1:]
                    mask_with_cardinalities = torch.repeat_interleave(mask_without_self_loop, torch.tensor(rep))
                    factor.module_class.prune(mask_with_cardinalities)
                new_factors.append(factor)
                seen_factors.add(factor)

        # --- 6) Update available_query_vars to reflect the unrolled graph ---
        self._unrolled_query_vars = set(v.concepts[0] for v in new_variables)

        return ProbabilisticGraphicalModel(new_variables, new_factors)


class DeterministicInference(ForwardInference):
    def get_results(self, results: torch.tensor, parent_variable: Variable) -> torch.Tensor:
        return results


class AncestralSamplingInference(ForwardInference):
    def __init__(self, pgm: ProbabilisticGraphicalModel, graph_learner: BaseGraphLearner = None, **dist_kwargs):
        super().__init__(pgm, graph_learner)
        self.dist_kwargs = dist_kwargs

    def get_results(self, results: torch.tensor, parent_variable: Variable) -> torch.Tensor:
        sig = inspect.signature(parent_variable.distribution.__init__)
        params = sig.parameters
        allowed = {
            name for name, p in params.items()
            if name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        # retain only allowed dist kwargs
        dist_kwargs = {k: v for k, v in self.dist_kwargs.items() if k in allowed}

        if parent_variable.distribution in [Bernoulli]:
            return parent_variable.distribution(logits=results, **dist_kwargs).sample()
        elif parent_variable.distribution in [RelaxedBernoulli, RelaxedOneHotCategorical]:
            return parent_variable.distribution(logits=results, **dist_kwargs).rsample()
        return parent_variable.distribution(results, **dist_kwargs).rsample()
