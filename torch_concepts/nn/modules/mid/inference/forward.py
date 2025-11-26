import inspect
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.distributions import RelaxedBernoulli, Bernoulli, RelaxedOneHotCategorical

from ..models.variable import Variable, EndogenousVariable
from ...low.base.graph import BaseGraphLearner
from typing import List, Dict, Union, Tuple, Set

from ...low.inference.intervention import _InterventionWrapper, _GlobalPolicyInterventionWrapper
from ..models.probabilistic_model import ProbabilisticModel
from ...low.base.inference import BaseInference


class ForwardInference(BaseInference):
    """
    Forward inference engine for probabilistic models.

    This class implements forward inference through a probabilistic model
    by topologically sorting variables and computing them in dependency order. It
    supports parallel computation within topological levels and can optionally use
    a learned graph structure.

    The inference engine:
    - Automatically sorts variables in topological order
    - Computes variables level-by-level (variables at same depth processed in parallel)
    - Supports GPU parallelization via CUDA streams
    - Supports CPU parallelization via threading
    - Handles interventions via _InterventionWrapper

    Attributes:
        probabilistic_model (ProbabilisticModel): The probabilistic model to perform inference on.
        graph_learner (BaseGraphLearner): Optional graph structure learner.
        concept_map (Dict[str, Variable]): Maps concept names to Variable objects.
        sorted_variables (List[Variable]): Variables in topological order.
        levels (List[List[Variable]]): Variables grouped by topological depth.

    Args:
        probabilistic_model: The probabilistic model to perform inference on.
        graph_learner: Optional graph learner for weighted adjacency structure.

    Raises:
        RuntimeError: If the model contains cycles (not a DAG).

    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch_concepts import InputVariable, EndogenousVariable
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts.nn import ForwardInference, ParametricCPD, ProbabilisticModel
        >>>
        >>> # Create a simple model: latent -> A -> B
        >>> # Where A is a root concept and B depends on A
        >>>
        >>> # Define variables
        >>> input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        >>> var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        >>> var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        >>>
        >>> # Define CPDs (modules that compute each variable)
        >>> from torch.nn import Identity, Linear
        >>> latent_cpd = ParametricCPD('input', parametrization=Identity())
        >>> cpd_A = ParametricCPD('A', parametrization=Linear(10, 1))  # latent -> A
        >>> cpd_B = ParametricCPD('B', parametrization=Linear(1, 1))   # A -> B
        >>>
        >>> # Create probabilistic model
        >>> pgm = ProbabilisticModel(
        ...     variables=[input_var, var_A, var_B],
        ...     parametric_cpds=[latent_cpd, cpd_A, cpd_B]
        ... )
        >>>
        >>> # Create forward inference engine
        >>> inference = ForwardInference(pgm)
        >>>
        >>> # Check topological order
        >>> print([v.concepts[0] for v in inference.sorted_variables])
        >>> # ['input', 'A', 'B']
        >>>
        >>> # Check levels (for parallel computation)
        >>> for i, level in enumerate(inference.levels):
        ...     print(f"Level {i}: {[v.concepts[0] for v in level]}")
        >>> # Level 0: ['input']
        >>> # Level 1: ['A']
        >>> # Level 2: ['B']
    """
    def __init__(self, probabilistic_model: ProbabilisticModel, graph_learner: BaseGraphLearner = None, *args, **kwargs):
        super().__init__()
        self.probabilistic_model = probabilistic_model
        self.graph_learner = graph_learner
        self.concept_map = {var.concepts[0]: var for var in probabilistic_model.variables}

        # topological order + levels (list of lists of Variables)
        self.sorted_variables, self.levels = self._topological_sort()

        if graph_learner is not None:
            self.row_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.row_labels)}
            self.col_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.col_labels)}

        if len(self.sorted_variables) != len(self.probabilistic_model.variables):
            raise RuntimeError("The ProbabilisticModel contains cycles and cannot be processed in topological order.")

    @abstractmethod
    def get_results(self, results: torch.tensor, parent_variable: Variable):
        """
        Process the raw output tensor from a CPD.

        This method should be implemented by subclasses to handle distribution-specific
        processing (e.g., sampling from Bernoulli, taking argmax from Categorical, etc.).

        Args:
            results: Raw output tensor from the CPD.
            parent_variable: The variable being computed.

        Returns:
            Processed output tensor.
        """
        pass

    def _topological_sort(self):
        """
        Sort variables topologically and compute levels.

        Variables are organized into levels where each level contains variables
        that have the same topological depth (can be computed in parallel).

        Returns:
            Tuple of (sorted_variables, levels) where:
            - sorted_variables: List of all variables in topological order
            - levels: List of lists, each containing variables at the same depth
        """
        in_degree = {var.concepts[0]: 0 for var in self.probabilistic_model.variables}
        adj = {var.concepts[0]: [] for var in self.probabilistic_model.variables}

        for var in self.probabilistic_model.variables:
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
        Compute the output tensor for a single variable.

        Args:
            var: The variable to compute.
            external_inputs: Dictionary of external input tensors for root variables.
            results: Dictionary of already computed variable outputs.

        Returns:
            Tuple of (concept_name, output_tensor).

        Raises:
            RuntimeError: If CPD is missing for the variable.
            ValueError: If root variable is missing from external_inputs.
            RuntimeError: If parent variable hasn't been computed yet.
        """
        concept_name = var.concepts[0]
        parametric_cpd = self.probabilistic_model.get_module_of_concept(concept_name)

        if parametric_cpd is None:
            raise RuntimeError(f"Missing parametric_cpd for variable/concept: {concept_name}")

        # 1. Root nodes (no parents)
        if not var.parents:
            if concept_name not in external_inputs:
                raise ValueError(f"Root variable '{concept_name}' requires an external input tensor in the 'external_inputs' dictionary.")
            input_tensor = external_inputs[concept_name]
            parent_kwargs = self.get_parent_kwargs(parametric_cpd, [input_tensor], [])
            output_tensor = parametric_cpd.forward(**parent_kwargs)
            output_tensor = self.get_results(output_tensor, var)

        # 2. Child nodes (has parents)
        else:
            parent_endogenous = []
            parent_input = []
            for parent_var in var.parents:
                parent_name = parent_var.concepts[0]
                if parent_name not in results:
                    # Should not happen with correct topological sort
                    raise RuntimeError(f"Parent data missing: Cannot compute {concept_name} because parent {parent_name} has not been computed yet.")

                if isinstance(parent_var, EndogenousVariable):
                    # For probabilistic parents, pass endogenous
                    weight = 1
                    if self.graph_learner is not None:
                        weight = self.graph_learner.weighted_adj[self.row_labels2id[parent_name], self.col_labels2id[concept_name]]
                    parent_endogenous.append(results[parent_name] * weight)
                else:
                    # For continuous parents, pass latent features
                    parent_input.append(results[parent_name])

            parent_kwargs = self.get_parent_kwargs(parametric_cpd, parent_input, parent_endogenous)
            output_tensor = parametric_cpd.forward(**parent_kwargs)
            if not isinstance(parametric_cpd.parametrization, _InterventionWrapper):
                output_tensor = self.get_results(output_tensor, var)

        return concept_name, output_tensor

    def predict(self, external_inputs: Dict[str, torch.Tensor], debug: bool = False, device: str = 'auto') -> Dict[str, torch.Tensor]:
        """
        Perform forward pass prediction across the entire probabilistic model.

        This method processes variables level-by-level, exploiting parallelism within
        each level. On GPU, uses CUDA streams for parallel computation. On CPU, uses
        ThreadPoolExecutor.

        Args:
            external_inputs: Dictionary mapping root variable names to input tensors.
            debug: If True, runs sequentially for easier debugging (disables parallelism).
            device: Device to use for computation. Options:
                - 'auto' (default): Automatically detect and use CUDA if available, else CPU
                - 'cuda' or 'gpu': Force use of CUDA (will raise error if not available)
                - 'cpu': Force use of CPU even if CUDA is available

        Returns:
            Dictionary mapping concept names to their output tensors.

        Raises:
            RuntimeError: If device='cuda'/'gpu' is specified but CUDA is not available.
        """
        # Determine which device to use
        if device == 'auto':
            use_cuda = torch.cuda.is_available()
        elif device in ['cuda', 'gpu']:
            if not torch.cuda.is_available():
                raise RuntimeError(f"device='{device}' was specified but CUDA is not available")
            use_cuda = True
        elif device == 'cpu':
            use_cuda = False
        else:
            raise ValueError(f"Invalid device '{device}'. Must be 'auto', 'cuda', 'gpu', or 'cpu'")

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

                # Apply global policy interventions if needed
                self._apply_global_interventions_for_level(level, results, debug=debug, use_cuda=use_cuda)
                continue

            # === PARALLEL MODE ===
            level_outputs = []

            # GPU: parallel via CUDA streams
            if use_cuda:
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

            # Apply global policy interventions if needed
            self._apply_global_interventions_for_level(level, results, debug=debug, use_cuda=use_cuda)

        return results

    def _apply_single_global_intervention(
            self,
            concept_name: str,
            wrapper: _GlobalPolicyInterventionWrapper,
            results: Dict[str, torch.Tensor]
    ) -> Tuple[str, torch.Tensor]:
        """
        Apply a global policy intervention for a single concept.

        Args:
            concept_name: Name of the concept to intervene on.
            wrapper: The global policy intervention wrapper.
            results: Dictionary of computed results.

        Returns:
            Tuple of (concept_name, intervened_output).
        """
        original_output = results[concept_name]
        intervened_output = wrapper.apply_intervention(original_output)
        return concept_name, intervened_output

    def _apply_global_interventions_for_level(self, level: List, results: Dict[str, torch.Tensor], debug: bool, use_cuda: bool) -> None:
        """
        Apply global policy interventions for all concepts in a level.

        This method checks if any concepts in the level have global policy wrappers,
        and if so, applies interventions after all concepts have been computed.
        Supports parallel execution via CUDA streams (GPU) or ThreadPoolExecutor (CPU).

        Args:
            level: List of variables in the current level
            results: Dictionary of computed results to update
            debug: If True, runs sequentially for easier debugging (disables parallelism)
            use_cuda: If True, uses CUDA streams for parallel execution; otherwise uses CPU threads
        """
        # Check if any concept in this level has a global policy wrapper
        global_wrappers = []
        for var in level:
            concept_name = var.concepts[0]
            parametric_cpd = self.probabilistic_model.get_module_of_concept(concept_name)
            if parametric_cpd is not None:
                if isinstance(parametric_cpd.parametrization, _GlobalPolicyInterventionWrapper):
                    global_wrappers.append((concept_name, parametric_cpd.parametrization))

        # If we found global wrappers, check if they're ready and apply interventions
        if global_wrappers:
            # Check if all wrappers in the shared state are ready
            first_wrapper = global_wrappers[0][1]
            if first_wrapper.shared_state.is_ready():

                # === DEBUG MODE or single wrapper: always run sequentially ===
                if debug or len(global_wrappers) <= 1:
                    for concept_name, wrapper in global_wrappers:
                        original_output = results[concept_name]
                        intervened_output = wrapper.apply_intervention(original_output)
                        results[concept_name] = intervened_output

                # === PARALLEL MODE ===
                else:
                    intervention_outputs = []

                    # GPU: parallel via CUDA streams
                    if use_cuda:
                        streams = [torch.cuda.Stream(device=torch.cuda.current_device()) for _ in global_wrappers]

                        for (concept_name, wrapper), stream in zip(global_wrappers, streams):
                            with torch.cuda.stream(stream):
                                concept_name_out, intervened_output = self._apply_single_global_intervention(
                                    concept_name, wrapper, results
                                )
                                intervention_outputs.append((concept_name_out, intervened_output))

                        torch.cuda.synchronize()

                    # CPU: parallel via threads
                    else:
                        with ThreadPoolExecutor(max_workers=len(global_wrappers)) as executor:
                            futures = [
                                executor.submit(self._apply_single_global_intervention, concept_name, wrapper, results)
                                for concept_name, wrapper in global_wrappers
                            ]
                            for fut in futures:
                                intervention_outputs.append(fut.result())

                    # Update results with intervened outputs
                    for concept_name, intervened_output in intervention_outputs:
                        results[concept_name] = intervened_output

                # Reset shared state for next batch/level
                first_wrapper.shared_state.reset()

    def get_parent_kwargs(self, parametric_cpd,
                          parent_input: Union[List[torch.Tensor], torch.Tensor] = None,
                          parent_endogenous: Union[List[torch.Tensor], torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare keyword arguments for CPD forward pass based on parent outputs.

        This method inspects the CPD's forward signature and constructs appropriate
        kwargs, separating endogenous (from probabilistic parents) and latent features
        (from continuous parents).

        Args:
            parametric_cpd: The CPD module to call.
            parent_input: List of continuous parent outputs (latent/exogenous).
            parent_endogenous: List of probabilistic parent outputs (concept endogenous).

        Returns:
            Dictionary of kwargs ready for parametric_cpd.forward(**kwargs).
        """
        parent_kwargs = {}
        if (isinstance(parametric_cpd.parametrization, _InterventionWrapper) or
                isinstance(parametric_cpd.parametrization, _GlobalPolicyInterventionWrapper)):
            forward_to_check = parametric_cpd.parametrization.forward_to_check
        else:
            forward_to_check = parametric_cpd.parametrization.forward

        sig = inspect.signature(forward_to_check)
        params = sig.parameters
        allowed = {
            name for name, p in params.items()
            if name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        if allowed not in [{'endogenous'}, {'endogenous', 'input'}, {'endogenous', 'exogenous'}, {'input'}, {'exogenous'}]:
             # standard torch module
            parent_kwargs[allowed.pop()] = torch.cat(parent_endogenous + parent_input, dim=-1)
        else:
            # this is a PyC layer: separate endogenous and latent inputs
            if 'endogenous' in allowed:
                parent_kwargs['endogenous'] = torch.cat(parent_endogenous, dim=-1)
            if 'input' in allowed:
                parent_kwargs['input'] = torch.cat(parent_input, dim=-1)
            elif 'exogenous' in allowed:
                parent_kwargs['exogenous'] = torch.cat(parent_input, dim=1)

        return parent_kwargs

    def query(self, query_concepts: List[str], evidence: Dict[str, torch.Tensor], debug: bool = False, device: str = 'auto') -> torch.Tensor:
        """
        Execute forward pass and return only specified concepts concatenated.

        This method runs full inference via predict() and then extracts and
        concatenates only the requested concepts in the specified order.

        Args:
            query_concepts: List of concept names to retrieve (e.g., ["C", "B", "A"]).
            evidence: Dictionary of {root_concept_name: input_tensor}.
            debug: If True, runs in debug mode (sequential execution).
            device: Device to use for computation. Options:
                - 'auto' (default): Automatically detect and use CUDA if available, else CPU
                - 'cuda' or 'gpu': Force use of CUDA (will raise error if not available)
                - 'cpu': Force use of CPU even if CUDA is available

        Returns:
            Single tensor containing concatenated predictions for requested concepts,
            ordered as requested (Batch x TotalFeatures).

        Raises:
            ValueError: If requested concept was not computed.
            RuntimeError: If batch sizes don't match across concepts.
            RuntimeError: If concatenation produces unexpected feature dimension.
            RuntimeError: If device='cuda'/'gpu' is specified but CUDA is not available.
        """
        # 1. Run the full forward pass to get all necessary predictions
        all_predictions = self.predict(evidence, debug=debug, device=device)

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
        Get all variable names available for querying.

        Returns:
            Set of concept names that can be queried.
        """
        if hasattr(self, "_unrolled_query_vars"):
            return self._unrolled_query_vars
        return set(var.concepts[0] for var in self.probabilistic_model.variables)

    def unrolled_probabilistic_model(self) -> ProbabilisticModel:
        """
        Build an 'unrolled' view of the ProbabilisticModel based on graph_learner adjacency.

        This method creates a modified PGM that reflects the learned graph structure,
        applying rules for keeping/dropping CPDs based on root/non-root status
        and recursively pruning unused variables.

        Rules:
        - For root columns (no incoming edges): keep row CPD, drop column CPD
        - For non-root columns: keep column CPD, drop row CPD
        - Recursively drop variables whose children are all dropped
        - Apply adjacency gating to remove zero-weight edges

        Returns:
            Modified ProbabilisticModel with unrolled structure.

        Raises:
            RuntimeError: If graph_learner is not set or lacks weighted_adj.
            RuntimeError: If adjacency shape doesn't match label lengths.
        """
        if self.graph_learner is None or not hasattr(self.graph_learner, "weighted_adj"):
            raise RuntimeError("unrolled_probabilistic_model requires a graph_learner with a 'weighted_adj' attribute.")

        adj = self.graph_learner.weighted_adj
        row_labels = list(self.graph_learner.row_labels)
        col_labels = list(self.graph_learner.col_labels)

        n_rows, n_cols = adj.shape
        if n_rows != len(row_labels) or n_cols != len(col_labels):
            raise RuntimeError("Mismatch between adjacency shape and row/col labels length.")

        # --- 0) Build children map from the raw ProbabilisticModel (no adjacency, no renaming) ---
        # children_map[parent_name] -> set(child_name)
        children_map: Dict[str, Set[str]] = defaultdict(set)
        for var in self.probabilistic_model.variables:
            child_name = var.concepts[0]
            for parent in var.parents:
                parent_name = parent.concepts[0]
                children_map[parent_name].add(child_name)

        # All variable names in the ProbabilisticModel
        all_names: Set[str] = {var.concepts[0] for var in self.probabilistic_model.variables}

        # --- 1) Determine which side we keep for each row/col pair (using adjacency) ---
        # Root CPD (in adjacency sense) = column with no incoming edges
        col_has_parent = (adj != 0).any(dim=0)  # bool per column

        rename_map: Dict[str, str] = {}   # old_name -> new_name
        keep_names_initial: Set[str] = set()
        drop_names: Set[str] = set()

        # For each index i, (row_labels[i], col_labels[i]) is a pair of copies
        for idx in range(min(n_rows, n_cols)):
            src = row_labels[idx]  # "row" CPD
            dst = col_labels[idx]  # "column" CPD

            is_root = not bool(col_has_parent[idx].item())
            if is_root:
                # Root column: keep row CPD, drop its column copy
                rename_map[dst] = src
                keep_names_initial.add(src)
                drop_names.add(dst)
            else:
                # Non-root column: keep column CPD, drop original row CPD
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
        for var in self.probabilistic_model.variables:
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

        # --- 5) Unique list of CPDs corresponding to these variables ---
        new_parametric_cpds: List[object] = []
        seen_parametric_cpds: Set[object] = set()

        repeats = [self.probabilistic_model.concept_to_variable[p].size for p in row_labels]
        for var in new_variables:
            parametric_cpd = self.probabilistic_model.parametric_cpds[var.concepts[0]]
            if parametric_cpd is not None and parametric_cpd not in seen_parametric_cpds:
                if parametric_cpd.concepts[0] in rename_map.values() and parametric_cpd.concepts[0] in col_labels:
                    col_id = self.col_labels2id[parametric_cpd.concepts[0]]
                    mask = adj[:, col_id] != 0
                    mask_without_self_loop = torch.cat((mask[:col_id], mask[col_id + 1:]))
                    rep = repeats[:col_id] + repeats[col_id + 1:]
                    mask_with_cardinalities = torch.repeat_interleave(mask_without_self_loop, torch.tensor(rep))
                    parametric_cpd.parametrization.prune(mask_with_cardinalities)
                new_parametric_cpds.append(parametric_cpd)
                seen_parametric_cpds.add(parametric_cpd)

        # --- 6) Update available_query_vars to reflect the unrolled graph ---
        self._unrolled_query_vars = set(v.concepts[0] for v in new_variables)

        return ProbabilisticModel(new_variables, new_parametric_cpds)


class DeterministicInference(ForwardInference):
    """
    Deterministic forward inference for probabilistic graphical models.

    This inference engine performs deterministic (maximum likelihood) inference by
    returning raw endogenous/outputs from CPDs without sampling. It's useful for
    prediction tasks where you want the most likely values rather than samples
    from the distribution.

    Inherits all functionality from ForwardInference but implements get_results()
    to return raw outputs without stochastic sampling.

    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch_concepts import InputVariable, EndogenousVariable
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts.nn import DeterministicInference, ParametricCPD, ProbabilisticModel, LinearCC
        >>>
        >>> # Create a simple PGM: latent -> A -> B
        >>> input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        >>> var_A = EndogenousVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        >>> var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        >>>
        >>> # Define CPDs
        >>> from torch.nn import Identity, Linear
        >>> cpd_emb = ParametricCPD('input', parametrization=Identity())
        >>> cpd_A = ParametricCPD('A', parametrization=Linear(10, 1))
        >>> cpd_B = ParametricCPD('B', parametrization=LinearCC(1, 1))
        >>>
        >>> # Create probabilistic model
        >>> pgm = ProbabilisticModel(
        ...     variables=[input_var, var_A, var_B],
        ...     parametric_cpds=[cpd_emb, cpd_A, cpd_B]
        ... )
        >>>
        >>> # Create deterministic inference engine
        >>> inference = DeterministicInference(pgm)
        >>>
        >>> # Perform inference - returns endogenous, not samples
        >>> x = torch.randn(4, 10)  # batch_size=4, latent_size=10
        >>> results = inference.predict({'input': x})
        >>>
        >>> # Results contain raw endogenous for Bernoulli variables
        >>> print(results['A'].shape)  # torch.Size([4, 1]) - endogenous, not {0,1}
        >>> print(results['B'].shape)  # torch.Size([4, 1]) - endogenous, not {0,1}
        >>>
        >>> # Query specific concepts - returns concatenated endogenous
        >>> output = inference.query(['B', 'A'], evidence={'input': x})
        >>> print(output.shape)  # torch.Size([4, 2])
        >>> # output contains [logit_B, logit_A] for each sample
        >>>
        >>> # Convert endogenous to probabilities if needed
        >>> prob_A = torch.sigmoid(results['A'])
        >>> print(prob_A.shape)  # torch.Size([4, 1])
        >>>
        >>> # Get hard predictions (0 or 1)
        >>> pred_A = (prob_A > 0.5).float()
        >>> print(pred_A)  # Binary predictions
    """
    def get_results(self, results: torch.tensor, parent_variable: Variable) -> torch.Tensor:
        """
        Return raw output without sampling.

        Args:
            results: Raw output tensor from the CPD.
            parent_variable: The variable being computed (unused in deterministic mode).

        Returns:
            torch.Tensor: Raw output tensor (endogenous for probabilistic variables).
        """
        return results


class AncestralSamplingInference(ForwardInference):
    """
    Ancestral sampling inference for probabilistic graphical models.

    This inference engine performs ancestral (forward) sampling by drawing samples
    from the distributions defined by each variable. It's useful for generating
    realistic samples from the model and for tasks requiring stochastic predictions.

    The sampling respects the probabilistic structure:
    - Samples from Bernoulli distributions using .sample()
    - Uses reparameterization (.rsample()) for RelaxedBernoulli and RelaxedOneHotCategorical
    - Supports custom distribution kwargs (e.g., temperature for Gumbel-Softmax)

    Args:
        probabilistic_model: The probabilistic model to perform inference on.
        graph_learner: Optional graph learner for weighted adjacency structure.
        **dist_kwargs: Additional kwargs passed to distribution constructors
                      (e.g., temperature for relaxed distributions).

    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch_concepts import InputVariable
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts.nn import AncestralSamplingInference, ParametricCPD, ProbabilisticModel
        >>> from torch_concepts import EndogenousVariable
        >>> from torch_concepts.nn import LinearCC
        >>>
        >>> # Create a simple PGM: embedding -> A -> B
        >>> embedding_var = InputVariable('embedding', parents=[], distribution=Delta, size=10)
        >>> var_A = EndogenousVariable('A', parents=['embedding'], distribution=Bernoulli, size=1)
        >>> var_B = EndogenousVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        >>>
        >>> # Define CPDs
        >>> from torch.nn import Identity, Linear
        >>> cpd_emb = ParametricCPD('embedding', parametrization=Identity())
        >>> cpd_A = ParametricCPD('A', parametrization=Linear(10, 1))
        >>> cpd_B = ParametricCPD('B', parametrization=LinearCC(1, 1))
        >>>
        >>> # Create probabilistic model
        >>> pgm = ProbabilisticModel(
        ...     variables=[embedding_var, var_A, var_B],
        ...     parametric_cpds=[cpd_emb, cpd_A, cpd_B]
        ... )
        >>>
        >>> # Create ancestral sampling inference engine
        >>> inference = AncestralSamplingInference(pgm)
        >>>
        >>> # Perform inference - returns samples, not endogenous
        >>> x = torch.randn(4, 10)  # batch_size=4, embedding_size=10
        >>> results = inference.predict({'embedding': x})
        >>>
        >>> # Results contain binary samples {0, 1} for Bernoulli variables
        >>> print(results['A'].shape)  # torch.Size([4, 1])
        >>> print(results['A'].unique())  # tensor([0., 1.]) - actual samples
        >>> print(results['B'].shape)  # torch.Size([4, 1])
        >>> print(results['B'].unique())  # tensor([0., 1.]) - actual samples
        >>>
        >>> # Query specific concepts - returns concatenated samples
        >>> samples = inference.query(['B', 'A'], evidence={'embedding': x})
        >>> print(samples.shape)  # torch.Size([4, 2])
        >>> # samples contains [sample_B, sample_A] for each instance
        >>> print(samples)  # All values are 0 or 1
        >>>
        >>> # Multiple runs produce different samples (stochastic)
        >>> samples1 = inference.query(['A'], evidence={'embedding': x})
        >>> samples2 = inference.query(['A'], evidence={'embedding': x})
        >>> print(torch.equal(samples1, samples2))  # Usually False (different samples)
        >>>
        >>> # With relaxed distributions (requires temperature)
        >>> from torch.distributions import RelaxedBernoulli
        >>> var_A_relaxed = InputVariable('A', parents=['embedding'],
        ...                               distribution=RelaxedBernoulli, size=1)
        >>> pgm = ProbabilisticModel(
        ...     variables=[embedding_var, var_A_relaxed, var_B],
        ...     parametric_cpds=[cpd_emb, cpd_A, cpd_B]
        ... )
        >>> inference_relaxed = AncestralSamplingInference(pgm, temperature=0.05)
        >>> # Now uses reparameterization trick (.rsample())
        >>>
        >>> # Query returns continuous values in [0, 1] for relaxed distributions
        >>> relaxed_samples = inference_relaxed.query(['A'], evidence={'embedding': x})
        >>> # relaxed_samples will be continuous, not binary
    """
    def __init__(self,
                 probabilistic_model: ProbabilisticModel,
                 graph_learner: BaseGraphLearner = None,
                 log_probs: bool = True,
                 **dist_kwargs):
        super().__init__(probabilistic_model, graph_learner)
        self.dist_kwargs = dist_kwargs
        self.log_probs = log_probs

    def get_results(self, results: torch.tensor, parent_variable: Variable) -> torch.Tensor:
        """
        Sample from the distribution parameterized by the results.

        This method creates a distribution using the variable's distribution type
        and the computed endogenous/parameters, then draws a sample.

        Args:
            results: Raw output tensor from the CPD (endogenous or parameters).
            parent_variable: The variable being computed (defines distribution type).

        Returns:
            torch.Tensor: Sampled values from the distribution.
        """
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

        if parent_variable.distribution in [Bernoulli, RelaxedBernoulli, RelaxedOneHotCategorical]:
            if self.log_probs:
                dist_kwargs['logits'] = results
            else:
                dist_kwargs['probs'] = results

            if parent_variable.distribution in [Bernoulli]:
                return parent_variable.distribution(**dist_kwargs).sample()
            elif parent_variable.distribution in [RelaxedBernoulli, RelaxedOneHotCategorical]:
                return parent_variable.distribution(**dist_kwargs).rsample()

        return parent_variable.distribution(results, **dist_kwargs).rsample()
