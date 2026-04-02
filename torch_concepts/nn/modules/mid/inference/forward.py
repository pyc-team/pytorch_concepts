import inspect
from abc import abstractmethod, ABC
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch

from ..models.variable import Variable, ConceptVariable
from ...low.base.graph import BaseGraphLearner
from typing import List, Dict, Union, Tuple, Set

from ...low.inference.intervention import _InterventionWrapper, _GlobalPolicyInterventionWrapper
from ..models.probabilistic_model import ProbabilisticModel
from ...low.base.inference import BaseInference


class ForwardInference(BaseInference, ABC):
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
        detach (bool): If True, concept predictions are detached before propagation
            to children.
        lazy (bool): If True, only compute ancestors of the queried concepts.
        variable_map (Dict[str, Variable]): Maps concept names to Variable objects.
        sorted_variables (List[Variable]): Variables in topological order.
        levels (List[List[Variable]]): Variables grouped by topological depth.

    Args:
        probabilistic_model: The probabilistic model to perform inference on.
        graph_learner: Optional graph learner for weighted adjacency structure.
        detach: If True, detach concept predictions before propagation to
            children. Each encoder then receives gradients only from its own loss.
            Exogenous / latent variables keep their gradients. Default: False.
        lazy: If True, only compute variables that are ancestors of the queried
            concepts, skipping unrelated branches. Default: False.

    Raises:
        RuntimeError: If the model contains cycles (not a DAG).

    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch_concepts import LatentVariable, ConceptVariable
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts.nn import ForwardInference, ParametricCPD, ProbabilisticModel
        >>>
        >>> # Create a simple model: latent -> A -> B
        >>> # Where A is a root concept and B depends on A
        >>>
        >>> # Define variables
        >>> input_var = LatentVariable('input', distribution=Delta, size=10)
        >>> var_A = ConceptVariable('A', distribution=Bernoulli, size=1)
        >>> var_B = ConceptVariable('B', distribution=Bernoulli, size=1)
        >>>
        >>> # Define CPDs (modules that compute each variable)
        >>> from torch.nn import Identity, Linear
        >>> latent_cpd = ParametricCPD('input', parametrization=Identity())
        >>> cpd_A = ParametricCPD('A', parametrization=Linear(10, 1), parents=[input_var])  # latent -> A
        >>> cpd_B = ParametricCPD('B', parametrization=Linear(1, 1), parents=[var_A])   # A -> B
        >>>
        >>> # Create probabilistic model
        >>> pgm = ProbabilisticModel(
        ...     variables=[input_var, var_A, var_B],
        ...     factors=[latent_cpd, cpd_A, cpd_B]
        ... )
        >>>
        >>> # Create forward inference engine
        >>> # Note: ForwardInference is abstract; use DeterministicInference or AncestralSamplingInference
        >>> from torch_concepts.nn import DeterministicInference
        >>> inference = DeterministicInference(pgm)
        >>>
        >>> # Check topological order
        >>> print([v.concept for v in inference.sorted_variables])
        >>> # ['input', 'A', 'B']
        >>>
        >>> # Check levels (for parallel computation)
        >>> for i, level in enumerate(inference.levels):
        ...     print(f"Level {i}: {[v.concept for v in level]}")
        >>> # Level 0: ['input']
        >>> # Level 1: ['A']
        >>> # Level 2: ['B']
    """
    def __init__(
        self, 
        probabilistic_model: ProbabilisticModel, 
        graph_learner: BaseGraphLearner = None, 
        detach: bool = False,
        lazy: bool = False,
        *args, 
        **kwargs
    ):
        super().__init__()
        self.probabilistic_model = probabilistic_model
        self.graph_learner = graph_learner
        self.detach = detach
        self.lazy = lazy
        self.variable_map = {var.concept: var for var in probabilistic_model.variables}

        # topological order + levels (list of lists of Variables)
        self.sorted_variables, self.levels = self._topological_sort()

        if graph_learner is not None:
            self.row_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.row_labels)}
            self.col_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.col_labels)}

        if len(self.sorted_variables) != len(self.probabilistic_model.variables):
            raise RuntimeError("The ProbabilisticModel contains cycles and cannot be processed in topological order.")

        # Cache forward-signature parameter names per CPD to avoid
        # calling inspect.signature() on every forward pass.
        self._cpd_allowed_params: Dict[str, Set[str]] = {}
        self._cached_parents: Dict[str, List] = {}
        _cpd_id_to_allowed: Dict[int, Set[str]] = {}
        for var in self.probabilistic_model.variables:
            cpd = self.probabilistic_model.get_module_of_concept(var.concept)
            if cpd is not None:
                cpd_id = id(cpd)
                if cpd_id not in _cpd_id_to_allowed:
                    _cpd_id_to_allowed[cpd_id] = self._get_allowed_params(cpd)
                self._cpd_allowed_params[var.concept] = _cpd_id_to_allowed[cpd_id]
                self._cached_parents[var.concept] = getattr(cpd, 'parents', [])
            else:
                self._cached_parents[var.concept] = []

        # Build shared CPD primary names for fast-path in _concatenate_results.
        self._shared_cpd_primaries: Set[str] = set()
        for cpd in self.probabilistic_model.factors.values():
            if getattr(cpd, 'shared', False):
                self._shared_cpd_primaries.add(cpd.concept)

    @abstractmethod
    def activate(self, pred: torch.Tensor, variable: Variable) -> torch.Tensor:
        """
        Apply the inference-specific transformation to raw CPD output.

        This is the single point where each inference strategy defines its
        semantics.  The activated value is:

        * propagated to child predictors (they receive already-activated inputs)
        * returned as the query output (unless ``return_logits=True``)

        Subclass contracts:

        * ``DeterministicInference`` — Bernoulli → sigmoid, Categorical → softmax
        * ``AncestralSamplingInference`` — sample from the distribution

        Args:
            pred: Raw output tensor from the CPD (logits).
            variable: The Variable whose prediction is being transformed.

        Returns:
            torch.Tensor: Activated / sampled prediction.
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
        in_degree = {var.concept: 0 for var in self.probabilistic_model.variables}
        adj = {var.concept: [] for var in self.probabilistic_model.variables}

        for var in self.probabilistic_model.variables:
            child_name = var.concept
            cpd = self.probabilistic_model.get_module_of_concept(child_name)
            if cpd:
                for parent_var in cpd.parents:
                    parent_name = parent_var.concept
                    adj[parent_name].append(child_name)
                    in_degree[child_name] += 1

        # Nodes with zero inbound edges = level 0
        queue = [self.variable_map[name] for name, deg in in_degree.items() if deg == 0]

        sorted_variables = []
        levels = []

        # Track current BFS frontier
        current_level = queue.copy()
        while current_level:
            levels.append(current_level)
            next_level = []

            for var in current_level:
                sorted_variables.append(var)

                for neighbour_name in adj[var.concept]:
                    in_degree[neighbour_name] -= 1
                    if in_degree[neighbour_name] == 0:
                        next_level.append(self.variable_map[neighbour_name])

            current_level = next_level

        return sorted_variables, levels

    def _compute_single_variable(
            self,
            var: Variable,
            evidence: Dict[str, torch.Tensor],
            results: Dict[str, torch.Tensor],
    ) -> Tuple[str, torch.Tensor]:
        """
        Compute the output tensor for a single variable.

        Args:
            var: The variable to compute.
            evidence: Dictionary of evidence tensors. For root variables this
                contains the external input; for non-root variables an entry
                means the value is observed and used directly as output.
            results: Dictionary of already computed variable outputs.

        Returns:
            Tuple of (concept_name, output_tensor).

        Raises:
            RuntimeError: If CPD is missing for the variable.
            ValueError: If root variable is missing from evidence.
            RuntimeError: If parent variable hasn't been computed yet.
        """
        concept_name = var.concept
        
        # Get parents from cached info (robust to intervention module replacement)
        parents = self._cached_parents.get(concept_name, [])
        
        # If evidence is provided for a non-root variable, use it directly as output
        # (Root nodes still pass their input through the CPD)
        if parents and concept_name in evidence:
            return concept_name, evidence[concept_name]
        
        parametric_cpd = self.probabilistic_model.get_module_of_concept(concept_name)

        if parametric_cpd is None:
            raise RuntimeError(f"Missing parametric_cpd for variable/concept: {concept_name}")

        # 1. Root nodes (no parents)
        if not parents:
            if concept_name not in evidence:
                raise ValueError(f"Root variable '{concept_name}' requires an input tensor in the 'evidence' dictionary.")
            input_tensor = evidence[concept_name]
            parent_kwargs = self.get_parent_kwargs(parametric_cpd, [input_tensor], [])
            output_tensor = parametric_cpd.forward(**parent_kwargs)

        # 2. Child nodes (has parents)
        else:
            parent_concepts = []
            parent_input = []
            for parent_var in parents:
                parent_name = parent_var.concept
                if parent_name not in results:
                    # Should not happen with correct topological sort
                    raise RuntimeError(f"Parent data missing: Cannot compute {concept_name} because parent {parent_name} has not been computed yet.")

                if isinstance(parent_var, ConceptVariable):
                    # For probabilistic parents, pass concepts
                    weight = 1
                    if self.graph_learner is not None:
                        weight = self.graph_learner.weighted_adj[self.row_labels2id[parent_name], self.col_labels2id[concept_name]]
                    parent_concepts.append(results[parent_name] * weight)
                else:
                    # For continuous parents, pass latent features
                    parent_input.append(results[parent_name])

            parent_kwargs = self.get_parent_kwargs(parametric_cpd, parent_input, parent_concepts)
            output_tensor = parametric_cpd.forward(**parent_kwargs)

        return concept_name, output_tensor

    def _resolve_device(self, device: str) -> bool:
        """Resolve device string to use_cuda boolean.
        
        Args:
            device: Device specification ('auto', 'cuda', 'gpu', or 'cpu').
            
        Returns:
            True if CUDA should be used, False otherwise.
            
        Raises:
            RuntimeError: If CUDA is requested but not available.
            ValueError: If device string is invalid.
        """
        if device == 'auto':
            return torch.cuda.is_available()
        elif device in ['cuda', 'gpu']:
            if not torch.cuda.is_available():
                raise RuntimeError(f"device='{device}' was specified but CUDA is not available")
            return True
        elif device == 'cpu':
            return False
        else:
            raise ValueError(f"Invalid device '{device}'. Must be 'auto', 'cuda', 'gpu', or 'cpu'")

    def _predict_level(
        self,
        level: List[Variable],
        evidence: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor],
        debug: bool = False,
        use_cuda: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute predictions for a single level.

        Iterates over unique CPDs rather than variables.  Shared CPDs are
        executed once and the output is sliced into per-concept tensors.
        """
        level_results = {}

        # Deduplicate CPDs: collect unique (cpd_id -> (cpd, primary_var))
        seen_cpd_ids: Set[int] = set()
        unique_cpds: List[Tuple[Variable, object]] = []   # (representative var, cpd)
        for var in level:
            cpd = self.probabilistic_model.get_module_of_concept(var.concept)
            if cpd is None:
                raise RuntimeError(f"Missing parametric_cpd for variable/concept: {var.concept}")
            cid = id(cpd)
            if cid in seen_cpd_ids:
                continue
            seen_cpd_ids.add(cid)
            # For shared CPDs the representative var is the primary (first concept)
            rep_var = self.variable_map[cpd.concept] if getattr(cpd, 'shared', False) else var
            unique_cpds.append((rep_var, cpd))

        def _run_cpd(var, cpd):
            concept_name, output_tensor = self._compute_single_variable(var, evidence, results)
            if getattr(cpd, 'shared', False):
                offset = 0
                for cname in cpd.concepts:
                    var_size = self.variable_map[cname].out_features
                    level_results[cname] = output_tensor[..., offset:offset + var_size]
                    offset += var_size
            else:
                level_results[concept_name] = output_tensor

        # Sequential execution
        if debug or len(unique_cpds) <= 1:
            for var, cpd in unique_cpds:
                _run_cpd(var, cpd)
            return level_results

        # Parallel execution
        if use_cuda:
            streams = [torch.cuda.Stream(device=torch.cuda.current_device()) for _ in unique_cpds]
            for (var, cpd), stream in zip(unique_cpds, streams):
                with torch.cuda.stream(stream):
                    _run_cpd(var, cpd)
            torch.cuda.synchronize()
        else:
            with ThreadPoolExecutor(max_workers=len(unique_cpds)) as executor:
                futures = [executor.submit(_run_cpd, var, cpd) for var, cpd in unique_cpds]
                for fut in futures:
                    fut.result()

        return level_results

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
            concept_name = var.concept
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

    @staticmethod
    def _get_allowed_params(parametric_cpd) -> Set[str]:
        """
        Extract the set of allowed parameter names from a CPD's forward signature.

        Args:
            parametric_cpd: The CPD module to inspect.

        Returns:
            Set of parameter names (excluding 'self').
        """
        if isinstance(parametric_cpd.parametrization, (_InterventionWrapper, _GlobalPolicyInterventionWrapper)):
            forward_to_check = parametric_cpd.parametrization.forward_to_check
        else:
            forward_to_check = parametric_cpd.parametrization.forward

        sig = inspect.signature(forward_to_check)
        return {
            name for name, p in sig.parameters.items()
            if name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

    # Known PyC parameter-name combinations
    _PYC_PARAM_SETS = [
        {'concepts'},
        {'concepts', 'latent'},
        {'concepts', 'exogenous'},
        {'latent'},
        {'exogenous'},
    ]

    def get_parent_kwargs(
        self,
        parametric_cpd,
        parent_input: Union[List[torch.Tensor], torch.Tensor] = None,
        parent_concepts: Union[List[torch.Tensor], torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare keyword arguments for CPD forward pass based on parent outputs.

        Uses the cached parameter names from ``__init__`` to avoid calling
        ``inspect.signature`` on every forward pass.

        Args:
            parametric_cpd: The CPD module to call.
            parent_input: List of continuous parent outputs (latent/exogenous).
            parent_concepts: List of probabilistic parent outputs (concept values).

        Returns:
            Dictionary of kwargs ready for ``parametric_cpd.forward(**kwargs)``.
        """
        allowed = self._cpd_allowed_params.get(parametric_cpd.concept)
        if allowed is None:
            # Fallback for dynamically added CPDs
            allowed = self._get_allowed_params(parametric_cpd)

        parent_kwargs: Dict[str, torch.Tensor] = {}

        if allowed in self._PYC_PARAM_SETS:
            # PyC layer: separate concepts and latent/exogenous inputs
            if 'concepts' in allowed:
                parent_kwargs['concepts'] = torch.cat(parent_concepts, dim=-1)
            if 'latent' in allowed:
                parent_kwargs['latent'] = torch.cat(parent_input, dim=-1)
            elif 'exogenous' in allowed:
                # Exogenous inputs typically have shape (batch, concepts, features).
                # When multiple exogenous parents are combined, concatenate along
                # the concept dimension (dim=1), not the feature dimension.
                parent_kwargs['exogenous'] = torch.cat(parent_input, dim=1)
        else:
            # Standard torch module: concatenate everything into a single tensor
            combined = torch.cat(parent_concepts + parent_input, dim=-1)
            # Feed into the first positional parameter
            first_param = next(iter(allowed))
            parent_kwargs[first_param] = combined

        return parent_kwargs

    def _concatenate_results(
        self,
        query_concepts: List[str],
        all_predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract and concatenate predictions for queried concepts.

        When all queried concepts belong to a single shared CPD and are
        requested in their original order, the full output tensor is returned
        directly (avoiding thousands of tiny ``torch.cat`` calls).

        Args:
            query_concepts: Ordered list of concept names to return.
            all_predictions: Dictionary mapping concept names to output tensors.

        Returns:
            Concatenated tensor (Batch x TotalFeatures).

        Raises:
            ValueError: If a queried concept is missing from predictions.
            RuntimeError: If batch sizes or feature dimensions are inconsistent.
        """
        # --- fast path: entire shared CPD queried in order ---
        if self._shared_cpd_primaries:
            shared_map = self.probabilistic_model._shared_cpd_map
            first = query_concepts[0]
            primary = shared_map.get(first, first)
            if primary in self._shared_cpd_primaries:
                cpd = self.probabilistic_model.factors[str(primary)]
                if query_concepts == list(cpd.concepts):
                    # All concepts of this shared CPD in original order → cat is unnecessary.
                    # The slices are contiguous views; stack them once.
                    return torch.cat(
                        [all_predictions[c] for c in cpd.concepts], dim=-1
                    )

        # --- general path ---
        result_tensors = []
        for concept_name in query_concepts:
            if concept_name not in all_predictions:
                raise ValueError(
                    f"Query concept '{concept_name}' was requested but could not be computed. "
                    f"Available predictions: {list(all_predictions.keys())}"
                )
            result_tensors.append(all_predictions[concept_name])

        batch_size = result_tensors[0].shape[0]
        if any(t.shape[0] != batch_size for t in result_tensors):
            raise RuntimeError("Batch size mismatch detected in query results before concatenation.")

        final_tensor = torch.cat(result_tensors, dim=-1)

        expected_feature_dim = sum(self.variable_map[c].out_features for c in query_concepts)
        if final_tensor.shape[-1] != expected_feature_dim:
            raise RuntimeError(
                f"Concatenation error. Expected total feature dimension of {expected_feature_dim}, "
                f"but got {final_tensor.shape[1]}. Check Variable.out_features logic."
            )

        return final_tensor

    def _get_ancestors(self, query_concepts: List[str]) -> Set[str]:
        """
        Get all ancestors (including query concepts) needed to compute the query.

        Args:
            query_concepts: List of concept names to query.

        Returns:
            Set of concept names that need to be computed (query + all ancestors).
        """
        needed = set(query_concepts)
        queue = list(query_concepts)
        while queue:
            concept_name = queue.pop()
            for parent_var in self._cached_parents.get(concept_name, []):
                parent_name = parent_var.concept
                if parent_name not in needed:
                    needed.add(parent_name)
                    queue.append(parent_name)
        return needed

    def query(
        self, 
        query: List[str], 
        evidence: Dict[str, torch.Tensor], 
        debug: bool = False, 
        device: str = 'auto',
        return_logits: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Execute forward pass and return only specified concepts concatenated.

        This method runs inference level-by-level (exploiting parallelism within
        each level) and then extracts and concatenates only the requested concepts
        in the specified order.

        Args:
            query: List of concept names to retrieve (e.g., ["C", "B", "A"]).
            evidence: Dictionary of {root_concept_name: input_tensor}.
            debug: If True, runs in debug mode (sequential execution).
            device: Device to use for computation. Options:
                - 'auto' (default): Automatically detect and use CUDA if available, else CPU
                - 'cuda' or 'gpu': Force use of CUDA (will raise error if not available)
                - 'cpu': Force use of CPU even if CUDA is available
            return_logits: If True, return raw CPD outputs (logits) instead of
                activated values.  Useful during training when the loss expects
                logits (e.g. ``BCEWithLogitsLoss``).
            **kwargs: Additional keyword arguments (ignored for forward compatibility).

        Returns:
            Single tensor containing concatenated predictions for requested concepts,
            ordered as requested (Batch x TotalFeatures).

        Raises:
            ValueError: If requested concept was not computed.
            RuntimeError: If batch sizes don't match across concepts.
            RuntimeError: If concatenation produces unexpected feature dimension.
            RuntimeError: If device='cuda'/'gpu' is specified but CUDA is not available.
        """
        # Filter kwargs to only those needed by this inference to save memory
        kwargs = self._filter_kwargs(kwargs)
        
        assert query, "Query list cannot be empty - at least one concept must be requested."
        self._validate_evidence(evidence)
        use_cuda = self._resolve_device(device)

        # When lazy=True, restrict to the ancestor sub-graph of the query.
        levels = self.levels
        if self.lazy:
            needed = self._get_ancestors(query)
            levels = [[v for v in lvl if v.concept in needed] for lvl in levels]
            levels = [lvl for lvl in levels if lvl]

        # Two dicts:
        #   `propagation` – activated values fed to children (detached when self.detach)
        #   `returned`    – allocated only when propagation can't serve as return
        #                   (i.e. when return_logits or self.detach)
        need_separate_return = return_logits or self.detach
        returned: Dict[str, torch.Tensor] | None = {} if need_separate_return else None
        propagation: Dict[str, torch.Tensor] = dict(evidence)

        for level in levels:
            level_output = self._predict_level(level, evidence, propagation, debug=debug, use_cuda=use_cuda)

            # FIXME: where to apply global interventions? extract this from the inference
            self._apply_global_interventions_for_level(level, level_output, debug=debug, use_cuda=use_cuda)

            for name, pred in level_output.items():
                variable = self.variable_map.get(name)
                if isinstance(variable, ConceptVariable):
                    activated = self.activate(pred, variable)
                    if returned is not None:
                        returned[name] = pred if return_logits else activated
                    propagation[name] = activated.detach() if self.detach else activated
                else:
                    if returned is not None:
                        returned[name] = pred
                    propagation[name] = pred

        return self._concatenate_results(query, returned if returned is not None else propagation)

    @property
    def available_query_vars(self) -> Set[str]:
        """
        Get all variable names available for querying.

        Returns:
            Set of concept names that can be queried.
        """
        if hasattr(self, "_unrolled_query_vars"):
            return self._unrolled_query_vars
        return set(var.concept for var in self.probabilistic_model.variables)

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
            child_name = var.concept
            cpd = self.probabilistic_model.get_module_of_concept(child_name)
            if cpd:
                for parent in cpd.parents:
                    parent_name = parent.concept
                    children_map[parent_name].add(child_name)

        # All variable names in the ProbabilisticModel
        all_names: Set[str] = {var.concept for var in self.probabilistic_model.variables}

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
            child_name = var.concept
            cpd = self.probabilistic_model.get_module_of_concept(child_name)
            if cpd is None:
                continue
            new_parents: List[Variable] = []
            seen: Set[str] = set()

            for parent in cpd.parents:
                parent_orig = parent.concept

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

                new_parents.append(self.variable_map[mapped_parent])
                seen.add(mapped_parent)

            cpd.parents = new_parents

        # --- 4) Build final ordered list of variables (unique, no duplicates) ---
        new_variables: List[Variable] = []
        seen_var_names: Set[str] = set()

        for var in self.sorted_variables:
            name = var.concept
            if name in keep_names and name not in seen_var_names:
                new_variables.append(var)
                seen_var_names.add(name)

        # --- 5) Unique list of CPDs corresponding to these variables ---
        new_parametric_cpds: List[object] = []
        seen_parametric_cpds: Set[object] = set()

        repeats = [self.probabilistic_model.concept_to_variable[p].size for p in row_labels]
        for var in new_variables:
            parametric_cpd = self.probabilistic_model.parametric_cpds[var.concept]
            if parametric_cpd is not None and parametric_cpd not in seen_parametric_cpds:
                if parametric_cpd.concept in rename_map.values() and parametric_cpd.concept in col_labels:
                    col_id = self.col_labels2id[parametric_cpd.concept]
                    mask = adj[:, col_id] != 0
                    mask_without_self_loop = torch.cat((mask[:col_id], mask[col_id + 1:]))
                    rep = repeats[:col_id] + repeats[col_id + 1:]
                    mask_with_cardinalities = torch.repeat_interleave(mask_without_self_loop, torch.tensor(rep))
                    parametric_cpd.parametrization.prune(mask_with_cardinalities)
                new_parametric_cpds.append(parametric_cpd)
                seen_parametric_cpds.add(parametric_cpd)

        # --- 6) Update available_query_vars to reflect the unrolled graph ---
        self._unrolled_query_vars = set(v.concept for v in new_variables)

        return ProbabilisticModel(new_variables, new_parametric_cpds)

