"""Independent training inference."""

from typing import List, Dict, Tuple, Optional

import torch

from .deterministic import DeterministicInference
from ..models.variable import ConceptVariable


class IndependentInference(DeterministicInference):
    """
    Independent training inference.

    This inference engine supports independent (level-by-level) training where
    ground truth concepts from previous levels are used during training.
    Ground truth is always required - for evaluation, use DeterministicInference.

    Processing is done level-by-level to exploit the same parallelization as
    ForwardInference (CUDA streams on GPU, ThreadPoolExecutor on CPU).

    Supports the ``detach`` and ``lazy`` parameters inherited from
    :class:`ForwardInference`.  When ``detach=True``, concept predictions
    that lack ground truth are detached before propagation.

    **No annotations required.** Ground truth is passed as a flat tensor in
    index format (one column per concept) with ``concept_names`` to specify order.

    Parameters
    ----------
    probabilistic_model : ProbabilisticModel
        The probabilistic model to perform inference on.
    graph_learner : BaseGraphLearner, optional
        Optional graph structure learner.
    detach : bool, default False
        If True, detach concept predictions (without GT) before propagation.
    lazy : bool, default False
        If True, only compute ancestors of the queried concepts.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import IndependentInference
        >>>
        >>> # Create inference engine
        >>> inference = IndependentInference(pgm)
        >>>
        >>> # Ground truth in index format: one column per concept
        >>> gt_tensor = torch.tensor([[1, 0]])  # [A_idx, B_idx]
        >>> out = inference.query(['A', 'B'], evidence={'input': x}, 
        ...                       ground_truth=gt_tensor, concept_names=['A', 'B'])
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index_cache: Tuple[Optional[tuple], Dict[str, int]] = (None, {})

    def _get_index_map(self, concept_names: List[str]) -> Dict[str, int]:
        """Get cached mapping from concept name to column index."""
        key = tuple(concept_names)
        if self._index_cache[0] == key:
            return self._index_cache[1]
        
        index_map = {name: i for i, name in enumerate(concept_names)}
        self._index_cache = (key, index_map)
        return index_map

    @property
    def query_kwargs(self) -> frozenset:
        """Kwargs accepted by this inference: ground_truth, concept_names."""
        return frozenset({'ground_truth', 'concept_names'})

    def query(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        ground_truth: torch.Tensor,
        concept_names: List[str],
        debug: bool = False,
        device: str = 'auto',
        return_logits: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Query concepts using independent training.

        Processing is level-by-level to exploit parallelization:
        1. Compute predictions for the level (parallel via CUDA streams or threads)
        2. Store raw logits for ``return_logits`` mode
        3. Use GT values (where available) for propagation to next level;
           non-GT concepts are activated and optionally detached when
           ``self.detach`` is True.
        4. Return **activated predictions** for queried concepts in query order
           (or raw logits when ``return_logits=True``)

        Parameters
        ----------
        query : List[str]
            List of concept names to retrieve. Output is in this order.
        evidence : Dict[str, torch.Tensor]
            Dictionary of {root_concept_name: input_tensor}.
        ground_truth : torch.Tensor
            Ground truth tensor in index format. Shape: (batch, num_concepts).
            One column per concept containing class index or binary value.
        concept_names : List[str]
            Ordered concept names matching ground_truth columns.
        debug : bool, default False
            If True, runs sequentially (no parallelism).
        device : str, default 'auto'
            Device for computation ('auto', 'cuda', 'cpu').
        return_logits : bool, default False
            If True, return raw CPD outputs (logits) instead of activated values.
            Useful during training when the loss expects logits.
        **kwargs
            Additional kwargs (ignored, allows any Learner to pass all kwargs).

        Returns
        -------
        torch.Tensor
            Concatenated predictions for requested concepts in query order.
        """
        # Filter kwargs to only those needed by this inference to save memory
        kwargs = self._filter_kwargs(kwargs)
        
        assert query, "Query list cannot be empty - at least one concept must be requested."
        self._validate_evidence(evidence)

        use_cuda = self._resolve_device(device)

        # Lazy: restrict to the ancestor sub-graph of the query.
        levels = self.levels
        if self.lazy:
            needed = self._get_ancestors(query)
            levels = [[v for v in lvl if v.concept in needed] for lvl in levels]
            levels = [lvl for lvl in levels if lvl]

        # Two dicts only:
        #   `propagation` – activated and fed to children (may contain GT overrides)
        #   `returned`    – model's own predictions (logits or activated)
        returned: Dict[str, torch.Tensor] = {}
        propagation: Dict[str, torch.Tensor] = dict(evidence)
        index_map = self._get_index_map(concept_names)
        
        for level in levels:
            level_output = self._predict_level(level, evidence, propagation, debug=debug, use_cuda=use_cuda)
            
            # FIXME: where to apply global interventions? extract this from the inference
            self._apply_global_interventions_for_level(level, level_output, debug=debug, use_cuda=use_cuda)

            for name, pred in level_output.items():
                variable = self.variable_map.get(name)
                if isinstance(variable, ConceptVariable):
                    # GT override: skip activation when only logits are needed
                    if name in index_map:
                        idx = index_map[name]
                        if return_logits:
                            returned[name] = pred
                        else:
                            returned[name] = self.activate(pred, variable)
                        propagation[name] = self.ground_truth_to_evidence(
                            value=ground_truth[:, idx:idx+1],
                            cardinality=variable.size,
                        )
                    else:
                        activated = self.activate(pred, variable)
                        returned[name] = pred if return_logits else activated
                        propagation[name] = activated.detach() if self.detach else activated
                else:
                    returned[name] = pred
                    propagation[name] = pred

        return self._concatenate_results(query, returned)
