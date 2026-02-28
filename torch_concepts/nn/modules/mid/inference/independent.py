"""Independent training inference."""

from typing import List, Dict

import torch

from .deterministic import DeterministicInference


class IndependentInference(DeterministicInference):
    """
    Independent training inference.

    This inference engine supports independent (level-by-level) training where
    ground truth concepts from previous levels are used during training,
    while predictions are cascaded during evaluation.

    Processing is done level-by-level to exploit the same parallelization as
    ForwardInference (CUDA streams on GPU, ThreadPoolExecutor on CPU).

    **No annotations required at construction.** Ground truth is passed as a dictionary
    mapping concept names to their GT tensors. Use `prepare_forward_kwargs` with
    annotations to convert flat GT tensors from batches.

    Parameters
    ----------
    probabilistic_model : ProbabilisticModel
        The probabilistic model to perform inference on.
    graph_learner : BaseGraphLearner, optional
        Optional graph structure learner.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import IndependentInference
        >>>
        >>> # Create inference engine (no annotations needed!)
        >>> inference = IndependentInference(pgm)
        >>>
        >>> # Training: use ground truth for parents
        >>> x = torch.randn(4, 10)
        >>> ground_truth = {
        ...     'A': torch.tensor([[1], [0], [1], [0]]).float(),
        ...     'B': torch.tensor([[0], [1], [1], [0]]).float(),
        ... }
        >>> output = inference.query(['A', 'B'], evidence={'input': x}, ground_truth=ground_truth)
        >>>
        >>> # Evaluation: cascade predictions (no ground_truth provided)
        >>> output = inference.query(['A', 'B'], evidence={'input': x})
    """

    def prepare_forward_kwargs(self, batch, annotations, **kwargs) -> dict:
        """Extract ground truth from the batch for independent training.

        Parameters
        ----------
        batch : dict
            Raw batch dictionary from the dataloader.
        annotations : AxisAnnotation, optional
            Concept annotations containing label indices and cardinalities.
        **kwargs
            Additional learner context.

        Returns
        -------
        dict
            ``{'ground_truth': {concept_name: tensor, ...}}`` if available, else ``{}``.
        """
        # Validate inputs
        assert batch is not None, "Batch cannot be None"
        assert annotations is not None, "Annotations cannot be None"
        assert hasattr(annotations, 'label_to_index'), "Annotations must have 'label_to_index' attribute"
        assert hasattr(annotations, 'cardinalities'), "Annotations must have 'cardinalities' attribute"

        concepts = batch.get('concepts')
        assert concepts is not None, "Batch must contain 'concepts' key"
        assert 'c' in concepts, "Concepts dictionary must contain 'c' key"

        gt_tensor = concepts['c']

        ground_truth_dict = {}
        for label, idx in annotations.label_to_index.items():
            card = annotations.cardinalities[idx]
            c_value = gt_tensor[:, idx]
            ground_truth_dict[label] = self.ground_truth_to_evidence(c_value, card)

        return {'ground_truth': ground_truth_dict}

    def query(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        debug: bool = False,
        device: str = 'auto'
    ) -> torch.Tensor:
        """
        Query concepts using independent training.

        Processing is level-by-level to exploit parallelization:
        1. Compute predictions for the level (parallel via CUDA streams or threads)
        2. Store original predictions for output
        3. Use GT values (where available) for propagation to next level
        4. Return **original predictions** for queried concepts in query order

        Parameters
        ----------
        query : List[str]
            List of concept names to retrieve. Output is in this order.
        evidence : Dict[str, torch.Tensor]
            Dictionary of {root_concept_name: input_tensor}.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary mapping concept names to their ground truth tensors.
        debug : bool, default False
            If True, runs sequentially (no parallelism).
        device : str, default 'auto'
            Device for computation ('auto', 'cuda', 'cpu').

        Returns
        -------
        torch.Tensor
            Concatenated predictions for requested concepts in query order.
        """
        assert query, "Query list cannot be empty - at least one concept must be requested."
        self._validate_evidence(evidence)

        # Independent training: level-by-level with GT replacement
        use_cuda = self._resolve_device(device)
        all_predictions = {}  # Original predictions (for loss computation)
        propagation = {}      # Values for propagation (GT where available)

        for level in self.levels:
            # Compute predictions for this level using parent's parallel execution
            level_predictions = self._predict_level(
                level, evidence, propagation, debug=debug, use_cuda=use_cuda
            )

            # Store original predictions for output
            all_predictions.update(level_predictions)

            # For propagation: use GT where available, else use predictions
            for concept_name, pred in level_predictions.items():
                if concept_name in ground_truth:
                    propagation[concept_name] = ground_truth[concept_name]
                else:
                    propagation[concept_name] = pred

        return self._concatenate_results(query, all_predictions)
