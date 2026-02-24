"""Independent training learner for concept-based models.

In independent training, each level of the concept graph is trained separately.
During training, ground truth concepts from previous levels are used as input.
During validation/testing, predicted concepts are cascaded forward.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

import torch
from ..base.learner import BaseLearner


class IndependentLearner(BaseLearner):
    """Independent training learner for concept-based models.
    
    Trains concept levels independently: uses ground truth from previous
    levels during training, predicted values during validation/testing.
    
    Parameters
    ----------
    parallel : bool, optional
        If True (default), use parallel level computation during training.
        If False, use sequential computation.
    **kwargs
        Additional arguments passed to BaseLearner.
    """
    
    def __init__(self, parallel: bool = False, **kwargs):
        super(IndependentLearner, self).__init__(**kwargs)
        self.parallel = parallel

    def _extract_parent_evidence(self, concepts_tensor, level):
        """Extract evidence for all concept parents of a level.
        
        Parameters
        ----------
        concepts_tensor : torch.Tensor
            Ground truth concepts tensor.
        level : List[str]
            Concept names in this level.
            
        Returns
        -------
        dict
            Evidence dict mapping parent concept names to evidence tensors
            in the format expected by the inference engine.
        """
        evidence = {}
        ann = self.concept_annotations
        pm = self.inference.probabilistic_model
        
        for concept_name in level:
            var = pm.concept_to_variable[concept_name]
            for p in var.parents:
                parent_name = p.concept
                # Skip non-concept parents and already-processed parents
                if parent_name == 'input' or parent_name in evidence:
                    continue
                if parent_name not in ann.label_to_index:
                    continue
                    
                idx = ann.label_to_index[parent_name]
                card = ann.cardinalities[idx]
                c_value = concepts_tensor[:, idx]
                
                # Use inference-specific conversion
                evidence[parent_name] = self.inference.ground_truth_to_evidence(c_value, card)
        
        return evidence

    def _scatter_level_predictions(self, level_out, level, out):
        """Scatter level predictions from level order to annotation order.
        
        Parameters
        ----------
        level_out : torch.Tensor
            Predictions for concepts in `level`, columns in level order.
        level : List[str]
            Concept names for this level, in the order they appear in level_out.
        out : torch.Tensor
            Pre-allocated output tensor to scatter into (annotation order).
        
        Returns
        -------
        torch.Tensor
            Updated output tensor with level predictions scattered into annotation order.
        """
        col_offset = 0
        for concept_name in level:
            s = self.concept_annotations.concept_slices[concept_name]
            card = s.stop - s.start
            out[:, s] = level_out[:, col_offset:col_offset + card]
            
            if self.accumulated_evidence is not None:
                self.accumulated_evidence[concept_name] = out[:, s]
            
            col_offset += card

        return out

    def _compute_single_level(
        self,
        level_idx: int,
        level: List[str],
        latent_input: torch.Tensor,
        concepts_tensor: torch.Tensor,
    ) -> Tuple[int, List[str], torch.Tensor]:
        """Compute predictions for a single level.
        
        Parameters
        ----------
        level_idx : int
            Index of this level in graph_levels.
        level : List[str]
            Concept names in this level.
        latent_input : torch.Tensor
            Pre-computed latent input tensor.
        concepts_tensor : torch.Tensor
            Ground truth concepts for extracting parent evidence.
            
        Returns
        -------
        Tuple[int, List[str], torch.Tensor]
            (level_idx, level, level_predictions) for later scattering.
        """
        evidence = self._extract_parent_evidence(concepts_tensor, level)
        evidence['input'] = latent_input
        level_out = self.forward(evidence=evidence, query=level)
        return level_idx, level, level_out

    def _prepare_prediction(self, inputs, use_ground_truth):
        """Prepare output tensor and latent input for prediction.
        
        Parameters
        ----------
        inputs : dict
            Input dictionary with 'x' key.
        use_ground_truth : bool
            If True, training mode (no accumulated evidence needed).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (out, latent_input) - pre-allocated output and latent input.
        """
        out = torch.empty(
            inputs['x'].size(0), 
            sum(self.concept_annotations.cardinalities), 
            device=inputs['x'].device
        )
        self.accumulated_evidence = {} if not use_ground_truth else None
        latent_input = self.forward(x=inputs['x'], query=['input'])
        return out, latent_input

    def predict_all_levels_sequential(self, inputs, concepts=None, use_ground_truth=True):
        """Predict concepts level by level (sequential version).
        
        Simple sequential implementation that processes levels one at a time.
        
        Parameters
        ----------
        inputs : dict
            Input dictionary with 'x' key.
        concepts : dict, optional
            Ground truth concepts dict with 'c' key.
        use_ground_truth : bool
            If True, use ground truth for previous levels (training).
            If False, cascade predictions (validation/testing).
        
        Returns
        -------
        torch.Tensor
            Predictions for all concepts in annotation order.
        """
        out, latent_input = self._prepare_prediction(inputs, use_ground_truth)
        
        for level in self.graph_levels:
            # Training: use ground truth evidence for parents
            if use_ground_truth and concepts is not None:
                evidence = self._extract_parent_evidence(concepts['c'], level)
            # Validation/test: use accumulated predictions as evidence
            else:
                evidence = self.accumulated_evidence.copy()
            
            evidence['input'] = latent_input
            level_out = self.forward(evidence=evidence, query=level)
            out = self._scatter_level_predictions(level_out, level, out)
        
        return out

    def predict_all_levels_parallel(self, inputs, concepts=None, use_ground_truth=True):
        """Predict concepts level by level (parallel version).
        
        During training with ground truth, all levels are computed in parallel
        since they don't depend on each other's predictions. Uses CUDA streams
        on GPU or ThreadPoolExecutor on CPU.
        
        During validation/testing (cascading), falls back to sequential since
        each level depends on previous predictions.
        
        Parameters
        ----------
        inputs : dict
            Input dictionary with 'x' key.
        concepts : dict, optional
            Ground truth concepts dict with 'c' key.
        use_ground_truth : bool
            If True, use ground truth for previous levels (training).
            If False, cascade predictions (validation/testing).
        
        Returns
        -------
        torch.Tensor
            Predictions for all concepts in annotation order.
        """
        out, latent_input = self._prepare_prediction(inputs, use_ground_truth)
        
        # Cascading mode must be sequential (each level depends on previous)
        if not use_ground_truth or concepts is None:
            for level in self.graph_levels:
                evidence = self.accumulated_evidence.copy()
                evidence['input'] = latent_input
                level_out = self.forward(evidence=evidence, query=level)
                out = self._scatter_level_predictions(level_out, level, out)
            return out
        
        # Training mode: all levels can be computed in parallel
        num_levels = len(self.graph_levels)
        
        # Single level: no parallelization overhead
        if num_levels == 1:
            level = self.graph_levels[0]
            _, level, level_out = self._compute_single_level(
                0, level, latent_input, concepts['c']
            )
            out = self._scatter_level_predictions(level_out, level, out)
            return out
        
        # Multiple levels: parallelize
        use_cuda = inputs['x'].is_cuda and torch.cuda.is_available()
        level_outputs = []
        
        if use_cuda:
            # GPU: parallel via CUDA streams
            streams = [torch.cuda.Stream(device=inputs['x'].device) for _ in self.graph_levels]
            
            for (level_idx, level), stream in zip(enumerate(self.graph_levels), streams):
                with torch.cuda.stream(stream):
                    result = self._compute_single_level(
                        level_idx, level, latent_input, concepts['c']
                    )
                    level_outputs.append(result)
            
            torch.cuda.synchronize()
        else:
            # CPU: parallel via threads
            with ThreadPoolExecutor(max_workers=num_levels) as executor:
                futures = [
                    executor.submit(
                        self._compute_single_level,
                        level_idx, level, latent_input, concepts['c']
                    )
                    for level_idx, level in enumerate(self.graph_levels)
                ]
                for fut in futures:
                    level_outputs.append(fut.result())
        
        # Scatter results in level order
        level_outputs.sort(key=lambda x: x[0])
        for _, level, level_out in level_outputs:
            out = self._scatter_level_predictions(level_out, level, out)
        
        return out


    def shared_step(self, batch, step='train'):
        """Shared logic for train/val/test steps.
        
        Parameters
        ----------
        batch : dict
            Batch dictionary with 'inputs' and 'concepts' keys.
        step : str
            One of 'train', 'val', or 'test'.
            
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        inputs, concepts, transforms = self.unpack_batch(batch)
        batch_size = batch['inputs']['x'].size(0)
        c = c_loss = concepts['c']

        # Training uses ground truth, validation/test use cascade
        use_ground_truth = (step == 'train')
        if self.parallel:
            out = self.predict_all_levels_parallel(inputs, concepts, use_ground_truth)
        else:
            out = self.predict_all_levels_sequential(inputs, concepts, use_ground_truth)

        # Compute loss
        if self.loss is not None:
            loss_args = self.filter_output_for_loss(out, c_loss)
            loss = self.loss(**loss_args)
            self.log_loss(step, loss, batch_size=batch_size)

        # Update and log metrics
        metrics_args = self.filter_output_for_metrics(out, c)
        self.update_and_log_metrics(metrics_args, step, batch_size)
        return loss

    def training_step(self, batch):
        """Training step for independent learning.
        
        Uses ground truth concepts from previous levels.
        
        Args:
            batch (dict): Training batch.
            
        Returns:
            torch.Tensor: Training loss.
        """
        loss = self.shared_step(batch, step='train')
        return loss

    def validation_step(self, batch):
        """Validation step with cascading predictions.
        
        Uses predicted concepts from previous levels.
        
        Args:
            batch (dict): Validation batch.
            
        Returns:
            torch.Tensor: Validation loss.
        """
        loss = self.shared_step(batch, step='val')
        return loss
    
    def test_step(self, batch):
        """Test step with cascading predictions.
        
        Uses predicted concepts from previous levels.
        
        Args:
            batch (dict): Test batch.
            
        Returns:
            torch.Tensor: Test loss.
        """
        loss = self.shared_step(batch, step='test')
        return loss
