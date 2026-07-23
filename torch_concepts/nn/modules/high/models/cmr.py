"""Concept-based Memory Reasoner (CMR)

    References: 
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024.
        https://arxiv.org/abs/2407.15527
"""

from typing import List, Optional, Union

import torch

from .....annotations import Annotations

from ...low.base.inference import BaseInference
from ...low.encoders.linear import LinearLatentToConcept
from ...low.encoders.selector import CategoricalSelectorLatentToExogenous
from ...low.predictors.exogenous import MixMemoryConceptExogenousToConcept
from ...low.lazy import LazyConstructor

from ...mid.inference.deterministic import DeterministicInference
from ...mid.constructors.bipartite import BipartiteModel

from ..base.bipartite import BaseBipartiteModel


class ConceptMemoryReasoner(BaseBipartiteModel):
    """Concept Memory Reasoner with configurable training mode.
    
    A unified CMR class that works as a pure PyTorch module by default,
    or as a Lightning module when lightning=True.
    
    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after backbone if used).
    annotations : Annotations
        Concept annotations with labels, cardinalities, and distributions.
    task_names : Union[List[str], str]
        Names of task variables (subset of annotation labels).
    n_rules : int, optional
        Number of candidate rules per task. Defaults to 10.
    memory_latent_size : int, optional
        Latent size of the task-specific rule memory. Defaults to 100.
    memory_decoder_hidden_layers : int, optional
        Number of hidden layers in the rule memory decoder. Defaults to 1.
    selector_hidden_layers : int, optional
        Number of hidden layers in the rule selector MLP. Defaults to 1.
    rec_weight : float, optional
        Reconstruction-weight exponent used by CMR reconstruction-aware
        task prediction. Defaults to 0.1.
    eps : float, optional
        Numerical scaling factor used in the memory decoder softmax.
        Defaults to 1e-3.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
        If False (default), works as pure PyTorch module.
    inference : BaseInference, optional
        Inference engine class for evaluation. Defaults to DeterministicInference.
    train_inference : BaseInference, optional
        Inference engine class for training. Only used when lightning=True.
        Defaults to DeterministicInference.
    variable_distributions : Mapping, optional
        Distribution classes for each concept if not in annotations.
    **kwargs
        Additional arguments passed to BaseBipartiteModel, including:
        
        - **backbone** : Feature extraction module (e.g., ResNet)
        - **latent_encoder** : Custom encoder for latent space
        - **latent_encoder_kwargs** : Arguments for latent encoder
        
        Lightning Training (when lightning=True):
        
        - **loss** : Loss function (nn.Module)
        - **metrics** : ConceptMetrics or dict of MetricCollections
        - **optim_class** : Optimizer class (e.g., torch.optim.Adam)
        - **optim_kwargs** : Optimizer arguments (e.g., {'lr': 0.001})
        - **scheduler_class** : LR scheduler class
        - **scheduler_kwargs** : Scheduler arguments
    
    Examples
    --------
    >>> # Pure PyTorch module (default)
    >>> model = ConceptMemoryReasoner(
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     n_rules=10,
    ...     rec_weight=0.1,
    ... )
    >>> out = model(x, query=['c1', 'task'])  # Direct forward pass
    
    >>> # Lightning training enabled
    >>> model = ConceptMemoryReasoner(
    ...     lightning=True,
    ...     input_size=8,
    ...     annotations=ann,
    ...     task_names=['task'],
    ...     n_rules=10,
    ...     rec_weight=0.1,
    ...     loss=my_loss,
    ...     optim_class=torch.optim.Adam,
    ...     optim_kwargs={'lr': 0.001}
    ... )
    """
    
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        n_rules: int = 10,
        memory_latent_size: int = 100,
        memory_decoder_hidden_layers: int = 1,
        selector_hidden_layers: int = 1,
        rec_weight: float = 0.1,
        eps: float = 1e-3,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = DeterministicInference,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            lightning=lightning,
            **kwargs
        )
        
        # Extract concept cardinalities (excluding tasks)
        concept_idxs = [self.concept_names.index(name) for name in self.concept_names
                        if name not in self.task_names]
        cardinalities = [self.concept_annotations.cardinalities[i] for i in concept_idxs]
        assert all(cardinality == 1 for cardinality in cardinalities), (
            "ConceptMemoryReasoner currently requires all concepts "
            f"to have cardinality 1, got {cardinalities}."
        )

        self.rec_weight = rec_weight

        # Build bipartite model architecture with CMR components
        self.model = BipartiteModel(
            task_names=task_names,
            input_size=self.latent_size,
            annotations=annotations,
            encoder=LazyConstructor(LinearLatentToConcept),
            internal_exogenous=LazyConstructor(
                CategoricalSelectorLatentToExogenous,
                out_exogenous=n_rules,
                selector_hidden_layers=selector_hidden_layers,
            ),
            predictor=LazyConstructor(
                MixMemoryConceptExogenousToConcept,
                memory_latent_size=memory_latent_size,
                memory_decoder_hidden_layers=memory_decoder_hidden_layers,
                eps=eps,
            ),
            use_source_exogenous=False,
        )

        self.eval_inference = inference(
            self.model.probabilistic_model, 
            **(inference_kwargs or {})
        )
        self.train_inference = train_inference(
            self.model.probabilistic_model, 
            **(train_inference_kwargs or {})
        )

    def forward(self, query, x=None, evidence=None, *inference_args, **inference_kwargs):
        """Forward pass for CMR.

        Returns standard predictions by default. When invoked by the generic
        learner path (which provides ``ground_truth``), returns both no-rec and
        with-rec predictions so ``filter_output_for_loss`` can build explicit
        CMR loss kwargs`.
        """
        include_rec = inference_kwargs.pop('include_rec', False)
        rec_weight = inference_kwargs.pop('rec_weight', self.rec_weight)
        # CMR predictor already produces task probabilities. Requesting raw CPD outputs avoids an extra sigmoid in inference for Bernoulli variables.
        inference_kwargs['return_logits'] = True

        # BaseLearner passes ground_truth in inference kwargs. Use that signal to produce both branches for CMRLoss.
        # TODO: this is a bit ad-hoc - consider a more explicit, stable way to signal this mode (e.g. a flag).
        learner_mode = 'ground_truth' in inference_kwargs

        if learner_mode:
            no_rec = super().forward(
                query=query,
                x=x,
                include_rec=False,
                evidence=evidence,
                *inference_args,
                rec_weight=rec_weight,
                **inference_kwargs,
            )

            with_rec = super().forward(
                query=query,
                x=x,
                include_rec=True,
                evidence=evidence,
                *inference_args,
                rec_weight=rec_weight,
                **inference_kwargs,
            )
            assert not torch.equal(no_rec, with_rec), (
                "Expected 'no_rec' and 'with_rec' predictions to be different tensors, "
                "but they are identical. Check that 'include_rec' is properly handled in forward pass."
            )
            return {'no_rec': no_rec, 'with_rec': with_rec}

        return super().forward(
            query=query,
            x=x,
            evidence=evidence,
            *inference_args,
            include_rec=include_rec,
            rec_weight=rec_weight,
            **inference_kwargs,
        )

    def filter_output_for_loss(self, forward_out, target):
        """Build explicit CMR loss kwargs for :class:`CMRLoss`.

        Parameters
        ----------
        forward_out : dict
            Dictionary with keys ``no_rec`` and ``with_rec`` containing full
            model predictions (concepts + tasks).
        target : torch.Tensor
            Ground-truth tensor aligned with ``self.concept_names``.

        Returns
        -------
        dict
            Explicit tensors required by ``CMRLoss``.
        """
        if not isinstance(forward_out, dict):
            raise ValueError(
                "ConceptMemoryReasoner.filter_output_for_loss expects a dict "
                "with 'no_rec' and 'with_rec' predictions."
            )

        if 'no_rec' not in forward_out or 'with_rec' not in forward_out:
            raise ValueError(
                "ConceptMemoryReasoner.filter_output_for_loss requires both "
                "'no_rec' and 'with_rec' entries in forward_out."
            )

        no_rec = forward_out['no_rec']
        with_rec = forward_out['with_rec']

        # assert that no_rec and with_rec are different tensors (eps)
        assert not torch.equal(no_rec, with_rec), (
            "Expected 'no_rec' and 'with_rec' predictions to be different tensors, "
            "but they are identical. Check that 'include_rec' is properly handled in forward pass."
        )

        task_indices = [
            i for i, name in enumerate(self.concept_names)
            if name in self.task_names
        ]
        concept_indices = [
            i for i, name in enumerate(self.concept_names)
            if name not in self.task_names
        ]

        # return {
        #     'concept_input': no_rec[:, concept_indices],
        #     'concept_target': target[:, concept_indices],
        #     'task_input': no_rec[:, task_indices],
        #     'task_input_with_rec': with_rec[:, task_indices],
        #     'task_target': target[:, task_indices],
        # }
        return {
            'input': no_rec,
            'target': target,
            'input_with_rec': with_rec,
        }

    def filter_output_for_metrics(self, forward_out, target):
        """Use no-rec predictions for metric computation in learner mode."""
        if isinstance(forward_out, dict):
            return {'preds': forward_out['no_rec'], 'target': target}
        return {'preds': forward_out, 'target': target}
