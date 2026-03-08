"""
Importance Sampling Inference for Probabilistic Graphical Models.

This module implements probabilistic inference using Pyro's importance sampling,
supporting both discrete and continuous variables in concept-based models.
"""

from typing import Callable, Dict, List, Optional, Union

import torch
from torch.distributions import (
    Bernoulli, Categorical,
    RelaxedBernoulli, RelaxedOneHotCategorical,
    Normal, LogNormal, Beta, Gamma,
)

# Pyro imports
import pyro
from pyro.infer import Importance, EmpiricalMarginal

from .forward import ForwardInference
from ..models.variable import Variable, ConceptVariable
from ..models.probabilistic_model import ProbabilisticModel
from ...low.base.graph import BaseGraphLearner
from .utils import (
    DISCRETE_DISTRIBUTIONS,
    CONTINUOUS_DISTRIBUTIONS,
    validate_pyro_distributions,
    get_pyro_distribution,
    build_obs_dict_from_evidence,
    build_pyro_model,
)


class ImportanceSamplingInference(ForwardInference):
    """
    Probabilistic inference using Pyro importance sampling.
    
    This inference engine computes marginal probabilities for variables in a
    probabilistic graphical model using importance sampling. It uses the standard
    Pyro pattern of passing evidence via the ``obs=`` parameter in ``pyro.sample()``.
    
    Key Features:
        - Marginal computation via importance sampling
        - Support for both discrete and continuous variables
        - Evidence conditioning via ``obs=`` parameter
        - Conditional probability queries
        - Integration with trained CBMs and PGMs
    
    Args:
        probabilistic_model: The probabilistic model to perform inference on.
        graph_learner: Optional graph learner for weighted adjacency structure.
        num_samples: Number of importance samples for marginal estimation.
        num_draws: Number of draws from empirical marginal for probability estimation.
        **kwargs: Additional arguments passed to ForwardInference.
    
    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch_concepts import LatentVariable, ConceptVariable
        >>> from torch_concepts.nn import (
        ...     ImportanceSamplingInference, ParametricCPD, ProbabilisticModel
        ... )
        >>>
        >>> # Create a simple PGM: input -> [A, B] -> C
        >>> input_var = LatentVariable('input', parents=[], size=10)
        >>> var_A = ConceptVariable('A', parents=['input'], distribution=Bernoulli)
        >>> var_B = ConceptVariable('B', parents=['input'], distribution=Bernoulli)
        >>> var_C = ConceptVariable('C', parents=['A', 'B'], distribution=Bernoulli)
        >>>
        >>> # ... define CPDs and create model ...
        >>>
        >>> # Create importance sampling inference engine
        >>> inference = ImportanceSamplingInference(model, num_samples=1000)
        >>>
        >>> # Query marginal: p(A | x)
        >>> p_A = inference.marginal(['A'], evidence={'input': x})
        >>>
        >>> # Conditional query: p(C | A=1, x)
        >>> p_C_given_A = inference.marginal(
        ...     ['C'],
        ...     evidence={'input': x, 'A': torch.ones(batch_size, 1)}
        ... )
    """
    
    def __init__(
        self,
        probabilistic_model: ProbabilisticModel,
        graph_learner: BaseGraphLearner = None,
        num_samples: int = 1000,
        num_draws: int = 100,
        detach: bool = False,
        lazy: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            probabilistic_model,
            graph_learner,
            *args,
            detach=detach,
            lazy=lazy,
            **kwargs,
        )
        self.num_samples = num_samples
        self.num_draws = num_draws

        # Validate that all variables have supported distributions
        validate_pyro_distributions(self.probabilistic_model.variables)

        # Build the Pyro model once; it accepts (evidence, obs_dict) at call time.
        self._pyro_model = build_pyro_model(self)
    
    def activate(self, pred: torch.Tensor, variable: Variable) -> torch.Tensor:
        """
        Return raw output (logits/parameters) unchanged.
        
        For Pyro-based inference, we keep raw CPD outputs
        which contain the distribution parameters needed to construct
        Pyro distributions.
        
        Args:
            pred: Raw output tensor from the CPD (logits or parameters).
            variable: The variable being computed.
        
        Returns:
            torch.Tensor: Raw output tensor unchanged.
        """
        return pred
    
    def ground_truth_to_evidence(
        self, 
        value: torch.Tensor, 
        cardinality: int
    ) -> torch.Tensor:
        """
        Convert ground truth to evidence format for conditioning.
        
        For importance sampling with plate notation, evidence should be
        ``(batch_size,)`` for scalar variables (Bernoulli, Categorical class
        indices) so that it is compatible with ``pyro.plate``.
        
        Args:
            value: Ground truth tensor. Shape: (batch_size,) or (batch_size, 1).
            cardinality: Number of classes for this variable.
        
        Returns:
            torch.Tensor: Evidence tensor (hard values), shape ``(batch_size,)``.
        """
        if value.dim() == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value
    
    def marginal(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        num_samples: Optional[int] = None,
        num_draws: Optional[int] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Compute marginal probabilities using Pyro importance sampling.
        
        This method computes p(query | evidence) by building a Pyro model
        with evidence passed via ``obs=`` parameter and running importance sampling.
        
        Args:
            query: List of variable names to compute marginals for.
            evidence: Dictionary of observed variables {name: tensor}.
                     Must include at least the input variables.
            num_samples: Number of importance samples (overrides default).
            num_draws: Number of draws from empirical marginal (overrides default).
            return_dict: If True, return dict with detailed statistics.
                        If False, return concatenated probability tensor.
        
        Returns:
            If return_dict=False (default):
                torch.Tensor: Marginal probabilities concatenated.
                    - For Bernoulli: p(variable=1), shape (batch_size, 1)
                    - For Categorical: [p(class_0), ..., p(class_K)], shape (batch_size, K)
                    - For Continuous: mean value, shape (batch_size, size)
            
            If return_dict=True:
                Dict[str, Dict[str, torch.Tensor]]: Per-variable statistics.
                    - Discrete: {'probs': tensor}
                    - Continuous: {'mean': tensor, 'std': tensor, 'samples': tensor}
        
        Example:
            >>> # Single marginal
            >>> p_A = inference.marginal(['A'], {'input': x})
            >>> 
            >>> # Multiple marginals
            >>> p_AB = inference.marginal(['A', 'B'], {'input': x})
            >>>
            >>> # Conditional marginal
            >>> p_C_given_A = inference.marginal(
            ...     ['C'], 
            ...     {'input': x, 'A': torch.ones(batch, 1)}
            ... )
            >>>
            >>> # Get detailed statistics
            >>> stats = inference.marginal(['A'], {'input': x}, return_dict=True)
            >>> print(stats['A']['probs'])
        """
        if num_samples is None:
            num_samples = self.num_samples
        if num_draws is None:
            num_draws = self.num_draws
        
        # Validate query variables exist
        for var_name in query:
            if var_name not in self.probabilistic_model.concept_to_variable:
                raise ValueError(f"Variable '{var_name}' not found in model.")
        
        batch_size = next(iter(evidence.values())).shape[0]
        
        # Build obs dict from concept evidence
        obs_dict = build_obs_dict_from_evidence(
            self.probabilistic_model, evidence
        )

        # Run importance sampling using the pre-built model
        importance = Importance(self._pyro_model, num_samples=num_samples)
        trace = importance.run(evidence, obs_dict=obs_dict)
        
        # Estimate marginals for queried variables
        results = {}
        for var_name in query:
            var = self.probabilistic_model.concept_to_variable[var_name]
            
            try:
                empirical = EmpiricalMarginal(trace, sites=var_name)
                
                # Draw samples from empirical marginal.
                # Each draw is (batch_size,) for scalar variables inside a plate.
                samples = torch.stack([empirical() for _ in range(num_draws)])
                # samples shape: (num_draws, batch_size) or (num_draws, batch_size, K)
                
                if var.distribution in (Bernoulli, RelaxedBernoulli):
                    # P(var=1) = mean over draws → (batch_size, 1)
                    probs = samples.float().mean(dim=0).unsqueeze(-1)
                    results[var_name] = {'probs': probs}
                
                elif var.distribution in (Categorical, RelaxedOneHotCategorical):
                    # Count class frequencies over draws
                    num_classes = var.size
                    probs = torch.zeros(batch_size, num_classes)
                    for k in range(num_classes):
                        probs[:, k] = (samples == k).float().mean(dim=0)
                    results[var_name] = {'probs': probs}
                
                else:
                    # Continuous: mean and std over draws
                    mean = samples.mean(dim=0)
                    std = samples.std(dim=0)
                    if mean.dim() == 1:
                        mean = mean.unsqueeze(-1)
                        std = std.unsqueeze(-1)
                    results[var_name] = {'mean': mean, 'std': std}
                
            except KeyError:
                # Variable not sampled (in evidence or disconnected)
                if var.distribution in (Categorical, RelaxedOneHotCategorical):
                    results[var_name] = {'probs': torch.zeros(batch_size, var.size)}
                elif var.distribution in (Bernoulli, RelaxedBernoulli):
                    results[var_name] = {'probs': torch.zeros(batch_size, 1)}
                else:
                    results[var_name] = {'mean': torch.zeros(batch_size, 1),
                                         'std': torch.zeros(batch_size, 1)}
        
        if return_dict:
            return results
        else:
            # Concatenate probabilities/means into single tensor
            tensors = []
            for var_name in query:
                var = self.probabilistic_model.concept_to_variable[var_name]
                if var.distribution in DISCRETE_DISTRIBUTIONS:
                    tensors.append(results[var_name]['probs'])
                else:
                    tensors.append(results[var_name]['mean'])
            return torch.cat(tensors, dim=-1)

    def _get_pyro_distribution(self, variable, params):
        """Convert distribution.  See :func:`.utils.get_pyro_distribution`."""
        return get_pyro_distribution(variable, params)

    def query(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Query interface — delegates to :meth:`marginal`."""
        return self.marginal(query, evidence, return_dict=False, **kwargs)
