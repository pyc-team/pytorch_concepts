"""Deterministic inference for probabilistic graphical models."""

import torch

from .forward import ForwardInference, LazyForwardInference
from ..models.variable import Variable


class DeterministicInference(ForwardInference):
    """
    Deterministic forward inference for probabilistic graphical models.

    This inference engine performs deterministic (maximum likelihood) inference by
    returning raw logits/outputs from CPDs without sampling. It's useful for
    prediction tasks where you want the most likely values rather than samples
    from the distribution.

    Inherits all functionality from ForwardInference but implements get_results()
    to return raw outputs without stochastic sampling.

    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch_concepts import LatentVariable, ConceptVariable
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts.nn import DeterministicInference, ParametricCPD, ProbabilisticModel, LinearConceptToConcept
        >>>
        >>> # Create a simple PGM: latent -> A -> B
        >>> input_var = LatentVariable('input', parents=[], distribution=Delta, size=10)
        >>> var_A = ConceptVariable('A', parents=['input'], distribution=Bernoulli, size=1)
        >>> var_B = ConceptVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        >>>
        >>> # Define CPDs
        >>> from torch.nn import Identity, Linear
        >>> cpd_emb = ParametricCPD('input', parametrization=Identity())
        >>> cpd_A = ParametricCPD('A', parametrization=Linear(10, 1))
        >>> cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(1, 1))
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
        >>> # Perform inference - returns logits, not samples
        >>> x = torch.randn(4, 10)  # batch_size=4, latent_size=10
        >>> results = inference.predict({'input': x})
        >>>
        >>> # Results contain raw logits for Bernoulli variables
        >>> print(results['A'].shape)  # torch.Size([4, 1]) - logits, not {0,1}
        >>> print(results['B'].shape)  # torch.Size([4, 1]) - logits, not {0,1}
        >>>
        >>> # Query specific concepts - returns concatenated logits
        >>> output = inference.query(['B', 'A'], evidence={'input': x})
        >>> print(output.shape)  # torch.Size([4, 2])
        >>> # output contains [logit_B, logit_A] for each sample
        >>>
        >>> # Convert logits to probabilities if needed
        >>> prob_A = torch.sigmoid(results['A'])
        >>> print(prob_A.shape)  # torch.Size([4, 1])
        >>>
        >>> # Get hard predictions (0 or 1)
        >>> pred_A = (prob_A > 0.5).float()
        >>> print(pred_A)  # Binary predictions
    """
    def get_results(self, results: torch.tensor, variable: Variable) -> torch.Tensor:
        """
        Return raw output without sampling.

        Args:
            results: Raw output tensor from the CPD.
            variable: The variable being computed (unused in deterministic mode).

        Returns:
            torch.Tensor: Raw output tensor (endogenous for probabilistic variables).
        """
        return results

    def ground_truth_to_evidence(self, value: torch.Tensor, cardinality: int) -> torch.Tensor:
        """
        Convert discrete ground truth to logits for propagation. 
        Supports both binary (cardinality=1) and categorical (cardinality>1) variables.
        DOES NOT SUPPORT CONTINUOUS VARIABLES.
        
        Parameters
        ----------
        value : torch.Tensor
            Ground truth tensor. Shape: (batch_size,) or (batch_size, 1).
            - Binary (cardinality=1): values in {0, 1}
            - Categorical (cardinality>1): class indices (converted to one-hot)
        cardinality : int
            Number of features/classes for this variable.
            
        Returns
        -------
        torch.Tensor
            Logits tensor. Shape: (batch_size, cardinality).
        """

        # TODO: add support for continuous variables
        # Allow (batch,) and unsqueeze to (batch, 1)
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        
        if value.dim() != 2 or value.shape[-1] != 1:
            raise ValueError(
                f"Expected shape (batch,) or (batch, 1), got {tuple(value.shape)}."
            )
        
        if cardinality == 1:
            # Binary: values in {0, 1}
            return torch.logit(value.float().clamp(1e-7, 1 - 1e-7))
        else:
            # Categorical: convert class indices to one-hot then to logits
            one_hot = torch.nn.functional.one_hot(
                value.squeeze(-1).long(), num_classes=cardinality
            ).float()
            return torch.logit(one_hot.clamp(1e-7, 1 - 1e-7))


class LazyDeterministicInference(LazyForwardInference, DeterministicInference):
    """
    Lazy deterministic inference that only computes ancestor variables.
    
    Combines the lazy query strategy (computing only ancestors of queried
    concepts) with deterministic inference (returning raw logits).
    
    Use this when:
    - You only need a subset of concepts (e.g., just tasks)
    - The graph has many independent branches
    - You want to avoid computing unused variables
    
    Example:
        >>> # Given model: input -> A -> B, input -> C -> D
        >>> inference = LazyDeterministicInference(pgm)
        >>> # Querying only ['B'] computes: input, A, B (not C, D)
        >>> out = inference.query(['B'], evidence={'input': x})
    """
    pass
