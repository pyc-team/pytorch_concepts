"""Ancestral sampling inference for probabilistic graphical models."""

import inspect

import torch
from torch.distributions import RelaxedBernoulli, Bernoulli, RelaxedOneHotCategorical

from .forward import ForwardInference, LazyForwardInference
from ..models.variable import Variable
from ..models.probabilistic_model import ProbabilisticModel
from ...low.base.graph import BaseGraphLearner


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
        >>> from torch_concepts import LatentVariable
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts.nn import AncestralSamplingInference, ParametricCPD, ProbabilisticModel
        >>> from torch_concepts import ConceptVariable
        >>> from torch_concepts.nn import LinearConceptToConcept
        >>>
        >>> # Create a simple PGM: embedding -> A -> B
        >>> embedding_var = LatentVariable('embedding', parents=[], distribution=Delta, size=10)
        >>> var_A = ConceptVariable('A', parents=['embedding'], distribution=Bernoulli, size=1)
        >>> var_B = ConceptVariable('B', parents=['A'], distribution=Bernoulli, size=1)
        >>>
        >>> # Define CPDs
        >>> from torch.nn import Identity, Linear
        >>> cpd_emb = ParametricCPD('embedding', parametrization=Identity())
        >>> cpd_A = ParametricCPD('A', parametrization=Linear(10, 1))
        >>> cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(1, 1))
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
        >>> var_A_relaxed = ConceptVariable('A', parents=['embedding'],
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

    # TODO: currently assumes discrete, to be extended to continuous 
    def ground_truth_to_evidence(self, value: torch.Tensor, cardinality: int) -> torch.Tensor:
        """
        Convert ground truth to raw states for ancestral sampling.
        
        For sampling inference, evidence should be in the same format as samples:
        - Binary: (batch_size, 1) with values 0.0 or 1.0
        - Categorical: (batch_size, cardinality) one-hot encoded
        
        Parameters
        ----------
        value : torch.Tensor
            Ground truth value tensor. Shape: (batch_size,).
        cardinality : int
            Number of classes (1 for binary, >1 for categorical).
            
        Returns
        -------
        torch.Tensor
            State tensor in sample format.
        """
        if cardinality > 1:
            return torch.nn.functional.one_hot(
                value.long(), num_classes=cardinality
            ).float()
        else:
            return value.unsqueeze(-1).float()

    def get_results(self, results: torch.tensor, variable: Variable) -> torch.Tensor:
        """
        Sample from the distribution parameterized by the results.

        This method creates a distribution using the variable's distribution type
        and the computed endogenous/parameters, then draws a sample.

        Args:
            results: Raw output tensor from the CPD (endogenous or parameters).
            variable: The variable being computed (defines distribution type).

        Returns:
            torch.Tensor: Sampled values from the distribution.
        """
        sig = inspect.signature(variable.distribution.__init__)
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

        if variable.distribution in [Bernoulli, RelaxedBernoulli, RelaxedOneHotCategorical]:
            if self.log_probs:
                dist_kwargs['logits'] = results
            else:
                dist_kwargs['probs'] = results

            if variable.distribution in [Bernoulli]:
                return variable.distribution(**dist_kwargs).sample()
            elif variable.distribution in [RelaxedBernoulli, RelaxedOneHotCategorical]:
                return variable.distribution(**dist_kwargs).rsample()

        return variable.distribution(results, **dist_kwargs).rsample()


class LazyAncestralSamplingInference(LazyForwardInference, AncestralSamplingInference):
    """
    Lazy ancestral sampling inference that only computes ancestor variables.
    
    Combines the lazy query strategy (computing only ancestors of queried
    concepts) with ancestral sampling (drawing samples from distributions).
    
    Use this when:
    - You only need samples for a subset of concepts
    - The graph has many independent branches
    - You want to avoid sampling unused variables
    
    Example:
        >>> # Given model: input -> A -> B, input -> C -> D
        >>> inference = LazyAncestralSamplingInference(pgm)
        >>> # Querying only ['B'] computes: input, A, B (not C, D)
        >>> samples = inference.query(['B'], evidence={'input': x})
    """
    pass
