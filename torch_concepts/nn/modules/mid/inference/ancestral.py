"""Ancestral sampling inference for probabilistic graphical models."""

import inspect
from typing import Dict, Set

import torch

from .forward import ForwardInference
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
    - Uses reparameterization (.rsample()) when the distribution supports it
    - Falls back to .sample() otherwise
    - Passes logits or probs depending on the log_probs flag
    - Supports custom distribution kwargs (e.g., temperature for Gumbel-Softmax)

    Args:
        probabilistic_model: The probabilistic model to perform inference on.
        graph_learner: Optional graph learner for weighted adjacency structure.
        detach: If True, detach concept predictions before propagation to
            children. Default: False.
        lazy: If True, only compute ancestors of the queried concepts. Default: False.
        log_probs: If True, pass logits to distributions; otherwise pass probs.

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
        ...     factors=[cpd_emb, cpd_A, cpd_B]
        ... )
        >>>
        >>> # Create ancestral sampling inference engine
        >>> inference = AncestralSamplingInference(pgm)
        >>>
        >>> # Perform inference - returns samples, not endogenous
        >>> x = torch.randn(4, 10)  # batch_size=4, embedding_size=10
        >>> results = inference.predict({'input': x})
        >>>
        >>> # Results contain binary samples {0, 1} for Bernoulli variables
        >>> print(results['A'].shape)  # torch.Size([4, 1])
        >>> print(results['A'].unique())  # tensor([0., 1.]) - actual samples
        >>> print(results['B'].shape)  # torch.Size([4, 1])
        >>> print(results['B'].unique())  # tensor([0., 1.]) - actual samples
        >>>
        >>> # Query specific concepts - returns concatenated samples
        >>> samples = inference.query(['B', 'A'], evidence={'input': x})
        >>> print(samples.shape)  # torch.Size([4, 2])
        >>> # samples contains [sample_B, sample_A] for each instance
        >>> print(samples)  # All values are 0 or 1
        >>>
        >>> # Multiple runs produce different samples (stochastic)
        >>> samples1 = inference.query(['A'], evidence={'input': x})
        >>> samples2 = inference.query(['A'], evidence={'input': x})
        >>> print(torch.equal(samples1, samples2))  # Usually False (different samples)
        >>>
        >>> # With relaxed distributions (requires temperature)
        >>> from torch.distributions import RelaxedBernoulli
        >>> var_A_relaxed = ConceptVariable('A', parents=['embedding'],
        ...                               distribution=RelaxedBernoulli, size=1)
        >>> pgm = ProbabilisticModel(
        ...     variables=[embedding_var, var_A_relaxed, var_B],
        ...     factors=[cpd_emb, cpd_A, cpd_B]
        ... )
        >>> inference_relaxed = AncestralSamplingInference(pgm)
        >>> # Now uses reparameterization trick (.rsample())
        >>>
        >>> # Query returns continuous values in [0, 1] for relaxed distributions
        >>> relaxed_samples = inference_relaxed.query(['A'], evidence={'input': x})
        >>> # relaxed_samples will be continuous, not binary
    """
    def __init__(self,
                 probabilistic_model: ProbabilisticModel,
                 graph_learner: BaseGraphLearner = None,
                 detach: bool = False,
                 lazy: bool = False,
                 log_probs: bool = True,
                 p: float = 0.0):
        super().__init__(probabilistic_model, graph_learner, detach=detach, lazy=lazy, p=p)
        self.log_probs = log_probs

        # Cache distribution constructor signatures to avoid calling
        # inspect.signature() on every forward pass.
        self._dist_allowed_params: Dict[type, Set[str]] = {}
        for var in probabilistic_model.variables:
            dist_cls = var.distribution
            if dist_cls not in self._dist_allowed_params:
                sig = inspect.signature(dist_cls.__init__)
                self._dist_allowed_params[dist_cls] = {
                    name for name, p in sig.parameters.items()
                    if name != "self" and p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }

    def activate(self, pred: torch.Tensor, variable: Variable) -> torch.Tensor:
        """
        Sample from the distribution parameterized by the raw CPD output.

        The method introspects the distribution's constructor to decide how
        to pass ``pred`` (as ``logits``, ``probs``, or positional arg) and
        uses ``has_rsample`` to choose between ``.rsample()`` and
        ``.sample()``.

        Distribution kwargs are read from ``variable.dist_kwargs``.

        Args:
            pred: Raw output tensor from the CPD (logits or parameters).
            variable: The variable being computed (defines distribution type
                and per-variable ``dist_kwargs``).

        Returns:
            torch.Tensor: Sampled values from the distribution.
        """
        allowed = self._dist_allowed_params.get(variable.distribution)
        if allowed is None:
            # Fallback for dynamically added distributions
            sig = inspect.signature(variable.distribution.__init__)
            allowed = {
                name for name, p in sig.parameters.items()
                if name != "self" and p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            self._dist_allowed_params[variable.distribution] = allowed

        # retain only allowed dist kwargs
        dist_kwargs = {k: v for k, v in variable.dist_kwargs.items() if k in allowed}
        dropped = set(variable.dist_kwargs) - set(dist_kwargs)
        if dropped:
            import warnings
            warnings.warn(
                f"Variable '{variable.concept}': dist_kwargs {dropped} are not "
                f"accepted by {variable.distribution.__name__} and were ignored.",
                stacklevel=2,
            )

        # Decide how to pass pred based on the distribution's accepted params
        if "logits" in allowed and self.log_probs:
            dist_kwargs["logits"] = pred
            dist = variable.distribution(**dist_kwargs)
        elif "probs" in allowed and not self.log_probs:
            dist_kwargs["probs"] = pred
            dist = variable.distribution(**dist_kwargs)
        else:
            dist = variable.distribution(pred, **dist_kwargs)

        sample = dist.rsample() if dist.has_rsample else dist.sample()
        if sample.dim() == 1:
            sample = sample.unsqueeze(-1)
        return sample

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
                value.squeeze(-1).long(), num_classes=cardinality
            ).float()
        else:
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            return value.float()
