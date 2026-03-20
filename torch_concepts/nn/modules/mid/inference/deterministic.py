"""Deterministic inference for probabilistic graphical models."""

import torch

from .forward import ForwardInference
from ..models.variable import Variable


class DeterministicInference(ForwardInference):
    """
    Deterministic forward inference for probabilistic graphical models.

    This inference engine performs deterministic (maximum likelihood) inference by
    returning raw logits/outputs from CPDs without sampling. It's useful for
    prediction tasks where you want the most likely values rather than samples
    from the distribution.

    Inherits all functionality from ForwardInference but implements activate()
    to map logits to probabilities without stochastic sampling.

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
        ...     factors=[cpd_emb, cpd_A, cpd_B]
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
    def activate(self, pred: torch.Tensor, variable: Variable) -> torch.Tensor:
        """
        Map logits to probabilities using the variable's activation.

        The activation function is stored on the :class:`Variable` instance
        (defaulting to sigmoid for Bernoulli, softmax for Categorical,
        identity for Delta, etc.).  Custom activations can be provided when
        constructing the variable.

        Args:
            pred: Prediction tensor (logits).
            variable: The Variable whose prediction is being propagated.

        Returns:
            torch.Tensor: Probability tensor.
        """
        return variable.activation(pred)

    def ground_truth_to_evidence(self, value: torch.Tensor, cardinality: int) -> torch.Tensor:
        """
        Convert discrete ground truth to activated probabilities for propagation.

        Since the inference engine now owns the activation, propagation values
        must be in the same representation as ``activate`` produces:
        probabilities for Bernoulli (0.0/1.0) and one-hot for Categorical.

        Supports both binary (cardinality=1) and categorical (cardinality>1)
        variables.  DOES NOT SUPPORT CONTINUOUS VARIABLES.

        Parameters
        ----------
        value : torch.Tensor
            Ground truth tensor. Shape: (batch_size,) or (batch_size, 1).
            - Binary (cardinality=1): values in {0, 1}
            - Categorical (cardinality>1): class indices
        cardinality : int
            Number of features/classes for this variable.

        Returns
        -------
        torch.Tensor
            Probability / one-hot tensor. Shape: (batch_size, cardinality).
        """

        # TODO: this step should invert whatever is done in the activate() implementation, 
        # which is currently hardcoded for Bernoulli/Categorical. 
        # To support custom distributions, we may need a more flexible way to convert GT to evidence format.
        # TODO: add support for continuous variables

        # Allow (batch,) and unsqueeze to (batch, 1)
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        if value.dim() != 2 or value.shape[-1] != 1:
            raise ValueError(
                f"Expected shape (batch,) or (batch, 1), got {tuple(value.shape)}."
            )

        if cardinality == 1:
            # Binary: return 0.0 / 1.0 probabilities
            return value.float()
        else:
            # Categorical: return one-hot probabilities
            return torch.nn.functional.one_hot(
                value.squeeze(-1).long(), num_classes=cardinality
            ).float()
