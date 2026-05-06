"""Deterministic inference for probabilistic graphical models."""

import warnings

import torch

from .....distributions import Delta
from .forward import ForwardInference
from ..models.variable import Variable

from ..models.variable import _DEFAULT_ACTIVATIONS 

class DeterministicInference(ForwardInference):
    """
    Deterministic forward inference for probabilistic graphical models.

    This inference engine propagates raw CPD outputs forward without sampling or
    using distribution-specific semantics. Because of that, it treats every
    variable as deterministic: if the provided probabilistic model contains
    variables whose distribution is not :class:`Delta`, the variables are
    converted to :class:`Delta` with identity activations during initialization.

    Example:
        >>> import torch
        >>> from torch_concepts import LatentVariable, ConceptVariable
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts.nn import DeterministicInference, ParametricCPD, ProbabilisticModel, LinearConceptToConcept
        >>>
        >>> # Create a simple PGM: latent -> A -> B
        >>> input_var = LatentVariable('input', distribution=Delta, size=10)
        >>> var_A = ConceptVariable('A', distribution=Delta, size=1)
        >>> var_B = ConceptVariable('B', distribution=Delta, size=1)
        >>>
        >>> # Define CPDs
        >>> from torch.nn import Identity, Linear
        >>> cpd_emb = ParametricCPD('input', parametrization=Identity())
        >>> cpd_A = ParametricCPD('A', parametrization=Linear(10, 1), parents=['input'])
        >>> cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(1, 1), parents=['A'])
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
        >>> output = inference.query(['B', 'A'], evidence={'input': x})
        >>> print(output.probs.shape)  # torch.Size([4, 2])
        >>> # output.probs contains [logit_B, logit_A] for each sample
    """
    def __init__(
        self,
        probabilistic_model,
        graph_learner=None,
        detach: bool = False,
        lazy: bool = False,
        log_probs: bool = False,
        p: float = 0.0,
        *args,
        **kwargs,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.log_probs = log_probs
        self._coerce_variables_to_delta(probabilistic_model)
        super().__init__(
            probabilistic_model,
            graph_learner,
            detach,
            lazy,
            p,
            *args,
            **kwargs,
        )

    def _coerce_variables_to_delta(self, probabilistic_model) -> None:
        var_to_change = [
            var for var in probabilistic_model.variables
            if var.distribution is not Delta
        ]

        if var_to_change:
            non_delta_summary = ", ".join(
                f"{var.concept} ({getattr(var.distribution, '__name__', var.distribution)})"
                for var in var_to_change
            )
            with warnings.catch_warnings():
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    "DeterministicInference assumes all variables are Delta() variables. " \
                    "All Variables will be changed to Delta(), with activations set to identity. "
                    f"Non-Delta variables: {non_delta_summary}.",
                    UserWarning,
                    stacklevel=3,
                )

        for var in probabilistic_model.variables:
            if var in var_to_change:
                if self.log_probs:
                    var.activation = lambda x: x
                else:
                    var.activation = _DEFAULT_ACTIVATIONS.get(var.distribution, lambda x: x)
                var.distribution = Delta
                var.dist_kwargs = {}

    def activate(self, pred: torch.Tensor, variable: Variable) -> torch.Tensor:
        """
        Apply activation function to raw CPD outputs.

        Args:
            pred: Prediction tensor (logits).
            variable: The Variable whose prediction is being activated.

        Returns:
            torch.Tensor: Activated prediction tensor.
        """
        return variable.activation(pred)

    def ground_truth_to_evidence(self, value: torch.Tensor, size: int, type: str) -> torch.Tensor:
        """
        Convert ground truth to tensors used for propagation.
        Supports binary (size=1), categorical (size>1), and dense continuous variables.

        Parameters
        ----------
        value : torch.Tensor
            Ground truth tensor. Shape: (batch_size,) or (batch_size, 1).
            - Binary (size=1): binary values with shape (batch_size, )
            - Categorical (size>1): class indices with shape (batch_size,) or one-hot vectors with shape (batch_size, size)
            - Continuous: values with shape (batch_size, size)
        size : int
            Number of features/classes for this variable.
        type : str
            Type of the variable ('binary', 'categorical', 'continuous' or 'delta').

        Returns
        -------
        torch.Tensor
            Value tensor. Shape: (batch_size, size).
        """

        # Allow (batch,) and unsqueeze to (batch, 1)
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        if value.dim() != 2:
            raise ValueError(
                f"Expected shape (batch,), (batch, 1), or "
                f"(batch, {size}), got {tuple(value.shape)}."
            )

        width = value.shape[-1]

        if type == 'binary':
            if width != 1:
                raise ValueError(
                    f"Expected shape (batch,) or (batch, 1) for binary variable, "
                    f"got {tuple(value.shape)}."
                )
            if not torch.all((value == 0) | (value == 1)):
                unique_vals = value.unique()
                warnings.warn(
                    f"Binary ground truth contains values outside {{0, 1}}: "
                    f"{unique_vals.tolist()}. Values will be used as-is.",
                    stacklevel=2,
                )
            probs = value.float()
            return torch.logit(probs, eps=1e-7) if self.log_probs else probs

        elif type == 'categorical':
            if width == size:
                # Already one-hot encoded
                one_hot = value.float()
            elif width != 1:
                raise ValueError(
                    f"Expected shape (batch,), (batch, 1), or "
                    f"(batch, {size}) for categorical variable, got {tuple(value.shape)}."
                )
            else:
                # Class indices → one-hot
                one_hot = torch.nn.functional.one_hot(
                    value.squeeze(-1).long(), num_classes=size
                ).float()
            return torch.logit(one_hot, eps=1e-7) if self.log_probs else one_hot
            
        else:  # 'continuous' or 'delta'
            if width == size:
                return value.float()
            raise ValueError(
                f"Expected shape (batch, {size}) for {type} variable, "
                f"got {tuple(value.shape)}."
            )
