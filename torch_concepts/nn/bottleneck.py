import copy
import numpy as np
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch_concepts.base import AnnotatedTensor
from torch_concepts.nn import Annotate
from torch_concepts.utils import numerical_stability_check
from torch_concepts.nn.functional import intervene, concept_embedding_mixture
from torch_concepts.nn.functional import ConfIntervalOptimalStrategy
from torch.distributions import MultivariateNormal
from typing import List, Dict, Callable, Union, Tuple


def _check_annotations(annotations: Union[List[str], int]):
    assert isinstance(
        annotations, (list, int, np.ndarray)
    ), "annotations must be either a single list of str or a single int"
    if isinstance(annotations, (list, np.ndarray)):
        assert all(
            isinstance(a, str) for a in annotations
        ), "all elements in the annotations list must be of type str"


class BaseConceptBottleneck(ABC, torch.nn.Module):
    """
    BaseConceptLayer is an abstract base class for concept layers.
    The output objects are annotated tensors.
    """

    def __init__(
        self,
        in_features: int,
        annotations: List[Union[List[str], int]],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features

        self.annotations = []
        shape = []
        self.annotated_axes = []
        for dim, annotation in enumerate(annotations):
            if isinstance(annotation, int):
                shape.append(annotation)
            else:
                self.annotations.append(annotation)
                shape.append(len(annotation))
                self.annotated_axes.append(dim + 1)

        self.concept_axis = 1
        self._shape = shape
        self.output_size = np.prod(self.shape())

        self.annotator = Annotate(self.annotations, self.annotated_axes)

    def shape(self):
        return self._shape

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        raise NotImplementedError("predict")

    @abstractmethod
    def intervene(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor = None,
        intervention_idxs: torch.Tensor = None,
        intervention_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        Intervene on concept scores.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (torch.Tensor): Ground truth concepts.
            intervention_idxs (torch.Tensor): Boolean Tensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            torch.Tensor: Intervened concept scores.
        """
        raise NotImplementedError("intervene")

    @abstractmethod
    def transform(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed tensor and dictionary with
                intermediate concepts tensors.
        """
        raise NotImplementedError("transform")

    def annotate(
        self,
        x: torch.Tensor,
    ) -> AnnotatedTensor:
        """
        Annotate tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            AnnotatedTensor: Annotated tensor.
        """
        return self.annotator(x)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Forward pass of a ConceptBottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor
                and dictionary with intermediate concepts tensors.
        """
        x_new, val_dict = self.transform(x, *args, **kwargs)
        return x_new, val_dict


class LinearConceptBottleneck(BaseConceptBottleneck):
    """
    ConceptBottleneck creates a bottleneck of supervised concepts.
    Main reference: `"Concept Bottleneck
    Models" <https://arxiv.org/pdf/2007.04612>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """

    def __init__(
        self,
        in_features: int,
        annotations: Union[List[str], int],
        activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        _check_annotations(annotations)

        if isinstance(annotations, int):
            annotations = [annotations]

        super().__init__(
            in_features=in_features,
            annotations=[annotations],
        )
        self.activation = activation
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
        )

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        c_emb = self.linear(x)
        return self.activation(c_emb)

    def intervene(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor = None,
        intervention_idxs: torch.Tensor = None,
        intervention_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        Intervene on concept scores.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (torch.Tensor): Ground truth concepts.
            intervention_idxs (torch.Tensor): Boolean Tensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            torch.Tensor: Intervened concept scores.
        """
        int_probs = torch.rand(x.shape[0], x.shape[1]) <= intervention_rate
        int_probs = int_probs.to(x.device)
        intervention_idxs = int_probs * intervention_idxs
        return intervene(x, c_true, intervention_idxs)

    def transform(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_pred = c_int = self.predict(x)
        if "c_true" in kwargs:
            c_int = self.intervene(c_pred, *args, **kwargs)
        c_int = self.annotate(c_int)
        c_pred = self.annotate(c_pred)
        return c_int, dict(c_pred=c_pred, c_int=c_int)


class StochasticConceptBottleneck(BaseConceptBottleneck):
    """
    StochasticConceptBottleneck creates a bottleneck of supervised concepts with their covariance matrix.
    Main reference: `"Stochastic Concept Bottleneck
    Models" <https://arxiv.org/pdf/2406.19272>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """

    def __init__(
        self,
        in_features: int,
        annotations: Union[List[str], int],
        activation: Callable = torch.sigmoid,
        level: float = 0.99,
        num_monte_carlo: int = 100,
        *args,
        **kwargs,
    ):
        _check_annotations(annotations)

        if isinstance(annotations, int):
            annotations = [annotations]

        super().__init__(
            in_features=in_features,
            annotations=[annotations],
        )
        self.num_monte_carlo = num_monte_carlo
        self.activation = activation
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
            ),
            torch.nn.Unflatten(-1, self.shape()),
        )
        self.sigma = torch.nn.Linear(
            in_features,
            int(self.output_size * (self.output_size + 1) / 2),
        )
        self.sigma.weight.data *= (
            0.01  # Prevent exploding precision matrix at initialization
        )
        self.interv_strat = ConfIntervalOptimalStrategy(level=level)

    def predict_sigma(self, x):
        c_sigma = self.sigma(x)
        # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
        c_triang_cov = torch.zeros(
            (c_sigma.shape[0], self.output_size, self.output_size),
            device=c_sigma.device,
        )
        rows, cols = torch.tril_indices(
            row=self.output_size, col=self.output_size, offset=0
        )
        diag_idx = rows == cols
        c_triang_cov[:, rows, cols] = c_sigma
        c_triang_cov[:, range(self.output_size), range(self.output_size)] = (
            F.softplus(c_sigma[:, diag_idx]) + 1e-6
        )
        return c_triang_cov

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        c_mu = self.mu(x)
        c_triang_cov = self.predict_sigma(x)
        # Sample from predicted normal distribution
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample(
            [self.num_monte_carlo]
            ).movedim(
            0, 
            -1
        )  # [batch_size,num_concepts,mcmc_size]
        return self.activation(c_mcmc_logit)

    def intervene(
        self,
        c_pred: torch.Tensor,
        c_true: torch.Tensor = None,
        intervention_idxs: torch.Tensor = None,
        c_cov: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate an intervention on an SCBM using the conditional normal distribution.
        First, this function computes the logits of the intervened-on concepts based on the intervention strategy.
        Then, using the predicted concept mean and covariance, it computes the conditional normal distribution, conditioned on
        the intervened-on concept logits. To this end, the order is permuted such that the intervened-on concepts form a block at the start.
        Finally, the method samples from the conditional normal distribution and permutes the results back to the original order.
        Args:
            c_pred (torch.Tensor): The predicted mean values of the concepts. Shape: (batch_size, num_concepts)
            c_cov (torch.Tensor): The predicted covariance matrix of the concepts. Shape: (batch_size, num_concepts, num_concepts)
            c_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts)
            c_mask (torch.Tensor): A mask indicating which concepts are intervened-on. Shape: (batch_size, num_concepts)
        Returns:
            tuple: A tuple containing the intervened-on concept means, covariances, MCMC sampled concept probabilities, and logits.
                    Note that the probabilities are set to 0/1 for the intervened-on concepts according to the ground-truth.
        """
        print("Intervention Strategy for SCBM in beta phase")
        c_mu = torch.logit(c_pred)
        num_intervened = intervention_idxs.sum(1)[0]
        device = intervention_idxs.device
        if num_intervened == 0:
            # No intervention
            interv_mu = c_mu
            interv_cov = c_cov
            # Sample from normal distribution
            dist = MultivariateNormal(interv_mu, covariance_matrix=interv_cov)
            mcmc_logits = dist.rsample([self.num_monte_carlo]).movedim(
                0, -1
            )  # [batch_size,bottleneck_size,mcmc_size]
        else:
            # Compute logits of intervened-on concepts
            c_intervened_logits = self.interv_strat.compute_intervened_logits(
                c_mu, c_cov, c_true, intervention_idxs
            )
            ## Compute conditional normal distribution sample-wise
            # Permute covariance s.t. intervened-on concepts are a block at start
            indices = torch.argsort(
                intervention_idxs, dim=1, descending=True, stable=True
            )
            perm_cov = c_cov.gather(
                1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2))
            )
            perm_cov = perm_cov.gather(
                2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1)
            )
            perm_mu = c_mu.gather(1, indices)
            perm_c_intervened_logits = c_intervened_logits.gather(1, indices)
            # Compute mu and covariance conditioned on intervened-on concepts
            # Intermediate steps
            perm_intermediate_cov = torch.matmul(
                perm_cov[:, num_intervened:, :num_intervened],
                torch.inverse(perm_cov[:, :num_intervened, :num_intervened]),
            )
            perm_intermediate_mu = (
                perm_c_intervened_logits[:, :num_intervened]
                - perm_mu[:, :num_intervened]
            )
            # Mu and Cov
            perm_interv_mu = perm_mu[:, num_intervened:] + torch.matmul(
                perm_intermediate_cov, perm_intermediate_mu.unsqueeze(-1)
            ).squeeze(-1)
            perm_interv_cov = perm_cov[
                :, num_intervened:, num_intervened:
            ] - torch.matmul(
                perm_intermediate_cov, perm_cov[:, :num_intervened, num_intervened:]
            )
            # Adjust for floating point errors in the covariance computation to keep it symmetric
            perm_interv_cov = numerical_stability_check(
                perm_interv_cov, device=device
            )  # Uncomment if Normal throws an error. Takes some time so maybe code it more smartly
            # Sample from conditional normal
            perm_dist = MultivariateNormal(
                perm_interv_mu, covariance_matrix=perm_interv_cov
            )
            perm_mcmc_logits = (
                perm_dist.rsample([self.num_monte_carlo])
                .movedim(0, -1)
                .to(torch.float32)
            )  # [bottleneck_size-num_intervened,mcmc_size]
            # Concat logits of intervened-on concepts
            perm_mcmc_logits = torch.cat(
                (
                    perm_c_intervened_logits[:, :num_intervened]
                    .unsqueeze(-1)
                    .repeat(1, 1, self.num_monte_carlo),
                    perm_mcmc_logits,
                ),
                dim=1,
            )
            # Permute back into original form and store
            indices_reversed = torch.argsort(indices)
            mcmc_logits = perm_mcmc_logits.gather(
                1,
                indices_reversed.unsqueeze(2).expand(-1, -1, perm_mcmc_logits.size(2)),
            )
            # Return conditional mu&cov
            assert (
                torch.argsort(indices[:, num_intervened:])
                == torch.arange(len(perm_interv_mu[0][:]), device=device)
            ).all(), "Non-intervened concepts were permuted, a permutation of interv_mu is needed"
            interv_mu = perm_interv_mu
            interv_cov = perm_interv_cov
        assert (
            (mcmc_logits.isnan()).any()
            == (interv_mu.isnan()).any()
            == (interv_cov.isnan()).any()
            == False
        ), "NaN values in intervened-on concepts"
        # Compute probabilities and set intervened-on probs to 0/1
        mcmc_probs = self.act_c(mcmc_logits)
        # Set intervened-on hard concepts to 0/1
        mcmc_probs = (c_true * intervention_idxs).unsqueeze(2).repeat(
            1, 1, self.num_monte_carlo
        ) + mcmc_probs * (1 - intervention_idxs).unsqueeze(2).repeat(
            1, 1, self.num_monte_carlo
        )
        return mcmc_probs

    def transform(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_pred = c_int = self.predict(x)
        if "c_true" in kwargs:
            c_int = self.intervene(c_pred, *args, **kwargs)
        c_int = self.annotate(c_int)
        c_pred = self.annotate(c_pred)
        return c_int, dict(c_pred=c_pred, c_int=c_int)


class LinearConceptResidualBottleneck(LinearConceptBottleneck):
    """
    ConceptResidualBottleneck is a layer where a first set of neurons is aligned
    with supervised concepts and a second set of neurons is free to encode
    residual information.
    Main reference: `"Promises and Pitfalls of Black-Box Concept Learning
    Models" <https://arxiv.org/abs/2106.13314>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """

    def __init__(
        self,
        in_features: int,
        annotations: Union[List[str], int],
        residual_size: int,
        activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            annotations=annotations,
            activation=activation,
            *args,
            **kwargs,
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(in_features, residual_size), torch.nn.LeakyReLU()
        )
        self.annotations_extended = list(copy.deepcopy(self.annotations))
        self.annotations_extended[0] = list(self.annotations_extended[0])
        self.annotations_extended[0].extend(
            [f"residual_{i}" for i in range(residual_size)]
        )
        self.annotator_extended = Annotate(
            self.annotations_extended,
            self.annotated_axes,
        )

    def transform(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_pred = c_int = self.predict(x)
        emb = self.residual(x)
        if "c_true" in kwargs:
            c_int = self.intervene(c_pred, *args, **kwargs)
        c_int = self.annotate(c_int)
        c_pred = self.annotate(c_pred)
        c_new = torch.hstack((c_pred, emb))
        c_new = self.annotator_extended(c_new)
        return c_new, dict(c_pred=c_pred, c_int=c_int)


class ConceptEmbeddingBottleneck(BaseConceptBottleneck):
    """
    ConceptEmbeddingBottleneck creates supervised concept embeddings.
    Main reference: `"Concept Embedding Models: Beyond the
    Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """

    def __init__(
        self,
        in_features: int,
        annotations: Union[List[str], int],
        embedding_size: int,
        activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        _check_annotations(annotations)
        annotations = [annotations, embedding_size]
        n_concepts = (
            len(annotations[0])
            if isinstance(annotations[0], (list, np.ndarray))
            else annotations[0]
        )

        super().__init__(
            in_features=in_features,
            annotations=annotations,
        )

        self._shape = [n_concepts, embedding_size * 2]
        self.output_size = np.prod(self.shape())

        self.activation = activation
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
            torch.nn.LeakyReLU(),
        )
        self.concept_score_bottleneck = torch.nn.Sequential(
            torch.nn.Linear(self.shape()[-1], 1),
            torch.nn.Flatten(),
        )

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        c_emb = self.linear(x)
        return self.activation(self.concept_score_bottleneck(c_emb))

    def intervene(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor = None,
        intervention_idxs: torch.Tensor = None,
        intervention_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        Intervene on concept scores.

        Args:
            x (torch.Tensor): Input tensor.
            c_true (torch.Tensor): Ground truth concepts.
            intervention_idxs (torch.Tensor): Boolean Tensor indicating
                which concepts to intervene on.
            intervention_rate (float): Rate at which perform interventions.

        Returns:
            torch.Tensor: Intervened concept scores.
        """
        int_probs = torch.rand(x.shape[0], x.shape[1]) <= intervention_rate
        int_probs = int_probs.to(x.device)
        intervention_idxs = int_probs * intervention_idxs
        return intervene(x, c_true, intervention_idxs)

    def transform(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[AnnotatedTensor, Dict]:
        """
        Transform input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[AnnotatedTensor, Dict]: Transformed AnnotatedTensor and
                dictionary with intermediate concepts tensors.
        """
        c_emb = self.linear(x)
        c_pred = c_int = self.activation(self.concept_score_bottleneck(c_emb))
        if "c_true" in kwargs:
            c_int = self.intervene(c_pred, *args, **kwargs)
        c_mix = concept_embedding_mixture(c_emb, c_int)
        c_mix = self.annotate(c_mix)
        c_int = self.annotate(c_int)
        c_pred = self.annotate(c_pred)
        return c_mix, dict(c_pred=c_pred, c_int=c_int)
