import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod

from .task import BaseClassifier


class Sequential(nn.Sequential):
    """
    Sequential is a class that extends the PyTorch Sequential class to handle concept layers.
    """
    def forward(self, x, *extra_params):
        for layer in self:
            if isinstance(layer, BaseConcept) or isinstance(layer, BaseClassifier):
                if extra_params:
                    x = layer(x, *extra_params)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x


class BaseConcept(ABC, nn.Module):
    """
    BaseConcept is an abstract base class for models that handle concept layers.

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        c_activation (function): Activation function for the concept layer.
    """
    def __init__(self, in_features, n_concepts, c_activation=F.sigmoid):
        super().__init__()
        self.in_features = in_features
        self.n_concepts = n_concepts
        self.c_activation = c_activation

    @abstractmethod
    def forward(self, x, **kwargs):
        pass


class ConceptLinear(BaseConcept):
    """
    ConceptLinear is a linear model for concept learning.
    Main reference: `"Concept Bottleneck Models" <https://arxiv.org/pdf/2007.04612>`_

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        c_activation (function): Activation function for the concept layer.
        emb_size (int): Size of the embedding vector.
    """
    def __init__(self, in_features, n_concepts, c_activation=F.sigmoid, emb_size=10):
        super().__init__(in_features, n_concepts, c_activation)
        self.emb_size = emb_size
        self.concept_embedder = nn.Linear(in_features, emb_size)
        self.concept_predictor = nn.Linear(emb_size, n_concepts)

    def forward(self, x, c=None, intervention_idxs=None, intervention_rate=0.0):
        # predict concepts
        emb = F.leaky_relu(self.concept_embedder(x))
        c_pred = self.c_activation(self.concept_predictor(emb))

        # intervene
        c_int = c_pred.clone()
        if c is not None and intervention_idxs is not None:
            if intervention_idxs.max() >= self.n_concepts:
                raise ValueError("Intervention indices must be less than the number of concepts.")

            c_int[:, intervention_idxs] = c[:, intervention_idxs]

        return {'c_pred': c_pred, 'c_int': c_int, 'emb_pre': emb}


class ConceptEmbeddingResidual(ConceptLinear):
    """
    ConceptEmbeddingResidual is a layer where a first set of neurons is aligned with supervised concepts and
    a second set of neurons is free to encode residual information.
    Main reference: `"Promises and Pitfalls of Black-Box Concept Learning Models" <https://arxiv.org/abs/2106.13314>`_

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        c_activation (function): Activation function for the concept layer.
        emb_size (int): Size of the embedding vector.
        n_residuals (int): Number of residual neurons to encode additional information.
    """
    def __init__(self, in_features, n_concepts, c_activation=F.sigmoid, emb_size=10, n_residuals=10):
        super().__init__(in_features, n_concepts, c_activation, emb_size)
        self.fc_residual = nn.Linear(emb_size, n_residuals)

    def forward(self, x, c=None, intervention_idxs=None, intervention_rate=0.0):
        # predict concepts
        emb = F.leaky_relu(self.concept_embedder(x))
        c_pred = self.c_activation(self.concept_predictor(emb))
        residuals = F.leaky_relu(self.fc_residual(emb))

        # intervene
        c_int = c_pred.clone()
        if c is not None and intervention_idxs is not None:
            if intervention_idxs.max() >= self.n_concepts:
                raise ValueError("Intervention indices must be less than the number of concepts.")

            c_int[:, intervention_idxs] = c[:, intervention_idxs]

        return {'c_pred': c_pred, 'c_int': c_int, 'emb_pre': emb, 'residuals': residuals}


class ConceptEmbedding(BaseConcept):
    """
    ConceptEmbedding is a model for learning concept embeddings.
    Main reference: `"Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        c_activation (function): Activation function for the concept layer.
        emb_size (int): Size of the embedding vector.
        active_intervention_values (torch.Tensor): Values used for active interventions.
        inactive_intervention_values (torch.Tensor): Values used for inactive interventions.
    """
    def __init__(
            self,
            in_features,
            n_concepts,
            c_activation=F.sigmoid,
            emb_size=10,
            active_intervention_values=None,
            inactive_intervention_values=None,
    ):
        super().__init__(in_features, n_concepts, c_activation)
        self.emb_size = emb_size
        self.ones = torch.ones(self.n_concepts)

        self.embedder = nn.Linear(in_features, emb_size)
        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(emb_size, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
            intervention_rate=0.0,
    ):
        if train and (intervention_rate != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * intervention_rate)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, c=None, intervention_idxs=None, intervention_rate=0.0, train=False):
        if c is not None and intervention_idxs is not None:
            if intervention_idxs.max() >= self.n_concepts:
                raise ValueError("Intervention indices must be less than the number of concepts.")

        c_emb_list, c_pred_list, c_int_list = [], [], []
        emb = F.leaky_relu(self.embedder(x))
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(emb)
            c_pred = self.c_activation(self.concept_prob_predictor(context))
            c_pred_list.append(c_pred)

            # Time to check for interventions
            c_pred = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=intervention_idxs,
                c_true=c,
                train=train,
                intervention_rate=intervention_rate,
            )
            c_int_list.append(c_pred)

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))

        c_emb = torch.cat(c_emb_list, axis=1)
        c_pred = torch.cat(c_pred_list, axis=1)
        c_int = torch.cat(c_int_list, axis=1)
        return {'c_emb': c_emb, 'c_pred': c_pred, 'c_int': c_int, 'emb_pre': emb}
