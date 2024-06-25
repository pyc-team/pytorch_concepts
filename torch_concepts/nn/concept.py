import torch
from torch import nn


class BaseConcept(nn.Module):
    """
    BaseConcept is an abstract base class for models that handle concept layers.

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
    """
    def __init__(self, in_features, n_concepts):
        super().__init__()
        self.in_features = in_features
        self.n_concepts = n_concepts

    def forward(self, x):
        raise NotImplementedError

    def intervene(self, x):
        return self.forward(x)

    def __call__(self, x):
        return self.forward(x)


class ConceptLinear(BaseConcept):
    """
    ConceptLinear is a linear model for concept learning.
    Main reference: `"Concept Bottleneck Models" <https://arxiv.org/pdf/2007.04612>`_

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        fc (nn.Linear): Fully connected linear layer for transforming input features to concepts.
    """
    def __init__(self, in_features, n_concepts):
        super().__init__(in_features, n_concepts)
        self.fc = nn.Linear(in_features, n_concepts)

    def forward(self, x):
        return self.fc(x)

    def intervene(self, x, c=None, intervention_idxs=None):
        c_pred = self.fc(x)
        if c is not None and intervention_idxs is not None:
            c_pred[:, intervention_idxs] = c[:, intervention_idxs]
        return c_pred


class ConceptEmbedding(BaseConcept):
    """
    ConceptEmbedding is a model for learning concept embeddings.
    Main reference: `"Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Attributes:
        in_features (int): Number of input features.
        n_concepts (int): Number of concepts to be learned.
        emb_size (int): Size of the embedding vector.
        intervention_idxs (list): Indices of the concepts to be intervened.
        training_intervention_prob (float): Probability of intervention during training.
        active_intervention_values (torch.Tensor): Values used for active interventions.
        inactive_intervention_values (torch.Tensor): Values used for inactive interventions.
        concept_context_generators (nn.ModuleList): List of context generators for each concept.
        concept_prob_predictor (nn.Sequential): Model to predict concept probabilities.
    """
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
    ):
        super().__init__(in_features, n_concepts)
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
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
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, c=None, intervention_idxs=None, train=False):
        c_emb_list, c_pred_list, c_int_list = [], [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            c_pred_list.append(c_pred)
            # Time to check for interventions
            c_pred = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )
            c_int_list.append(c_pred)

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1), torch.cat(c_int_list, axis=1)

    def __call__(self, x, c=None, intervention_idxs=None, train=False):
        c_emb, c_pred, c_int = self.forward(x, c, intervention_idxs, train)
        self.saved_c_pred = c_pred
        self.saved_c_int = c_int
        return c_emb
