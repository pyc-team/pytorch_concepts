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
    Main reference: `"Concept Bottleneck Models" <https://arxiv.org/abs/1901.11468>`_

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
