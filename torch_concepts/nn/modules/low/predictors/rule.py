import torch

from ..base.layer import BasePredictor
from ....functional import grouped_concept_exogenous_mixture, replace_expand_cols
from typing import List


class RuleMemory(torch.nn.Module):
    """Learnable rule memory decoded into categorical role probabilities.

    During training the decoded roles remain soft probabilities. During eval,
    ``hard_at_eval=True`` converts each 3-way role categorical to its argmax
    one-hot mode.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024.
        https://arxiv.org/abs/2407.15527
    """
    def __init__(self, n_tasks, n_rules, n_concepts, latent_size=100, hidden_layers=1, hard_at_eval=False):
        super().__init__()
        self.hard_at_eval = hard_at_eval
        self.shape = (n_tasks, n_rules, n_concepts, 3)
        width = n_rules * n_concepts * 3
        self.memory = torch.nn.Embedding(n_tasks, latent_size)
        layers = [torch.nn.Linear(latent_size, width)]
        for _ in range(hidden_layers):
            layers += [torch.nn.LeakyReLU(), torch.nn.Linear(width, width)]
        layers += [torch.nn.Unflatten(-1, (n_rules, n_concepts, 3))]
        self.decoder = torch.nn.Sequential(*layers)
    def forward(self):
        pred = torch.softmax(self.decoder(self.memory.weight), dim=-1)
        if (not self.training) and self.hard_at_eval:
            idx = pred.argmax(dim=-1)
            pred = torch.nn.functional.one_hot(idx, num_classes=pred.shape[-1]).to(pred.dtype)
        assert torch.all((pred >= 0) & (pred <= 1)), "Decoded memory should be in [0, 1]"
        return pred


class RuleTaskPredictor(BasePredictor):
    """Compute the ordinary CMR task probability from concepts, selector and roles.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024.
        https://arxiv.org/abs/2407.15527
    """
    def forward(self, concepts, selector, roles):
        c = concepts.detach().unsqueeze(1).unsqueeze(1)
        per_rule = (c * roles[..., 0] + (1.0 - c) * roles[..., 1] + roles[..., 2]).prod(dim=-1)
        pred = (per_rule * selector).sum(dim=-1)
        eps = 0.0001
        pred = eps + (1 - 2 * eps) * pred  # numerical stability
        return pred


class RuleReconstructionPredictor(BasePredictor):
    """Compute the reconstruction-aware CMR task probability.

    For each rule, this predictor multiplies its task satisfaction probability
    by its reconstruction probability raised to ``rec_weight``. A weight of
    zero disables reconstruction within this branch; larger non-negative
    weights make reconstruction agreement more influential.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024.
        https://arxiv.org/abs/2407.15527
    """
    def __init__(self, rec_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        if rec_weight < 0:
            raise ValueError("rec_weight must be non-negative.")
        self.rec_weight = rec_weight
    def forward(self, concepts, selector, roles):
        c = concepts.detach().unsqueeze(1).unsqueeze(1)
        task_per_rule = (c * roles[..., 0] + (1.0 - c) * roles[..., 1] + roles[..., 2]).prod(dim=-1)
        reconstruction_per_rule = (c * roles[..., 0] + (1.0 - c) * roles[..., 1] + 0.5 * roles[..., 2]).prod(dim=-1)
        reconstruction_per_rule = torch.pow(reconstruction_per_rule + 1e-6, self.rec_weight)
        pred = (task_per_rule * reconstruction_per_rule * selector).sum(dim=-1)
        eps = 0.0001
        pred = eps + (1 - 2 * eps) * pred  # numerical stability
        return pred
