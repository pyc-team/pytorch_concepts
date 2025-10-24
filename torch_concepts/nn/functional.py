import torch

from collections import defaultdict

from torch import Tensor

from torch_concepts.semantic import CMRSemantic
from typing import List, Dict
from torch_concepts.utils import numerical_stability_check
from scipy.stats import chi2
from torch_concepts.nn.minimize_constraint import minimize_constr
from torch.distributions import MultivariateNormal


def _default_concept_names(shape: List[int]) -> Dict[int, List[str]]:
    concept_names = {}
    for dim in range(len(shape)):
        concept_names[dim+1] = [
            f"concept_{dim+1}_{i}" for i in range(shape[dim])
        ]
    return concept_names


def intervene(
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
    indexes: torch.Tensor,
) -> torch.Tensor:
    """
    Intervene on concept embeddings.

    Args:
        c_pred (Tensor): Predicted concepts.
        c_true (Tensor): Ground truth concepts.
        indexes (Tensor): Boolean Tensor indicating which concepts to intervene
            on.

    Returns:
        Tensor: Intervened concepts.
    """
    if c_true is None or indexes is None:
        return c_pred

    if c_pred.shape != c_true.shape:
        raise ValueError(
            "Predicted and true concepts must have the same shape."
        )

    if c_true is not None and indexes is not None:
        if indexes.max() >= c_pred.shape[1]:
            raise ValueError(
                "Intervention indices must be less than the number of concepts."
            )

    return torch.where(indexes, c_true, c_pred)


def concept_embedding_mixture(
    c_emb: torch.Tensor,
    c_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Mixes concept embeddings and concept predictions.
    Main reference: `"Concept Embedding Models: Beyond the
    Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Args:
        c_emb (Tensor): Concept embeddings with shape (batch_size, n_concepts,
            emb_size).
        c_scores (Tensor): Concept scores with shape (batch_size, n_concepts).
        concept_names (List[str]): Concept names.

    Returns:
        Tensor: Mix of concept embeddings and concept scores with shape
            (batch_size, n_concepts, emb_size//2)
    """
    emb_size = c_emb[0].shape[1] // 2
    c_mix = (
        c_scores.unsqueeze(-1) * c_emb[:, :, :emb_size] +
        (1 - c_scores.unsqueeze(-1)) * c_emb[:, :, emb_size:]
    )
    return c_mix


def intervene_on_concept_graph(
    c_adj: torch.Tensor,
    indexes: List[int],
) -> torch.Tensor:
    """
    Intervene on a Tensor adjacency matrix by zeroing out specified
    concepts representing parent nodes.

    Args:
        c_adj: torch.Tensor adjacency matrix.
        indexes: List of indices to zero out.

    Returns:
        Tensor: Intervened Tensor adjacency matrix.
    """
    # Check if the tensor is a square matrix
    if c_adj.shape[0] != c_adj.shape[1]:
        raise ValueError(
            "The Tensor must be a square matrix (it represents an "
            "adjacency matrix)."
        )

    # Zero out specified columns
    c_adj = c_adj.clone()
    c_adj[:, indexes] = 0

    return c_adj


def selection_eval(
    selection_weights: torch.Tensor,
    *predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate predictions as a weighted product based on selection weights.

    Args:
        selection_weights (Tensor): Selection weights with at least two
            dimensions (D1, ..., Dn).
        predictions (Tensor): Arbitrary number of prediction tensors, each with
            the same shape as selection_weights (D1, ..., Dn).

    Returns:
        Tensor: Weighted product sum with shape (D1, ...).
    """
    if len(predictions) == 0:
        raise ValueError("At least one prediction tensor must be provided.")

    product = selection_weights
    for pred in predictions:
        assert pred.shape == product.shape, \
            "Prediction shape mismatch the selection weights."
        product = product * pred

    result = product.sum(dim=-1)

    return result


def linear_equation_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Function to evaluate a set of linear equations with concept predictions.
    In this case we have one equation (concept_weights) for each sample in the
    batch.

    Args:
        concept_weights: Parameters representing the weights of multiple linear
            models with shape (batch_size, memory_size, n_concepts, n_classes).
        c_pred: Concept predictions with shape (batch_size, n_concepts).
        bias: Bias term to add to the linear models (batch_size,
            memory_size, n_classes).

    Returns:
        Tensor: Predictions made by the linear models with shape (batch_size,
            n_classes, memory_size).
    """
    assert concept_weights.shape[-2] == c_pred.shape[-1]
    assert bias is None or bias.shape[-1] == concept_weights.shape[-1]
    y_pred = torch.einsum('bmcy,bc->bym', concept_weights, c_pred)
    if bias is not None:
        # the bias is (b,m,y) while y_pred is (bym) so we invert bias dimension
        y_pred += torch.transpose(bias, -1, -2)
    return y_pred


def linear_equation_expl(
    concept_weights: torch.Tensor,
    bias: torch.Tensor = None,
    concept_names: Dict[int, List[str]] = None,
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extract linear equations from decoded equations embeddings as strings.
    Args:
        concept_weights: Equation embeddings with shape (batch_size,
            memory_size, n_concepts, n_tasks).
        bias: Bias term to add to the linear models (batch_size,
            memory_size, n_tasks).
        concept_names: Concept and task names. If the bias is included, the
            concept names should include the bias name.
    Returns:
        List[Dict[str, Dict[str, str]]]: List of predicted equations as strings.
    """
    if len(concept_weights.shape) != 4:
        raise ValueError(
            "The concept weights must have 4 dimensions (batch_size, "
            "memory_size, n_concepts, n_tasks)."
        )
    if (concept_names is not None
            and concept_weights.shape[-2] != len(concept_names[1])):
        raise ValueError(
            "The concept names must have the same length as the number of "
            "concepts."
        )

    if hasattr(concept_weights, 'concept_names'):
        names = concept_weights.concept_names.copy()
        c_names = names[1]
        t_names = names[2]
    else:
        names = _default_concept_names(concept_weights.shape[1:3])
        if concept_names is None:
            c_names = names[1]
            t_names = names[2]
        else:
            c_names = concept_names[1]
            t_names = concept_names[2]

    # add the bias to the concept_weights and c_names
    if bias is not None:
        concept_weights = torch.cat(
            (concept_weights, bias.unsqueeze(-2)),
            dim=-2,
        )
        c_names = c_names + ["bias"]

    batch_size = concept_weights.size(0)
    memory_size = concept_weights.size(1)
    n_concepts = concept_weights.size(2)
    n_tasks = concept_weights.size(3)
    explanation_list = []
    for s_idx in range(batch_size):
        equations_str = defaultdict(dict)  # batch, task, memory_size
        for t_idx in range(n_tasks):
            for mem_idx in range(memory_size):
                eq = []
                for c_idx in range(n_concepts):
                    weight = concept_weights[s_idx, mem_idx, c_idx, t_idx]
                    name = c_names[c_idx]
                    if torch.round(weight.abs(), decimals=2) > 0.1:
                        eq.append(f"{weight.item():.1f} * {name}")
                eq = " + ".join(eq)
                eq = eq.replace(" + -", " - ")
                equations_str[t_names[t_idx]][f"Equation {mem_idx}"] = eq

        explanation_list.append(dict(equations_str))
    return explanation_list


def logic_rule_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
    memory_idxs: torch.Tensor = None,
    semantic=CMRSemantic()
) -> torch.Tensor:
    """
    Use concept weights to make predictions based on logic rules.

    Args:
        concept_weights: concept weights with shape (batch_size,
            memory_size, n_concepts, n_tasks, n_roles) with n_roles=3.
        c_pred: concept predictions with shape (batch_size, n_concepts).
        memory_idxs: Indices of rules to evaluate with shape (batch_size,
            n_tasks). Default is None (evaluate all).
        semantic: Semantic function to use for rule evaluation.

    Returns:
        torch.Tensor: Rule predictions with shape (batch_size, n_tasks,
            memory_size)
    """

    assert len(concept_weights.shape) == 5, \
        ("Size error, concept weights should be batch_size x memory_size "
         f"x n_concepts x n_tasks x n_roles. Received {concept_weights.shape}")
    memory_size = concept_weights.size(1)
    n_tasks = concept_weights.size(3)

    # to avoid numerical problem
    concept_weights = concept_weights * 0.999

    pos_polarity, neg_polarity, irrelevance = (
        concept_weights[..., 0],
        concept_weights[..., 1],
        concept_weights[..., 2],
    )

    if memory_idxs is None:
        # cast all to (batch_size, memory_size, n_concepts, n_tasks)
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(
            -1,
            memory_size,
            -1,
            n_tasks,
        )
    else:  # cast all to (batch_size, memory_size=1, n_concepts, n_tasks)
        # TODO: memory_idxs never used!
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n_tasks)

    # batch_size, mem_size, n_tasks
    y_per_rule = semantic.disj(
        irrelevance,
        semantic.conj((1 - x), neg_polarity),
        semantic.conj(x, pos_polarity)
    )
    assert (y_per_rule < 1.0).all(), "y_per_rule should be in [0, 1]"

    # performing a conj while iterating over concepts of y_per_rule
    y_per_rule = semantic.conj(
        *[y for y in y_per_rule.split(1, dim=2)]
    ).squeeze(dim=2)

    return y_per_rule.permute(0, 2, 1)


def logic_memory_reconstruction(
    concept_weights: torch.Tensor,
    c_true: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct tasks based on concept reconstructions, ground truth concepts
    and ground truth tasks.

    Args:
        concept_weights: concept reconstructions with shape (batch_size,
            memory_size, n_concepts, n_tasks).
        c_true: concept ground truth with shape (batch_size, n_concepts).
        y_true: task ground truth with shape (batch_size, n_tasks).

    Returns:
        torch.Tensor: Reconstructed tasks with shape (batch_size, n_tasks,
            memory_size).
    """
    pos_polarity, neg_polarity, irrelevance = (
        concept_weights[..., 0],
        concept_weights[..., 1],
        concept_weights[..., 2],
    )

    # batch_size, mem_size, n_tasks, n_concepts
    c_rec_per_classifier = 0.5 * irrelevance + pos_polarity

    reconstruction_mask = torch.where(
        c_true[:, None, :, None] == 1,
        c_rec_per_classifier,
        1 - c_rec_per_classifier,
    )
    c_rec_per_classifier = reconstruction_mask.prod(dim=2).pow(
        y_true[:, None, :]
    )
    return c_rec_per_classifier.permute(0, 2, 1)


def logic_rule_explanations(
    concept_logic_weights: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extracts rules from rule concept weights as strings.

    Args:
        concept_logic_weights: Rule embeddings with shape
            (batch_size, memory_size, n_concepts, n_tasks, 3).
        concept_names: Concept and task names.

    Returns:
        List[Dict[str, Dict[str, str]]]: Rules as strings.
    """
    if len(concept_logic_weights.shape) != 5 or (
        concept_logic_weights.shape[-1] != 3
    ):
        raise ValueError(
            "The concept logic weights must have 5 dimensions "
            "(batch_size, memory_size, n_concepts, n_tasks, 3)."
        )

    if hasattr(concept_logic_weights, 'concept_names'):
        names = concept_logic_weights.concept_names.copy()
        c_names = names[1]
        t_names = names[2]
    else:
        names = _default_concept_names(concept_logic_weights.shape[1:3])
        if concept_names is None:
            c_names = names[1]
            t_names = names[2]
        else:
            c_names = concept_names[1]
            t_names = concept_names[2]

    batch_size = concept_logic_weights.size(0)
    memory_size = concept_logic_weights.size(1)
    n_concepts = concept_logic_weights.size(2)
    n_tasks = concept_logic_weights.size(3)
    # memory_size, n_concepts, n_tasks
    concept_roles = torch.argmax(concept_logic_weights, dim=-1)
    rule_list = []
    for sample_id in range(batch_size):
        rules_str = defaultdict(dict)  # task, memory_size
        for task_id in range(n_tasks):
            for mem_id in range(memory_size):
                rule = []
                for concept_id in range(n_concepts):
                    role = concept_roles[sample_id, mem_id, concept_id, task_id].item()
                    if role == 0:
                        rule.append(c_names[concept_id])
                    elif role == 1:
                        rule.append(f"~ {c_names[concept_id]}")
                    else:
                        continue
                rules_str[t_names[task_id]][f"Rule {mem_id}"] = " & ".join(rule)
        rule_list.append(dict(rules_str))
    return rule_list


def selective_calibration(
    c_confidence: torch.Tensor,
    target_coverage: float,
) -> torch.Tensor:
    """
    Selects concepts based on confidence scores and target coverage.

    Args:
        c_confidence: Concept confidence scores.
        target_coverage: Target coverage.

    Returns:
        Tensor: Thresholds to select confident predictions.
    """
    theta = torch.quantile(
        c_confidence, 1 - target_coverage,
        dim=0,
        keepdim=True,
    )
    return theta


def confidence_selection(
    c_confidence: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Selects concepts with confidence above a selected threshold.

    Args:
        c_confidence: Concept confidence scores.
        theta: Threshold to select confident predictions.

    Returns:
        Tensor: mask selecting confident predictions.
    """
    return torch.where(c_confidence > theta, True, False)


def soft_select(values, temperature, dim=1) -> torch.Tensor:
    """
    Soft selection function, a special activation function for a network
    rescaling the output such that, if they are uniformly distributed, then we
    will select only half of them. A higher temperature will select more
    concepts, a lower temperature will select fewer concepts.

    Args:
        values: Output of the network.
        temperature: Temperature for the softmax function [-inf, +inf].
        dim: dimension to apply the softmax function. Default is 1.

    Returns:
        Tensor: Soft selection scores.
    """

    softmax_scores = torch.log_softmax(values, dim=dim)
    soft_scores = torch.sigmoid(softmax_scores - temperature *
                               softmax_scores.mean(dim=dim, keepdim=True))
    return soft_scores

class ConfIntervalOptimalStrategy:
    """
    A strategy for intervening on concepts using confidence interval bounds.
    Args:
        level (float, optional): The confidence level for the confidence interval.
    """
    # Set intervened concept logits to bounds of 90% confidence interval
    def __init__(self, level=0.9):
        from torchmin import minimize
        self.level = level
    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        """
        Compute the logits for the intervened-on concepts based on the confidence interval bounds.
        This method finds values that lie on the confidence region boundary and maximize the likelihood
        of the intervened concepts.
        Args:
            c_mu (torch.Tensor): The predicted mean values of the concepts. Shape: (batch_size, num_concepts)
            c_cov (torch.Tensor): The predicted covariance matrix of the concepts. Shape: (batch_size, num_concepts, num_concepts)
            c_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts)
            c_mask (torch.Tensor): A mask indicating which concepts are intervened-on. Shape: (batch_size, num_concepts)
        Returns:
            torch.Tensor: The logits for the intervened-on concepts, rest filled with NaN. Shape: (batch_size, num_concepts)
        Step-by-step procedure:
            - The method first separates the intervened-on concepts from the others.
            - It finds a good initial point on the confidence region boundary, that is spanned in the logit space.
                It is defined as a vector with equal magnitude in each dimension, originating from c_mu and oriented
                in the direction of the ground truth. Thus, only the scale factor of this vector needs to be found
                s.t. it lies on the confidence region boundary.
            - It defines the confidence region bounds on the logits, as well as defining some objective and derivatives
              for faster optimization.
            - It performs sample-wise constrained optimization to find the intervention logits by minimizing the concept BCE
              while ensuring they lie within the boundary of the confidence region. The starting point from before is used as
              initialization. Note that this is done sequentially for each sample, and therefore very slow.
              The optimization problem also scales with the number of intervened-on concepts. There are certainly ways to make it much faster.
            - After having found the optimal points at the confidence region bound, it permutes determined concept logits back into the original order.
        """
        # Find values that lie on confidence region ball
        # Approach: Find theta s.t.  Λn(θ)= −2(ℓ(θ)−ℓ(θ^))=χ^2_{1-α,n} and minimize concept loss of intervened concepts.
        # Note, theta^ is = mu, evaluated for the N(mu,Sigma) distribution, while theta is point on the boundary of the confidence region
        # Then, we make theta by arg min Concept BCE(θ) s.t. Λn(θ) <= holds with 1-α = self.level for theta~N(0,Sigma) (not fully correct explanation, but intuition).
        n_intervened = c_mask.sum(1)[0]
        # Separate intervened-on concepts from others
        indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
        perm_cov = c_cov.gather(1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2)))
        perm_cov = perm_cov.gather(
            2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1)
        )
        marginal_interv_cov = perm_cov[:, :n_intervened, :n_intervened]
        marginal_interv_cov = numerical_stability_check(
            marginal_interv_cov.float(), device=marginal_interv_cov.device
        ).cpu()
        target = (c_true * c_mask).gather(1, indices)[:, :n_intervened].float().cpu()
        marginal_c_mu = c_mu.gather(1, indices)[:, :n_intervened].float().cpu()
        interv_direction = (
            ((2 * c_true - 1) * c_mask)
            .gather(1, indices)[:, :n_intervened]
            .float()
            .cpu()
        )  # direction
        quantile_cutoff = chi2.ppf(q=self.level, df=n_intervened.cpu())
        # Finding good init point on confidence region boundary (each dim with equal magnitude)
        dist = MultivariateNormal(torch.zeros(n_intervened), marginal_interv_cov)
        loglikeli_theta_hat = dist.log_prob(torch.zeros(n_intervened))
        def conf_region(scale):
            loglikeli_theta_star = dist.log_prob(scale * interv_direction)
            log_likelihood_ratio = -2 * (loglikeli_theta_star - loglikeli_theta_hat)
            return ((quantile_cutoff - log_likelihood_ratio) ** 2).sum(-1)
        scale = minimize(
            conf_region,
            x0=torch.ones(c_mu.shape[0], 1),
            method="bfgs",
            max_iter=50,
            tol=1e-5,
        ).x
        scale = (
            scale.abs()
        )  # in case negative root was found (note that both give same log-likelihood as its point-symmetric around 0)
        x0 = marginal_c_mu + (interv_direction * scale)
        # Define bounds on logits
        lb_interv = torch.where(
            interv_direction > 0, marginal_c_mu + 1e-4, torch.tensor(float("-inf"))
        )
        ub_interv = torch.where(
            interv_direction < 0, marginal_c_mu - 1e-4, torch.tensor(float("inf"))
        )
        # Define confidence region
        dist_logits = MultivariateNormal(marginal_c_mu, marginal_interv_cov)
        loglikeli_theta_hat = dist_logits.log_prob(marginal_c_mu)
        loglikeli_goal = -quantile_cutoff / 2 + loglikeli_theta_hat
        # Initialize variables
        cov_inverse = torch.linalg.inv(marginal_interv_cov)
        interv_vector = torch.empty_like(marginal_c_mu)
        #### Sample-wise constrained optimization (as there are no batched functions available out-of-the-box). Can surely be optimized
        for i in range(marginal_c_mu.shape[0]):
            # Define variables required for optimization
            dist_logits_uni = MultivariateNormal(
                marginal_c_mu[i], marginal_interv_cov[i]
            )
            loglikeli_goal_uni = loglikeli_goal[i]
            target_uni = target[i]
            inverse = cov_inverse[i]
            marginal = marginal_c_mu[i]
            # Define minimization objective and jacobian
            def loglikeli_bern_uni(marginal_interv_vector):
                return F.binary_cross_entropy_with_logits(
                    input=marginal_interv_vector, target=target_uni, reduction="sum"
                )
            def jac_min_fct(x):
                return torch.sigmoid(x) - target_uni
            # Define confidence region constraint and its jacobian
            def conf_region_uni(marginal_interv_vector):
                loglikeli_theta_star = dist_logits_uni.log_prob(marginal_interv_vector)
                return loglikeli_theta_star - loglikeli_goal_uni
            def jac_constraint(x):
                return -(inverse @ (x - marginal).unsqueeze(-1)).squeeze(-1)
            # Wrapper for scipy "minimize" function
            # Find intervention logits by minimizing the concept BCE s.t. they still lie on the boundary of the confidence region
            minimum = minimize_constr(
                f=loglikeli_bern_uni,
                x0=x0[i],
                jac=jac_min_fct,
                method="SLSQP",
                constr={
                    "fun": conf_region_uni,
                    "lb": 0,
                    "ub": float("inf"),
                    "jac": jac_constraint,
                },
                bounds={"lb": lb_interv[i], "ub": ub_interv[i]},
                max_iter=50,
                tol=1e-4 * n_intervened.cpu(),
            )
            interv_vector[i] = minimum.x
        # Permute intervened concept logits back into original order
        indices_reversed = torch.argsort(indices)
        interv_vector_unordered = torch.full_like(
            c_mu, float("nan"), device=c_mu.device, dtype=torch.float32
        )
        interv_vector_unordered[:, :n_intervened] = interv_vector
        c_intervened_logits = interv_vector_unordered.gather(1, indices_reversed)
        return c_intervened_logits
