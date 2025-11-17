"""
Functional utilities for concept-based neural networks.

This module provides functional operations for concept manipulation, intervention,
embedding mixture, and evaluation metrics for concept-based models.
"""
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from typing import Callable, List, Union, Dict
from torch.nn import Linear

from ..semantic import CMRSemantic


def _default_concept_names(shape: List[int]) -> Dict[int, List[str]]:
    """
    Generate default concept names for a given shape.

    Args:
        shape: List of integers representing the shape of concept dimensions.

    Returns:
        Dict mapping dimension index to list of concept names.
    """
    concept_names = {}
    for dim in range(len(shape)):
        concept_names[dim+1] = [
            f"concept_{dim+1}_{i}" for i in range(shape[dim])
        ]
    return concept_names


def grouped_concept_embedding_mixture(c_emb: torch.Tensor,
                                      c_scores: torch.Tensor,
                                      groups: list[int]) -> torch.Tensor:
    """
    Vectorized version of grouped concept embedding mixture.

    Extends concept_embedding_mixture to handle grouped concepts where
    some groups may contain multiple related concepts. Adapted from "Concept Embedding Models:
    Beyond the Accuracy-Explainability Trade-Off" (Espinosa Zarlenga et al., 2022).

    Args:
        c_emb: Concept embeddings of shape (B, n_concepts, emb_size * sum(groups)).
        c_scores: Concept scores of shape (B, sum(groups)).
        groups: List of group sizes (e.g., [3, 4] for two groups).

    Returns:
        Tensor: Mixed embeddings of shape (B, n_concepts, emb_size * len(groups)).

    Raises:
        AssertionError: If group sizes don't sum to n_concepts.
        AssertionError: If embedding dimension is not even.

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    B, C, D = c_emb.shape
    assert sum(groups) == C, "group_sizes must sum to n_concepts"
    assert D % 2 == 0, "embedding dim must be even (two halves)"
    E = D // 2

    # Split concept embeddings into two halves
    emb_a, emb_b = c_emb[..., :E], c_emb[..., E:]         # [B, C, E], [B, C, E]
    s = c_scores.unsqueeze(-1)                            # [B, C, 1]

    # Build group ids per concept: [0,0,...,0, 1,1,...,1, ...]
    device = c_emb.device
    G = len(groups)
    gs = torch.as_tensor(groups, device=device)
    group_id = torch.repeat_interleave(torch.arange(G, device=device), gs)  # [C]

    # For singleton groups, do the two-half mixture; otherwise use emb_a weighted by the score
    is_singleton_concept = (gs == 1)[group_id].view(1, C, 1)               # [1, C, 1], bool
    eff = torch.where(is_singleton_concept, s * emb_a + (1 - s) * emb_b,   # singleton: two-half mix
                      s * emb_a)                                           # multi: weight base embedding

    # Sum weighted embeddings within each group (no loops)
    out = torch.zeros(B, G, E, device=device, dtype=eff.dtype)
    index = group_id.view(1, C, 1).expand(B, C, E)                         # [B, C, E]
    out = out.scatter_add(1, index, eff)                                   # [B, G, E]
    return out


def selection_eval(
    selection_weights: torch.Tensor,
    *predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate concept selection by computing weighted predictions.

    Args:
        selection_weights: Weights for selecting between predictions.
        *predictions: Variable number of prediction tensors to combine.

    Returns:
        Tensor: Weighted combination of predictions.
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

def completeness_score(
    y_true,
    y_pred_blackbox,
    y_pred_whitebox,
    scorer=roc_auc_score,
    average='macro',
):
    """
    Calculate the completeness score for the given predictions and true labels.
    Main reference: `"On Completeness-aware Concept-Based Explanations in
    Deep Neural Networks" <https://arxiv.org/abs/1910.07969>`_

    Parameters:
        y_true (torch.Tensor): True labels.
        y_pred_blackbox (torch.Tensor): Predictions from the blackbox model.
        y_pred_whitebox (torch.Tensor): Predictions from the whitebox model.
        scorer (function): Scoring function to evaluate predictions. Default is
            roc_auc_score.
        average (str): Type of averaging to use. Default is 'macro'.

    Returns:
        float: Completeness score.
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_blackbox_np = y_pred_blackbox.cpu().detach().numpy()
    y_pred_whitebox_np = y_pred_whitebox.cpu().detach().numpy()

    # Compute accuracy or other score using scorer
    blackbox_score = scorer(y_true_np, y_pred_blackbox_np, average=average)
    whitebox_score = scorer(y_true_np, y_pred_whitebox_np, average=average)

    return (whitebox_score) / (blackbox_score + 1e-10)


def intervention_score(
    y_predictor: torch.nn.Module,
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
    y_true: torch.Tensor,
    intervention_groups: List[List[int]],
    activation: Callable = torch.sigmoid,
    scorer: Callable = roc_auc_score,
    average: str = 'macro',
    auc: bool = True,
) -> Union[float, List[float]]:
    """
    Compute the effect of concept interventions on downstream task predictions.

    Given  set of intervention groups, the intervention score measures the
    effectiveness of each intervention group on the model's task predictions.

    Main reference: `"Concept Bottleneck
    Models" <https://arxiv.org/abs/2007.04612>`_

    Parameters:
        y_predictor (torch.nn.Module): Model that predicts downstream task
            abels.
        c_pred (torch.Tensor): Predicted concept values.
        c_true (torch.Tensor): Ground truth concept values.
        y_true (torch.Tensor): Ground truth task labels.
        intervention_groups (List[List[int]]): List of intervention groups.
        activation (Callable): Activation function to apply to the model's
            predictions. Default is torch.sigmoid.
        scorer (Callable): Scoring function to evaluate predictions. Default is
            roc_auc_score.
        average (str): Type of averaging to use. Default is 'macro'.
        auc (bool): Whether to return the average score across all intervention
            groups. Default is True.

    Returns:
        Union[float, List[float]]: The intervention effectiveness for each
            intervention group or the average score across all groups.
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().detach().numpy()

    # Re-compute the model's predictions for each intervention group
    intervention_effectiveness = []
    for group in intervention_groups:
        # Intervene on the concept values
        c_pred_group = c_pred.clone()
        c_pred_group[:, group] = c_true[:, group]

        # Compute the new model's predictions
        y_pred_group = activation(y_predictor(c_pred_group))

        # Compute the new model's task performance
        intervention_effectiveness.append(scorer(
            y_true_np,
            y_pred_group.cpu().detach().numpy(),
            average=average,
        ))

    # Compute the area under the curve of the intervention curve
    if auc:
        intervention_effectiveness = (
            sum(intervention_effectiveness) / len(intervention_groups)
        )
    return intervention_effectiveness


def cace_score(y_pred_c0, y_pred_c1):
    """
    Compute the Average Causal Effect (ACE) also known as the Causal Concept
    Effect (CaCE) score.

    The ACE/CaCE score measures the causal effect of a concept on the
    predictions of a model. It is computed as the absolute difference between
    the expected predictions when the concept is inactive (c0) and active (c1).

    Main reference: `"Explaining Classifiers with Causal Concept Effect
    (CaCE)" <https://arxiv.org/abs/1907.07165>`_

    Parameters:
        y_pred_c0 (torch.Tensor): Predictions of the model when the concept is
            inactive. Shape: (batch_size, num_classes).
        y_pred_c1 (torch.Tensor): Predictions of the model when the concept is
            active. Shape: (batch_size, num_classes).

    Returns:
        torch.Tensor: The ACE/CaCE score for each class. Shape: (num_classes,).
    """
    if y_pred_c0.shape != y_pred_c1.shape:
        raise RuntimeError(
            "The shapes of y_pred_c0 and y_pred_c1 must be the same but got "
            f"{y_pred_c0.shape} and {y_pred_c1.shape} instead."
        )
    return y_pred_c1.mean(dim=0) - y_pred_c0.mean(dim=0)


def residual_concept_causal_effect(cace_before, cace_after):
    """
    Compute the residual concept causal effect between two concepts.
    Args:
        cace_metric_before: ConceptCausalEffect metric before the do-intervention on the inner concept
        cace_metric_after: ConceptCausalEffect metric after do-intervention on the inner concept
    """
    return cace_after / cace_before

def edge_type(graph, i, j):
    if graph[i,j]==1 and graph[j,i]==0:
        return 'i->j'
    elif graph[i,j]==0 and graph[j,i]==1:
        return 'i<-j'
    elif (graph[i,j]==-1 and graph[j,i]==-1) or (graph[i,j]==1 and graph[j,i]==1):
        return 'i-j'
    elif graph[i,j]==0 and graph[j,i]==0:
        return '/'
    else:
        raise ValueError(f'invalid edge type {i}, {j}')

# graph similairty metrics
def hamming_distance(first, second):
    """Compute the graph edit distance between two partially direceted graphs"""
    first = first.loc[[row for row in first.index if '#virtual_' not in row],
                      [col for col in first.columns if '#virtual_' not in col]]
    first = torch.Tensor(first.values)
    second = second.loc[[row for row in second.index if '#virtual_' not in row],
                        [col for col in second.columns if '#virtual_' not in col]]
    second = torch.Tensor(second.values)
    assert (first.diag() == 0).all() and (second.diag() == 0).all()
    assert first.size() == second.size()
    N = first.size(0)
    cost = 0
    count = 0
    for i in range(N):
        for j in range(i, N):
            if i==j: continue
            if edge_type(first, i, j)==edge_type(second, i, j): continue
            else:
                count += 1
                # edge was directed
                if edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i-j': cost += 1./5.
                elif edge_type(first, i, j)=='i->j' and edge_type(second, i, j)=='i<-j': cost += 1./3.
                elif edge_type(first, i, j)=='i<-j' and edge_type(second, i, j)=='i->j': cost += 1./3.
                # edge was undirected
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='/': cost += 1./4.
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i->j': cost += 1./4. 
                elif edge_type(first, i, j)=='i-j' and edge_type(second, i, j)=='i<-j': cost += 1./4.
                # there was no edge
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i-j': cost += 1./2.
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i->j': cost += 1
                elif edge_type(first, i, j)=='/' and edge_type(second, i, j)=='i<-j': cost += 1

                else:  
                    raise ValueError(f'invalid combination of edge types {i}, {j}')
    
    # cost = cost / (N*(N-1))/2
    return cost, count


def prune_linear_layer(linear: Linear, mask: torch.Tensor, dim: int = 0) -> Linear:
    """
    Return a new nn.Linear where inputs (dim=0) or outputs (dim=1)
    have been pruned according to `mask`.

    Args
    ----
    linear : nn.Linear
        Layer to prune.
    mask : 1D Tensor[bool] or 0/1
        Mask over features. True/1 = keep, False/0 = drop.
        - If dim=0: length == in_features
        - If dim=1: length == out_features
    dim : int
        0 -> prune input features (columns of weight)
        1 -> prune output units (rows of weight)
    """
    if not isinstance(linear, Linear):
        raise TypeError("`linear` must be an nn.Linear")

    mask = mask.to(dtype=torch.bool)
    weight = linear.weight
    device = weight.device
    dtype = weight.dtype

    idx = mask.nonzero(as_tuple=False).view(-1)  # indices to KEEP

    if dim == 0:
        if mask.numel() != linear.in_features:
            raise ValueError("mask length must equal in_features when dim=0")

        new_in = idx.numel()
        new_linear = Linear(
            in_features=new_in,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            # keep all rows (outputs), select only kept input columns
            new_linear.weight.copy_(weight[:, idx])
            if linear.bias is not None:
                new_linear.bias.copy_(linear.bias)

    elif dim == 1:
        if mask.numel() != linear.out_features:
            raise ValueError("mask length must equal out_features when dim=1")

        new_out = idx.numel()
        new_linear = Linear(
            in_features=linear.in_features,
            out_features=new_out,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            # select only kept output rows
            new_linear.weight.copy_(weight[idx, :])
            if linear.bias is not None:
                new_linear.bias.copy_(linear.bias[idx])

    else:
        raise ValueError("dim must be 0 (inputs) or 1 (outputs)")

    return new_linear
