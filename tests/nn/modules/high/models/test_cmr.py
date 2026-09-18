import torch
from torch_concepts import Annotations
from torch_concepts.nn import (
    CMRTaskLoss,
    CompositeLoss,
    ConceptLoss,
    ConceptSubset,
)
from torch_concepts.nn.modules.high.models.cmr import ConceptMemoryReasoner


def make_cmr_loss(task_names, concept_weight=1.0, task_weight=1.0):
    return CompositeLoss(
        terms=[
            ConceptSubset(
                ConceptLoss(
                    binary=torch.nn.BCELoss(),
                    binary_param="probs",
                ),
                exclude=task_names,
            ),
            CMRTaskLoss(task_names),
        ],
        weights=[concept_weight, task_weight],
        names=["concepts", "tasks"],
    )


def test_cmr_routes_reconstruction_prediction_as_auxiliary_value():
    model = ConceptMemoryReasoner(
        input_size=2,
        annotations=Annotations(labels=["c1", "c2", "xor"], cardinalities=[1, 1, 1]),
        task_names=["xor"],
        n_rules=3,
    )
    target = torch.tensor([[0., 1., 1.], [1., 0., 0.]])
    query = model.fully_observed_query(target)
    query["tasks_with_rec"] = None
    output = model(query=query, evidence={"input": torch.randn(2, 2)})

    assert output.probs["xor"].shape == (2, 1)
    assert output.value["tasks_with_rec"].shape == (2, 1)
    assert "tasks_with_rec" not in output.probs.annotation.label_to_index

    loss = make_cmr_loss(task_names=["xor"])(output, model.prepare_target(target))
    loss.backward()
    assert torch.isfinite(loss)


def test_cmr_cpd_parametrizations_match_layer_output_domains():
    model = ConceptMemoryReasoner(
        input_size=2,
        annotations=Annotations(
            labels=["c1", "c2", "xor"], cardinalities=[1, 1, 1]
        ),
        task_names=["xor"],
        n_rules=3,
    )
    factors = {factor.variable.name: factor for factor in model.pgm.factors.values()}

    assert set(factors["rule_selector"].parametrization) == {"logits"}
    assert set(factors["tasks"].parametrization) == {"probs"}
    assert set(factors["tasks_with_rec"].parametrization) == {"value"}


def test_cmr_composite_loss_matches_original_value_and_gradients():
    torch.manual_seed(7)
    task_names = ["y1", "y2"]
    model = ConceptMemoryReasoner(
        input_size=4,
        annotations=Annotations(
            labels=["c1", "c2", "c3", *task_names],
            cardinalities=[1, 1, 1, 1, 1],
        ),
        task_names=task_names,
        n_rules=4,
        hard_roles_at_eval=False,
    )
    model.train()

    raw_target = torch.randint(0, 2, (9, 5)).float()
    target = model.prepare_target(raw_target)
    query = {name: None for name in model.fully_observed_query(raw_target)}
    query["tasks_with_rec"] = None
    output = model(query=query, evidence={"input": torch.randn(9, 4)})

    concept_weight, task_weight = 0.7, 1.3
    composed = make_cmr_loss(
        task_names,
        concept_weight=concept_weight,
        task_weight=task_weight,
    )
    actual = composed(output, target)

    concept_names = [
        name for name in target.annotation.labels if name not in task_names
    ]
    concept_loss = torch.nn.functional.binary_cross_entropy(
        output.probs[concept_names],
        target[concept_names].to(output.probs.dtype),
    )
    task_target = target[task_names].to(output.probs.dtype)
    task_pred = output.probs[task_names]
    rec_pred = output.value["tasks_with_rec"].to(task_pred.dtype)
    ordinary_bce = torch.nn.functional.binary_cross_entropy(
        task_pred, task_target, reduction="none"
    )
    reconstruction_bce = torch.nn.functional.binary_cross_entropy(
        rec_pred, task_target, reduction="none"
    )
    switched_task_loss = (
        (1.0 - task_target) * ordinary_bce
        + task_target * reconstruction_bce
    ).mean()
    expected = (
        concept_weight * concept_loss
        + task_weight * switched_task_loss
    )

    assert torch.equal(actual, expected)
    assert set(composed.breakdown(output, target)) == {"concepts", "tasks"}

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    actual_gradients = torch.autograd.grad(
        actual, parameters, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected, parameters, allow_unused=True
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        assert (actual_gradient is None) == (expected_gradient is None)
        if actual_gradient is not None:
            assert torch.equal(actual_gradient, expected_gradient)
