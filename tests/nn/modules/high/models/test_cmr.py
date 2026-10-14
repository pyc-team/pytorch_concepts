import torch
from torch_concepts import Annotations, AnnotatedTensor
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


def test_cmr_exposes_reconstruction_prediction_beside_the_task():
    annotations = Annotations(labels=["c1", "c2", "xor"], cardinalities=[1, 1, 1])
    model = ConceptMemoryReasoner(
        input_size=2,
        annotations=annotations,
        task_names=["xor"],
        n_rules=3,
    )
    target = AnnotatedTensor(
        torch.tensor([[0., 1., 1.], [1., 0., 0.]]),
        annotations.to_concept_space(),
        axis=-1,
    )
    batch = {"inputs": {"x": torch.randn(2, 2)}, "concepts": {"c": target}}
    query = model.prepare_query(batch)
    assert query["tasks_with_rec"] is None
    output = model(query=query, evidence=model.prepare_evidence(batch))

    assert output.probs["xor"].shape == (2, 1)
    assert output.probs["tasks_with_rec"].shape == (2, 1)

    loss = make_cmr_loss(task_names=["xor"])(output, model.prepare_target(batch))
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
    assert set(factors["tasks_with_rec"].parametrization) == {"probs"}
    # The reconstruction head scores the same quantity as the task head, so it
    # must not drift to a different family.
    assert (
        factors["tasks_with_rec"].variable.distribution
        == factors["tasks"].variable.distribution
    )


def test_cmr_composite_loss_matches_original_value_and_gradients():
    torch.manual_seed(7)
    task_names = ["y1", "y2"]
    annotations = Annotations(
        labels=["c1", "c2", "c3", *task_names],
        cardinalities=[1, 1, 1, 1, 1],
    )
    model = ConceptMemoryReasoner(
        input_size=4,
        annotations=annotations,
        task_names=task_names,
        n_rules=4,
        hard_roles_at_eval=False,
    )
    model.train()

    batch = {
        "inputs": {"x": torch.randn(9, 4)},
        "concepts": {"c": AnnotatedTensor(
            torch.randint(0, 2, (9, 5)).float(),
            annotations.to_concept_space(),
            axis=-1,
        )},
    }
    target = model.prepare_target(batch)
    # Latent concepts, as at evaluation: every prediction depends on the input.
    query = model.prepare_query(batch, step="val")
    output = model(query=query, evidence=model.prepare_evidence(batch))

    concept_weight, task_weight = 0.7, 1.3
    composed = make_cmr_loss(
        task_names,
        concept_weight=concept_weight,
        task_weight=task_weight,
    )
    actual = composed(output, target)

    c = target["c"]
    concept_names = [
        name for name in c.annotations.labels if name not in task_names
    ]
    concept_loss = torch.nn.functional.binary_cross_entropy(
        output.probs[concept_names],
        c[concept_names].to(output.probs.dtype),
    )
    task_target = c[task_names].to(output.probs.dtype)
    task_pred = output.probs[task_names]
    rec_pred = output.probs["tasks_with_rec"].to(task_pred.dtype)
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
