import torch
from torch_concepts import Annotations
from torch_concepts.nn import CMRBlendedLoss
from torch_concepts.nn.modules.high.models.cmr import ConceptMemoryReasoner


def test_cmr_routes_reconstruction_prediction_through_modeloutput_extra():
    model = ConceptMemoryReasoner(
        input_size=2,
        annotations=Annotations(labels=["c1", "c2", "xor"], cardinalities=[1, 1, 1]),
        task_names=["xor"],
        n_rules=3,
    )
    target = torch.tensor([[0., 1., 1.], [1., 0., 0.]])
    query = model.build_query(target)
    query["tasks_with_rec"] = None
    output = model(query=query, evidence={"input": torch.randn(2, 2)})

    assert output.probs["xor"].shape == (2, 1)
    assert output.extra["task_input"].shape == (2, 1)
    assert output.extra["input_with_rec"].shape == (2, 1)
    assert "tasks_with_rec" not in output.probs.annotation.label_to_index

    loss = CMRBlendedLoss(task_names=["xor"])(output, model.prepare_target(target))
    loss.backward()
    assert torch.isfinite(loss)
