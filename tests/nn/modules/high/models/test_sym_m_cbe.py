import pytest
import torch
from torch import nn

from torch_concepts import AnnotatedTensor, Annotations
from torch_concepts.nn import (
    ConceptLoss,
    DeterministicInference,
    SymbolicConceptToConcept,
    SymbolicMixtureOfConceptBottleneckExperts,
)


ANN = Annotations(labels=["c1", "c2", "y"], cardinalities=[1, 1, 1])


def _batch(n=8):
    c = torch.randint(0, 2, (n, 3)).float()
    return {
        "inputs": {"x": torch.randn(n, 4)},
        "concepts": {"c": AnnotatedTensor(c, ANN.to_concept_space(), axis=-1)},
    }


def _model(**kwargs):
    kwargs.setdefault("n_experts", 3)
    return SymbolicMixtureOfConceptBottleneckExperts(
        input_size=4, annotations=ANN, task_names=["y"], **kwargs
    )


def _selector_seen_by_experts(model, x):
    mixture = model.experts()["tasks"]
    seen = {}
    forward = mixture.forward

    def spy(concepts, embeddings):
        seen["selector"] = embeddings.detach()
        return forward(concepts, embeddings)

    mixture.forward = spy
    model(["y"], input=x)
    return seen["selector"]


def test_forward_shapes_and_backward():
    model = _model()
    batch = _batch()
    out = model(query=model.prepare_query(batch), evidence=model.prepare_evidence(batch))
    assert out.logits["y"].shape == (8, 1)
    loss = ConceptLoss(binary=nn.BCEWithLogitsLoss())(out, model.prepare_target(batch))
    loss.backward()
    assert torch.isfinite(loss)


def test_selector_is_hard_in_training_and_eval_soft_with_deterministic():
    model = _model()
    x = torch.randn(5, 4)
    model.train()
    hard = _selector_seen_by_experts(model, x)
    assert torch.equal(hard.sum(-1), torch.ones(5)) and set(hard.unique().tolist()) <= {0.0, 1.0}
    model.eval()
    assert set(_selector_seen_by_experts(model, x).unique().tolist()) <= {0.0, 1.0}
    model.setup_inference(DeterministicInference)
    model.eval()
    soft = _selector_seen_by_experts(model, x)
    assert torch.allclose(soft.sum(-1), torch.ones(5)) and (soft < 1).all()


def test_linear_experts_and_invalid_arguments():
    model = _model(expert_hidden_layers=0)
    assert all(isinstance(e, nn.Linear) for e in model.experts()["tasks"].experts)
    with pytest.raises(ValueError):
        _model(n_experts=0)
    with pytest.raises(ValueError):
        _model(selector_entropy_weight=-1.0)


def test_set_symbolic_experts_then_train():
    model = _model(lightning=True, loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()),
                   optim_class=torch.optim.Adam, optim_kwargs={"lr": 0.01})
    assert model.expert_input_names == ["c1", "c2"]
    model.set_symbolic_experts({"tasks": [["2.0*c1 - c2"], ["0.5*c2"], ["1.0"]]})
    experts = model.experts()["tasks"].experts
    assert all(isinstance(e, SymbolicConceptToConcept) for e in experts)
    model.train()
    loss = model.training_step(_batch())
    loss.backward()
    assert all(p.grad is not None for e in experts for p in e.parameters())


def test_distill_experts_recovers_a_linear_expert(tmp_path):
    pytest.importorskip("symtorch")
    model = _model(expert_hidden_layers=0)
    for expert in model.experts()["tasks"].experts:
        with torch.no_grad():
            expert.weight.copy_(torch.tensor([[2.0, -1.0]]))
            expert.bias.zero_()
    batch = _batch(64)
    exprs = model.distill_experts(
        batch["inputs"]["x"], batch["concepts"]["c"], save_path=str(tmp_path),
        sr_params={"niterations": 20, "binary_operators": ["+", "-", "*"],
                   "unary_operators": []},
    )
    assert len(exprs["tasks"]) == 3
    model.set_symbolic_experts(exprs)
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    for expert, exprs_m in zip(model.experts()["tasks"].experts, exprs["tasks"]):
        if exprs_m[0].free_symbols:  # an expert that received samples
            assert torch.allclose(expert(c), torch.tensor([[2.0], [-1.0]]), atol=1e-2)


def test_lightning_fit_runs_the_three_stages(tmp_path, monkeypatch):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    model = _model(lightning=True, loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()),
                   optim_class=torch.optim.Adam, optim_kwargs={"lr": 0.01},
                   mlp_epochs=1)
    # Stage 2 without PySR: fixed expressions.
    monkeypatch.setattr(model, "distill_experts",
                        lambda x, c: {"tasks": [["2.0*c1 - c2"], ["0.5*c2"], ["1.0"]]})
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="train_loss")
    trainer = Trainer(max_epochs=2, logger=False, callbacks=[checkpoint],
                      enable_progress_bar=False, enable_model_summary=False)
    batches = torch.utils.data.DataLoader([_batch(16) for _ in range(4)], collate_fn=lambda b: b[0])
    trainer.fit(model, train_dataloaders=batches)

    assert model.is_symbolic
    assert model.optim_kwargs["lr"] == pytest.approx(0.05)
    optimized = {id(p) for g in trainer.optimizers[0].param_groups for p in g["params"]}
    constants = [p for e in model.experts()["tasks"].experts for p in e.parameters()]
    assert constants and all(id(p) in optimized for p in constants)
    assert "epoch=1" in checkpoint.best_model_path  # the best restarts with stage 3


def test_selector_entropy_regularization_uses_the_batch_average():
    kwargs = dict(lightning=True, loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()))
    torch.manual_seed(0)
    plain = _model(**kwargs)
    torch.manual_seed(0)
    regularized = _model(selector_entropy_weight=0.5, **kwargs)
    batch = _batch(32)

    torch.manual_seed(1)
    plain_loss = plain.training_step(batch)
    torch.manual_seed(1)
    loss = regularized.training_step(batch)

    seen = regularized._selector_value  # what the predictor received
    assert set(seen.unique().tolist()) <= {0.0, 1.0}  # hard samples in training
    p = seen.mean(0)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    assert torch.allclose(loss, plain_loss - 0.5 * entropy)
    loss.backward()
    selector = regularized.pgm.factors["selector"].parametrization["logits"]
    assert all(p.grad is not None for p in selector.parameters())
    # validation is not regularized
    plain.eval(), regularized.eval()
    assert torch.allclose(regularized.validation_step(batch), plain.validation_step(batch))
