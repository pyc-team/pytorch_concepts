"""Same model as ``0.2_cbm_plate.py``, but every tensor carries an extra
leading dimension.

The engines treat the last axis as the only operating one: a tensor is
``(*leading, event)`` for any number of leading (batch-like) dimensions, and
results come back with the same leading shape. Here the extra dimension is
obtained trivially — ``unsqueeze(0)`` + ``expand`` — so the ``n_replicas``
copies are identical and the run must match the single-batch one; in real use
the replicas would differ (e.g. Monte-Carlo repetitions or a grid of settings).
"""

import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.data import ToyDataset
from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept, \
    ParametricCPD, BayesianNetwork, DeterministicInference, LearnablePrior, Sequential


def main():
    seed_everything(42)

    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    n_replicas = 3          # the extra leading dimension
    concept_reg = 0.5

    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    y_train = torch.cat([y_train, 1 - y_train], dim=1)

    # Add the extra leading dimension: (N, d) -> (R, N, d). ``expand`` is a
    # broadcast view, so the replicas share storage (no copy). Every tensor
    # handed to the engine must agree on the leading shape, so the input is
    # expanded the same way as the targets.
    x_rep = x_train.unsqueeze(0).expand(n_replicas, -1, -1)
    c_rep = c_train.unsqueeze(0).expand(n_replicas, -1, -1)
    y_rep = y_train.unsqueeze(0).expand(n_replicas, -1, -1)

    # Variable setup — identical to the single-batch example: the variables
    # only declare the event (last-axis) layout, never the batch dims.
    input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=latent_dims)
    concepts = ConceptVariable("concepts", distribution=Bernoulli, size=1, members=["c1", "c2"])
    tasks = ConceptVariable("xor", distribution=OneHotCategorical, size=2)

    # ParametricCPD setup
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(x_train.shape[1]),
                              parents=[])  # learnable prior parametrization is automatically set

    backbone = ParametricCPD(latent_var,
                             parametrization=torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims),
                                                                 torch.nn.LeakyReLU()), parents=[input_var])

    c_encoder = ParametricCPD(
        concepts,
        parametrization={'logits': Sequential(LinearEmbeddingToConcept(in_embeddings=latent_dims, out_concepts=2))},
        parents=[latent_var]
    )

    y_predictor = ParametricCPD(
        tasks,
        parametrization={'logits': Sequential(LinearConceptToConcept(in_concepts=2, out_concepts=2))},
        parents=[concepts]
    )

    # ProbabilisticModel Initialization
    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, concepts, tasks],
        factors=[input_cpd, backbone, c_encoder, y_predictor]
    )

    # Inference Initialization — the replicated tensors go in as-is.
    inference_engine = DeterministicInference(concept_model, activate_before_propagation=True)
    evidence = {'input': x_rep}
    query_concepts = {"concepts": c_rep, "xor": y_rep}

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions: (R, N, width) out
        cy_pred = inference_engine.query(
            query=query_concepts,
            evidence=evidence
        )
        c_pred = cy_pred.logits['concepts']
        y_pred = cy_pred.logits['xor']
        assert c_pred.shape == (n_replicas, n_samples, 2)
        assert y_pred.shape == (n_replicas, n_samples, 2)

        # compute loss — BCEWithLogitsLoss is elementwise, so the extra
        # leading dimension just averages over R*N instead of N.
        concept_loss = loss_fn(c_pred.tensor, c_rep)
        task_loss = loss_fn(y_pred.tensor, y_rep)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            # sklearn wants 2-D inputs: fold the replica axis into the batch.
            y_flat = y_pred.tensor.detach().reshape(-1, 2)
            c_flat = c_pred.tensor.detach().reshape(-1, 2)
            task_accuracy = accuracy_score(y_rep.reshape(-1, 2), y_flat > 0.)
            concept_accuracy = accuracy_score(c_rep.reshape(-1, 2), c_flat > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    # The replicas were exact copies, so every replica's prediction must agree.
    with torch.no_grad():
        out = inference_engine.query(query=["concepts", "xor"], evidence={'input': x_rep})
    assert torch.allclose(out.logits['xor'].tensor[0], out.logits['xor'].tensor[-1])
    print(f"Replicas agree: output shape {tuple(out.logits['xor'].shape)} "
          f"with identical slices along the leading axis.")

    return


if __name__ == "__main__":
    main()
