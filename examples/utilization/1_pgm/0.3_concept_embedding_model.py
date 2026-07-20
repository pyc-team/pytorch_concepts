"""
Concept Embedding Model (CEM) as a PGM.

Where a CBM (see ``0.1_cbm.py``) reduces every concept to a single scalar before
the task predictor, a CEM keeps a *per-concept embedding* and predicts the task
from the concept activations **and** their embeddings jointly. This lets the
task head recover information the concept bottleneck would otherwise discard.

Graph:  input -> latent -> {emb1, emb2} -> {c1, c2} -> xor
                                    └──────────────────┘
        the task predictor sees both the concepts and their embeddings.

Experiment settings:
- Dataset: XOR toy dataset, two concepts (c1, c2) and one task (xor).
- Model: MLP backbone -> per-concept embeddings -> concept encoder ->
  ``MixConceptEmbeddingToConcept`` task head.
- Inference engine: Deterministic inference.

Expected outcome:
- Both concept and task accuracy reach ~1.0: unlike the linear-head CBM, the
  embedding-mixing task head can solve XOR.
"""

from copy import deepcopy

import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli

import torch_concepts as pyc
from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.data import ToyDataset
from torch_concepts.nn import MLP, LinearEmbeddingToConcept, \
    MixConceptEmbeddingToConcept, ParametricCPD, BayesianNetwork, \
    DeterministicInference, LearnablePrior, Sequential


def main():
    seed_everything(42)

    emb_dims = 7
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5

    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    # Variable setup. Each concept gets its own embedding variable of shape
    # (1, emb_dims); the embeddings are what makes this a CEM rather than a CBM.
    input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=latent_dims)
    embs = EmbeddingVariable(['emb1', 'emb2'], distribution=Delta, shape=(1, emb_dims))
    concepts = ConceptVariable(['c1', 'c2'], distribution=Bernoulli)
    tasks = ConceptVariable("xor", distribution=Bernoulli)

    # Parametrization modules for each CPD.
    backbone = MLP(input_size=x_train.shape[1], hidden_size=latent_dims,
                   n_layers=1, activation='leaky_relu')
    # latent -> (batch, 1, emb_dims): one embedding per concept.
    emb_encoder = pyc.nn.Sequential(
        torch.nn.Linear(latent_dims, emb_dims),
        torch.nn.Unflatten(unflattened_size=(1, emb_dims), dim=1),
    )
    # (batch, 1, emb_dims) -> (batch, 1) concept activation.
    concept_encoder = pyc.nn.Sequential(
        LinearEmbeddingToConcept(in_embeddings=emb_dims, out_concepts=1),
        torch.nn.Flatten(),
    )
    # concepts + embeddings -> task logit. Sequential (not torch.nn.Sequential)
    # so the first layer can take both concepts and embeddings.
    task_predictor = Sequential(
        MixConceptEmbeddingToConcept(
            in_concepts=pyc.Annotations.empty(2, types=['binary', 'binary']),
            in_embeddings=emb_dims,
            out_concepts=1,
        ),
    )

    # CPD setup: `parents` wires the graph. Roots use a LearnablePrior.
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(input_var.size), parents=[])
    backbone_cpd = ParametricCPD(latent_var, parametrization=backbone, parents=[input_var])
    emb_cpd = ParametricCPD(embs, parametrization=emb_encoder, parents=[latent_var])
    c1_encoder = ParametricCPD(concepts[0], parametrization={'logits': concept_encoder}, parents=[embs[0]])
    c2_encoder = ParametricCPD(concepts[1], parametrization={'logits': deepcopy(concept_encoder)}, parents=[embs[1]])
    # The task head consumes both concepts and embeddings, so its aggregate
    # returns the two groups the mixing layer expects.
    y_predictor = ParametricCPD(
        tasks,
        parametrization={'logits': task_predictor},
        parents=[*concepts, *embs],
        aggregate=lambda concepts, embeddings: {
            'concepts': torch.cat(list(concepts.values()), dim=-1),
            'embeddings': torch.cat(list(embeddings.values()), dim=1),
        },
    )

    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, *embs, *concepts, tasks],
        factors=[input_cpd, backbone_cpd, *emb_cpd, c1_encoder, c2_encoder, y_predictor],
    )

    inference_engine = DeterministicInference(concept_model, activate_before_propagation=True)
    evidence = {'input': x_train}
    query_concepts = {"c1": c_train[:, 0], "c2": c_train[:, 1], "xor": y_train}

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        cy_pred = inference_engine.query(query=query_concepts, evidence=evidence)
        c_pred = cy_pred.logits['c1', 'c2']
        y_pred = cy_pred.logits['xor']

        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")


if __name__ == "__main__":
    main()
