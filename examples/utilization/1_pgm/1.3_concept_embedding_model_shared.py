"""Concept Embedding Model with a *shared* concept CPD.

The N concepts are parametrized **jointly** by one CPD and produced stacked
(one forward of the encoder for all of them), yet each concept stays an
individually addressable graph node — you can query, observe, or read out a
single concept. ``shared_key`` on the CPD is the only thing needed:

    concepts  = ConceptVariable(['c1', 'c2'], distribution=Bernoulli, size=1)
    c_encoder = ParametricCPD(concepts, parametrization=encoder,
                              parents=[embs], shared_key='concepts')  # -> one facade per member

The sharing lives entirely inside the CPD (a cached core + per-member facades),
so the BayesianNetwork and every inference engine are unchanged — it works with
any backend. Members are ordinary nodes, hence ``*concepts`` is unpacked into
``variables``/``factors``/``parents``, and they are addressed by member name.
"""

import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli

import torch_concepts as pyc
from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.data import ToyDataset
from torch_concepts.nn import MLP, LinearEmbeddingToConcept, MixConceptEmbeddingToConcept, \
    ParametricCPD, BayesianNetwork, DeterministicInference, LearnablePrior, Sequential


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
    c_train = dataset.concepts[:, concept_idx]   # (N, n_concepts)
    y_train = dataset.concepts[:, task_idx]       # (N, n_tasks)
    n_concepts = c_train.shape[1]
    concept_names = [f"c{i + 1}" for i in range(n_concepts)]

    # Variable setup
    input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=latent_dims)
    embs = EmbeddingVariable("embs", distribution=Delta, shape=(n_concepts, emb_dims))
    # Shared concept group: one plate variable with named members c1..cN.
    concepts = ConceptVariable("concepts", members=concept_names, distribution=Bernoulli)
    tasks = ConceptVariable("xor", distribution=Bernoulli)

    layers = {
        # input encoding: (batch, n_features) -> (batch, latent_dims)
        "backbone": MLP(
            input_size=x_train.shape[1],
            hidden_size=latent_dims,
            n_layers=1,
            activation='leaky_relu',
        ),
        # embedding encoder: (batch, latent_dims) -> (batch, n_concepts, embedding_size)
        "emb_encoder": pyc.nn.Sequential(
            torch.nn.Linear(latent_dims, n_concepts * emb_dims),
            torch.nn.Unflatten(unflattened_size=(n_concepts, emb_dims), dim=1),
        ),
        # concept encoder: (batch, n_concepts, embedding_size) -> (batch, n_concepts).
        # ONE encoder shared across all concepts (per-concept embedding -> per-concept logit).
        "concept_encoder": pyc.nn.Sequential(
            LinearEmbeddingToConcept(in_embeddings=emb_dims, out_concepts=1),
            torch.nn.Sigmoid(),
            torch.nn.Flatten(),
        ),
        # predictor: (batch, n_concepts) + (batch, n_concepts, embedding_size) -> (batch, n_tasks)
        "task_predictor": Sequential(
            MixConceptEmbeddingToConcept(
                in_concepts=pyc.AxisAnnotation.empty(n_concepts, types=['binary'] * n_concepts),
                in_embeddings=emb_dims,
                out_concepts=1,
            ),
            torch.nn.Sigmoid(),        
        ),
    }

    # ParametricCPD setup
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(input_var.size), parents=[])
    backbone = ParametricCPD(latent_var, parametrization=layers["backbone"], parents=[input_var])
    emb_encoder = ParametricCPD(embs, parametrization=layers["emb_encoder"], parents=[latent_var])
    # ONE CPD over the concept plate: the encoder runs once and emits all members.
    c_encoder = ParametricCPD(concepts, parametrization=layers["concept_encoder"], parents=[embs])
    y_predictor = ParametricCPD(tasks, parametrization=layers["task_predictor"], parents=[concepts, embs])

    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, embs, concepts, tasks],
        factors=[input_cpd, backbone, emb_encoder, c_encoder, y_predictor],
    )

    # Inference Initialization
    inference_engine = DeterministicInference(concept_model)

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        cy_pred = inference_engine.query(
            query = {"c1": c_train[:, 0],
                     "c2": c_train[:, 1],
                     "xor": None }, 
            evidence={"input": x_train}
        )

        c_pred = torch.cat([
            cy_pred.params['c1']['probs'], 
            cy_pred.params['c2']['probs']
        ], dim=1)
        y_pred = cy_pred.params['xor']['probs']

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_acc = accuracy_score(y_train, y_pred.detach() > 0.5)
            concept_acc = accuracy_score(c_train, c_pred.detach() > 0.5)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_acc:.2f} | Concept Acc: {concept_acc:.2f}")

    # ------------------------------------------------------------------
    # The shared CPD is addressable per member (the encoder still runs once).
    # ------------------------------------------------------------------
    concept_model.eval()

    # (1) Query a single concept — only its slice of the shared CPD is needed.
    only_c1 = inference_engine.query({"c1": None}, evidence={"input": x_train})
    print(f"\nP(c1 | input)         [:5]: {only_c1.params['c1']['probs'][:5].flatten().tolist()}")

    only_c1 = inference_engine.query({"c1": c_train[:, 0]}, evidence={"input": x_train})
    print(f"\nP(c1 | input) with data        [:5]: {only_c1.params['c1']['probs'][:5].flatten().tolist()}")

    # (2) Query one concept with the others observed (partial evidence on the group).
    others = {name: c_train[:, i:i + 1] for i, name in enumerate(concept_names) if name != "c1"}
    cond_c1 = inference_engine.query({"c1": None}, evidence={**{"input": x_train}, **others})
    print(f"P(c1 | input, c2) [:5]: {cond_c1.params['c1']['probs'][:5].flatten().tolist()}")

    out = inference_engine.query({"xor": None}, evidence={"input": x_train})
    print(f"\nP(xor | input) [:5]: {out.params['xor']['probs'][:5].flatten().tolist()}")

    out = inference_engine.query({"xor": None}, evidence={"input": x_train, **others})
    print(f"\nP(xor | input, c1, c2) [:5]: {out.params['xor']['probs'][:5].flatten().tolist()}")

    return


if __name__ == "__main__":
    main()
