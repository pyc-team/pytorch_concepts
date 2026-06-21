from copy import deepcopy

import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, OneHotCategorical

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
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    # Variable setup
    input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=latent_dims)
    embs = EmbeddingVariable(['emb1', 'emb2'], distribution=Delta, shape=(1, emb_dims))
    concepts = ConceptVariable(['c1', 'c2'], distribution=Bernoulli)
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
            torch.nn.Linear(latent_dims, emb_dims),
            torch.nn.Unflatten(unflattened_size=(1, emb_dims), dim=1),
        ),
        # concept encoder: (batch, n_concepts, embedding_size) -> (batch, n_concepts)
        "concept_encoder": pyc.nn.Sequential(
            LinearEmbeddingToConcept(in_embeddings=emb_dims, out_concepts=1),
            torch.nn.Flatten()
        ),
        # predictor: (batch, n_concepts) + (batch, n_concepts, embedding_size) -> (batch, n_tasks)
        # Sequential (not torch.nn.Sequential) so the first layer can take both
        # concepts and embeddings; the result is threaded through the Sigmoid.
        "task_predictor": Sequential(
            MixConceptEmbeddingToConcept(
                in_concepts=pyc.AxisAnnotation.empty(2, types=['binary', 'binary']),
                in_embeddings=emb_dims,
                out_concepts=1,
            ),
        )
    }
    
    # ParametricCPD setup
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(input_var.size), parents=[])
    backbone = ParametricCPD(latent_var, parametrization=layers['backbone'], parents=[input_var])
    emb_encoder = ParametricCPD(embs, parametrization=layers['emb_encoder'], parents=[latent_var])
    c1_encoder = ParametricCPD(
        concepts[0], 
        parametrization={'logits': layers['concept_encoder']}, 
        parents=[embs[0]]
    )
    c2_encoder = ParametricCPD(
        concepts[1], 
        parametrization={'logits': deepcopy(layers['concept_encoder'])}, 
        parents=[embs[1]]
    )
    y_predictor = ParametricCPD(
        tasks, 
        parametrization={'logits': layers['task_predictor']}, 
        parents=[*concepts, *embs], 
        aggregate=lambda concepts, embeddings: {
            'concepts': torch.cat(list(concepts.values()), dim=-1), 
            'embeddings': torch.cat(list(embeddings.values()), dim=1)
        }
    )

    # ProbabilisticModel Initialization
    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, *embs, *concepts, tasks], 
        factors=[input_cpd, backbone, *emb_encoder, c1_encoder, c2_encoder, y_predictor]
    )

    # Inference Initialization
    inference_engine = DeterministicInference(concept_model, activate_before_propagation=True)
    evidence = {'input': x_train}
    query_concepts = {"c1": c_train[:, 0], "c2": c_train[:, 1], "xor": y_train}

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        cy_pred = inference_engine.query(
            query = query_concepts,
            evidence = evidence
        )
        c_pred = torch.cat([cy_pred.params['c1']['logits'], cy_pred.params['c2']['logits']], dim=1)
        y_pred = cy_pred.params['xor']['logits']

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    # print("=== Interventions ===")
    # print(cy_pred.logits[:5])

    # int_policy_c = RandomPolicy(out_concepts=concept_model.concept_to_variable["c1"].size, scale=100)
    # int_strategy_c = DoIntervention(model=concept_model.parametric_cpds, constants=-10)
    # with intervention(policies=int_policy_c,
    #                   strategies=int_strategy_c,
    #                   target_concepts=["c1", "c2"],
    #                   quantiles=1):
    #     # intervention affect the layer output 
    #     # -> the parametrization of the distribution 
    #     # -> the logits for discrete variables
    #     cy_pred = inference_engine.query(query_concepts, evidence=initial_input, return_logits=True)
    #     print(cy_pred.logits[:5])

    return


if __name__ == "__main__":
    main()
