"""
Example: Interventions with Low-Level API

This example demonstrates how to use intervention strategies with
the low-level concept encoder and predictor layers.
"""
import torch
from sklearn.metrics import accuracy_score
from torch.nn import ModuleDict

import torch_concepts as pyc
from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.nn import (
    LinearConceptToConcept,
    LinearEmbeddingToConcept,
    UniformPolicy,
    GroundTruthIntervention,
    intervene,
    intervention,
    PositiveWeightsIntervention,
    DistributionIntervention,
    DoIntervention,
)


def main():
    embedding_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5

    seed_everything(42)

    dataset = ToyDataset(dataset='xor', n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    # Get dimensions
    n_features = x_train.shape[1]
    concept_dims = c_train.shape[1]
    task_dims = y_train.shape[1]

    embedding_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, embedding_dims),
        torch.nn.LeakyReLU(),
    )

    # PyC layers
    out_concepts = pyc.AxisAnnotation(["c1", "c2"])
    c_encoder = LinearEmbeddingToConcept(in_embeddings=embedding_dims, out_concepts=out_concepts)
    y_predictor = LinearConceptToConcept(in_concepts=concept_dims, out_concepts=task_dims)

    # these are equivalent to the following torch layers
    # c_encoder = torch.nn.Linear(embedding_dims, concept_dims)
    # y_predictor = torch.nn.Linear(concept_dims, task_dims)

    model = ModuleDict({
        "embedding_encoder": embedding_encoder,
         "concept_encoder": c_encoder,
         "task_predictor": y_predictor
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        embedding = model["embedding_encoder"](x_train)
        c_pred = model["concept_encoder"](embeddings=embedding)
        y_pred = model["task_predictor"](concepts=c_pred)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | "
                  f"Concept Acc: {concept_accuracy:.2f}")

    # ==================== Raw Intervention Examples ====================

    # Uniform Policy + Ground Truth Intervention
    print("\n" + "="*60)
    print("Interventions!")
    print("="*60)
    print("\nConcept predictions before intervention (first 5):")
    print(c_pred[:5])

    # Context manager example
    with intervention(
        model["concept_encoder"],
        GroundTruthIntervention(ground_truth=c_train),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c1"]
    ) as concept_encoder_intervened:
        c_pred_intervened = concept_encoder_intervened(embedding)
        print("\n\n## Ground Truth Intervention Example (intervene on c1):")
        print(c_pred_intervened[:5])

    # Module intervention example (intervene on the module parameters instead of the outputs)
    with intervention(
        model["concept_encoder"],
        PositiveWeightsIntervention(),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c2"]
    ) as concept_encoder_intervened:
        c_pred_intervened = concept_encoder_intervened(embedding)
        print("\n\n## Module Intervention Example (Positive Weights Intervention on c2):")
        print(c_pred_intervened[:5])

    # Do Intervention example (set concepts to constant value)
    with intervention(
        model["concept_encoder"],
        DoIntervention(constants=[-100., 50]),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c1", "c2"]
    ) as concept_encoder_intervened:
        c_pred_intervened = concept_encoder_intervened(embedding)
        print("\n\n## Do Intervention Example (set concepts to -100 and 50):")
        print(c_pred_intervened[:5])

    # Distribution Intervention example (sample concepts from distribution)
    with intervention(
        model["concept_encoder"],
        DistributionIntervention(dist=[
            torch.distributions.Normal(loc=-100, scale=1),
            torch.distributions.Normal(loc=50, scale=1)
        ]),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c1", "c2"]
    ) as concept_encoder_intervened:
        c_pred_intervened = concept_encoder_intervened(embedding)
        print("\n\n## Do Intervention Example (set concepts to -100 and 50):")
        print(c_pred_intervened[:5])










    # ===================== Intervention Context Manager Examples =====================

    #
    # with intervention(
    #     policies=int_policy_c,
    #     strategies=int_strategy_c,
    #     target_concepts=[0, 1]
    # ) as new_c_encoder:
    #     emb = model["latent_encoder"](x_train)
    #     c_pred = new_c_encoder(latent=emb)
    #     y_pred = model["y_predictor"](concepts=c_pred)
    #     print("\nConcept predictions (first 5):")
    #     print(c_pred[:5])
    #     print("\nGround truth (first 5):")
    #     print(torch.logit(c_train, eps=1e-6)[:5])
    #
    # # Example 2: Uniform Policy + Do Intervention (set concepts to constant)
    # print("\n" + "="*60)
    # print("Uniform Policy + Do Intervention (set to -10):")
    # print("="*60)
    #
    # int_policy_c = UniformPolicy(out_concepts=n_concepts)
    # int_strategy_c = DoIntervention(model=model["c_encoder"], constants=-10)
    #
    # with intervention(
    #     policies=int_policy_c,
    #     strategies=int_strategy_c,
    #     target_concepts=[1],
    # ) as new_c_encoder:
    #     emb = model["latent_encoder"](x_train)
    #     c_pred = new_c_encoder(latent=emb)
    #     y_pred = model["y_predictor"](concepts=c_pred)
    #     print("\nConcept predictions (first 5, columns 0-1):")
    #     print(c_pred[:5, :2])
    #
    # # Example 3: Random Policy + Do Intervention (selective intervention)
    # print("\n" + "="*60)
    # print("Random Policy + Do Intervention (50% quantile):")
    # print("="*60)
    #
    # int_policy_c = RandomPolicy(out_concepts=n_concepts)
    # int_strategy_c = DoIntervention(model=model["c_encoder"], constants=-10)
    #
    # with intervention(
    #     policies=int_policy_c,
    #     strategies=int_strategy_c,
    #     target_concepts=[0, 1],
    #     quantiles=0.5
    # ) as new_c_encoder:
    #     emb = model["latent_encoder"](x_train)
    #     c_pred = new_c_encoder(latent=emb)
    #     y_pred = model["y_predictor"](concepts=c_pred)
    #     print("\nConcept predictions (first 5, columns 0-1):")
    #     print(c_pred[:5, :2])
    #
    # # Example 4: Distribution Intervention (sample from distribution)
    # print("\n" + "="*60)
    # print("Random Policy + Distribution Intervention (Normal(50, 1)):")
    # print("="*60)
    #
    # int_strategy_c = DistributionIntervention(model=model["c_encoder"],
    #                                           dist=torch.distributions.Normal(loc=50, scale=1))
    #
    # with intervention(
    #     policies=int_policy_c,
    #     strategies=int_strategy_c,
    #     target_concepts=[1, 3],
    #     quantiles=.5
    # ) as new_c_encoder:
    #     emb = model["latent_encoder"](x_train)
    #     c_pred = new_c_encoder(latent=emb)
    #     y_pred = model["y_predictor"](concepts=c_pred)
    #     print("\nConcept predictions (first 5):")
    #     print(c_pred[:5])

    return


if __name__ == "__main__":
    main()
