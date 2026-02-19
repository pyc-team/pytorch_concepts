"""
Example: Interventions with Low-Level API

This example demonstrates how to use intervention strategies with
the low-level concept encoder and predictor layers.
"""
import torch
from sklearn.metrics import accuracy_score

from torch_concepts import seed_everything
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import LinearLatentToConcept, LinearConceptToConcept, \
    GroundTruthIntervention, UncertaintyInterventionPolicy, intervention, \
    DoIntervention, DistributionIntervention, UniformPolicy, RandomPolicy


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5

    seed_everything(42)

    # Load dataset
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    # Duplicate concepts for demonstration
    c_train = torch.concat([c_train, c_train, c_train], dim=1)
    
    # Get dimensions
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]

    # Build model using low-level layers
    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    c_encoder = LinearLatentToConcept(latent_dims, n_concepts)
    y_predictor = LinearConceptToConcept(n_concepts, n_tasks)

    # All models in a ModuleDict for easier intervention
    model = torch.nn.ModuleDict({
        "latent_encoder": latent_encoder,
        "c_encoder": c_encoder,
        "y_predictor": y_predictor,
    })

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Generate concept and task predictions
        emb = model["latent_encoder"](x_train)
        c_pred = model["c_encoder"](latent=emb)
        y_pred = model["y_predictor"](concepts=c_pred)

        # Compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    # ==================== Intervention Examples ====================
    
    # Example 1: Uniform Policy + Ground Truth Intervention
    print("\n" + "="*60)
    print("Uniform Policy + Ground Truth Intervention:")
    print("="*60)
    
    int_policy_c = UniformPolicy(out_concepts=n_concepts)
    int_strategy_c = GroundTruthIntervention(model=model["c_encoder"], ground_truth=torch.logit(c_train, eps=1e-6))

    with intervention(
        policies=int_policy_c,
        strategies=int_strategy_c,
        target_concepts=[0, 1]
    ) as new_c_encoder:
        emb = model["latent_encoder"](x_train)
        c_pred = new_c_encoder(latent=emb)
        y_pred = model["y_predictor"](concepts=c_pred)
        print("\nConcept predictions (first 5):")
        print(c_pred[:5])
        print("\nGround truth (first 5):")
        print(torch.logit(c_train, eps=1e-6)[:5])

    # Example 2: Uniform Policy + Do Intervention (set concepts to constant)
    print("\n" + "="*60)
    print("Uniform Policy + Do Intervention (set to -10):")
    print("="*60)
    
    int_policy_c = UniformPolicy(out_concepts=n_concepts)
    int_strategy_c = DoIntervention(model=model["c_encoder"], constants=-10)

    with intervention(
        policies=int_policy_c,
        strategies=int_strategy_c,
        target_concepts=[1],
    ) as new_c_encoder:
        emb = model["latent_encoder"](x_train)
        c_pred = new_c_encoder(latent=emb)
        y_pred = model["y_predictor"](concepts=c_pred)
        print("\nConcept predictions (first 5, columns 0-1):")
        print(c_pred[:5, :2])

    # Example 3: Random Policy + Do Intervention (selective intervention)
    print("\n" + "="*60)
    print("Random Policy + Do Intervention (50% quantile):")
    print("="*60)
    
    int_policy_c = RandomPolicy(out_concepts=n_concepts)
    int_strategy_c = DoIntervention(model=model["c_encoder"], constants=-10)

    with intervention(
        policies=int_policy_c,
        strategies=int_strategy_c,
        target_concepts=[0, 1],
        quantiles=0.5
    ) as new_c_encoder:
        emb = model["latent_encoder"](x_train)
        c_pred = new_c_encoder(latent=emb)
        y_pred = model["y_predictor"](concepts=c_pred)
        print("\nConcept predictions (first 5, columns 0-1):")
        print(c_pred[:5, :2])

    # Example 4: Distribution Intervention (sample from distribution)
    print("\n" + "="*60)
    print("Random Policy + Distribution Intervention (Normal(50, 1)):")
    print("="*60)
    
    int_strategy_c = DistributionIntervention(model=model["c_encoder"], 
                                              dist=torch.distributions.Normal(loc=50, scale=1))

    with intervention(
        policies=int_policy_c,
        strategies=int_strategy_c,
        target_concepts=[1, 3],
        quantiles=.5
    ) as new_c_encoder:
        emb = model["latent_encoder"](x_train)
        c_pred = new_c_encoder(latent=emb)
        y_pred = model["y_predictor"](concepts=c_pred)
        print("\nConcept predictions (first 5):")
        print(c_pred[:5])

    return


if __name__ == "__main__":
    main()
