"""
Example: Interventions with Low-Level API

This example demonstrates how to use intervention strategies with
the low-level concept encoder and predictor layers.
"""
from typing import Dict

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import ModuleDict

import torch_concepts as pyc
from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.nn import (
    LinearConceptToConcept,
    LinearEmbeddingToConcept,
    UniformPolicy,
    GroundTruthIntervention,
    PositiveWeightsIntervention,
    DistributionIntervention,
    DoIntervention,
    GradientPolicy,
    InterventionModule,
    BaseInterventionModule,
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

    # Ground Truth Intervention example (intervene on the layer outputs)
    # FIXME: this replace the layer output with the ground truth values for the specified concepts
    # regardeless of whether the parametrization is done via 'logits' or 'probs'. 
    concept_encoder_intervened = InterventionModule(
        model["concept_encoder"],
        GroundTruthIntervention(ground_truth=c_train),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c1"]
    )
    c_pred_intervened = concept_encoder_intervened(embedding)
    print("\n\n## Ground Truth Intervention Example (intervene on c1):")
    print(c_pred_intervened[:5])

    # Module intervention example (intervene on the module parameters instead of the outputs)
    concept_encoder_intervened = InterventionModule(
        model["concept_encoder"],
        PositiveWeightsIntervention(),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c2"]
    )
    c_pred_intervened = concept_encoder_intervened(embedding)
    print("\n\n## Module Intervention Example (Positive Weights Intervention on c2):")
    print(c_pred_intervened[:5])

    # Do Intervention example (set concepts to constant value)
    concept_encoder_intervened = InterventionModule(
        model["concept_encoder"],
        DoIntervention(constants=[-100., 50]),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c1", "c2"]
    )
    c_pred_intervened = concept_encoder_intervened(embedding)
    print("\n\n## Do Intervention Example (set concepts to -100 and 50):")
    print(c_pred_intervened[:5])

    # Distribution Intervention example (sample concepts from distribution)
    concept_encoder_intervened = InterventionModule(
        model["concept_encoder"],
        DistributionIntervention(dist=[
            torch.distributions.Normal(loc=-100, scale=1),
            torch.distributions.Normal(loc=50, scale=1)
        ]),
        UniformPolicy(),
        out_concepts_to_intervene_on=["c1", "c2"]
    )
    c_pred_intervened = concept_encoder_intervened(embedding)
    print("\n\n## Distribution Intervention Example with resampling (set concepts to -100 and 50):")
    print(c_pred_intervened[:5])


    # Example of using a custom build_context function to combine gradients from both the task predictor and
    # a random head on the concept encoder.
    c_pred = model["concept_encoder"](embeddings=embedding)  # pre-compute c_pred
    y_pred = model["task_predictor"](concepts=c_pred)  # pre-compute y_pred
    random_head = torch.nn.Linear(concept_dims, task_dims)

    def build_context_combined(original_module_predictions, original_module, original_module_inputs, 
                               extra_tensors, extra_modules):
        original_module_predictions = original_module(embeddings=original_module_inputs["embeddings"])
        
        pred = original_module_predictions.detach().requires_grad_(True)
        grads_random = torch.autograd.grad(extra_modules["random_head"](pred).sum(), pred)[0]

        grads_task = torch.autograd.grad(
            extra_tensors["y_pred"].sum(), extra_tensors["c_pred"], retain_graph=True
        )[0]

        concept_grads = (grads_random.detach().abs() + grads_task.detach().abs()) / 2.0
        return {"concept_grads": concept_grads}

    int_module_combined = InterventionModule(
        original_module=model["concept_encoder"],
        intervention_strategy=GroundTruthIntervention(ground_truth=c_train),
        intervention_policy=GradientPolicy(),
        build_context=build_context_combined,
        extra_modules={"random_head": random_head},
        quantile=0.5,
    )
    c_pred_combined = int_module_combined(
        embeddings=embedding,
        extra_tensors={"c_pred": c_pred, "y_pred": y_pred},
    )
    print("\nConcept predictions with combined gradient intervention (first 5):")
    print(c_pred_combined[:5])

    # Example subclassing BaseInterventionModule to implement the same custom build_context function as above.
    class CombinedGradientInterventionModule(BaseInterventionModule):
        def build_context(
                self,
                original_module_inputs: Dict[str, torch.Tensor],
                original_module: nn.Module,
                original_module_predictions: torch.Tensor,
                extra_tensors: Dict[str, torch.Tensor] = None,
                extra_modules: Dict[str, nn.Module] = None,
        ) -> Dict[str, torch.Tensor]:
            original_module_predictions = original_module(embeddings=original_module_inputs["embeddings"])

            pred = original_module_predictions.detach().requires_grad_(True)
            grads_random = torch.autograd.grad(extra_modules["random_head"](pred).sum(), pred)[0]

            grads_task = torch.autograd.grad(
                extra_tensors["y_pred"].sum(), extra_tensors["c_pred"], retain_graph=True
            )[0]

            concept_grads = (grads_random.detach().abs() + grads_task.detach().abs()) / 2.0
            return {"concept_grads": concept_grads}

    int_module_combined_subclass = CombinedGradientInterventionModule(
        original_module=model["concept_encoder"],
        intervention_strategy=GroundTruthIntervention(ground_truth=c_train),
        intervention_policy=GradientPolicy(),
        extra_modules={"random_head": random_head},
        quantile=0.5,
    )
    c_pred_combined_subclass = int_module_combined_subclass(
        embeddings=embedding,
        extra_tensors={"c_pred": c_pred, "y_pred": y_pred},
    )
    print("\nConcept predictions with combined gradient intervention (using subclass) (first 5):")
    print(c_pred_combined_subclass[:5])

    return


if __name__ == "__main__":
    main()
