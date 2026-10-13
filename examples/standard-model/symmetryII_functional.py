import torch
from torch_concepts.nn.functional import prediction_concept_dependency_score


def main():
    input_dim = 1000
    batch_size = 5
    output_dims = [10, 100, 1000]
    concept_dims = [10, 100, 1000]

    print("=" * 80)
    print("CASE 1: Chained architecture (x → concepts → preds)")
    print("Expected: metric ≈ 0 (preds Jacobian fully contained in concepts Jacobian)")
    print("=" * 80)

    for concept_dim in concept_dims:
        for output_dim in output_dims:
            # Case 1: x → concepts → preds (chained dependency)
            # concept_jacobian = ∂c/∂x, preds_jacobian = ∂p/∂x = (∂p/∂c)(∂c/∂x)
            # The row span of ∂p/∂x is contained in the row span of ∂c/∂x

            concept_module = torch.nn.Linear(input_dim, concept_dim, bias=False)
            pred_module = torch.nn.Linear(concept_dim, output_dim, bias=False)

            x = torch.randn(batch_size, input_dim, requires_grad=True)

            # Forward pass
            concepts = concept_module(x)
            preds = pred_module(concepts)

            # Compute Jacobians using functorch
            from torch.func import jacrev, vmap

            # Jacobian of concepts w.r.t. x: [batch, num_concepts, input_dim]
            def concepts_fn(x_single):
                return concept_module(x_single)
            concept_jacobian = vmap(jacrev(concepts_fn))(x)

            # Jacobian of preds w.r.t. x: [batch, num_preds, input_dim]
            def preds_fn(x_single):
                c = concept_module(x_single)
                return pred_module(c)
            preds_jacobian = vmap(jacrev(preds_fn))(x)

            metric = prediction_concept_dependency_score(
                preds_jacobian,
                concept_jacobian,
                method="energy",
                fraction=1.0,
                reduction="mean",
            )

            print(f"ConceptDim={concept_dim:4d}, OutputDim={output_dim:4d} | Metric: {metric.item():.6f}")

    print("\n" + "=" * 80)
    print("CASE 2: Parallel architecture (x → concepts, x → preds independently)")
    print("Expected: metric ≠ 0 (preds and concepts span different subspaces)")
    print("=" * 80)

    for concept_dim in concept_dims:
        for output_dim in output_dims:
            # Case 2: x → concepts and x → preds (independent paths)
            # concept_jacobian = ∂c/∂x, preds_jacobian = ∂p/∂x
            # The row spans are generally unrelated (random projections of x)

            concept_module = torch.nn.Linear(input_dim, concept_dim, bias=False)
            pred_module = torch.nn.Linear(input_dim, output_dim, bias=False)

            x = torch.randn(batch_size, input_dim, requires_grad=True)

            # Jacobian of concepts w.r.t. x: [batch, num_concepts, input_dim]
            def concepts_fn(x_single):
                return concept_module(x_single)
            concept_jacobian = vmap(jacrev(concepts_fn))(x)

            # Jacobian of preds w.r.t. x: [batch, num_preds, input_dim]
            def preds_fn(x_single):
                return pred_module(x_single)
            preds_jacobian = vmap(jacrev(preds_fn))(x)

            metric = prediction_concept_dependency_score(
                preds_jacobian,
                concept_jacobian,
                method="energy",
                fraction=1.0,
                reduction="mean",
            )

            print(f"ConceptDim={concept_dim:4d}, OutputDim={output_dim:4d} | Metric: {metric.item():.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
