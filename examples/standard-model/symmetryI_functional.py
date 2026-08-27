import torch
import torch.nn.functional as F
from torch_concepts.nn.functional import shared_concept_semantics_score, shared_concept_semantics_loss

def main():
    """Demonstrate shared_concept_semantics_loss with two training scenarios."""

    input_dim = 1000
    concept_dim = 100
    batch_size = 100

    # Generate ordered input data (naturally ordered by index)
    x = torch.randn(batch_size, input_dim)
    # Sort by first dimension to create natural ordering
    x, _ = torch.sort(x, dim=0)

    print("=" * 80)
    print("SCENARIO 1: Train with L1 loss, then evaluate shared_concept_semantics_loss")
    print(f"Batch size: {batch_size}, Concept dim: {concept_dim}")
    print("=" * 80)

    # Scenario 1: Train with L1 loss
    model1 = torch.nn.Linear(input_dim, concept_dim, bias=False)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)

    # Create target: simple linear function that preserves order
    target = torch.linspace(0, 1, batch_size).unsqueeze(1).expand(-1, concept_dim)

    epochs = 100
    print(f"\nTraining with L1 loss for {epochs} epochs...")

    for epoch in range(epochs):
        optimizer1.zero_grad()
        concepts = model1(x)
        loss = F.l1_loss(concepts, target)
        loss.backward()
        optimizer1.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                concepts_eval = model1(x)
                sem_loss = shared_concept_semantics_loss(concepts_eval, target)
                order_metric = shared_concept_semantics_score(concepts_eval, target)
            print(f"Epoch {epoch+1:3d} | L1 Loss: {loss.item():.6f} | "
                  f"Semantics Loss: {sem_loss.item():.6f} | "
                  f"Order Metric: {order_metric:.4f}")

    # Final evaluation
    with torch.no_grad():
        concepts_final1 = model1(x)
        final_sem_loss1 = shared_concept_semantics_loss(concepts_final1, target)
        final_order_metric1 = shared_concept_semantics_score(concepts_final1, target)

    print(f"\n[FINAL] L1-trained model:")
    print(f"  Semantics Loss: {final_sem_loss1.item():.6f}")
    print(f"  Order Preservation: {final_order_metric1:.4f}")

    print("\n" + "=" * 80)
    print("SCENARIO 2: Train with shared_concept_semantics_loss")
    print("=" * 80)

    # Scenario 2: Train with shared_concept_semantics_loss
    model2 = torch.nn.Linear(input_dim, concept_dim, bias=False)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    print(f"\nTraining with shared_concept_semantics_loss for {epochs} epochs...")

    for epoch in range(epochs):
        optimizer2.zero_grad()
        concepts = model2(x)
        loss = shared_concept_semantics_loss(concepts, target, chunk_size=100)

        # Add small L1 regularization to prevent trivial solution (all zeros)
        loss = loss + 0.01 * concepts.abs().mean()

        loss.backward()
        optimizer2.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                concepts_eval = model2(x)
                l1_loss = F.l1_loss(concepts_eval, target)
                sem_loss = shared_concept_semantics_loss(concepts_eval, target)
                order_metric = shared_concept_semantics_score(concepts_eval, target)
            print(f"Epoch {epoch+1:3d} | Semantics Loss: {sem_loss.item():.6f} | "
                  f"L1 Loss: {l1_loss.item():.6f} | "
                  f"Order Metric: {order_metric:.4f}")

    # Final evaluation
    with torch.no_grad():
        concepts_final2 = model2(x)
        final_l1_loss2 = F.l1_loss(concepts_final2, target)
        final_sem_loss2 = shared_concept_semantics_loss(concepts_final2, target)
        final_order_metric2 = shared_concept_semantics_score(concepts_final2, target)

    print(f"\n[FINAL] Semantics-loss-trained model:")
    print(f"  Semantics Loss: {final_sem_loss2.item():.6f}")
    print(f"  L1 Loss: {final_l1_loss2.item():.6f}")
    print(f"  Order Preservation: {final_order_metric2:.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"L1-trained:         Order Preservation = {final_order_metric1:.4f}, Semantics Loss = {final_sem_loss1.item():.6f}")
    print(f"Semantics-trained:  Order Preservation = {final_order_metric2:.4f}, Semantics Loss = {final_sem_loss2.item():.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
