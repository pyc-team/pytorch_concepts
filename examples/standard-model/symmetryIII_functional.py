"""
Example: Training with linear_pde constraint
Visualizes how function evolves with linear vs cubic targets
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch_concepts.nn.functional import bounded_reasoning_loss, linear_pde

sns.set_style("whitegrid")
sns.set_palette("husl")


def train_with_snapshots(model, x_data, y_target, pde_func, reg_weight,
                         num_epochs=500, lr=0.01, snapshot_epochs=None):
    """Train model and capture snapshots for visualization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if snapshot_epochs is None:
        snapshot_epochs = [1, 50, 150, 300, 500]

    history = {'mse': [], 'pde': [], 'total': []}
    snapshots = {}

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # Forward pass
        x_input = x_data.clone().requires_grad_(True)
        y_pred = model(x_input)

        # Losses
        mse_loss = ((y_pred - y_target) ** 2).mean()
        pde_loss = bounded_reasoning_loss(y_pred, x_input, pde_func)
        total_loss = mse_loss + reg_weight * pde_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # Record
        history['mse'].append(mse_loss.item())
        history['pde'].append(pde_loss.item())
        history['total'].append(total_loss.item())

        # Snapshot
        if epoch in snapshot_epochs:
            with torch.no_grad():
                snapshots[epoch] = model(x_data).cpu().numpy()

    return history, snapshots


def plot_training_evolution(x_data, y_target, snapshots, history, title, save_path=None):
    """Create visualization of training evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Function evolution
    ax1 = axes[0]

    # Plot target data
    x_np = x_data.cpu().numpy()
    y_np = y_target.cpu().numpy()
    ax1.scatter(x_np, y_np, c='black', s=30, alpha=0.6,
                label='Target Data', zorder=10)

    # Plot predictions at different epochs with increasing alpha
    epochs = sorted(snapshots.keys())
    colors = sns.color_palette("rocket", len(epochs))

    for i, epoch in enumerate(epochs):
        y_snap = snapshots[epoch]
        alpha = 0.3 + (i / len(epochs)) * 0.7  # Fade from 0.3 to 1.0

        # Sort for smooth line
        sort_idx = np.argsort(x_np.flatten())
        x_sorted = x_np[sort_idx]
        y_sorted = y_snap[sort_idx]

        ax1.plot(x_sorted, y_sorted, c=colors[i], alpha=alpha,
                linewidth=2.5 if i == len(epochs)-1 else 1.5,
                label=f'Epoch {epoch}', zorder=5)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'{title}\nModel Evolution with Linear PDE Constraint',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Right plot: Loss history
    ax2 = axes[1]

    epochs_range = range(1, len(history['mse']) + 1)
    ax2.plot(epochs_range, history['mse'], label='MSE Loss', linewidth=2, alpha=0.8)
    ax2.plot(epochs_range, history['pde'], label='PDE Loss', linewidth=2, alpha=0.8)
    ax2.plot(epochs_range, history['total'], label='Total Loss',
             linewidth=2, alpha=0.8, linestyle='--')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss History', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def main():
    print("=" * 80)
    print("TRAINING WITH LINEAR PDE CONSTRAINT")
    print("Comparing Linear vs Cubic Target Functions")
    print("=" * 80)

    # ============================================================================
    # Example 1: Linear Target with Linear PDE
    # ============================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 1: LINEAR TARGET with linear_pde")
    print("Expected: MSE stays low, PDE loss decreases → perfect fit!")
    print("=" * 60)

    torch.manual_seed(42)
    input_dim = 1
    output_dim = 1

    # Create LINEAR target data
    x_data = torch.linspace(-3, 3, 50).unsqueeze(1)
    y_target = 2.0 * x_data + 1.0 + 0.2 * torch.randn_like(x_data)

    print(f"\nTarget function: y = 2x + 1 + noise")
    print(f"Number of data points: {len(x_data)}")

    # Test different regularization weights
    reg_weights = [0.0, 0.1, 10.0]

    for reg in reg_weights:
        print(f"\n{'='*60}")
        print(f"Training with regularization weight λ={reg}")
        print(f"{'='*60}")

        # Fresh model
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

        # Train
        pde = linear_pde(strength=1.0)
        history, snapshots = train_with_snapshots(
            model, x_data, y_target, pde,
            reg_weight=reg, num_epochs=500, lr=0.01,
            snapshot_epochs=[1, 50, 150, 300, 500]
        )

        # Print final metrics
        print(f"\nFinal MSE: {history['mse'][-1]:.6f}")
        print(f"Final PDE Loss: {history['pde'][-1]:.6f}")
        print(f"Final Total Loss: {history['total'][-1]:.6f}")

        if reg == 0.0:
            print("  → No constraint: Fits well naturally (target is linear)")
        else:
            print(f"  → With constraint: PDE loss decreases, becomes more linear")

        # Visualize
        plot_training_evolution(
            x_data, y_target, snapshots, history,
            title=f"Linear Target (λ={reg})"
        )

    # ============================================================================
    # Example 2: Cubic Target with Linear PDE
    # ============================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CUBIC TARGET with linear_pde")
    print("Expected: Trade-off between MSE and PDE loss!")
    print("=" * 60)

    torch.manual_seed(42)

    # Create CUBIC (3rd order polynomial) target data
    x_data = torch.linspace(-2, 2, 50).unsqueeze(1)
    y_target = 0.5 * x_data**3 - x_data + 0.15 * torch.randn_like(x_data)

    print(f"\nTarget function: y = 0.5x³ - x + noise (CUBIC)")
    print(f"Number of data points: {len(x_data)}")

    # Test different regularization weights
    reg_weights = [0.0, 0.1, 10.0]

    for reg in reg_weights:
        print(f"\n{'='*60}")
        print(f"Training with regularization weight λ={reg}")
        print(f"{'='*60}")

        # Fresh model
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

        # Train
        pde = linear_pde(strength=1.0)
        history, snapshots = train_with_snapshots(
            model, x_data, y_target, pde,
            reg_weight=reg, num_epochs=500, lr=0.01,
            snapshot_epochs=[1, 50, 150, 300, 500]
        )

        # Print final metrics
        print(f"\nFinal MSE: {history['mse'][-1]:.6f}")
        print(f"Final PDE Loss: {history['pde'][-1]:.6f}")
        print(f"Final Total Loss: {history['total'][-1]:.6f}")

        if reg == 0.0:
            print("  → No constraint: Fits cubic perfectly, high curvature")
        else:
            print(f"  → With constraint: Model forced to be more linear")
            print(f"     MSE increases as model can't fit nonlinear target")

        # Visualize
        plot_training_evolution(
            x_data, y_target, snapshots, history,
            title=f"Cubic Target (λ={reg})"
        )

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. LINEAR TARGET: PDE constraint helps → lower MSE, lower PDE loss")
    print("   - Function evolution shows smooth convergence to linear fit")
    print("   - Alpha visualization shows consistent improvement across epochs")
    print("\n2. CUBIC TARGET: Trade-off → lower PDE loss, higher MSE")
    print("   - With λ=0: Model fits cubic curve (high curvature)")
    print("   - With λ>0: Model becomes more linear (can't fit cubic target)")
    print("   - Alpha visualization clearly shows linearization process")
    print("\n3. Regularization weight λ controls the trade-off strength")
    print("=" * 80)


if __name__ == "__main__":
    main()
