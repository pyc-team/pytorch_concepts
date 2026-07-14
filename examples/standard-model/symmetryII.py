import os
import torch
import torch.nn.functional as F
from matplotlib import colors
from torch.func import jacrev, vmap
import seaborn as sns
import torch_concepts as pyc
import matplotlib.pyplot as plt

pyc.seed_everything(42)

# Define your hex color
# dark_page = '#2C2C2C'

# Update Matplotlib configuration
plt.rcParams.update({
    "text.usetex": True,           # Enables external LaTeX rendering
    # 'figure.facecolor': dark_page,
    # 'axes.facecolor': dark_page,
    # 'savefig.facecolor': dark_page,
    # --- TEXT SIZE SETTINGS ---
    "font.size": 14,                # Base font size for all text
    "axes.titlesize": 18,           # Specifically for the title
    "axes.labelsize": 16,           # Specifically for X and Y labels
    # "xtick.labelsize": 12,          # Size for the tick numbers
    # "ytick.labelsize": 12,
    "legend.fontsize": 14,          # Size for the legend text
    # ---------------------------
    # 'text.color': 'white',
    # 'axes.labelcolor': 'white',
    # 'xtick.color': 'white',
    # 'ytick.color': 'white',
    'axes.edgecolor': 'gray'
})

# generate synthetic data
def generate_data(n_samples=1000):
    # generate random 2D points in [0, 1]^2
    x = torch.rand(n_samples, 2)
    c = ((x[:, 0] > 0.4) & (x[:, 1] < 0.5)).float().unsqueeze(1)
    y = (x[:, 0] * x[:, 1] > 0.3).float().unsqueeze(1)
    return x, c, y

def generate_4_points_in_2D():
    # Generate 4 points in 2D space
    points = torch.tensor([[0.7, 0.8],
                           [0.8, 0.7],
                           [0.2, 0.8],
                           [0.4, 0.4]])
    labels = torch.tensor([[1], [1], [0], [0]])  # Label based on the product of coordinates
    return points, labels.float()

# visualize the data with scatter plot
def scatter_xy(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y.squeeze(), cmap='coolwarm', edgecolor='k', zorder=3)
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

def countour_xy(model, x_range=(0, 1), y_range=(0, 1), resolution=100):
    x1 = torch.linspace(x_range[0], x_range[1], resolution)
    x2 = torch.linspace(y_range[0], y_range[1], resolution)
    x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')
    grid_points = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)

    with torch.no_grad():
        outputs = model(grid_points).reshape(resolution, resolution)

    return outputs

def countour_plot(outputs, z_test, g_f, g_phi, x_range=(0, 1), y_range=(0, 1), resolution=100):
    x1 = torch.linspace(x_range[0], x_range[1], resolution)
    x2 = torch.linspace(y_range[0], y_range[1], resolution)
    norm = colors.TwoSlopeNorm(vcenter=0, vmin=outputs.min(), vmax=outputs.max())
    plt.quiver(z_test[0, 0].item(), z_test[0, 1].item(), g_f[1], g_f[0], color='red', label=r"$\nabla f$", zorder=2)
    plt.quiver(z_test[0, 0].item(), z_test[0, 1].item(), g_phi[1], g_phi[0], color='black', label=r"$\nabla c$", zorder=2)
    plt.contourf(x1, x2, outputs.numpy(), levels=20, cmap='coolwarm', norm=norm, alpha=0.4, zorder=0)
    plt.colorbar()
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    plt.legend()


def _compute_jacobian(model, x):
    def model_fn(input_vec):
        return model(input_vec)

    jacobian_op = vmap(jacrev(model_fn))
    return jacobian_op(x)


def gradient_alignment_loss(g_c, g_y, normalize=True):
    # 2. Get Orthonormal Bases for the subspaces using QR decomposition
    # We transpose because we want the basis for the row-space (the gradients)
    # g_c.transpose(1, 2) is (Batch, dim_z, dim_out)
    q_c, r_c = torch.linalg.qr(g_c.transpose(1, 2))
    q_y, r_y = torch.linalg.qr(g_y.transpose(1, 2))

    # 3. Compute the Projection Matrices: P = Q @ Q.T
    # This matrix represents the subspace spanned by the gradients
    p_c = torch.bmm(q_c, q_c.transpose(1, 2))

    # 3. Project each row of J_f onto row_space(J_c)
    # g_y @ p_c: (B, n, m) @ (B, m, m) -> (B, n, m)
    g_y_proj = torch.bmm(g_y, p_c)

    # 4. Residual: the component of J_f orthogonal to row_space(J_c)
    # Equivalent to g_y @ (I - P_c)
    residual = g_y - g_y_proj  # (B, n, m)

    # 5. Loss: Frobenius norm of the residual
    loss = torch.linalg.matrix_norm(residual, ord='fro')  # (B,)

    # 6. Optional: normalize by ||J_f||_F so loss is in [0, 1]
    # 0 means row_space(J_f) ⊆ row_space(J_c) exactly
    # 1 means J_f is entirely orthogonal to row_space(J_c)
    if normalize:
        g_y_norm = torch.linalg.matrix_norm(g_y, ord='fro')  # (B,)
        loss = loss / (g_y_norm + 1e-8)

    return loss


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    os.makedirs('symmetryII', exist_ok=True)

    method_ids = ["mlp", "constrained_mlp", "architectural_mlp"]
    method_names = ["MLP", "MLP+L", "MLP+A"]

    g_cy_losses = {"MLP": [], "MLP+L": [], "MLP+A": []}
    for method_id, method_name in zip(method_ids, method_names):
        x, c, y = generate_data()
        x_train, y_train = generate_4_points_in_2D()
        x_eval = x_train[0:1].detach().clone().requires_grad_(True)

        c_model = MLP(input_size=x.shape[1], hidden_size=200, output_size=y.shape[1])
        y_model = MLP(input_size=x.shape[1], hidden_size=200, output_size=y.shape[1])

        if method_id == "architectural_mlp":
            y_model2 = MLP(input_size=c.shape[1], hidden_size=200, output_size=y.shape[1])
            y_model = torch.nn.Sequential(c_model, y_model2)

        optimizer = torch.optim.Adam(list(c_model.parameters()) + list(y_model.parameters()), lr=0.001)
        criterion = torch.nn.BCELoss()
        epochs = 1000

        c_model.train()
        y_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            c_pred = c_model(x)
            y_pred = y_model(x_train)

            loss_c = criterion(torch.sigmoid(c_pred), c)
            loss_y = criterion(torch.sigmoid(y_pred), y_train)

            g_c = _compute_jacobian(c_model, x_eval)
            g_y = _compute_jacobian(y_model, x_eval)

            g_cy_loss = gradient_alignment_loss(g_c, g_y)

            loss = loss_c + loss_y

            if method_id == "constrained_mlp":
                loss += 0.05 * g_cy_loss.mean()

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss c: {loss_c.item():.4f}, Loss y: {loss_y.item():.4f}, Gradient CY Loss: {g_cy_loss.mean().item():.4f}')

            g_cy_losses[method_name].append(g_cy_loss.mean().item())

        x_range = (0, 1)
        y_range = (0, 1)
        resolution = 1000
        x1 = torch.linspace(x_range[0], x_range[1], resolution)
        x2 = torch.linspace(y_range[0], y_range[1], resolution)
        g_c = torch.autograd.grad(c_model(x_eval).sum(), x_eval)[0][0].detach().numpy()
        g_y = torch.autograd.grad(y_model(x_eval).sum(), x_eval)[0][0].detach().numpy()
        c_out = countour_xy(c_model, x_range=(0, 1), y_range=(0, 1), resolution=1000)
        y_out = countour_xy(y_model, x_range=(0, 1), y_range=(0, 1), resolution=1000)

        # plot overlayed scatter + contour after training
        plt.figure(figsize=(7, 4))
        plt.title(f"{method_name}")
        countour_plot(y_out, x_eval, g_y, g_c, x_range=(0, 1), y_range=(0, 1), resolution=1000)
        plt.contour(x1, x2, c_out.detach().numpy(), levels=20, colors='black', alpha=0.3)
        scatter_xy(x_train, y_train)
        plt.tight_layout()
        plt.savefig(f'symmetryII/contour_{method_id}.png')
        plt.savefig(f'symmetryII/contour_{method_id}.pdf')
        plt.show()

        with torch.no_grad():
            x_evals = x_eval.repeat(500, 1) + 0.1 * torch.randn(500, 2)
            c_pred = c_model(x_evals).detach().numpy()
            y_pred = y_model(x_evals).detach().numpy()

            plt.figure(figsize=(7, 4))
            plt.scatter(c_pred, y_pred, c='orange', edgecolor='k', alpha=0.7, label='Functionally aligned', marker='d')
            plt.xlabel(r'Model $c$ (logits)')
            plt.ylabel(r'Model $f$ (logits)')
            # plt.title('Model vs Explanation')
            plt.tight_layout()
            plt.savefig(f'symmetryII/{method_id}_vs_explanation.png')
            plt.savefig(f'symmetryII/{method_id}_vs_explanation.pdf')
            plt.show()


    # create lineplot of gcy_losses across epochs for each method
    plt.figure(figsize=(5, 3))
    sns.lineplot(data=g_cy_losses)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Gradient Alignment Loss', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'symmetryII/gradient_alignment_comparison.png')
    plt.savefig(f'symmetryII/gradient_alignment_comparison.pdf')
    plt.show()

    return


if __name__ == "__main__":
    main()
