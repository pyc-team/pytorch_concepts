import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_concepts as pyc

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


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class LinearMLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc1(x)


def linear_pde_loss(y_pred, x):
    dy_dx = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,  # This allows us to differentiate again
        retain_graph=True  # This keeps the graph for the final loss.backward()
    )[0]
    # If dy_dx is None or doesn't depend on x, grad will fail
    d2y_dx2 = torch.autograd.grad(
        outputs=dy_dx,
        inputs=x,
        grad_outputs=torch.ones_like(dy_dx),
        create_graph=True,
        allow_unused=True  # This stops the error
    )[0]

    if d2y_dx2 is None:
        return torch.tensor(0.0, requires_grad=True)

    return torch.mean(d2y_dx2**2)


def main():
    os.makedirs("./symmetryIII", exist_ok=True)

    # create x as batch x 1 tensor of values from -3 to 3 and y as a very wiggly 8th degree polynomial of x with all powers from 0 to 8
    x = torch.linspace(-3, 3, 100).unsqueeze(1).requires_grad_(True)
    y = x**3 + 1.5 * torch.randn_like(x)

    criterion = torch.nn.MSELoss()
    epochs = 1000
    method_ids = ["constrained_mlp", "mlp", "architectural_mlp"]
    method_names = ["MLP with Interpretable Learning", "MLP", "MLP with Interpretable Architecture"]
    outputs_dict = {"constrained_mlp": [], "mlp": [], "architectural_mlp": []}

    for method_id, method_name in zip(method_ids, method_names):
        if method_id == "constrained_mlp":
            constraint_regularization = torch.logspace(-4, 1, steps=20).tolist()
        else:
            constraint_regularization = [0.0]

        if method_id == "architectural_mlp":
            model = LinearMLP(input_size=1, output_size=1)
        else:
            model = MLP(input_size=1, hidden_size=200, output_size=1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for regularization in constraint_regularization:
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(x)
                pde_term = linear_pde_loss(outputs, x)
                loss = criterion(outputs, y)

                if method_id == "constrained_mlp":
                    loss += regularization * pde_term

                loss.backward(retain_graph=True)
                optimizer.step()

                if epoch % 100 == 0:
                    print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, PDE Loss: {pde_term.item():.4f}')

            outputs_dict[method_id].append(outputs)

    colors = ['green', 'black', 'orange']
    plt.figure(figsize=(8, 5))
    plt.scatter(x.detach().numpy(), y.detach().numpy(), label='Data Points', color='blue', alpha=0.2)
    for method_id, method_name, color in zip(method_ids, method_names, colors):
        for i, outputs in enumerate(outputs_dict[method_id]):
            alpha = .2 if method_id == "constrained_mlp" else 1.0
            label = method_name if i == 0 else None
            linestyle = 'dotted' if method_id == "mlp" else '-'
            plt.plot(x.detach().numpy(), outputs.detach().numpy(), label=label, color=color, alpha=alpha,
                     linewidth=2, linestyle=linestyle)
            plt.legend()
    plt.xlabel(r'$c$')
    plt.ylabel(r'$y$')
    plt.tight_layout()
    plt.savefig('symmetryIII/symmetryIII_fit.png')
    plt.savefig('symmetryIII/symmetryIII_fit.pdf')
    plt.show()

    return


if __name__ == "__main__":
    main()
