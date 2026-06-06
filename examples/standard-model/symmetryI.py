import os
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
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
    "xtick.labelsize": 11,          # Size for the tick numbers
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
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CumulativeMLP(torch.nn.Module):
    def __init__(self, x, y):
        super(CumulativeMLP, self).__init__()
        self.params = torch.nn.Embedding(y.shape[1], y.shape[0])
        self.sorted_indices = torch.argsort(y.squeeze(), descending=False)
        self.prototypes = x[self.sorted_indices]

    def forward(self, x):
        positive_scores = torch.relu(self.params.weight) + 0.01
        cumulative_output = torch.cumsum(positive_scores, dim=1)

        # compute similarity using softmax between input x and prototypes
        similarity = F.softmax(-torch.cdist(x, self.prototypes) / 0.00001, dim=1)

        # weighted sum of cumulative output using similarity as weights
        output = similarity @ cumulative_output.t()

        return output


def order_loss(y_pred, y):
    y_pred = y_pred.view(-1, 1)
    y = y.view(-1, 1)

    # All-pairs differences
    diff_pred = y_pred - y_pred.t()
    diff_y = y - y.t()

    # 1. Order Penalty: Enforce y_j > y_i => y_pred_j > y_pred_i
    # We use (diff_y > 0) to find pairs that should have a positive diff_pred
    order_mask = (diff_y > 0).float()
    # softplus(-diff_pred) is high if diff_pred <= 0
    order_loss = torch.nn.functional.softplus(-diff_pred) * order_mask

    # Combine losses
    total_loss = order_loss.sum()

    # Normalize by number of pairs (excluding diagonal)
    num_pairs = y.size(0) * (y.size(0) - 1)
    return total_loss / num_pairs if num_pairs > 0 else y_pred.sum() * 0


def order_metric(y, y_pred):
    sorted_indices = torch.argsort(y.squeeze())
    diff_y = torch.diff(y_pred[sorted_indices], dim=0)
    # We want diff_outputs > 0.
    # This counts how many pairs are in the correct order.
    correct_order = (diff_y > 0).sum().item()
    total_pairs = len(y) - 1
    return correct_order / total_pairs if total_pairs > 0 else 1.0


def main():
    os.makedirs("symmetryI", exist_ok=True)

    folds = 5
    batch_size = 20
    input_size = 1

    # Define and train the MLP
    hidden_size = 200
    output_size = 1
    epochs = 3000
    criterion_list = [F.l1_loss, order_loss, "No loss"]
    lr = 1e-3

    model_names = ["MLP", "MLP+L", "MLP+A"]
    mae_scores = {"mlp": [], "constrained_mlp": [], "architectural_mlp": []}
    rank_scores = {"mlp": [], "constrained_mlp": [], "architectural_mlp": []}
    for fold in range(folds):

        # Generate synthetic data for each fold
        x = torch.randn(batch_size, input_size)
        y = x + torch.randn(batch_size, input_size)
        x_domain = torch.linspace(-3, 3, 100).unsqueeze(1)

        for model_id, model_name, criterion in zip(mae_scores.keys(), model_names, criterion_list):
            print(f"Fold {fold+1}, Model: {model_name}")

            if model_id != "architectural_mlp":
                model = MLP(input_size, hidden_size, output_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    if epoch % 10 == 0:
                        mae_score = F.l1_loss(outputs, y).item()
                        rank_score = order_metric(y, outputs)
                        print(f'Epoch [{epoch}/{epochs}], MAE: {mae_score:.4f}, Same Order Score: {rank_score:.4f}')

            else:
                model = CumulativeMLP(x, y)
                cumulative_pred = model(x)
                mae_score = F.l1_loss(cumulative_pred, y).item()
                rank_score = order_metric(y, cumulative_pred)

            mae_scores[model_id].append(mae_score)
            rank_scores[model_id].append(rank_score)

            # draw the MLP curve in the scatterplot
            with torch.no_grad():
                y_pred = model(x_domain)
                y_pred2 = model(x)

            plt.figure(figsize=(5, 5))
            plt.scatter(x.numpy(), y.numpy(), color='blue')
            plt.plot(x_domain.numpy(), y_pred.numpy(), color='red')
            plt.title(f"{model_name}")
            plt.xlabel(r'$z$')
            plt.ylabel(r'$c$')
            plt.tight_layout()
            plt.savefig(f"symmetryI/{model_id}_{fold}.png")
            plt.savefig(f"symmetryI/{model_id}_{fold}.pdf")
            plt.show()

            sorted_indices = torch.argsort(y.squeeze())
            y_sorted = y[sorted_indices]
            # pred_sorted_indices = torch.argsort(y_pred2.squeeze())
            y_pred_sorted = y_pred2[sorted_indices].detach()

            plt.figure(figsize=(5, 5))
            plt.scatter(torch.arange(0, len(sorted_indices)).numpy(), y_sorted.numpy(), color='blue')
            plt.plot(torch.arange(0, len(sorted_indices)), y_pred_sorted.numpy(), color='red')
            plt.title(f"{model_name}")
            plt.xlabel(r'sorted data indices')
            plt.ylabel(r'$c$')
            plt.tight_layout()
            plt.savefig(f"symmetryI/sorted_{model_id}_{fold}.png")
            plt.savefig(f"symmetryI/sorted_{model_id}_{fold}.pdf")
            plt.show()

    # create single dataframe with all results
    results = []
    model_id_to_name = {"mlp": "MLP", "constrained_mlp": "MLP+L", "architectural_mlp": "MLP+A"}
    for model_id in mae_scores.keys():
        for mae, rank, fold in zip(mae_scores[model_id], rank_scores[model_id], range(folds)):
            results.append({"model": model_id_to_name[model_id], "mae": mae, "rank_score": rank, "fold": fold})

    df = pd.DataFrame(results)

    # Melt the full dataframe to keep all fold entries for CI calculation
    df_melted_mae = df.melt(id_vars=['model', 'fold'], value_vars=['mae'],
                        var_name='metric', value_name='value')
    df_melted_rank = df.melt(id_vars=['model', 'fold'], value_vars=['rank_score'],
                        var_name='metric', value_name='value')

    plt.figure(figsize=(3, 3))
    sns.barplot(data=df_melted_mae, x='model', y='value', errorbar=('ci', 95), capsize=.1)
    plt.ylabel(r'MAE')
    plt.xlabel(r'Model')
    plt.tight_layout()
    plt.savefig(f"symmetryI/mae.png")
    plt.savefig(f"symmetryI/mae.pdf")
    plt.show()

    plt.figure(figsize=(3, 3))
    sns.barplot(data=df_melted_rank, x='model', y='value', errorbar=('ci', 95), capsize=.1)
    plt.ylabel('Rank Score')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(f"symmetryI/rank.png")
    plt.savefig(f"symmetryI/rank.pdf")
    plt.show()

    df.to_csv(f"symmetryI/raw_results.csv")

    return



if __name__ == "__main__":
    main()
