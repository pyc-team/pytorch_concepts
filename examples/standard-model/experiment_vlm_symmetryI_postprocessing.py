"""
Post-processing: compare VLM pairwise predictions against ground truth.

CSV format (one row per comparison):
  model, image_a, image_b, prediction

Ground truth: image with higher rank number is more red.
  e.g. mnist_digit_05.png vs mnist_digit_12.png -> GT winner is B (rank 12 > 5)

Metric: fraction of pairs where model prediction matches ground truth.
"""

import re
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from itertools import permutations, combinations

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


def extract_gt_rank(filename: str) -> int:
    """Extract the ground-truth rank (1-20) from the image filename."""
    match = re.search(r"mnist_digit_(\d+)", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract rank from filename: {filename}")


def gt_winner(image_a: str, image_b: str) -> str:
    """Return 'A' if image_a is more red, 'B' if image_b is more red."""
    return "A" if extract_gt_rank(image_a) > extract_gt_rank(image_b) else "B"


def pairwise_accuracy(group: pd.DataFrame) -> tuple[float, int, int]:
    """
    For each row, check if model prediction matches ground truth winner.
    Returns (accuracy, correct_count, total_count).
    Rows where prediction is None (parse failure) are skipped.
    """
    valid = group.dropna(subset=["prediction"])
    if len(valid) == 0:
        return 0.0, 0, 0
    correct = sum(
        row["prediction"] == gt_winner(row["image_a"], row["image_b"])
        for _, row in valid.iterrows()
    )
    return correct / len(valid), correct, len(valid)


def compute_win_ranks(image_files: list[str], pairs_df: pd.DataFrame, col: str) -> dict[str, int]:
    """
    Given a dataframe of pairwise comparisons in `col` (values 'A' or 'B'),
    count wins per image and return a dict {image: rank} (rank 1 = most wins).
    """
    wins = {img: 0 for img in image_files}
    for _, row in pairs_df.iterrows():
        if row[col] == "A":
            wins[row["image_a"]] += 1
        elif row[col] == "B":
            wins[row["image_b"]] += 1
    # rank: highest wins = rank 1
    sorted_imgs = sorted(wins, key=lambda x: wins[x], reverse=True)
    return {img: rank + 1 for rank, img in enumerate(sorted_imgs)}


def slope_chart(ranks_left: dict, ranks_right: dict,
                label_left: str, label_right: str,
                title: str, ax: plt.Axes,
                gt_ranks: dict | None = None) -> None:
    """
    Draw a slope chart on `ax` comparing two rankings.
    Each image is a line from its left rank to its right rank.
    Lines are colored and labelled by ground-truth rank position.
    """
    image_files = list(ranks_left.keys())
    n = len(image_files)
    gt_ranks_arr = np.array([extract_gt_rank(img) for img in image_files], dtype=float)
    norm = (gt_ranks_arr - gt_ranks_arr.min()) / (gt_ranks_arr.max() - gt_ranks_arr.min())
    colors = cm.get_cmap("RdYlGn_r")(norm)

    # Label = GT rank position (1 = least red, 20 = most red)
    sorted_by_gt = sorted(image_files, key=extract_gt_rank)
    gt_position = {img: pos + 1 for pos, img in enumerate(sorted_by_gt)}

    for img, color in zip(image_files, colors):
        rl = ranks_left[img]
        rr = ranks_right[img]
        lbl = str(gt_position[img])
        ax.plot([0, 1], [rl, rr], color=color, linewidth=1.5, alpha=0.85, solid_capstyle="round")
        ax.text(-0.08, rl, lbl, ha="right", va="center", fontsize=7, color=color)
        ax.text(1.08, rr, lbl, ha="left",  va="center", fontsize=7, color=color)

    # Extra padding top and bottom so labels are never clipped
    ax.set_ylim(n + 1.2, -0.2)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_left, label_right], fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, pad=24)


def create_pairwise_heatmap(entity_name: str, predictions: dict, image_files: list, image_folder: str) -> None:
    """
    Create a heatmap matrix where:
    - Rows and columns are images (sorted by GT rank)
    - Red color means row image < column image (A < B)
    - Green color means row image > column image (A > B)
    - Image thumbnails are used as tick labels
    - Only upper triangular matrix is shown
    """
    # Sort images by ground truth rank for consistent ordering
    image_files_sorted = sorted(image_files, key=extract_gt_rank)
    n = len(image_files_sorted)

    # Initialize matrix: 0 = no comparison, 1 = A wins, -1 = B wins
    matrix = np.zeros((n, n))

    # Fill the matrix based on predictions
    for (img_a, img_b), prediction in predictions.items():
        if prediction not in ['A', 'B']:
            continue

        try:
            i = image_files_sorted.index(img_a)  # row
            j = image_files_sorted.index(img_b)  # column

            if prediction == 'A':  # img_a wins (row > column)
                matrix[i, j] = 1
            else:  # img_b wins (column > row)
                matrix[i, j] = -1

        except ValueError:
            continue  # Skip if image not in sorted list

    # Mask lower triangular matrix (including diagonal)
    mask = np.eye(matrix.shape[0], dtype=bool)
    matrix_masked = np.ma.masked_where(mask, matrix)

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(4, 4))

    plt.title(rf'{entity_name.capitalize()}', fontsize=25)

    # Create custom colormap: red for negative (A < B), green for positive (A > B)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['red', 'white', 'green']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    cmap.set_bad('lightgray')  # Color for masked (lower triangle) areas

    # Plot heatmap
    im = ax.imshow(matrix_masked, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

    # Set up basic ticks and labels (will be replaced with images)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([''] * n)
    ax.set_yticklabels([''] * n)

    # Add axis labels
    ax.set_xlabel(r'B', fontsize=20, fontweight='bold', labelpad=20)
    ax.set_ylabel(r'A', fontsize=20, fontweight='bold', labelpad=20)

    # Add text annotations to cells
    for i in range(n):
        for j in range(n):
            # Only annotate upper triangular matrix
            if i != j:  # All cells except diagonal
                value = matrix[i, j]
                if value != 0:  # Only annotate non-zero cells
                    if value > 0:  # A wins (row > column)
                        text = r'A'
                    else:  # B wins (column > row)
                        text = r'B'
                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=16, fontweight='bold', color='black')

    # Remove title as requested

    # Add image thumbnails as tick labels with larger size
    thumb_size = 0.08  # Increased from 0.03
    ax_pos = ax.get_position()

    for i, img_name in enumerate(image_files_sorted):
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            img = plt.imread(img_path)

            # Calculate exact tick positions to align thumbnails
            # Matrix pixels are centered at integer coordinates, so we need to map properly
            # The matrix goes from -0.5 to n-0.5 in data coordinates

            # For a matrix of size n x n, the actual data extent is from -0.5 to n-0.5
            # But the visible ticks are at 0, 1, 2, ..., n-1
            # We need to convert tick position i to the correct relative position

            if n > 1:
                # Map tick i to position within the axes
                tick_x_rel = (i + 0.5) / n  # +0.5 centers on the pixel
                tick_y_rel = (i + 0.5) / n  # +0.5 centers on the pixel
            else:
                tick_x_rel = 0.5
                tick_y_rel = 0.5

            # Convert to figure coordinates
            x_fig = ax_pos.x0 + tick_x_rel * ax_pos.width
            y_fig = ax_pos.y0 + (1 - tick_y_rel) * ax_pos.height  # Invert Y for matrix coordinates

            # X-axis thumbnails (bottom) - aligned with matrix columns
            y_bottom = ax_pos.y0 - thumb_size - 0.01
            inset_x = fig.add_axes([x_fig - thumb_size/2, y_bottom, thumb_size, thumb_size])
            inset_x.imshow(img)
            inset_x.axis('off')

            # Y-axis thumbnails (left) - aligned with matrix rows
            x_left = ax_pos.x0 - thumb_size - 0.01
            inset_y = fig.add_axes([x_left, y_fig - thumb_size/2, thumb_size, thumb_size])
            inset_y.imshow(img)
            inset_y.axis('off')

        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue

    # Save the figure
    safe_name = entity_name.replace(' ', '_').replace('/', '_')
    plt.savefig(f'vlm_experiment/{safe_name}_heatmap.pdf', bbox_inches='tight')  # Increased DPI
    plt.savefig(f'vlm_experiment/{safe_name}_heatmap.png', bbox_inches='tight')  # Increased DPI
    plt.close()


def create_prototype_distances_heatmap(csv_path: str, train_image_folder: str, test_image_folder: str) -> None:
    """
    Create a heatmap from prototype distances CSV where:
    - Rows and columns are images from the CSV index/headers
    - Image thumbnails are used as tick labels on both axes
    - Color represents distance values (darker = closer, lighter = farther)
    """
    # Load the CSV
    df = pd.read_csv(csv_path, index_col=0)

    # Get image names from index and columns
    row_images = df.index.tolist()
    col_images = df.columns.tolist()

    # Convert to numpy array for plotting
    distance_matrix = df.values

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # plt.title('Prototype Distances Heatmap', fontsize=16, pad=20)

    # Create heatmap with custom colormap (lighter = farther distance)
    im = ax.imshow(distance_matrix, cmap='viridis_r', aspect='equal')

    # Set up basic ticks
    n_rows = len(row_images)
    n_cols = len(col_images)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([''] * n_cols)
    ax.set_yticklabels([''] * n_rows)

    # Add image thumbnails as tick labels
    thumb_size = 0.06
    ax_pos = ax.get_position()

    # X-axis thumbnails (bottom) - columns from train_data
    for j, img_name in enumerate(col_images):
        img_path = os.path.join(train_image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            img = plt.imread(img_path)

            if n_cols > 1:
                tick_x_rel = (j + 0.5) / n_cols
            else:
                tick_x_rel = 0.5

            x_fig = ax_pos.x0 + tick_x_rel * (ax_pos.width - 0.05)
            y_bottom = ax_pos.y0 - thumb_size - 0.02

            inset_x = fig.add_axes([x_fig - thumb_size/2, y_bottom, thumb_size, thumb_size])
            inset_x.imshow(img)
            inset_x.axis('off')

        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue

    # Y-axis thumbnails (left) - rows from test_data
    for i, img_name in enumerate(row_images):
        img_path = os.path.join(test_image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            img = plt.imread(img_path)

            if n_rows > 1:
                tick_y_rel = (i + 0.5) / n_rows
            else:
                tick_y_rel = 0.5

            y_fig = ax_pos.y0 + (1 - tick_y_rel) * ax_pos.height
            x_left = ax_pos.x0 - thumb_size - 0.02

            inset_y = fig.add_axes([x_left, y_fig - thumb_size/2, thumb_size, thumb_size])
            inset_y.imshow(img)
            inset_y.axis('off')

        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


    # Add colorbar with same height as heatmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('distance', rotation=270, labelpad=-1)

    # Set only min and max ticks
    vmin, vmax = im.get_clim()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['min', 'max'])

    # Add axis labels
    ax.set_xlabel(r'moonstream2_proto prototypes', fontsize=20, fontweight='bold', labelpad=40)
    ax.set_ylabel(r'test samples', fontsize=20, fontweight='bold', labelpad=40)

    # Save the figure
    plt.savefig('vlm_experiment/prototype_distances_heatmap.pdf', bbox_inches='tight')
    plt.savefig('vlm_experiment/prototype_distances_heatmap.png', bbox_inches='tight')
    plt.close()


def main():
    df = pd.read_csv("./vlm_experiment/results.csv")

    print("=== Per-model accuracy vs ground truth ===")
    print(f"{'Model':<20} {'Accuracy':>9}  {'Correct':>8}  {'Total':>7}")
    print("-" * 50)

    model_results = {}
    for model, group in df.groupby("model"):
        acc, correct, total = pairwise_accuracy(group)
        model_results[model] = {"acc": acc, "group": group}
        print(f"{model:<20} {acc:>9.3f}  {correct:>8}  {total:>7}")

    # --- Cross-model agreement ---
    model_names = list(model_results.keys())
    if len(model_names) >= 2:
        print()
        print("=== Cross-model pairwise agreement ===")
        print(f"{'Model A':<20} {'Model B':<20} {'Agreement':>10}  {'Matching':>9}  {'Total':>7}")
        print("-" * 70)

        # Pivot so we can compare predictions on the same pairs
        pivot = df.pivot_table(
            index=["image_a", "image_b"], columns="model", values="prediction", aggfunc="first"
        ).reset_index()

        for m1, m2 in permutations(model_names, 2):
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            valid = pivot.dropna(subset=[m1, m2])
            if len(valid) == 0:
                continue
            matching = (valid[m1] == valid[m2]).sum()
            agreement = matching / len(valid)
            print(f"{m1:<20} {m2:<20} {agreement:>10.3f}  {matching:>9}  {len(valid):>7}")

    # --- Ranking summary ---
    print()
    print("=== Ranking by accuracy vs ground truth ===")
    ranked = sorted(model_results.items(), key=lambda x: x[1]["acc"], reverse=True)
    for rank, (model, data) in enumerate(ranked, start=1):
        print(f"  #{rank}  {model:<20}  accuracy = {data['acc']:.3f}")

    # --- Confusion matrices ---
    image_files = sorted(set(df["image_a"].tolist() + df["image_b"].tolist()))

    # Build prediction series per entity, indexed by (image_a, image_b)
    # Ground truth
    all_pairs = list(df[["image_a", "image_b"]].drop_duplicates().itertuples(index=False, name=None))
    gt_preds = {(a, b): gt_winner(a, b) for a, b in all_pairs}

    entity_preds = {"Ground Truth": gt_preds}
    for model in model_names:
        mdf = df[df["model"] == model].set_index(["image_a", "image_b"])["prediction"]
        entity_preds[str(model)] = mdf.to_dict()

    entity_names = list(entity_preds.keys())
    pairs_to_plot = list(combinations(entity_names, 2))
    n_plots = len(pairs_to_plot)

    # --- Ranking line plot with image thumbnails on x-axis ---
    image_folder = os.path.join(os.path.dirname(__file__), "test_data")

    # --- NEW: Generate heatmaps for each model and ground truth ---
    print("\n=== Generating pairwise comparison heatmaps ===")

    # Create heatmap for ground truth
    create_pairwise_heatmap("Ground Truth", gt_preds, image_files, image_folder)

    # Create heatmap for each model
    for model in model_names:
        model_preds = entity_preds[str(model)]
        create_pairwise_heatmap(str(model), model_preds, image_files, image_folder)

    # Sort images by GT rank (x-axis order)
    image_files_sorted = sorted(image_files, key=extract_gt_rank)
    n_imgs = len(image_files_sorted)
    x = np.arange(n_imgs)

    # Recompute model win-ranks (already done above but scope-safe here)
    all_entity_ranks = {"Ground Truth": {img: extract_gt_rank(img) for img in image_files}}
    for model in model_names:
        mdf = df[df["model"] == model][["image_a", "image_b", "prediction"]].copy()
        mdf = mdf.rename(columns={"prediction": str(model)})
        all_entity_ranks[str(model)] = compute_win_ranks(image_files, mdf, str(model))

    fig2, ax = plt.subplots(figsize=(8, 4))

    colors_lines = ["black", "tab:blue", "tab:red", "tab:green", "tab:orange"]
    for (entity, ranks), color in zip(all_entity_ranks.items(), colors_lines):
        ys = [ranks[img] for img in image_files_sorted]
        if entity != "Ground Truth":
            ax.plot(x, ys, marker="o", markersize=5, linewidth=2,
                    label=rf"{entity}", color=color)

    ax.set_xlim(-0.5, n_imgs - 0.5)
    ax.set_ylabel("Rank (1 = most red)", fontsize=14)
    # ax.set_title("Ranking per image (x = GT order, y = assigned rank)", fontsize=11)
    ax.legend(fontsize=13)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.invert_yaxis()  # rank 1 at top

    # Add thumbnail images as x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n_imgs)  # hide default labels

    thumb_size = 0.06   # axes fraction
    y_offset = -0.01    # just below x-axis in axes coords
    for i, img_name in enumerate(image_files_sorted):
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue
        img = plt.imread(img_path)
        # Convert axes x position to figure coords
        ax_pos = ax.get_position()
        x_fig = ax_pos.x0 + (i / (n_imgs - 1)) * ax_pos.width + 0.005
        y_fig = ax_pos.y0 - thumb_size - 0.08
        inset = fig2.add_axes([x_fig - thumb_size / 2, y_fig, thumb_size, thumb_size])
        inset.imshow(img)
        inset.axis("off")

    plt.tight_layout()
    fig2.savefig('vlm_experiment/ranking.pdf', bbox_inches="tight")
    fig2.savefig('vlm_experiment/ranking.png', bbox_inches="tight")


    print("\n=== Generating prototype distances heatmap ===")
    prototype_csv_path = "vlm_experiment/prototype_distances.csv"
    if os.path.exists(prototype_csv_path):
        # Use both folders: train_data for columns (X-axis) and test_data for rows (Y-axis)
        train_image_folder = os.path.join(os.path.dirname(__file__), "train_data")
        test_image_folder = os.path.join(os.path.dirname(__file__), "test_data")
        create_prototype_distances_heatmap(prototype_csv_path, train_image_folder, test_image_folder)
        print("Prototype distances heatmap saved to vlm_experiment/prototype_distances_heatmap.pdf/png")
    else:
        print(f"Warning: {prototype_csv_path} not found, skipping prototype distances heatmap")


if __name__ == "__main__":
    main()
