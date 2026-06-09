import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 30,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 14,
    "axes.edgecolor": "gray",
})

results_file = "./clip_experiment/results.csv"
image_folder = "./test_data"
output_file = "./clip_experiment/clip_ranking.pdf"
heatmap_output = "./clip_experiment/clip_ranking_heatmap.pdf"

# Load results
df = pd.read_csv(results_file)
df = df.sort_values("image").reset_index(drop=True)
df["rank"] = df["dot_product"].rank().astype(int)

n = len(df)
fig, ax = plt.subplots(figsize=(max(n * 1.2, 8), 5))

# Palette from black to red, intensity proportional to rank
norm = mcolors.Normalize(vmin=df["rank"].min(), vmax=df["rank"].max())
cmap = mcolors.LinearSegmentedColormap.from_list("black_to_red", ["black", "red"])
palette = [cmap(norm(v)) for v in df["rank"]]
sns.barplot(data=df, x="image", y="rank", order=df["image"].tolist(),
            palette=palette, ax=ax)

ax.set_xlabel("")
ax.set_ylabel("CLIP ranking (\\texttt{red})")
ax.set_xticks([])

# Place images below each bar as custom x-tick labels
for i, fname in enumerate(df["image"]):
    img = mpimg.imread(os.path.join(image_folder, fname))
    imagebox = OffsetImage(img, zoom=0.6)
    ab = AnnotationBbox(
        imagebox, (i, 0),
        xycoords=("data", "axes fraction"),
        box_alignment=(0.5, 1.1),
        frameon=False,
        annotation_clip=False,
    )
    ax.add_artist(ab)

min_y_axis = df["rank"].min() - 0.1
max_y_axis = df["rank"].max() + 0.1
# set only minimum of y-axis to ensure all bars are visible and there's some space below the lowest bar
ax.set_ylim(min_y_axis, max_y_axis)
plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
plt.show()
print(f"Saved to {output_file}")

def create_pairwise_heatmap_from_ranks(df: pd.DataFrame, image_folder: str, output_path: str) -> None:
    """
    Create a pairwise heatmap where each cell (i, j) shows which image is
    ranked higher by CLIP. A = row image is more red, B = column image is more red.
    Images sorted by lexicographic filename order (same as barplot).
    """
    image_files_sorted = df.sort_values("image")["image"].tolist()
    rank_map = dict(zip(df["image"], df["rank"]))
    n = len(image_files_sorted)

    matrix = np.zeros((n, n))
    for i, img_a in enumerate(image_files_sorted):
        for j, img_b in enumerate(image_files_sorted):
            if i == j:
                continue
            if rank_map[img_a] > rank_map[img_b]:
                matrix[i, j] = 1   # A wins
            else:
                matrix[i, j] = -1  # B wins

    mask = np.eye(matrix.shape[0], dtype=bool)
    matrix_masked = np.ma.masked_where(mask, matrix)

    fig, ax = plt.subplots(figsize=(4, 4))
    plt.title("CLIP", fontsize=25)

    colors = ["red", "white", "green"]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
    cmap.set_bad("lightgray")

    ax.imshow(matrix_masked, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([""] * n)
    ax.set_yticklabels([""] * n)
    ax.set_xlabel(r"B", fontsize=20, fontweight="bold", labelpad=20)
    ax.set_ylabel(r"A", fontsize=20, fontweight="bold", labelpad=20)

    for i in range(n):
        for j in range(n):
            if i != j:
                value = matrix[i, j]
                if value != 0:
                    text = "A" if value > 0 else "B"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=16, fontweight="bold", color="black")

    thumb_size = 0.08
    ax_pos = ax.get_position()

    for i, img_name in enumerate(image_files_sorted):
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue
        img = plt.imread(img_path)
        tick_rel = (i + 0.5) / n
        x_fig = ax_pos.x0 + tick_rel * ax_pos.width
        y_fig = ax_pos.y0 + (1 - tick_rel) * ax_pos.height

        inset_x = fig.add_axes([x_fig - thumb_size / 2, ax_pos.y0 - thumb_size - 0.01, thumb_size, thumb_size])
        inset_x.imshow(img)
        inset_x.axis("off")

        inset_y = fig.add_axes([ax_pos.x0 - thumb_size - 0.01, y_fig - thumb_size / 2, thumb_size, thumb_size])
        inset_y.imshow(img)
        inset_y.axis("off")

    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_path}")

create_pairwise_heatmap_from_ranks(df, image_folder, heatmap_output)
