#!/usr/bin/env python
"""Visualize analysis results as tables and paper-ready plotly bar plots.

Usage:
    python visualize.py
"""

import re
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent / "results"

# ── Style constants ──────────────────────────────────────────────────────────
MODEL_LABELS = {
    "conceptbottleneckmodel": "CBM",
    "conceptembeddingmodel": "CEM",
    "causallyreliableconceptbottleneckmodel": "C2BM",
}
MODEL_ORDER = ["CBM", "CEM", "C2BM"]

METRIC_LABELS = {"accuracy": "Accuracy", "mcc": "MCC"}

DATASET_LABELS = {"asia": "Asia", "sachs": "Sachs", "insurance": "Insurance"}
DATASET_ORDER = ["Asia", "Sachs", "Insurance"]

COLORS = {
    "CBM": "#4C78A8",
    "CEM": "#E45756",
    "C2BM": "#AA0AAD",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_mean_sem(cell: str):
    """Parse a 'mean ± sem' string into (mean, sem) floats."""
    m = re.match(r"([\d.eE+-]+|nan)\s*±\s*([\d.eE+-]+|nan)", str(cell).strip())
    if m is None:
        return float("nan"), float("nan")
    mean = float(m.group(1))
    sem = float(m.group(2))
    return mean, sem


def load_analysis_csv(name: str) -> pd.DataFrame:
    """Read an analysis CSV with its multi-level header and row index."""
    path = BASE_DIR / f"analysis_{name}.csv"
    return pd.read_csv(path, header=[0, 1], index_col=[0, 1, 2])


def build_numeric_tables(df: pd.DataFrame):
    """Split a formatted DataFrame into separate mean / sem DataFrames.

    Only keeps the 'mean' concept columns (the aggregate per-metric results).
    """
    mean_vals, sem_vals = {}, {}
    for metric in df.columns.get_level_values(0).unique():
        if ("mean",) == df[metric].columns.tolist():
            col = (metric, "mean")
        elif "mean" in df[metric].columns:
            col = (metric, "mean")
        else:
            continue
        means, sems = [], []
        for val in df[col]:
            m, s = parse_mean_sem(val)
            means.append(m)
            sems.append(s)
        mean_vals[metric] = means
        sem_vals[metric] = sems

    idx = df.index
    df_mean = pd.DataFrame(mean_vals, index=idx)
    df_sem = pd.DataFrame(sem_vals, index=idx)
    return df_mean, df_sem


def pretty_index(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw class names with human-readable labels."""
    new_idx = []
    for row in df.index:
        dataset = DATASET_LABELS.get(row[0], row[0])
        model = MODEL_LABELS.get(row[1], row[1])
        new_idx.append((dataset, model))
    df = df.copy()
    df.index = pd.MultiIndex.from_tuples(new_idx, names=["Dataset", "Model"])
    df.columns = [METRIC_LABELS.get(c, c) for c in df.columns]
    return df


# ── Tables ───────────────────────────────────────────────────────────────────
def print_table(group: str):
    """Print a clean table for a given analysis group."""
    df = load_analysis_csv(group)
    df_mean, df_sem = build_numeric_tables(df)
    df_pretty = pretty_index(df_mean)
    # Format with ± inline
    df_sem_pretty = pretty_index(df_sem)
    formatted = df_pretty.combine(
        df_sem_pretty,
        lambda m, s: m.map(lambda v: f"{v:.4f}") + " ± " + s.map(lambda v: f"{v:.4f}"),
    )
    print(f"  {group.upper()} METRICS  (mean \u00b1 sem over seeds)\n{formatted.to_string()}")
    return df_mean, df_sem


# ── Plotly bar chart ─────────────────────────────────────────────────────────
def bar_chart(group: str, df_mean: pd.DataFrame, df_sem: pd.DataFrame):
    """Create a grouped bar chart: one subplot per metric, bars = models, groups = datasets."""
    metrics = list(df_mean.columns)
    n_metrics = len(metrics)

    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        shared_yaxes=False,
        horizontal_spacing=0.05,
    )

    for col_i, metric in enumerate(metrics, start=1):
        for model in MODEL_ORDER:
            means, sems, datasets = [], [], []
            for dataset in DATASET_ORDER:
                mask = (df_mean.index.get_level_values(0) == dataset.lower()) & (
                    df_mean.index.get_level_values(1).map(
                        lambda x: MODEL_LABELS.get(x, x)
                    )
                    == model
                )
                if mask.any():
                    means.append(df_mean.loc[mask, metric].values[0])
                    sems.append(df_sem.loc[mask, metric].values[0])
                    datasets.append(dataset)

            fig.add_trace(
                go.Bar(
                    name=model,
                    x=datasets,
                    y=means,
                    error_y=dict(type="data", array=sems, visible=True),
                    marker_color=COLORS[model],
                    showlegend=(col_i == 1),
                    legendgroup=model,
                ),
                row=1,
                col=col_i,
            )

    fig.update_layout(
        barmode="group",
        title_text=f"{group.capitalize()} Prediction Performance",
        title_x=0.5,
        font=dict(family="Times New Roman", size=14),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        width=450 * n_metrics,
        height=450,
        margin=dict(l=60, r=30, t=60, b=100),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showline=True, linecolor="black", mirror=True)
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        mirror=True,
        gridcolor="lightgrey",
    )
    # Label each subplot y-axis with the metric name and clip to its domain
    for col_i, metric in enumerate(metrics, start=1):
        label = METRIC_LABELS.get(metric, metric)
        # Compute data-driven range for this metric
        vals = df_mean[metric].dropna().values
        lo = max(0, float(vals.min()) - 0.1) if len(vals) else 0.0
        hi = min(1.02, float(vals.max()) + 0.1) if len(vals) else 1.02
        fig.update_yaxes(
            title_text=label, range=[lo, hi],
            constrain="domain", row=1, col=col_i,
        )

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    out = BASE_DIR / f"plot_{group}.pdf"
    fig.write_image(str(out), format="pdf")

    out_html = BASE_DIR / f"plot_{group}.html"
    fig.write_html(str(out_html))

    return fig


# ── Intervention line plots ──────────────────────────────────────────────────
MARKERS = {"CBM": "square", "CEM": "triangle-up", "C2BM": "circle"}
LINE_STYLES = {"CBM": "solid", "CEM": "dash", "C2BM": "dot"}


def intervention_plot(kind: str, y_col: str, y_label: str, title: str, filename: str):
    """Line plot: accuracy vs % concepts intervened, one line per model, faceted by dataset.

    Parameters
    ----------
    kind : str
        'task' or 'concept' — selects which accuracy column.
    """
    path = BASE_DIR / "analysis_interventional.csv"
    if not path.exists():
        logger.debug("Skipping intervention plot (%s): %s not found", kind, path)
        return

    df = pd.read_csv(path)

    # Rename for pretty labels
    df["dataset"] = df["dataset.name"].map(lambda x: DATASET_LABELS.get(x, x))
    df["model"] = df["model._target_"].map(lambda x: MODEL_LABELS.get(x, x))

    datasets_present = [d for d in DATASET_ORDER if d in df["dataset"].values]
    n_datasets = len(datasets_present)
    if n_datasets == 0:
        logger.debug("No datasets found for intervention plot (%s)", kind)
        return

    fig = make_subplots(
        rows=1,
        cols=n_datasets,
        shared_yaxes=True,
        subplot_titles=datasets_present,
        horizontal_spacing=0.06,
    )

    mean_col = f"{y_col}_mean"
    sem_col = f"{y_col}_sem"

    for col_i, dataset in enumerate(datasets_present, start=1):
        ds_df = df[df["dataset"] == dataset]
        for model in MODEL_ORDER:
            m_df = ds_df[ds_df["model"] == model].sort_values("step")
            if m_df.empty or mean_col not in m_df.columns:
                continue
            # Drop NaN rows
            valid = m_df.dropna(subset=[mean_col])
            if valid.empty:
                continue

            x_vals = valid["step"].astype(int).tolist()
            y_vals = (valid[mean_col] * 100).tolist()
            sem_vals = (valid[sem_col] * 100).tolist() if sem_col in valid.columns else None

            fig.add_trace(
                go.Scatter(
                    name=model,
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    line=dict(color=COLORS[model], dash=LINE_STYLES[model], width=2),
                    marker=dict(symbol=MARKERS[model], size=6),
                    error_y=dict(type="data", array=sem_vals, visible=True, thickness=1)
                    if sem_vals
                    else None,
                    showlegend=(col_i == 1),
                    legendgroup=model,
                ),
                row=1,
                col=col_i,
            )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        font=dict(family="Times New Roman", size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        width=400 * n_datasets,
        height=420,
        margin=dict(l=60, r=30, t=100, b=60),
        plot_bgcolor="white",
    )
    fig.update_xaxes(
        title_text="Number of Intervened Concepts",
        showline=True,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        mirror=True,
        gridcolor="lightgrey",
    )
    fig.update_yaxes(title_text=y_label, row=1, col=1)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    out = BASE_DIR / f"{filename}.pdf"
    fig.write_image(str(out), format="pdf")

    out_html = BASE_DIR / f"{filename}.html"
    fig.write_html(str(out_html))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    for group in ("concept", "task", "joint"):
        path = BASE_DIR / f"analysis_{group}.csv"
        if not path.exists():
            logger.debug("Skipping %s: %s not found", group, path)
            continue
        if group == "task":
            df_mean, df_sem = print_table(group)
        else:
            df = load_analysis_csv(group)
            df_mean, df_sem = build_numeric_tables(df)
        bar_chart(group, df_mean, df_sem)

    # Intervention plots
    intervention_plot(
        kind="task",
        y_col="task_accuracy",
        y_label="Task Accuracy (%)",
        title="Task Accuracy under Progressive Interventions",
        filename="plot_intervention_task",
    )
    intervention_plot(
        kind="joint",
        y_col="joint_accuracy",
        y_label="Joint Accuracy (%)",
        title="Joint Accuracy under Progressive Interventions",
        filename="plot_intervention_joint",
    )


if __name__ == "__main__":
    main()
