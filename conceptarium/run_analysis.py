#!/usr/bin/env python
"""Run post-hoc analysis on trained models tracked in the run registry.

Usage:
    # Evaluate all runs in the default CSV
    python run_analysis.py --config-name analysis

    # Use a custom CSV
    python run_analysis.py --config-name analysis csv_path=my_runs.csv

    # Filter to a subset
    python run_analysis.py --config-name analysis 'filters={dataset: asia, model: cbm}'
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd

from conceptarium.registry import load_registry
from conceptarium.evaluate import load_job, evaluate_job, job_metadata
from conceptarium.resolvers import register_custom_resolvers


def results_to_dataframes(all_results, meta_keys, seed_key="seed"):
    """Build one DataFrame per evaluation group (concept, task, joint).

    Parameters
    ----------
    all_results : list[dict]
        Each entry has ``meta`` and ``results``.
    meta_keys : list[str]
        Keys used in ``meta`` dicts (determines row index levels).
    seed_key : str
        Which meta key is the seed (averaged over in summary).

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    # Collect per-group data
    groups = {}
    for entry in all_results:
        meta = entry["meta"]
        key = tuple(meta[k] for k in meta_keys)
        for group, metrics in entry["results"].items():
            if group == "interventional" or not isinstance(metrics, dict):
                continue
            row = {}
            for mname, values in metrics.items():
                if not isinstance(values, dict):
                    continue
                for cname, val in values.items():
                    row[(mname, cname)] = val
            groups.setdefault(group, []).append((key, row))

    dfs = {}
    for group, records in groups.items():
        all_cols = sorted(
            {k for _, row in records for k in row if k[1] != "mean"}
        )
        mean_cols = sorted(
            {k for _, row in records for k in row if k[1] == "mean"}
        )
        ordered_cols = all_cols + mean_cols
        col_index = pd.MultiIndex.from_tuples(ordered_cols, names=["metric", "concept"])
        idx = [k for k, _ in records]
        data = [[row.get(c) for c in ordered_cols] for _, row in records]
        row_index = pd.MultiIndex.from_tuples(idx, names=meta_keys)
        dfs[group] = pd.DataFrame(data, index=row_index, columns=col_index)

    return dfs


def interventional_to_dataframe(all_results, meta_keys, seed_key="seed"):
    """Build a long-form DataFrame from interventional evaluation results.

    Returns
    -------
    pd.DataFrame or None
        Columns: meta_keys (minus seed) + step, n_total, n_concepts,
        pct_intervened, task_accuracy, joint_accuracy.
    """
    rows = []
    for entry in all_results:
        meta = entry["meta"]
        intv = entry["results"].get("interventional")
        if intv is None:
            continue
        n_total = intv["n_total"]
        n_concepts = intv["n_concepts"]
        for step, step_data in intv["results"].items():
            step = int(step)
            pct = step / n_total if n_total > 0 else 0.0
            task_acc = None
            joint_acc = None
            if "task" in step_data:
                task_acc = step_data["task"].get("accuracy", {}).get("mean")
            if "joint" in step_data:
                joint_acc = step_data["joint"].get("accuracy", {}).get("mean")
            row = {k: meta[k] for k in meta_keys}
            row.update({
                "step": step,
                "n_total": n_total,
                "n_concepts": n_concepts,
                "pct_intervened": round(pct, 4),
                "task_accuracy": task_acc,
                "joint_accuracy": joint_acc,
            })
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Average over seeds
    group_cols = [k for k in meta_keys if k != seed_key] + ["step", "n_total", "n_concepts", "pct_intervened"]
    agg = df.groupby(group_cols).agg(
        task_accuracy_mean=("task_accuracy", "mean"),
        task_accuracy_sem=("task_accuracy", lambda x: x.std() / len(x)**0.5 if len(x) > 1 else 0.0),
        joint_accuracy_mean=("joint_accuracy", "mean"),
        joint_accuracy_sem=("joint_accuracy", lambda x: x.std() / len(x)**0.5 if len(x) > 1 else 0.0),
    ).reset_index()
    return agg


@hydra.main(config_path="conf", config_name="analysis", version_base="1.3")
def main(cfg: DictConfig) -> None:

    # ---- Resolve meta keys ----
    meta_keys = list(cfg.meta_keys)
    seed_key = cfg.get("seed_key", "seed")

    # ---- Load runs from CSV (with optional filters) ----
    csv_path = os.path.join(get_original_cwd(), "conceptarium", cfg.csv_path)
    filters = (
        OmegaConf.to_container(cfg.filters, resolve=True)
        if cfg.get("filters")
        else None
    )
    rows = load_registry(csv_path, filters=filters)
    logger.info("Found %d run(s) in %s", len(rows), csv_path)

    # ---- Evaluate each run ----
    all_results = []
    n_runs = len(rows)
    for i, row in enumerate(rows, 1):
        job_dir = row["run_dir"]
        logger.info("Evaluating run %d/%d", i, n_runs)
        try:
            job_cfg, _ = load_job(job_dir)
            meta = job_metadata(job_cfg, meta_keys)
            meta["run_dir"] = job_dir

            # ----------------------------------------------------------
            # Evaluate all metrics for this run and collect results
            # ----------------------------------------------------------
            results = evaluate_job(job_dir, metrics_cfg=cfg.metrics)
            all_results.append({"meta": meta, "results": results})

        except Exception as e:
            logger.error("  Failed: %s", e)

    # ---- Build summary DataFrames ----
    if all_results:
        logger.info("Building summary tables...")
        dfs = results_to_dataframes(all_results, meta_keys, seed_key=seed_key)
        out_dir = os.path.join(get_original_cwd(), "conceptarium", "results")
        os.makedirs(out_dir, exist_ok=True)
        for group, df in dfs.items():
            # Average across seeds when multiple are present
            group_levels = [n for n in df.index.names if n != seed_key]
            df_avg = df.groupby(level=group_levels).mean()
            df_count = df.groupby(level=group_levels).count()
            df_sem = (df.groupby(level=group_levels).std() / df_count**0.5).fillna(0)

            # Format as "mean ± sem"
            df_summary = df_avg.combine(df_sem, lambda m, s: m.map("{:.4f}".format) + " ± " + s.map("{:.4f}".format))

            out_path = os.path.join(out_dir, f"analysis_{group}.csv")
            df_summary.to_csv(out_path)

        # Build and save interventional results
        df_intv = interventional_to_dataframe(all_results, meta_keys, seed_key=seed_key)
        if df_intv is not None:
            intv_path = os.path.join(out_dir, "analysis_interventional.csv")
            df_intv.to_csv(intv_path, index=False)

        # Run visualization
        import importlib.util
        _viz_path = os.path.join(get_original_cwd(), "conceptarium", "visualize.py")
        _spec = importlib.util.spec_from_file_location("visualize", _viz_path)
        _viz = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_viz)
        _viz.main()
        logger.info("Done. Results saved to %s", out_dir)
    else:
        logger.warning("No runs found or all evaluations failed.")


if __name__ == "__main__":
    register_custom_resolvers()
    main()
