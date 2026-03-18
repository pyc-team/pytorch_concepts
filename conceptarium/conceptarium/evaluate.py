"""Evaluation functions for concept-based model benchmarking.

Provides:
- Standard accuracy: concept, task, and joint (concept + task).
- Progressive interventional accuracy: clamp concepts to ground truth
  one-by-one in topological order and measure task accuracy.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from torch_concepts.nn import UniformPolicy, GroundTruthIntervention, intervention
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.annotations import Annotations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_test_data(datamodule, device="cpu"):
    """Collect all test batches into single tensors.

    Returns
    -------
    dict
        ``'x'``: input tensor [N, ...], ``'c'``: concept tensor [N, D].
    """
    xs, cs = [], []
    for batch in datamodule.test_dataloader():
        xs.append(batch["inputs"]["x"])
        cs.append(batch["concepts"]["c"])
    return {
        "x": torch.cat(xs, dim=0).to(device),
        "c": torch.cat(cs, dim=0).to(device),
    }


def compute_metrics(preds, targets, annotations, names, fn_collection):
    """Compute metrics for a subset of concepts.

    Uses :class:`ConceptMetrics` for type-aware evaluation
    (binary vs categorical).

    Parameters
    ----------
    preds : Tensor [N, D]
        Model output logits.
    targets : Tensor [N, D]
        Ground truth (0/1 for binary, class indices for categorical).
    annotations : AxisAnnotation
        Concept metadata (labels, cardinalities, slices).
    names : list[str]
        Concept names to evaluate.
    fn_collection : GroupConfig
        Metric collection (binary/categorical metric specs).

    Returns
    -------
    dict
        ``{metric_name: {concept_name: float, ..., 'mean': float}}``.
    """

    sub_ann = annotations.subset(names)
    sub_preds = preds[:, annotations.get_slice(names)]
    # targets have one column per concept (concept-level indices),
    # not one column per logit, so use concept indices, not logit slices.
    concept_idx = [annotations.get_index(n) for n in names]
    sub_targets = targets[:, concept_idx]

    cm = ConceptMetrics(
        annotations=Annotations({1: sub_ann}),
        binary=fn_collection.get('binary'),
        categorical=fn_collection.get('categorical'),
        summary=True,
        per_concept=True,
    )
    cm.to(preds.device)
    cm.update(sub_preds, sub_targets)
    computed = cm.compute()

    # Collect metric names from the fn_collection
    all_metric_names = set()
    if fn_collection.get("binary"):
        all_metric_names.update(fn_collection["binary"].keys())
    if fn_collection.get("categorical"):
        all_metric_names.update(fn_collection["categorical"].keys())

    results = {}
    for mname in sorted(all_metric_names):
        metric_results = {}
        vals = []
        for cname in names:
            key = f"{cname}_{mname}"
            val = computed[key].item() if key in computed else 0.0
            metric_results[cname] = val
            vals.append(val)
        metric_results["mean"] = sum(vals) / len(vals) if vals else 0.0
        results[mname] = metric_results

    return results


def _gt_to_logits(c_true, annotations, name):
    """Convert ground-truth for one concept to logit space.

    Binary  (card 1): ``logit(p)``  — expects p in [0, 1].
    Categorical (card k): class index → one-hot logits.

    Parameters
    ----------
    c_true : Tensor [N, n_concepts]
        Ground-truth with one column per concept (class index for
        categorical, 0/1 for binary).
    annotations : AxisAnnotation
        Concept metadata.
    name : str
        Single concept name.
    """
    idx = annotations.labels.index(name)
    card = annotations.cardinalities[idx]
    concept_col = c_true[:, idx]  # [N]

    if card == 1:
        return torch.logit(concept_col.float().clamp(1e-6, 1 - 1e-6)).unsqueeze(-1)
    else:
        # Class index → one-hot → logit-like scores
        one_hot = torch.nn.functional.one_hot(concept_col.long(), num_classes=card).float()
        return one_hot * 20.0 - 10.0  # 1 → 10, 0 → -10


def get_intervention_order(model, task_names):
    """Determine full intervention order (concepts then tasks).

    Concepts come first in topological order (or natural label order
    if no DAG is available), followed by task nodes at the end.
    """
    all_names = list(model.concept_names)
    concept_only = [n for n in all_names if n not in task_names]

    graph = getattr(model, "graph", None)
    if graph is not None and hasattr(graph, "is_dag") and graph.is_dag():
        topo = graph.topological_sort()
        concept_only = [n for n in topo if n in concept_only]

    # Tasks come last
    tasks = [n for n in all_names if n in task_names]
    return concept_only + tasks


# ---------------------------------------------------------------------------
# Standard evaluation
# ---------------------------------------------------------------------------

def evaluate_standard(model, datamodule, fn_collection, task_names=None):
    """Concept, task, and joint metrics on the test set.

    Parameters
    ----------
    model : nn.Module
    datamodule : LightningDataModule
    fn_collection : GroupConfig
        Metric collection (binary/categorical metric specs).
    task_names : list[str] or None
        Names of concepts that are tasks.  When *None*, falls back to
        ``model.task_names`` (empty for models that don't define it).
        Prefer passing the dataset's ``default_task_names`` explicitly.

    Returns
    -------
    dict
        Keys: ``concept``, ``task``, ``joint``.  Each maps to
        ``{metric_name: {concept: float, …, 'mean': float}}``.
    """
    model.eval()
    device = next(model.parameters()).device
    data = collect_test_data(datamodule, device=device)

    all_names = list(model.concept_names)
    if task_names is None:
        task_names = list(getattr(model, "task_names", []))
    else:
        task_names = list(task_names)
    concept_only = [n for n in all_names if n not in task_names]
    ann = model.concept_annotations

    with torch.no_grad():
        preds = model.forward(x=data["x"], query=all_names, return_logits=True)

    results = {}
    if concept_only:
        results["concept"] = compute_metrics(
            preds, data["c"], ann, concept_only, fn_collection=fn_collection
        )
    results["task"] = compute_metrics(
        preds, data["c"], ann, task_names, fn_collection=fn_collection
    )
    results["joint"] = compute_metrics(
        preds, data["c"], ann, all_names, fn_collection=fn_collection
    )
    return results


# ---------------------------------------------------------------------------
# Interventional evaluation
# ---------------------------------------------------------------------------

def evaluate_interventional(model, datamodule, fn_collection, task_names=None):
    """Progressive ground-truth interventions on concepts.

    At each step *i* the first *i* concepts (in topological / natural order)
    are clamped to their ground-truth values and task accuracy is measured.

    Parameters
    ----------
    model : nn.Module
    datamodule : LightningDataModule
    fn_collection : GroupConfig
        Metric collection (binary/categorical metric specs).
    task_names : list[str] or None
        Names of concepts that are tasks.  When *None*, falls back to
        ``model.task_names``.

    Returns
    -------
    dict or None
        ``'order'``: list of names in intervention order (concepts then tasks).
        ``'n_total'``: total number of variables (concepts + tasks).
        ``'n_concepts'``: number of concept-only variables.
        ``'results'``: ``{step: {'task': {...}, 'joint': {...}}}``,
        where *task* covers task-only metrics and *joint* covers all variables.
        ``None`` if the model has no probabilistic model.
    """
    model.eval()
    device = next(model.parameters()).device
    data = collect_test_data(datamodule, device=device)

    all_names = list(model.concept_names)
    if task_names is None:
        task_names = list(getattr(model, "task_names", []))
    else:
        task_names = list(task_names)
    ann = model.concept_annotations

    # Need the ProbabilisticModel for interventions
    pgm = None
    if hasattr(model, "model") and hasattr(model.model, "probabilistic_model"):
        pgm = model.model.probabilistic_model
    if pgm is None:
        logger.warning(
            "Model has no probabilistic model — skipping interventional evaluation."
        )
        return None

    cpds = pgm.parametric_cpds
    intervention_order = get_intervention_order(model, task_names)
    n_concepts = len([n for n in intervention_order if n not in task_names])
    logger.debug(
        "Intervention order (%d total, %d concepts): %s",
        len(intervention_order), n_concepts, intervention_order,
    )

    results = {}

    def _eval_step(preds):
        """Evaluate task-only and joint (all variables) accuracy."""
        return {
            "task": compute_metrics(
                preds, data["c"], ann, task_names, fn_collection=fn_collection
            ),
            "joint": compute_metrics(
                preds, data["c"], ann, all_names, fn_collection=fn_collection
            ),
        }

    with torch.no_grad():
        # Step 0 — no intervention
        preds_0 = model.forward(x=data["x"], query=all_names, return_logits=True)
        results[0] = _eval_step(preds_0)

        # Progressive interventions
        for i in range(1, len(intervention_order) + 1):
            names_to_intervene = intervention_order[:i]

            policies = []
            strategies = []
            for cname in names_to_intervene:
                var = pgm.concept_to_variable[cname]
                policies.append(UniformPolicy(out_concepts=var.size))
                strategies.append(
                    GroundTruthIntervention(
                        model=cpds,
                        ground_truth=_gt_to_logits(data["c"], ann, cname),
                    )
                )

            with intervention(
                policies=policies,
                strategies=strategies,
                target_concepts=names_to_intervene,
                quantiles=1.0,
            ):
                preds_i = model.forward(
                    x=data["x"], query=all_names, return_logits=True
                )
            results[i] = _eval_step(preds_i)

    return {
        "order": intervention_order,
        "n_total": len(intervention_order),
        "n_concepts": n_concepts,
        "results": results,
    }
# ---------------------------------------------------------------------------

def evaluate_all(model, datamodule, fn_collection, task_names=None):
    """Run standard + interventional evaluation.

    Returns a single dict with all results (JSON-serialisable after
    :func:`save_results`).
    """
    results = evaluate_standard(model, datamodule, fn_collection, task_names=task_names)
    try:
        interventional = evaluate_interventional(
            model, datamodule, fn_collection, task_names=task_names
        )
        if interventional is not None:
            results["interventional"] = interventional
    except Exception as e:
        logger.warning("Interventional evaluation failed: %s", e)
    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """Recursively convert to JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def save_results(results, path="results.json"):
    """Dump evaluation results to a JSON file."""
    with open(path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# Job-level helpers (used by run_analysis)
# ---------------------------------------------------------------------------

def iter_jobs(run_dir):
    """Yield numbered job sub-directories from a Hydra multirun sweep."""
    for d in sorted(Path(run_dir).iterdir()):
        if d.is_dir() and d.name.isdigit():
            yield d


def load_job(job_dir):
    """Load a job's saved config and best checkpoint path.

    Returns
    -------
    cfg : DictConfig
    ckpt_path : Path or None
    """
    from omegaconf import OmegaConf

    job_dir = Path(job_dir)
    cfg = OmegaConf.load(job_dir / ".hydra" / "config.yaml")

    ckpt_path = None
    ckpt_dir = job_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            ckpt_path = ckpts[0]

    return cfg, ckpt_path


def job_metadata(cfg, meta_keys):
    """Extract metadata from a job config using dot-path keys.

    Parameters
    ----------
    cfg : DictConfig
        The job's Hydra config.
    meta_keys : list[str]
        Dot-separated paths into the config, e.g.
        ``["dataset.name", "model._target_", "seed"]``.
        Paths ending in ``._target_`` are resolved to the class name.

    Returns
    -------
    dict
        ``{key: value}`` for each path.
    """
    from omegaconf import OmegaConf

    meta = {}
    for key in meta_keys:
        parts = key.split(".")
        node = cfg
        try:
            for p in parts:
                node = node[p] if isinstance(node, dict) else getattr(node, p)
        except (KeyError, AttributeError, TypeError):
            node = ""
        # Resolve _target_ to class name
        if isinstance(node, str) and parts[-1] == "_target_":
            node = node.split(".")[-1].lower()
        meta[key] = node
    return meta


def evaluate_job(job_dir, metrics_cfg=None):
    """Reconstruct a trained model from *job_dir* and evaluate it.

    Parameters
    ----------
    job_dir : str or Path
        Job output directory (must contain ``.hydra/config.yaml``
        and ``checkpoints/``).
    metrics_cfg : DictConfig, optional
        Hydra metrics config (with ``_target_``, ``binary``, etc.).
        If *None*, the job's own ``cfg.metrics`` is used.

    Returns
    -------
    dict
        Nested evaluation results (see :func:`evaluate_standard`).
    """
    from hydra.utils import instantiate
    from conceptarium.utils import (
        setup_run_env,
        update_config_from_data,
        instantiate_loss,
    )

    job_dir = Path(job_dir)
    cfg, ckpt_path = load_job(job_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint in {job_dir / 'checkpoints'}")

    # Reconstruct datamodule (suppress noisy dataset loggers)
    _data_logger = logging.getLogger("torch_concepts.data")
    _prev_level = _data_logger.level
    _data_logger.setLevel(logging.WARNING)
    cfg = setup_run_env(cfg)
    datamodule = instantiate(cfg.dataset, _convert_="all")
    datamodule.setup("fit", verbose=False)
    _data_logger.setLevel(_prev_level)
    cfg = update_config_from_data(cfg, datamodule)

    # Reconstruct model and load weights
    loss = instantiate_loss(cfg, datamodule.annotations)
    metrics = instantiate(
        cfg.metrics, annotations=datamodule.annotations, _convert_="all"
    )
    model = instantiate(
        cfg.model,
        annotations=datamodule.annotations,
        graph=datamodule.graph,
        loss=loss,
        metrics=metrics,
        _convert_="all",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    # Instantiate ConceptMetrics from config to get fn_collection
    mcfg = metrics_cfg if metrics_cfg is not None else cfg.metrics
    test_metrics = instantiate(
        mcfg, annotations=datamodule.annotations, _convert_="all"
    )
    fn_collection = test_metrics.fn_collection

    task_names = list(cfg.dataset.get("default_task_names", []))

    # Evaluate (standard + interventional)
    res = evaluate_all(model, datamodule, fn_collection, task_names=task_names)

    return res


def flatten_results(results):
    """Flatten nested evaluation results to ``{group_metric: mean}``."""
    flat = {}
    for group, metrics in results.items():
        if group == "interventional":
            continue
        if isinstance(metrics, dict):
            for mname, values in metrics.items():
                if isinstance(values, dict) and "mean" in values:
                    flat[f"{group}_{mname}"] = values["mean"]
    return flat
