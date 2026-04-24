"""Run registry: track experiment runs in a CSV file.

After each experiment, a row is appended with the run path and key
configuration values.  The CSV can later be read (optionally filtered)
by the analysis script.
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DEFAULT_CSV = "runs.csv"

# Top-level sections and leaf keys to drop from the flattened config.
_EXCLUDE_SECTIONS = {"metrics"}
_EXCLUDE_KEYS = {
    "dataset.label_descriptions",
    "trainer.logger",
    "trainer.log_model",
    "trainer.save_top_k",
    "trainer.save_last",
    "trainer.save_weights_only",
    "notes",
    "causal_discovery",
    "llm",
    "rag"
}
# Suffixes that are Hydra internals, not real hyperparameters.
_EXCLUDE_SUFFIXES = ("._partial_",)

# Meta columns always written first.
_META_COLUMNS = ["run_dir", "timestamp", "status"]


# ------------------------------------------------------------------
# Flatten
# ------------------------------------------------------------------

def _flatten_cfg(cfg: DictConfig) -> dict:
    """Recursively flatten a Hydra config into ``{'dot.path': value}``
    while skipping excluded sections/keys."""
    container = OmegaConf.to_container(cfg, resolve=True)
    flat = {}

    def _recurse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                full = f"{prefix}.{k}" if prefix else k
                # skip entire sections
                top = full.split(".")[0]
                if top in _EXCLUDE_SECTIONS:
                    continue
                # skip specific keys / subtrees
                if full in _EXCLUDE_KEYS:
                    continue
                # skip Hydra internals
                if any(full.endswith(s) for s in _EXCLUDE_SUFFIXES):
                    continue
                _recurse(v, full)
        elif isinstance(obj, (list, tuple)):
            flat[prefix] = str(obj)
        else:
            flat[prefix] = obj

    _recurse(container)
    return flat


# ------------------------------------------------------------------
# Write
# ------------------------------------------------------------------

def _extract_row(run_dir: str, cfg: DictConfig, status: str = "success") -> dict:
    """Build a flat row from meta info + flattened config."""
    row = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
    }
    row.update(_flatten_cfg(cfg))
    return row


def register_run(
    run_dir: str,
    cfg: DictConfig,
    status: str = "success",
    csv_path: str = DEFAULT_CSV,
):
    """Append a completed (or failed) run to the CSV registry.

    If the new row introduces columns not yet in the CSV, the file is
    rewritten with the expanded header so every row stays aligned.
    """
    row = _extract_row(run_dir, cfg, status)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_cols = list(reader.fieldnames) if reader.fieldnames else []
    else:
        existing_rows = []
        existing_cols = []

    new_cols = [k for k in row if k not in existing_cols]
    all_cols = existing_cols + new_cols

    # Rewrite only when new columns appear; otherwise just append.
    if new_cols or not existing_rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(row)
    else:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            writer.writerow(row)

    logger.info("Registered run in %s", csv_path)


# ------------------------------------------------------------------
# Read
# ------------------------------------------------------------------

def load_registry(csv_path: str = DEFAULT_CSV, filters: dict = None):
    """Read the run registry, optionally keeping only matching rows.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    filters : dict, optional
        ``{column: value}`` or ``{column: [val1, val2, ...]}``.
        Example: ``{"dataset": "asia", "seed": [1, 2]}``.

    Returns
    -------
    list[dict]
        Rows that pass all filters.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Registry not found: %s", csv_path)
        return []

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if filters:
        for col, val in filters.items():
            vals = {str(v) for v in val} if isinstance(val, (list, tuple)) else {str(val)}
            rows = [r for r in rows if r.get(col) in vals]

    return rows
