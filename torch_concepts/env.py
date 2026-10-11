"""Environment configuration shared by PyC, its examples and Conceptarium.

This module sets up configuration including:
- Cache directory for storing artifacts, embeddings, and checkpoints
- Data root directory for datasets
- API keys for external services (HuggingFace, OpenAI)
- Project name and W&B entity for Conceptarium's logging

Configuration can be customized by setting environment variables:
- PYC_CACHE: Override default cache location
- XDG_CACHE_HOME: Base cache directory (follows XDG Base Directory spec)
"""

from os import environ as env
from pathlib import Path

# Project name used for Conceptarium's logging
PROJECT_NAME = "conceptarium"

# W&B entity/username for experiment tracking
# Set this to your W&B username or team name
WANDB_ENTITY = "" 

# Cache directory for artifacts, embeddings, checkpoints and datasets, shared
# with the PyC examples. Can be overridden with the PYC_CACHE environment variable
# Default: $XDG_CACHE_HOME/pyc, or ~/.cache/pyc
CACHE = Path(
    env.get(
        "PYC_CACHE",
        Path(
            env.get("XDG_CACHE_HOME", Path("~", ".cache")),
            "pyc",
        ),
    )
).expanduser()
CACHE.mkdir(parents=True, exist_ok=True)

# Directory where datasets are stored
# By default, uses CACHE directory
# Customize this if you want datasets in a different location
DATA_ROOT = CACHE

# HuggingFace Hub token for accessing private models/datasets
# Set this if you need to download from private HF repositories
HUGGINGFACEHUB_TOKEN = (
    env.get("HF_TOKEN")
    or env.get("HUGGINGFACE_HUB_TOKEN")
    or env.get("HUGGINGFACEHUB_TOKEN", "")
)
if HUGGINGFACEHUB_TOKEN:
    env.setdefault("HF_TOKEN", HUGGINGFACEHUB_TOKEN)
    env.setdefault("HUGGINGFACE_HUB_TOKEN", HUGGINGFACEHUB_TOKEN)

# OpenAI API key for GPT models
# Set this if you're using OpenAI models for concept generation or evaluation
OPENAI_API_KEY = env.get("OPENAI_API_KEY", "")