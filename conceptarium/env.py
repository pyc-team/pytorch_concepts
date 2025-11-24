"""Environment configuration for the conceptarium project.

This module sets up project-level configuration including:
- Project name and W&B entity for logging
- Cache directory for storing artifacts, embeddings, and checkpoints
- Data root directory for datasets
- API keys for external services (HuggingFace, OpenAI)

Configuration can be customized by setting environment variables:
- CONCEPTARIUM_CACHE: Override default cache location
- XDG_CACHE_HOME: Base cache directory (follows XDG Base Directory spec)
"""

from os import environ as env
from pathlib import Path

# Project name used for logging and caching
PROJECT_NAME = "conceptarium" 

# W&B entity/username for experiment tracking
# Set this to your W&B username or team name
WANDB_ENTITY = "" 

# Cache directory for artifacts, embeddings, and checkpoints
# Can be overridden with CONCEPTARIUM_CACHE environment variable
# Default: ~/.cache/conceptarium (Linux/macOS) or %LOCALAPPDATA%/conceptarium (Windows)
CACHE = Path(
    env.get(
        f"{PROJECT_NAME.upper()}_CACHE",
        Path(
            env.get("XDG_CACHE_HOME", Path("~", ".cache")),
            PROJECT_NAME,
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
HUGGINGFACEHUB_TOKEN = ''   

# OpenAI API key for GPT models
# Set this if you're using OpenAI models for concept generation or evaluation
OPENAI_API_KEY = ''