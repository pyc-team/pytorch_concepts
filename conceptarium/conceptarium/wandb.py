"""Weights & Biases (W&B) integration utilities for model and data loading.

This module provides functions to interact with W&B for loading trained models,
datasets, and checkpoints from logged runs. Useful for model evaluation, 
deployment, and experiment reproduction.
"""

from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from torch import cuda

from env import CACHE, PROJECT_NAME, WANDB_ENTITY
from hydra.utils import instantiate
from wandb.apis.public import Run


wandb_project = f"{PROJECT_NAME}"
wandb_entity = WANDB_ENTITY


def run_from_id(run_id: str) -> Run:
    """Retrieve a W&B run object from its run ID.
    
    Args:
        run_id (str): W&B run identifier (8-character alphanumeric string).
        
    Returns:
        wandb.apis.public.Run: W&B run object with access to config, 
            metrics, and artifacts.
            
    Example:
        >>> run = run_from_id("abc12xyz")
        >>> print(run.name, run.state)
        my-experiment finished
    """
    from wandb import Api

    api = Api()
    return api.run(f"{wandb_entity}/{wandb_project}/{run_id}")


def checkpoint_from_run(run: Run | str, target_device: str = None) -> dict:
    """Download and load a PyTorch checkpoint from a W&B run.
    
    Downloads the model checkpoint artifact from W&B (if not already cached)
    and loads it into memory. Checkpoints are cached locally to avoid 
    repeated downloads.
    
    Args:
        run (Run or str): W&B run object or run ID string.
        
    Returns:
        dict: PyTorch checkpoint dictionary containing:
            - state_dict: Model weights
            - optimizer_states: Optimizer state
            - epoch: Training epoch
            - And other training metadata
            
    Example:
        >>> checkpoint = checkpoint_from_run("abc12xyz")
        >>> print(checkpoint.keys())
        dict_keys(['state_dict', 'optimizer_states', 'epoch', ...])
    """
    if isinstance(run, str):
        run = run_from_id(run)
    checkpoint_path = CACHE.joinpath(
        "artifacts", run.entity, run.project, run.id, "model.ckpt"
    )
    if not checkpoint_path.exists():
        from wandb import Api

        api = Api()
        artifact = api.artifact(
            f"{run.entity}/{run.project}/model-{run.id}:best", type="model"
        )
        artifact.download(root=str(checkpoint_path.parent))
    from torch import load

    if target_device is None:
        target_device = "cuda" if cuda.is_available() else "cpu"
    checkpoint = load(checkpoint_path, map_location=target_device)
    return checkpoint


def model_from_run(run: Run | str, target_device: str = None) -> LightningModule:
    """Load a trained PyTorch Lightning model from a W&B run.
    
    Reconstructs the model from the W&B config, loads trained weights from
    the checkpoint, and sets it to evaluation mode. Useful for inference
    and model analysis.
    
    Args:
        run (Run or str): W&B run object or run ID string.
        
    Returns:
        LightningModule: Trained model in evaluation mode.
        
    Example:
        >>> model = model_from_run("abc12xyz")
        >>> predictions = model(test_inputs)
    """
    if isinstance(run, str):
        run = run_from_id(run)
    checkpoint = checkpoint_from_run(run, target_device=target_device)
    config = OmegaConf.create(run.config["hydra_cfg"])
    model = instantiate(config.engine, _convert_="all")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def dataset_from_run(run: Run | str) -> LightningDataModule:
    """Reconstruct the dataset/datamodule from a W&B run's configuration.
    
    Instantiates the LightningDataModule using the configuration saved in
    the W&B run. Useful for reproducing experiments with identical data splits.
    
    Args:
        run (Run or str): W&B run object or run ID string.
        
    Returns:
        LightningDataModule: DataModule configured as in the original run.
        
    Example:
        >>> datamodule = dataset_from_run("abc12xyz")
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
    """
    if isinstance(run, str):
        run = run_from_id(run)
    config = OmegaConf.create(run.config["hydra_cfg"])
    datamodule = instantiate(config.dataset, _convert_="all")
    return datamodule


def iter_runs(
    entity: str | None = None,
    project: str | None = None,
    filters: dict[str, str] | None = None,
):
    """Iterator over W&B runs in a project with optional filtering.
    
    Args:
        entity (str, optional): W&B entity/username. Defaults to PROJECT_ENTITY.
        project (str, optional): W&B project name. Defaults to current project.
        filters (dict[str, str], optional): W&B API filters for querying runs.
            Examples: {"state": "finished"}, {"tags": "production"}.
            
    Yields:
        wandb.apis.public.Run: W&B run objects matching the filters.
        
    Example:
        >>> # Find all finished runs with specific tag
        >>> for run in iter_runs(filters={"state": "finished", "tags": "best"}):
        ...     print(run.name, run.summary["val_accuracy"])
        experiment-1 0.95
        experiment-2 0.97
    """
    from wandb import Api

    entity = entity if entity is not None else wandb_entity
    project = project if project is not None else wandb_project

    api = Api(overrides=dict(entity=entity, project=project))
    runs = api.runs(filters=filters or {})
    for run in runs:
        yield run
