from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import cuda

from env import CACHE, PROJECT_NAME, VERSION, PROJECT_ENTITY
from hydra.utils import instantiate
from wandb.apis.public import Run


wandb_project = f"{PROJECT_NAME}_v{VERSION}"
wandb_entity = PROJECT_ENTITY


def run_from_id(run_id: str) -> Run:
    from wandb import Api

    api = Api()
    return api.run(f"{wandb_entity}/{wandb_project}/{run_id}")


def checkpoint_from_run(run: Run | str) -> dict:
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

    map_location = "cuda" if cuda.is_available() else "cpu"
    checkpoint = load(checkpoint_path, map_location=map_location)
    return checkpoint


def model_from_run(run: Run | str) -> LightningModule:
    if isinstance(run, str):
        run = run_from_id(run)
    checkpoint = checkpoint_from_run(run)
    config = OmegaConf.create(run.config["hydra_cfg"])
    model = instantiate(config.engine, _convert_="all")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def dataset_from_run(run: Run | str) -> LightningDataModule:
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
    from wandb import Api

    entity = entity if entity is not None else wandb_entity
    project = project if project is not None else wandb_project

    api = Api(overrides=dict(entity=entity, project=project))
    runs = api.runs(filters=filters or {})
    for run in runs:
        yield run
