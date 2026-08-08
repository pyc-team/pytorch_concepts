"""Utility functions for configuration, seeding, and class instantiation.

This module provides helper functions for:
- Setting random seeds across all libraries (re-exported from torch_concepts)
- Configuring runtime environment from Hydra configs
- Dynamic class loading and instantiation
- Managing concept annotations and distributions
"""
import math
import os
import torch
import logging
import pandas as pd
from torch import nn
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate
from torch_concepts import seed_everything, Backbone, ConceptGraph
from torch_concepts.nn import MLP
from torch_concepts.utils import ensure_list
from torch_concepts.data.base import ConceptDataModule

logger = logging.getLogger(__name__)

from env import DATA_ROOT


def setup_run_env(cfg: DictConfig):
    """Configure runtime environment from Hydra configuration.
    
    Sets up threading, random seeds and matrix multiplication precision.
    
    Args:
        cfg: Hydra DictConfig containing runtime parameters:
            - num_threads: Number of PyTorch threads (default: 1)
            - seed: Random seed for reproducibility
            - matmul_precision: Float32 matmul precision ('highest', 'high', 'medium')
            
    Returns:
        Updated cfg
    """
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    if cfg.get("matmul_precision", None) is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    # set data root
    if not cfg.dataset.datamodule.get("root"):
        if "name" not in cfg.dataset:
            raise ValueError("If data root is not set, dataset name must be "
            "specified in cfg.dataset.name to set data root.")
        data_root = os.path.join(DATA_ROOT, cfg.dataset.get("name"))
        with open_dict(cfg):
            cfg.dataset.datamodule.update(root = data_root)
    return cfg

def maybe_precompute_embeddings(cfg: DictConfig, dm: ConceptDataModule, backbone: Backbone = None):
    """Decide what to do with ``backbone`` given ``cfg.precompute_embeddings``.

    Two mutually exclusive modes (plus the no-backbone case):

    - ``precompute_embeddings=True``: run the (frozen) backbone once to cache
      embeddings on the datamodule. The model then consumes cached features, so
      it receives no backbone.
    - ``precompute_embeddings=False``: hand the backbone (frozen or not) to the
      model, which runs it end-to-end.
    - ``backbone is None`` (e.g. tabular / bayesian dag datasets): nothing to do;
      the model uses raw features.

    Returns:
        (dm, model_backbone) where ``model_backbone`` is the backbone the model
        should use — ``None`` when embeddings were precomputed or no backbone exists.
    """
    if backbone is None:
        return dm, None

    if cfg.precompute_embeddings:  # default to True
        if not getattr(backbone, "frozen", True):
            raise ValueError(
                "precompute_embeddings=True requires a frozen backbone (embeddings "
                "are cached and never re-run). Set dataset.backbone.freeze=True to "
                "precompute, or precompute_embeddings=False to fine-tune end-to-end."
            )
        dm.precompute_embeddings(
            backbone=backbone,
            cache=True,
            cache_dir=cfg.dataset.datamodule.root,
            force=cfg.force_precompute,  # default to False
        )
        return dm, None  # model consumes cached features, not the backbone

    # End-to-end: the model runs the backbone (frozen or unfrozen).
    return dm, backbone


def resolve_graph(graph, annotations, task_names) -> ConceptGraph:
    """Return ``graph`` if given, else a default bipartite concept->task graph.

    When the datamodule provides no graph, build the same adjacency as
    :class:`~torch_concepts.nn.modules.high.base.bipartite.BipartiteMixin`: every
    concept points to every task and tasks have no outgoing edges. Lets a model
    that needs a graph fall back gracefully when none is supplied.

    Args:
        graph: An explicit :class:`ConceptGraph`, or ``None`` for the fallback.
        annotations: Annotations providing the ordered concept ``labels``.
        task_names: Task label(s) every concept should point to.
    """
    if graph is not None:
        return graph

    labels = list(annotations.labels)
    task_names = ensure_list(task_names)
    missing = [t for t in task_names if t not in labels]
    if missing:
        raise ValueError(
            f"task_names {missing} are not annotation labels {labels}."
        )
    adjacency = pd.DataFrame(0, index=labels, columns=labels)
    adjacency.loc[:, task_names] = 1             # concepts -> tasks
    adjacency.loc[task_names, task_names] = 0    # tasks do not self-loop
    return ConceptGraph(torch.FloatTensor(adjacency.values), node_names=labels)


def attach_latent_encoder(cfg: DictConfig, backbone: nn.Module) -> nn.Module:
    """Assemble the feature extractor the model receives as its ``backbone``.

    Call *after* :func:`maybe_precompute_embeddings` and :func:`update_config_from_data`.
    Combines the (optional) ``backbone`` with an (optional) trainable latent
    encoder configured under ``cfg.model.latent_encoder`` (a hydra config, e.g.
    an :class:`~torch_concepts.nn.MLP`; ``null`` disables it). Four cases:

    - both        -> ``Sequential(backbone, latent_encoder)``
    - backbone    -> the backbone unchanged (already maps input -> latent)
    - encoder     -> the latent encoder alone (runs on cached embeddings / raw features)
    - none        -> ``None`` (model uses its input directly, via ``nn.Identity``)

    The returned module always exposes ``out_features`` so the model can infer
    its ``latent_size`` (a bare encoder does not expose it on its own).
    """
    encoder_cfg = cfg.model.get("latent_encoder")
    has_backbone = backbone is not None
    has_encoder = encoder_cfg is not None

    # Case "none": nothing to attach; the model consumes its input directly.
    if not has_backbone and not has_encoder:
        return None

    # Case "backbone": it already maps input -> latent and exposes out_features.
    if has_backbone and not has_encoder:
        return backbone

    # Encoder is present. It runs on the backbone's embeddings when a live
    # backbone exists, otherwise on the model's input (cached embeddings when
    # precomputed, or raw features).
    in_features = backbone.out_features if has_backbone else cfg.model.model_cls.input_size
    latent_encoder = instantiate(encoder_cfg, input_size=in_features, _convert_="all")
    # MLP outputs hidden_size features unless an output_size readout is set.
    out_features = encoder_cfg.get("output_size") or encoder_cfg["hidden_size"]

    # Case "encoder": the latent encoder is the whole feature extractor.
    if not has_backbone:
        latent_encoder.out_features = out_features
        return latent_encoder

    # Case "both": backbone then encoder.
    extractor = nn.Sequential(backbone, latent_encoder)
    extractor.out_features = out_features
    return extractor


def update_config_from_data(cfg: DictConfig, dm: ConceptDataModule) -> DictConfig:
    """Update model configuration from datamodule properties.
    
    Automatically configures model input size, backbone, and embedding settings
    based on the datamodule's dataset properties. This ensures model architecture
    matches the data dimensions.
    
    Args:
        cfg: Hydra DictConfig containing model configuration.
        dm: ConceptDataModule instance with dataset information.
        
    Also publishes the data-derived sizes under a model-neutral ``data_dims``
    node, so a config can interpolate them where plain YAML cannot compute them:

    * ``data_dims.n_pixels`` — flattened input width (``prod(n_features)``);
    * ``data_dims.n_concepts`` — number of annotated concepts.

    A generative model's decoder needs both — its output is ``n_pixels`` wide and
    its input is a function of ``n_concepts`` — and neither is knowable before the
    datamodule is built.

    Args:
        cfg: Hydra DictConfig containing model configuration.
        dm: ConceptDataModule instance with dataset information.

    Returns:
        Updated cfg with model.input_size, model.backbone, and
        model.embs_precomputed set from datamodule.
    """
    with open_dict(cfg):
        cfg.model.model_cls.update(
            input_size = dm.n_features[-1] if len(dm.n_features)==1 else dm.n_features,
        )
        cfg.data_dims = {
            "n_pixels": int(math.prod(dm.n_features)),
            "n_concepts": len(dm.annotations.labels),
        }
    return cfg
