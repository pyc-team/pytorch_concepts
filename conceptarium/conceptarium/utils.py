from copy import deepcopy
import torch
import numpy as np
import random
import os
import torch
import importlib
from omegaconf import DictConfig, open_dict
from typing import Mapping

from torch_concepts import Annotations

def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_run_env(cfg: DictConfig):
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    if cfg.get("matmul_precision", None) is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    with open_dict(cfg): 
        cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    return cfg

def clean_empty_configs(cfg: DictConfig) -> DictConfig:
    """ can be used to set default values for missing keys """
    with open_dict(cfg):
        if not cfg.get('causal_discovery'):
            cfg.update(causal_discovery = None)
        if not cfg.get('llm'):
            cfg.update(llm = None)
        if not cfg.get('rag'):
            cfg.update(rag = None)
    
    if cfg.engine.train_inference['_target_'] is None:
        with open_dict(cfg):
            cfg.engine.update(train_inference = None)
    return cfg

def update_config_from_data(cfg: DictConfig, dm) -> DictConfig:
    """ can be used to update the config based on the data, e.g., set input and output size """
    with open_dict(cfg):
        cfg.model.update(
            input_size = dm.backbone.output_size if dm.backbone else dm.n_features[-1], # FIXME: backbone.output_size might not exist
            # output_size = sum(dm.concept_metadata.values()),   # check if this is needed
            backbone = dm.backbone,
            embs_precomputed = dm.embs_precomputed
        )
        # if cfg.engine.metrics.get('accuracy'):
        #     if cfg.engine.metrics.accuracy.get('_target_') == 'conceptarium.metrics.PerConceptClassificationAccuracy':
        #         cfg.engine.metrics.accuracy.update(
        #             n_concepts = dm.n_concepts,
        #             concept_names = dm.concept_names
        #         )
        # cfg.engine.update(
        #    concept_names = dm.concept_names,
        #    concept_metadata = dm.concept_metadata
        # )
    return cfg

def instantiate_from_string(class_path: str, **kwargs):
    """Instantiate a class from its string path."""
    cls = get_from_string(class_path)
    return cls(**kwargs)

def get_from_string(class_path: str):
    """Return a class from its string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls
    
def add_distribution_to_annotations(annotations: Annotations,
                                    variable_distributions: Mapping) -> Annotations:
    concepts_annotations = deepcopy(annotations[1])
    metadatas = concepts_annotations.metadata
    cardinalities = concepts_annotations.cardinalities
    for (concept_name, metadata), cardinality in zip(metadatas.items(), cardinalities):
        if 'distribution' in metadata:
            raise ValueError(f"Concept {concept_name} already has a 'distribution' field.")
        else:
            if metadata['type'] == 'discrete' and cardinality==1: distribution_flag = 'discrete_card1'
            elif metadata['type'] == 'discrete' and cardinality>1: distribution_flag = 'discrete_cardn'
            elif metadata['type'] == 'continuous' and cardinality==1: distribution_flag = 'continuous_card1'
            elif metadata['type'] == 'continuous' and cardinality>1: distribution_flag = 'continuous_cardn'
            else: raise ValueError(f"Cannot set distribution type for concept {concept_name}.")

            metadatas[concept_name]['distribution'] = get_from_string(variable_distributions[distribution_flag]['path'])

    annotations[1].metadata = metadatas
    return annotations
    
