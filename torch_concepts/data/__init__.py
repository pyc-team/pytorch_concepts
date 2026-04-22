"""
Data module for concept-based datasets.

This module provides dataset classes and utilities for working with concept-annotated
data, including various benchmark datasets (MNIST, CelebA, CUB, etc.) and custom
concept datasets.
"""

# Submodules
from . import base
from . import datasets
from . import datamodules
from . import preprocessing
from . import scalers
from . import splitters

# Utilities
from . import utils

# Backbone utilities
from . import backbone

# IO utilities
from . import io

# Re-export datasets for convenient access
from .datasets.bnlearn import BnLearnDataset
from .datasets.toy import ToyDataset, CompletenessDataset
from .datasets.categorical_toy_dag import ToyDAGDataset
from .datasets.celeba import CelebADataset
from .datasets.pendulum import PendulumDataset
from .datasets.mnist_arithmetic import MNISTArithmeticDataset
from .datasets.dsprites_regression import DSpritesRegressionDataset
from .datasets.awa2 import AWA2Dataset
from .datasets.cebab import CEBaBDataset

# Re-export datamodules for convenient access
from .datamodules.bnlearn import BnLearnDataModule
from .datamodules.categorical_toy_dag import ToyDAGDataModule
from .datamodules.completeness import CompletenessDataModule
from .datamodules.celeba import CelebADataModule
from .datamodules.pendulum import PendulumDataModule
from .datamodules.mnist_arithmetic import MNISTArithmeticDataModule
from .datamodules.dsprites_regression import DSpritesRegressionDataModule
from .datamodules.awa2 import AWA2DataModule
from .datamodules.cebab import CEBaBDataModule

__all__ = [
    # Submodules
    "base",
    "datasets",
    "datamodules",
    "preprocessing",
    "scalers",
    "splitters",

    "utils",
    "backbone",
    "io",
    
    # Datasets
    "BnLearnDataset",
    "ToyDataset",
    "CompletenessDataset",
    "ToyDAGDataset",
    "CelebADataset",
    "PendulumDataset",
    "MNISTArithmeticDataset",
    "DSpritesRegressionDataset",
    "AWA2Dataset",
    "CEBaBDataset",
    
    # DataModules
    "BnLearnDataModule",
    "ToyDAGDataModule",
    "CompletenessDataModule",
    "CelebADataModule",
    "PendulumDataModule",
    "MNISTArithmeticDataModule",
    "DSpritesRegressionDataModule",
    "AWA2DataModule",
    "CEBaBDataModule",
]
