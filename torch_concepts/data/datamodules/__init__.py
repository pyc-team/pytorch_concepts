from .bnlearn import BnLearnDataModule
from .categorical_toy_dag import ToyDAGDataModule
from .celeba import CelebADataModule
from .completeness import CompletenessDataModule
from .pendulum import PendulumDataModule
from .mnist_arithmetic import MNISTArithmeticDataModule
from .dsprites_regression import DSpritesRegressionDataModule

__all__: list[str] = [
    "BnLearnDataModule",
    "ToyDAGDataModule",
    "CompletenessDataModule",
    "CelebADataModule",
    "PendulumDataModule",
    "MNISTArithmeticDataModule",
    "DSpritesRegressionDataModule",
]

