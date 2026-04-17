from .bnlearn import BnLearnDataset
from .toy import ToyDataset, CompletenessDataset
from .categorical_toy_dag import ToyDAGDataset
from .celeba import CelebADataset
from .pendulum import PendulumDataset
from .mnist_arithmetic import MNISTArithmeticDataset
from .dsprites_regression import DSpritesRegressionDataset

__all__: list[str] = [
    "BnLearnDataset",
    "ToyDataset",
    "ToyDAGDataset",
    "ToyFunctionDAGDataset",
    "CompletenessDataset",
    "CelebADataset",
    "PendulumDataset",
    "MNISTArithmeticDataset",
    "DSpritesRegressionDataset",
]

