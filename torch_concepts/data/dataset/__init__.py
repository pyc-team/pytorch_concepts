from .awa2 import AwA2Dataset
from .bnlearn import BnLearnDataset
from .cebab import CEBaBDataset
from .celeba import CelebADataset
from .colormnist import ColorMNISTDataset
from .cub import CUBDataset
from .fashionmnist import FashionMNISTDataset
from .mnist import MNIST, MNISTAddition, MNISTEvenOdd, PartialMNISTAddition
from .toy import ToyDataset, CompletenessDataset
from .traffic import TrafficLights

__all__: list[str] = [
    "AwA2Dataset",
    "BnLearnDataset",
    "CEBaBDataset",
    "CelebADataset",
    "ColorMNISTDataset",
    "CUBDataset",
    "FashionMNISTDataset",
    "MNIST",
    "MNISTAddition",
    "MNISTEvenOdd",
    "PartialMNISTAddition",
    "ToyDataset",
    "CompletenessDataset",
    "TrafficLights",
]

