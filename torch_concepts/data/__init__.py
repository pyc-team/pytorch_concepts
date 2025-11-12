from .dataset.awa2 import AwA2Dataset
from .dataset.bnlearn import BnLearnDataset
from .dataset.cebab import CEBaBDataset
from .dataset.celeba import CelebADataset
from .dataset.colormnist import ColorMNISTDataset
from .dataset.cub import CUBDataset
from .dataset.fashionmnist import FashionMNISTDataset
from .dataset.mnist import ColorMNISTDataset, MNIST, MNISTAddition, MNISTEvenOdd, PartialMNISTAddition
from .dataset.toy import ToyDataset, CompletenessDataset
from .dataset.traffic import TrafficLights

__all__ = [
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
