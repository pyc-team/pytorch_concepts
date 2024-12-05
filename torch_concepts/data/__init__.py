from .celeba import CelebADataset
from .mnist import ColorMNISTDataset
from .toy import ToyDataset, CompletenessDataset
from .traffic import TrafficLights

__all__ = [
    'TrafficLights',
    'ToyDataset',
    'CompletenessDataset',
    'ColorMNISTDataset',
    'CelebADataset',
]
