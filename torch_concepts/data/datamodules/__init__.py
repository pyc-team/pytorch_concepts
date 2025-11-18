from .bnlearn import BnLearnDataModule
from .colormnist import ColorMNISTDataModule
from .fashionmnist import FashionMNISTDataModule

__all__: list[str] = [
    "BnLearnDataModule",
    "ColorMNISTDataModule",
    "FashionMNISTDataModule",
]

