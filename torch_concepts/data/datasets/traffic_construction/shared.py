"""
Shared global variables for this dataset generation.
"""
from importlib import resources

def SPRITES_DIRECTORY(x: str) -> str:
    return str(resources.files("torch_concepts.data.datasets.traffic_construction") / "assets" / x)
