"""
Shared global variables for this dataset generation.
"""
import pkg_resources

# Directory where all the useful sprites are stored
SPRITES_DIRECTORY = pkg_resources.resource_filename(
    'torch_concepts',
    'data/traffic_construction/assets/',
)