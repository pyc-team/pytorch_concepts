"""
CUB-200-2011 (Caltech-UCSD Birds) Dataset

Adapted from:
    - Koh et al.'s paper Concept Bottleneck Models 
    - Espinosa Zarlenga and Barbiero et al.'s repository https://github.com/mateoespinosa/cem/blob/main/cem/data/CUB200/cub_loader.py.
"""

import os
import logging
from pathlib import Path
import tarfile
from anyio import Path
import pickle
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from collections import defaultdict
from PIL import Image
from typing import List, Mapping, Optional
import zipfile
import shutil

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset
from torch_concepts.data.io import download_url

logger = logging.getLogger(__name__)

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 200

URLS = [
    # NOTE: we retrieve the .pkl split files from the CEM repository since I cannot find the m in the CBM repo.
    "https://raw.githubusercontent.com/mateoespinosa/cem/main/cem/data/CUB200/class_attr_data_10",
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
]

# CUB Class names

CLASS_NAMES = [
    "Black_footed_Albatross",
    "Laysan_Albatross",
    "Sooty_Albatross",
    "Groove_billed_Ani",
    "Crested_Auklet",
    "Least_Auklet",
    "Parakeet_Auklet",
    "Rhinoceros_Auklet",
    "Brewer_Blackbird",
    "Red_winged_Blackbird",
    "Rusty_Blackbird",
    "Yellow_headed_Blackbird",
    "Bobolink",
    "Indigo_Bunting",
    "Lazuli_Bunting",
    "Painted_Bunting",
    "Cardinal",
    "Spotted_Catbird",
    "Gray_Catbird",
    "Yellow_breasted_Chat",
    "Eastern_Towhee",
    "Chuck_will_Widow",
    "Brandt_Cormorant",
    "Red_faced_Cormorant",
    "Pelagic_Cormorant",
    "Bronzed_Cowbird",
    "Shiny_Cowbird",
    "Brown_Creeper",
    "American_Crow",
    "Fish_Crow",
    "Black_billed_Cuckoo",
    "Mangrove_Cuckoo",
    "Yellow_billed_Cuckoo",
    "Gray_crowned_Rosy_Finch",
    "Purple_Finch",
    "Northern_Flicker",
    "Acadian_Flycatcher",
    "Great_Crested_Flycatcher",
    "Least_Flycatcher",
    "Olive_sided_Flycatcher",
    "Scissor_tailed_Flycatcher",
    "Vermilion_Flycatcher",
    "Yellow_bellied_Flycatcher",
    "Frigatebird",
    "Northern_Fulmar",
    "Gadwall",
    "American_Goldfinch",
    "European_Goldfinch",
    "Boat_tailed_Grackle",
    "Eared_Grebe",
    "Horned_Grebe",
    "Pied_billed_Grebe",
    "Western_Grebe",
    "Blue_Grosbeak",
    "Evening_Grosbeak",
    "Pine_Grosbeak",
    "Rose_breasted_Grosbeak",
    "Pigeon_Guillemot",
    "California_Gull",
    "Glaucous_winged_Gull",
    "Heermann_Gull",
    "Herring_Gull",
    "Ivory_Gull",
    "Ring_billed_Gull",
    "Slaty_backed_Gull",
    "Western_Gull",
    "Anna_Hummingbird",
    "Ruby_throated_Hummingbird",
    "Rufous_Hummingbird",
    "Green_Violetear",
    "Long_tailed_Jaeger",
    "Pomarine_Jaeger",
    "Blue_Jay",
    "Florida_Jay",
    "Green_Jay",
    "Dark_eyed_Junco",
    "Tropical_Kingbird",
    "Gray_Kingbird",
    "Belted_Kingfisher",
    "Green_Kingfisher",
    "Pied_Kingfisher",
    "Ringed_Kingfisher",
    "White_breasted_Kingfisher",
    "Red_legged_Kittiwake",
    "Horned_Lark",
    "Pacific_Loon",
    "Mallard",
    "Western_Meadowlark",
    "Hooded_Merganser",
    "Red_breasted_Merganser",
    "Mockingbird",
    "Nighthawk",
    "Clark_Nutcracker",
    "White_breasted_Nuthatch",
    "Baltimore_Oriole",
    "Hooded_Oriole",
    "Orchard_Oriole",
    "Scott_Oriole",
    "Ovenbird",
    "Brown_Pelican",
    "White_Pelican",
    "Western_Wood_Pewee",
    "Sayornis",
    "American_Pipit",
    "Whip_poor_Will",
    "Horned_Puffin",
    "Common_Raven",
    "White_necked_Raven",
    "American_Redstart",
    "Geococcyx",
    "Loggerhead_Shrike",
    "Great_Grey_Shrike",
    "Baird_Sparrow",
    "Black_throated_Sparrow",
    "Brewer_Sparrow",
    "Chipping_Sparrow",
    "Clay_colored_Sparrow",
    "House_Sparrow",
    "Field_Sparrow",
    "Fox_Sparrow",
    "Grasshopper_Sparrow",
    "Harris_Sparrow",
    "Henslow_Sparrow",
    "Le_Conte_Sparrow",
    "Lincoln_Sparrow",
    "Nelson_Sharp_tailed_Sparrow",
    "Savannah_Sparrow",
    "Seaside_Sparrow",
    "Song_Sparrow",
    "Tree_Sparrow",
    "Vesper_Sparrow",
    "White_crowned_Sparrow",
    "White_throated_Sparrow",
    "Cape_Glossy_Starling",
    "Bank_Swallow",
    "Barn_Swallow",
    "Cliff_Swallow",
    "Tree_Swallow",
    "Scarlet_Tanager",
    "Summer_Tanager",
    "Artic_Tern",
    "Black_Tern",
    "Caspian_Tern",
    "Common_Tern",
    "Elegant_Tern",
    "Forsters_Tern",
    "Least_Tern",
    "Green_tailed_Towhee",
    "Brown_Thrasher",
    "Sage_Thrasher",
    "Black_capped_Vireo",
    "Blue_headed_Vireo",
    "Philadelphia_Vireo",
    "Red_eyed_Vireo",
    "Warbling_Vireo",
    "White_eyed_Vireo",
    "Yellow_throated_Vireo",
    "Bay_breasted_Warbler",
    "Black_and_white_Warbler",
    "Black_throated_Blue_Warbler",
    "Blue_winged_Warbler",
    "Canada_Warbler",
    "Cape_May_Warbler",
    "Cerulean_Warbler",
    "Chestnut_sided_Warbler",
    "Golden_winged_Warbler",
    "Hooded_Warbler",
    "Kentucky_Warbler",
    "Magnolia_Warbler",
    "Mourning_Warbler",
    "Myrtle_Warbler",
    "Nashville_Warbler",
    "Orange_crowned_Warbler",
    "Palm_Warbler",
    "Pine_Warbler",
    "Prairie_Warbler",
    "Prothonotary_Warbler",
    "Swainson_Warbler",
    "Tennessee_Warbler",
    "Wilson_Warbler",
    "Worm_eating_Warbler",
    "Yellow_Warbler",
    "Northern_Waterthrush",
    "Louisiana_Waterthrush",
    "Bohemian_Waxwing",
    "Cedar_Waxwing",
    "American_Three_toed_Woodpecker",
    "Pileated_Woodpecker",
    "Red_bellied_Woodpecker",
    "Red_cockaded_Woodpecker",
    "Red_headed_Woodpecker",
    "Downy_Woodpecker",
    "Bewick_Wren",
    "Cactus_Wren",
    "Carolina_Wren",
    "House_Wren",
    "Marsh_Wren",
    "Rock_Wren",
    "Winter_Wren",
    "Common_Yellowthroat",
]
# Set of CUB attributes selected by Koh et al. [CBM Paper]
SELECTED_CONCEPTS = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]

# Names of all CUB attributes
CONCEPT_SEMANTICS = [
    "has_bill_shape::curved_(up_or_down)",
    "has_bill_shape::dagger",
    "has_bill_shape::hooked",
    "has_bill_shape::needle",
    "has_bill_shape::hooked_seabird",
    "has_bill_shape::spatulate",
    "has_bill_shape::all-purpose",
    "has_bill_shape::cone",
    "has_bill_shape::specialized",
    "has_wing_color::blue",
    "has_wing_color::brown",
    "has_wing_color::iridescent",
    "has_wing_color::purple",
    "has_wing_color::rufous",
    "has_wing_color::grey",
    "has_wing_color::yellow",
    "has_wing_color::olive",
    "has_wing_color::green",
    "has_wing_color::pink",
    "has_wing_color::orange",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_wing_color::red",
    "has_wing_color::buff",
    "has_upperparts_color::blue",
    "has_upperparts_color::brown",
    "has_upperparts_color::iridescent",
    "has_upperparts_color::purple",
    "has_upperparts_color::rufous",
    "has_upperparts_color::grey",
    "has_upperparts_color::yellow",
    "has_upperparts_color::olive",
    "has_upperparts_color::green",
    "has_upperparts_color::pink",
    "has_upperparts_color::orange",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_upperparts_color::red",
    "has_upperparts_color::buff",
    "has_underparts_color::blue",
    "has_underparts_color::brown",
    "has_underparts_color::iridescent",
    "has_underparts_color::purple",
    "has_underparts_color::rufous",
    "has_underparts_color::grey",
    "has_underparts_color::yellow",
    "has_underparts_color::olive",
    "has_underparts_color::green",
    "has_underparts_color::pink",
    "has_underparts_color::orange",
    "has_underparts_color::black",
    "has_underparts_color::white",
    "has_underparts_color::red",
    "has_underparts_color::buff",
    "has_breast_pattern::solid",
    "has_breast_pattern::spotted",
    "has_breast_pattern::striped",
    "has_breast_pattern::multi-colored",
    "has_back_color::blue",
    "has_back_color::brown",
    "has_back_color::iridescent",
    "has_back_color::purple",
    "has_back_color::rufous",
    "has_back_color::grey",
    "has_back_color::yellow",
    "has_back_color::olive",
    "has_back_color::green",
    "has_back_color::pink",
    "has_back_color::orange",
    "has_back_color::black",
    "has_back_color::white",
    "has_back_color::red",
    "has_back_color::buff",
    "has_tail_shape::forked_tail",
    "has_tail_shape::rounded_tail",
    "has_tail_shape::notched_tail",
    "has_tail_shape::fan-shaped_tail",
    "has_tail_shape::pointed_tail",
    "has_tail_shape::squared_tail",
    "has_upper_tail_color::blue",
    "has_upper_tail_color::brown",
    "has_upper_tail_color::iridescent",
    "has_upper_tail_color::purple",
    "has_upper_tail_color::rufous",
    "has_upper_tail_color::grey",
    "has_upper_tail_color::yellow",
    "has_upper_tail_color::olive",
    "has_upper_tail_color::green",
    "has_upper_tail_color::pink",
    "has_upper_tail_color::orange",
    "has_upper_tail_color::black",
    "has_upper_tail_color::white",
    "has_upper_tail_color::red",
    "has_upper_tail_color::buff",
    "has_head_pattern::spotted",
    "has_head_pattern::malar",
    "has_head_pattern::crested",
    "has_head_pattern::masked",
    "has_head_pattern::unique_pattern",
    "has_head_pattern::eyebrow",
    "has_head_pattern::eyering",
    "has_head_pattern::plain",
    "has_head_pattern::eyeline",
    "has_head_pattern::striped",
    "has_head_pattern::capped",
    "has_breast_color::blue",
    "has_breast_color::brown",
    "has_breast_color::iridescent",
    "has_breast_color::purple",
    "has_breast_color::rufous",
    "has_breast_color::grey",
    "has_breast_color::yellow",
    "has_breast_color::olive",
    "has_breast_color::green",
    "has_breast_color::pink",
    "has_breast_color::orange",
    "has_breast_color::black",
    "has_breast_color::white",
    "has_breast_color::red",
    "has_breast_color::buff",
    "has_throat_color::blue",
    "has_throat_color::brown",
    "has_throat_color::iridescent",
    "has_throat_color::purple",
    "has_throat_color::rufous",
    "has_throat_color::grey",
    "has_throat_color::yellow",
    "has_throat_color::olive",
    "has_throat_color::green",
    "has_throat_color::pink",
    "has_throat_color::orange",
    "has_throat_color::black",
    "has_throat_color::white",
    "has_throat_color::red",
    "has_throat_color::buff",
    "has_eye_color::blue",
    "has_eye_color::brown",
    "has_eye_color::purple",
    "has_eye_color::rufous",
    "has_eye_color::grey",
    "has_eye_color::yellow",
    "has_eye_color::olive",
    "has_eye_color::green",
    "has_eye_color::pink",
    "has_eye_color::orange",
    "has_eye_color::black",
    "has_eye_color::white",
    "has_eye_color::red",
    "has_eye_color::buff",
    "has_bill_length::about_the_same_as_head",
    "has_bill_length::longer_than_head",
    "has_bill_length::shorter_than_head",
    "has_forehead_color::blue",
    "has_forehead_color::brown",
    "has_forehead_color::iridescent",
    "has_forehead_color::purple",
    "has_forehead_color::rufous",
    "has_forehead_color::grey",
    "has_forehead_color::yellow",
    "has_forehead_color::olive",
    "has_forehead_color::green",
    "has_forehead_color::pink",
    "has_forehead_color::orange",
    "has_forehead_color::black",
    "has_forehead_color::white",
    "has_forehead_color::red",
    "has_forehead_color::buff",
    "has_under_tail_color::blue",
    "has_under_tail_color::brown",
    "has_under_tail_color::iridescent",
    "has_under_tail_color::purple",
    "has_under_tail_color::rufous",
    "has_under_tail_color::grey",
    "has_under_tail_color::yellow",
    "has_under_tail_color::olive",
    "has_under_tail_color::green",
    "has_under_tail_color::pink",
    "has_under_tail_color::orange",
    "has_under_tail_color::black",
    "has_under_tail_color::white",
    "has_under_tail_color::red",
    "has_under_tail_color::buff",
    "has_nape_color::blue",
    "has_nape_color::brown",
    "has_nape_color::iridescent",
    "has_nape_color::purple",
    "has_nape_color::rufous",
    "has_nape_color::grey",
    "has_nape_color::yellow",
    "has_nape_color::olive",
    "has_nape_color::green",
    "has_nape_color::pink",
    "has_nape_color::orange",
    "has_nape_color::black",
    "has_nape_color::white",
    "has_nape_color::red",
    "has_nape_color::buff",
    "has_belly_color::blue",
    "has_belly_color::brown",
    "has_belly_color::iridescent",
    "has_belly_color::purple",
    "has_belly_color::rufous",
    "has_belly_color::grey",
    "has_belly_color::yellow",
    "has_belly_color::olive",
    "has_belly_color::green",
    "has_belly_color::pink",
    "has_belly_color::orange",
    "has_belly_color::black",
    "has_belly_color::white",
    "has_belly_color::red",
    "has_belly_color::buff",
    "has_wing_shape::rounded-wings",
    "has_wing_shape::pointed-wings",
    "has_wing_shape::broad-wings",
    "has_wing_shape::tapered-wings",
    "has_wing_shape::long-wings",
    "has_size::large_(16_-_32_in)",
    "has_size::small_(5_-_9_in)",
    "has_size::very_large_(32_-_72_in)",
    "has_size::medium_(9_-_16_in)",
    "has_size::very_small_(3_-_5_in)",
    "has_shape::upright-perching_water-like",
    "has_shape::chicken-like-marsh",
    "has_shape::long-legged-like",
    "has_shape::duck-like",
    "has_shape::owl-like",
    "has_shape::gull-like",
    "has_shape::hummingbird-like",
    "has_shape::pigeon-like",
    "has_shape::tree-clinging-like",
    "has_shape::hawk-like",
    "has_shape::sandpiper-like",
    "has_shape::upland-ground-like",
    "has_shape::swallow-like",
    "has_shape::perching-like",
    "has_back_pattern::solid",
    "has_back_pattern::spotted",
    "has_back_pattern::striped",
    "has_back_pattern::multi-colored",
    "has_tail_pattern::solid",
    "has_tail_pattern::spotted",
    "has_tail_pattern::striped",
    "has_tail_pattern::multi-colored",
    "has_belly_pattern::solid",
    "has_belly_pattern::spotted",
    "has_belly_pattern::striped",
    "has_belly_pattern::multi-colored",
    "has_primary_color::blue",
    "has_primary_color::brown",
    "has_primary_color::iridescent",
    "has_primary_color::purple",
    "has_primary_color::rufous",
    "has_primary_color::grey",
    "has_primary_color::yellow",
    "has_primary_color::olive",
    "has_primary_color::green",
    "has_primary_color::pink",
    "has_primary_color::orange",
    "has_primary_color::black",
    "has_primary_color::white",
    "has_primary_color::red",
    "has_primary_color::buff",
    "has_leg_color::blue",
    "has_leg_color::brown",
    "has_leg_color::iridescent",
    "has_leg_color::purple",
    "has_leg_color::rufous",
    "has_leg_color::grey",
    "has_leg_color::yellow",
    "has_leg_color::olive",
    "has_leg_color::green",
    "has_leg_color::pink",
    "has_leg_color::orange",
    "has_leg_color::black",
    "has_leg_color::white",
    "has_leg_color::red",
    "has_leg_color::buff",
    "has_bill_color::blue",
    "has_bill_color::brown",
    "has_bill_color::iridescent",
    "has_bill_color::purple",
    "has_bill_color::rufous",
    "has_bill_color::grey",
    "has_bill_color::yellow",
    "has_bill_color::olive",
    "has_bill_color::green",
    "has_bill_color::pink",
    "has_bill_color::orange",
    "has_bill_color::black",
    "has_bill_color::white",
    "has_bill_color::red",
    "has_bill_color::buff",
    "has_crown_color::blue",
    "has_crown_color::brown",
    "has_crown_color::iridescent",
    "has_crown_color::purple",
    "has_crown_color::rufous",
    "has_crown_color::grey",
    "has_crown_color::yellow",
    "has_crown_color::olive",
    "has_crown_color::green",
    "has_crown_color::pink",
    "has_crown_color::orange",
    "has_crown_color::black",
    "has_crown_color::white",
    "has_crown_color::red",
    "has_crown_color::buff",
    "has_wing_pattern::solid",
    "has_wing_pattern::spotted",
    "has_wing_pattern::striped",
    "has_wing_pattern::multi-colored",
]

# Generate a mapping containing all concept groups in CUB generated
# using a simple prefix tree
CONCEPT_GROUP_MAP = defaultdict(list)
for i, concept_name in enumerate(list(
    np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]
)):
    group = concept_name[:concept_name.find("::")]
    CONCEPT_GROUP_MAP[group].append(i)
CONCEPT_GROUP_MAP = dict(CONCEPT_GROUP_MAP)

# Ordered names for the 112 selected concepts (matches order in pkl files)
SELECTED_CONCEPT_NAMES: List[str] = [CONCEPT_SEMANTICS[i] for i in SELECTED_CONCEPTS]


class CUBDataset(ConceptDataset):
    """Dataset class for CUB-200-2011 (Caltech-UCSD Birds).

    CUB-200-2011 contains 11,788 bird images across 200 species classes,
    annotated with 112 binary semantic attributes selected by Koh et al.
    [CBM Paper] from the full set of 312 CUB attributes.

    Official train / val / test splits from the pre-processed pickle files
    are preserved; use :class:`~torch_concepts.data.splitters.NativeSplitter`
    in the corresponding datamodule.

    The concept vector per sample contains:

    - columns 0-111: 112 binary semantic attributes (cardinality 1 each)
    - column 112:    bird species index 0-199 (cardinality 200)

    Parameters
    ----------
    root : str, optional
        Root directory that contains ``class_attr_data_10/`` and
        ``CUB_200_2011/``.  Defaults to ``./data/CUB200``.
    image_size : int, optional
        Side length (px) images are resized to.  Defaults to 224.
    concept_subset : list of str, optional
        Subset of concept names to retain.  ``None`` keeps all 113.
    label_descriptions : dict, optional
        Mapping from concept name to human-readable description.
    """

    def __init__(
        self,
        root: str = None,
        image_size: int = 224,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[Mapping] = None,
    ):
        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'CUB200')
        self.root = root
        self.image_size = image_size
        self.label_descriptions = label_descriptions

        filenames, concepts, annotations, graph = self.load()

        super().__init__(
            input_data=filenames,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name='CUBDataset',
        )

    # ------------------------------------------------------------------
    # ConceptDataset interface
    # ------------------------------------------------------------------

    @property
    def raw_filenames(self) -> List[str]:
        return [
            "attributes",
            "images",
            "parts",
            "attributes.txt",
            "bounding_boxes.txt",
            "classes.txt",
            "image_class_labels.txt",
            "images.txt",
            "train_test_split.txt", # split with left out classes (not used in our setting)
            "class_attr_data_10/train.pkl", # splits with all classes, from Koh et al.'s pre-processing
            "class_attr_data_10/val.pkl",
            "class_attr_data_10/test.pkl",
        ]

    @property
    def processed_filenames(self) -> List[str]:
        return [
            'filenames.txt',
            'concepts.pt',
            'annotations.pt',
            'split_mapping.h5',
        ]

    def download(self) -> None:
        """Downloads the CUB dataset if it is not already present."""
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            
        # store the Koh et al. pre-processed pickle files in a subfolder "class_attr_data_10"
        class_attr_dir = os.path.join(self.root, "class_attr_data_10")
        if not os.path.exists(class_attr_dir):
            os.makedirs(class_attr_dir)
            for split_name in ('train', 'val', 'test'):
                url = f"{URLS[0]}/{split_name}.pkl"
                download_url(url, class_attr_dir)

        tgz_path = download_url(URLS[1], self.root)
        
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=self.root)
        os.unlink(tgz_path)

        # Move all the files outside of the nested "CUB_200_2011" folder to the root
        extracted_folder = os.path.join(self.root, "CUB_200_2011")
        for item in os.listdir(extracted_folder):
            src = os.path.join(extracted_folder, item)
            dst = os.path.join(self.root, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(extracted_folder)

    def _remap_image_path(self, img_path: str) -> str:
        """Remap the absolute path stored in a pkl entry to the local root.

        The Koh et al. pkl files embed absolute paths from their cluster
        (``/juice/scr/.../datasets/``).  We extract the ``CUB_200_2011/``
        subtree and join it with the local root.
        """
        marker = 'CUB_200_2011'
        idx = img_path.find(marker)
        if idx >= 0:
            relative = img_path[idx:]  # e.g. "CUB_200_2011/images/.../file.jpg"
            # Eliminate CUB_200_2011 from path
            relative = relative[len(marker) + 1:]  # e.g. "images/.../file.jpg"
            return os.path.abspath(os.path.join(self.root, relative))
        # Fallback: replace the known cluster prefix
        return img_path.replace(
            '/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/',
            self.root + os.sep,
        )

    def build(self):
        """Process raw CUB pickle files and save cached dataset artefacts."""
        self.maybe_download()

        logger.info(f"Building CUB dataset from {self.root} ...")

        all_paths: List[str] = []
        all_attrs: List[List[int]] = []
        all_classes: List[int] = []
        split_labels: List[str] = []

        for split_name in ('train', 'val', 'test'):
            pkl_path = os.path.join(
                self.root, 'class_attr_data_10', f'{split_name}.pkl'
            )
            with open(pkl_path, 'rb') as fh:
                entries = pickle.load(fh)

            for entry in entries:
                img_path = self._remap_image_path(entry['img_path'])
                all_paths.append(img_path)
                all_attrs.append(entry['attribute_label'])   # 112-dim list
                all_classes.append(int(entry['class_label']))
                split_labels.append(split_name)

        n = len(all_paths)
        logger.info(f"Loaded {n} samples (train/val/test)")

        # Build concept tensor: 112 binary attrs + class index
        attr_array = np.array(all_attrs, dtype=np.float32)               # (n, 112)
        class_array = np.array(all_classes, dtype=np.float32).reshape(-1, 1)  # (n, 1)
        all_concepts_np = np.concatenate([attr_array, class_array], axis=1)   # (n, 113)
        concepts_tensor = torch.tensor(all_concepts_np, dtype=torch.float32)

        # Build Annotations
        concept_names = SELECTED_CONCEPT_NAMES + ['class']
        binary_states = [['0'] for _ in SELECTED_CONCEPT_NAMES]
        states = binary_states + [CLASS_NAMES]
        cardinalities = [1] * len(SELECTED_CONCEPT_NAMES) + [N_CLASSES]
        concept_metadata = {name: {'type': 'discrete'} for name in concept_names}

        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                states=states,
                cardinalities=cardinalities,
                metadata=concept_metadata,
            )
        })

        # Build split mapping (native train/val/test)
        split_series = pd.Series(split_labels, name='split')

        # Save artefacts
        os.makedirs(self.root, exist_ok=True)
        logger.info(f"Saving CUB dataset artefacts to {self.root}")

        with open(self.processed_paths[0], 'w') as fh:
            fh.write('\n'.join(all_paths))

        torch.save(concepts_tensor, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])
        split_series.to_hdf(self.processed_paths[3], key='split_mapping')

        logger.info(f"CUB dataset saved ({n} samples)")

    def load_raw(self):
        """Load processed artefacts from disk."""
        self.maybe_build()

        logger.info(f"Loading CUB dataset from {self.root}")

        with open(self.processed_paths[0], 'r') as fh:
            filenames = fh.read().strip().split('\n')

        concepts = torch.load(self.processed_paths[1], weights_only=False)
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = None

        return filenames, concepts, annotations, graph

    def load(self):
        return self.load_raw()

    def __getitem__(self, item: int) -> dict:
        if self.embs_precomputed:
            x = self.input_data[item]
        else:
            img_path = self.input_data[item]
            x = Image.open(img_path).convert('RGB')
            x = transforms.Resize((self.image_size, self.image_size))(x)
            x = transforms.ToTensor()(x)
        c = self.concepts[item]
        return {'inputs': {'x': x}, 'concepts': {'c': c}}

    # ------------------------------------------------------------------
    # Properties — override base class which assumes input_data is a Tensor
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return len(self.input_data)

    @property
    def n_features(self) -> tuple:
        return tuple(self[0]['inputs']['x'].shape)

    @property
    def shape(self) -> tuple:
        return (self.n_samples, *self.n_features)

