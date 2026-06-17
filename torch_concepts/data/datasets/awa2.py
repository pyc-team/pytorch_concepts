"""
Animals with Attributes 2 (AwA2) Dataset

Adapted from https://github.com/xmed-lab/ECBM/blob/main/data/awa2.py and
https://github.com/mateoespinosa/cem/blob/mateo/probcbm/cem/data/awa2_loader.py

Credit goes to Xinyue Xu, Yi Qin, Lu Mi, Hao Wang, and Xiaomeng Li and the
code accompanying their paper "Energy-Based Concept Bottleneck Models:
Unifying Prediction, Concept Intervention, and Probabilistic Interpretations".
"""

import os
import logging
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import List, Mapping, Optional
import zipfile
from pathlib import Path

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset
from torch_concepts.data.io import download_url_wget, zip_is_valid


logger = logging.getLogger(__name__)


def _import_torchvision():
    """Lazily import torchvision, raising a clear error if it is not installed."""
    try:
        import torchvision as tv
        return tv
    except ImportError as exc:
        raise ImportError(
            "AWA2Dataset image loading requires `torchvision`. "
            "Install it with: pip install torchvision"
        ) from exc

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

URLS = [
    "https://cvml.ista.ac.at/AwA2/AwA2-base.zip",
    "https://cvml.ista.ac.at/AwA2/AwA2-features.zip",
    "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip",
    "https://cvml.ista.ac.at/AwA2/AwA2-data.zip",
]

N_CLASSES = 50

CLASS_NAMES = [
    'antelope',
    'grizzly+bear',
    'killer+whale',
    'beaver',
    'dalmatian',
    'persian+cat',
    'horse',
    'german+shepherd',
    'blue+whale',
    'siamese+cat',
    'skunk',
    'mole',
    'tiger',
    'hippopotamus',
    'leopard',
    'moose',
    'spider+monkey',
    'humpback+whale',
    'elephant',
    'gorilla',
    'ox',
    'fox',
    'sheep',
    'seal',
    'chimpanzee',
    'hamster',
    'squirrel',
    'rhinoceros',
    'rabbit',
    'bat',
    'giraffe',
    'wolf',
    'chihuahua',
    'rat',
    'weasel',
    'otter',
    'buffalo',
    'zebra',
    'giant+panda',
    'deer',
    'bobcat',
    'pig',
    'lion',
    'mouse',
    'polar+bear',
    'collie',
    'walrus',
    'raccoon',
    'cow',
    'dolphin',
]

CONCEPT_SEMANTICS = [
    'black',
    'white',
    'blue',
    'brown',
    'gray',
    'orange',
    'red',
    'yellow',
    'patches',
    'spots',
    'stripes',
    'furry',
    'hairless',
    'toughskin',
    'big',
    'small',
    'bulbous',
    'lean',
    'flippers',
    'hands',
    'hooves',
    'pads',
    'paws',
    'longleg',
    'longneck',
    'tail',
    'chewteeth',
    'meatteeth',
    'buckteeth',
    'strainteeth',
    'horns',
    'claws',
    'tusks',
    'smelly',
    'flys',
    'hops',
    'swims',
    'tunnels',
    'walks',
    'fast',
    'slow',
    'strong',
    'weak',
    'muscle',
    'bipedal',
    'quadrapedal',
    'active',
    'inactive',
    'nocturnal',
    'hibernate',
    'agility',
    'fish',
    'meat',
    'plankton',
    'vegetation',
    'insects',
    'forager',
    'grazer',
    'hunter',
    'scavenger',
    'skimmer',
    'stalker',
    'newworld',
    'oldworld',
    'arctic',
    'coastal',
    'desert',
    'bush',
    'plains',
    'forest',
    'fields',
    'jungle',
    'mountains',
    'ocean',
    'ground',
    'water',
    'tree',
    'cave',
    'fierce',
    'timid',
    'smart',
    'group',
    'solitary',
    'nestspot',
    'domestic',
]

CONCEPT_GROUPS = {
    'color': ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow'],
    'fur_pattern': ['patches', 'spots', 'stripes', 'furry', 'hairless', 'toughskin'],
    'size': ['big', 'small', 'bulbous', 'lean'],
    'limb_shape': ['flippers', 'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck'],
    'tail': ['tail'],
    'teeth_type': ['chewteeth','meatteeth','buckteeth','strainteeth'],
    'horns': ['horns'],
    'claws': ['claws'],
    'tusks': ['tusks'],
    'smelly': ['smelly'],
    'transport_mechanism': ['flys', 'hops', 'swims', 'tunnels', 'walks'],
    'speed': ['fast', 'slow'],
    'strength': ['strong', 'weak'],
    'muscle': ['muscle'],
    'movement_move': ['bipedal', 'quadrapedal'],
    'active': ['active', 'inactive'],
    'nocturnal': ['nocturnal'],
    'hibernate': ['hibernate'],
    'agility': ['agility'],
    'diet': ['fish', 'meat', 'plankton', 'vegetation', 'insects'],
    'feeding_type': ['forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker'],
    'general_location': ['newworld', 'oldworld', 'arctic'],
    'biome': ['coastal', 'desert', 'bush', 'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean', 'ground', 'water', 'tree', 'cave'],
    'fierceness': ['fierce', 'timid'],
    'smart': ['smart'],
    'social_mode': ['group', 'solitary'],
    'nestspot': ['nestspot'],
    'domestic': ['domestic'],
}
CONCEPT_GROUPS = {
    key: [CONCEPT_SEMANTICS.index(name) for name in concept_names]
    for key, concept_names in CONCEPT_GROUPS.items()
}


class AWA2Dataset(ConceptDataset):
    """Dataset class for Animals with Attributes 2 (AwA2).

    AwA2 pairs 37,322 animal images across 50 classes with 85 binary semantic
    attributes (colour, shape, behaviour, habitat, ...).

    The concept vector per sample contains:

    - columns 0-84: 85 binary semantic attributes (cardinality 1 each)
    - column 85:    animal class index 0-49 (cardinality 50)

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored.
        Defaults to ``./data/AWA2``.
    image_size : int, optional
        Side length (px) to resize images to.  Default: 224.
    concept_subset : list of str, optional
        Subset of concept names to retain. ``None`` keeps all 86.
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
            root = os.path.join(os.getcwd(), 'data', 'AWA2')
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
            name='AWA2Dataset',
        )

    # ------------------------------------------------------------------
    # ConceptDataset interface
    # ------------------------------------------------------------------

    @property
    def raw_filenames(self) -> List[str]:
        return [
            "classes.txt",
            "predicate-matrix-binary.txt",
            "predicate-matrix.png",
            "README-attributes.txt",
            "testclasses.txt",
            "predicate-matrix-continuous.txt",
            "predicates.txt",
            "README-images.txt",
            "trainclasses.txt",

            "Features/ResNet101/AwA2-features.txt",
            "Features/ResNet101/AwA2-filenames.txt",
            "Features/ResNet101/AwA2-labels.txt",

            # NOTE: We are not checking if each folder contains all images. Be sure that JPEG images are contained in each class folder.
            *[f"JPEGImages/{x}" for x in CLASS_NAMES],
        ]

    @property
    def processed_filenames(self) -> List[str]:
        return [
            'filenames.txt',
            'concepts.pt',
            'annotations.pt',
        ]

    def download(self):
        """Download raw AwA2 data from official sources.

        Each zip is verified with ``zipfile.testzip()`` after download.  If the
        CRC check fails the corrupted file is deleted and re-downloaded from
        scratch (up to ``_MAX_RETRIES`` times).  When ``wget`` is available it
        is preferred over urllib because it handles large files, retries, and
        resume much more reliably.
        """
        _MAX_RETRIES = 3
        Path(self.root).mkdir(parents=True, exist_ok=True)
        for url in URLS:
            dest = os.path.join(self.root, url.split("/")[-1])
            for attempt in range(1, _MAX_RETRIES + 1):
                download_url_wget(url, dest)
                print(f"  Verifying {os.path.basename(dest)} (attempt {attempt}/{_MAX_RETRIES}) ...")
                if zip_is_valid(dest):
                    break
                print(f"  CRC check failed — deleting corrupted file and retrying ...")
                os.remove(dest)
            else:
                raise RuntimeError(
                    f"Failed to download a valid '{os.path.basename(dest)}' "
                    f"after {_MAX_RETRIES} attempts.  "
                    "Check your network connection or disk space."
                )
            print(f"  Extracting {os.path.basename(dest)} ...")
            with zipfile.ZipFile(dest) as z:
                z.extractall(self.root)
            os.remove(dest)

        # Move all the files outside of the nested "Animals_with_Attributes2" folder to the root
        extracted_folder = os.path.join(self.root, "Animals_with_Attributes2")
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


    def build(self):
        """Process raw AwA2 files and save cached dataset artefacts."""
        self.maybe_download()

        logger.info(f"Building AWA2 dataset from {self.root} ...")

        # Load predicate matrix (50 classes x 85 attributes)
        predicate_mat = np.array(
            np.genfromtxt(
                os.path.join(self.root, 'predicate-matrix-binary.txt'),
                dtype='int',
            )
        )

        # Build class-name -> index mapping
        class_to_index = {}
        with open(os.path.join(self.root, 'classes.txt')) as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    class_to_index[parts[1].strip()] = len(class_to_index)

        # Walk JPEGImages/ to collect image paths and class labels
        all_paths: List[str] = []
        all_class_indices: List[int] = []
        img_dir = os.path.join(self.root, 'JPEGImages')

        for dirpath, _, files in os.walk(img_dir):
            class_name = os.path.basename(dirpath)
            if class_name not in class_to_index:
                continue
            cls_idx = class_to_index[class_name]
            for fname in sorted(files):
                if fname.lower().endswith('.jpg'):
                    all_paths.append(
                        os.path.abspath(os.path.join(dirpath, fname))
                    )
                    all_class_indices.append(cls_idx)

        all_paths_arr = np.array(all_paths)
        all_class_arr = np.array(all_class_indices, dtype=np.int64)
        n = len(all_paths_arr)

        # Build concept tensor: 85 binary attrs + 1 class label
        binary_attrs = predicate_mat[all_class_arr, :]        # (n, 85)
        class_col = all_class_arr.reshape(-1, 1)               # (n, 1)
        all_concepts = np.concatenate([binary_attrs, class_col], axis=1)
        concepts_tensor = torch.tensor(all_concepts, dtype=torch.float32)

        # Build Annotations
        concept_names = CONCEPT_SEMANTICS + ['class']
        binary_states = [['0'] for _ in CONCEPT_SEMANTICS]
        class_states = [CLASS_NAMES]
        states = binary_states + class_states
        cardinalities = [1] * len(CONCEPT_SEMANTICS) + [N_CLASSES]
        concept_metadata = {name: {'type': 'discrete'} for name in concept_names}

        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                states=states,
                cardinalities=cardinalities,
                metadata=concept_metadata,
            )
        })

        # Save artefacts
        os.makedirs(self.root, exist_ok=True)
        logger.info(f"Saving AWA2 dataset to {self.root}")

        with open(self.processed_paths[0], 'w') as fh:
            fh.write('\n'.join(all_paths_arr.tolist()))

        torch.save(concepts_tensor, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])

        logger.info(f"AWA2 dataset saved ({n} samples)")

    def load_raw(self):
        """Load processed artefacts from disk."""
        self.maybe_build()

        logger.info(f"Loading AWA2 dataset from {self.root}")

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
            tv = _import_torchvision()
            x = Image.open(img_path)
            x = x.convert('RGB')  # Ensure 3 channels
            x = tv.transforms.Resize((self.image_size, self.image_size))(x)  # Resize to 224x224
            x = tv.transforms.ToTensor()(x)  # Convert to tensor and scale to [0, 1]
        c = self.concepts[item]
        return {'inputs': {'x': x}, 'concepts': {'c': c}}

    @property
    def n_samples(self) -> int:
        return len(self.input_data)

    @property
    def n_features(self) -> tuple:
        return tuple(self[0]['inputs']['x'].shape)

    @property
    def shape(self) -> tuple:
        return (self.n_samples, *self.n_features)