"""
Animals with Attributes 2 (AwA2) Dataset

Adapted from https://github.com/xmed-lab/ECBM/blob/main/data/awa2.py and
https://github.com/mateoespinosa/cem/blob/mateo/probcbm/cem/data/awa2_loader.py

Credit goes to Xinyue Xu, Yi Qin, Lu Mi, Hao Wang, and Xiaomeng Li and the
code accompanying their paper "Energy-Based Concept Bottleneck Models:
Unifying Prediction, Concept Intervention, and Probabilistic Interpretations".

**Download instructions**

The dataset must be downloaded manually from https://cvml.ista.ac.at/AwA2/

The following files/directories are expected in ``root``:

- ``JPEGImages/``                  — images organised by class directory
- ``predicate-matrix-binary.txt``  — 50 x 85 binary attribute matrix
- ``classes.txt``                  — tab-separated (index, class_name) rows
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Mapping, Optional

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset

logger = logging.getLogger(__name__)

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50


#########################################################
## CONCEPT INFORMATION REGARDING AwA2
#########################################################

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

    The dataset is **not** downloaded automatically.  Please download the data
    from https://cvml.ista.ac.at/AwA2/ and place the following in ``root``:

    - ``JPEGImages/``                  — images organised by class directory
    - ``predicate-matrix-binary.txt``  — 50 x 85 binary attribute matrix
    - ``classes.txt``                  — list of (index, class_name) rows

    The concept vector per sample contains:

    - columns 0-84: 85 binary semantic attributes (cardinality 1 each)
    - column 85:    animal class index 0-49 (cardinality 50)

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored.
        Defaults to ``./data/AWA2``.
    image_size : int, optional
        Side length (pixels) for images loaded without a backbone. Default: 224.
    seed : int, optional
        Random seed for the train / val / test split. Default: 42.
    train_size : float, optional
        Fraction of samples for training. Default: 0.6.
    val_size : float, optional
        Fraction of samples for validation. Default: 0.2.
    concept_subset : list of str, optional
        Subset of concept names to retain. ``None`` keeps all 86.
    label_descriptions : dict, optional
        Mapping from concept name to human-readable description.
    """

    def __init__(
        self,
        root: str = None,
        image_size: int = 224,
        seed: int = 42,
        train_size: float = 0.6,
        val_size: float = 0.2,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[Mapping] = None,
    ):
        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'AWA2')
        self.root = root
        self.image_size = image_size
        self.seed = seed
        self.train_size = train_size
        self._val_frac = val_size
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
            'JPEGImages',
            'predicate-matrix-binary.txt',
            'classes.txt',
        ]

    @property
    def processed_filenames(self) -> List[str]:
        return [
            'filenames.txt',
            'concepts.pt',
            'annotations.pt',
            'split_mapping.h5',
        ]

    def download(self):
        raise FileNotFoundError(
            f"AwA2 data not found at '{self.root_dir}'.  "
            "Please download the dataset from https://cvml.ista.ac.at/AwA2/ "
            "and place JPEGImages/, predicate-matrix-binary.txt, and classes.txt "
            "in that directory."
        )

    def build(self):
        """Process raw AwA2 files and save cached dataset artefacts."""
        self.maybe_download()

        logger.info(f"Building AWA2 dataset from {self.root_dir} ...")

        # Load predicate matrix (50 classes x 85 attributes)
        predicate_mat = np.array(
            np.genfromtxt(
                os.path.join(self.root_dir, 'predicate-matrix-binary.txt'),
                dtype='int',
            )
        )

        # Build class-name -> index mapping
        class_to_index = {}
        with open(os.path.join(self.root_dir, 'classes.txt')) as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    class_to_index[parts[1].strip()] = len(class_to_index)

        # Walk JPEGImages/ to collect image paths and class labels
        all_paths: List[str] = []
        all_class_indices: List[int] = []
        img_dir = os.path.join(self.root_dir, 'JPEGImages')

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

        # Generate random train / val / test split
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        train_end = int(self.train_size * n)
        val_end = train_end + int(self._val_frac * n)

        split_labels = np.empty(n, dtype=object)
        split_labels[perm[:train_end]] = 'train'
        split_labels[perm[train_end:val_end]] = 'val'
        split_labels[perm[val_end:]] = 'test'

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
        os.makedirs(self.root_dir, exist_ok=True)
        logger.info(f"Saving AWA2 dataset to {self.root_dir}")

        with open(self.processed_paths[0], 'w') as fh:
            fh.write('\n'.join(all_paths_arr.tolist()))

        torch.save(concepts_tensor, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])
        pd.Series(split_labels.tolist()).to_hdf(
            self.processed_paths[3], key='split_mapping', mode='w'
        )

        logger.info(
            f"AWA2 dataset saved "
            f"(train={train_end}, val={val_end - train_end}, test={n - val_end})"
        )

    def load_raw(self):
        """Load processed artefacts from disk."""
        self.maybe_build()

        logger.info(f"Loading AWA2 dataset from {self.root_dir}")

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
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            x = get_transform_awa2(
                train=False, augment_data=False, image_size=self.image_size
            )(img)

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


# ---------------------------------------------------------------------------
# Legacy alias kept for backward compatibility
# ---------------------------------------------------------------------------
AwA2Dataset = AWA2Dataset


# ---------------------------------------------------------------------------
# Image-transform helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def get_transform_awa2(
    train: bool,
    augment_data: bool,
    image_size: int = 224,
    sample_transform=None,
):
    """Return a torchvision transform pipeline for AwA2 images.

    Parameters
    ----------
    train : bool
        Whether this transform is for the training fold.
    augment_data : bool
        Whether to apply random crop / flip augmentations.
    image_size : int, optional
        Target image side length.  Default: 224.
    sample_transform : callable, optional
        Additional per-sample transform inserted before normalisation.

    Returns
    -------
    torchvision.transforms.Compose
    """
    scale = 256.0 / 224.0
    sample_transform = sample_transform if sample_transform is not None else (lambda x: x)

    if (not train) or (not augment_data):
        return transforms.Compose([
            transforms.Resize((int(image_size * scale), int(image_size * scale))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            sample_transform,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            sample_transform,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])