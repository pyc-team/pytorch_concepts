"""
General utils for training, evaluation and data loading

Heavily adapted from https://github.com/xmed-lab/ECBM/blob/main/data/awa2.py and
https://github.com/mateoespinosa/cem/blob/mateo/probcbm/cem/data/awa2_loader.py

Credit goes to Xinyue Xu, Yi Qin, Lu Mi, Hao Wang, and Xiaomeng Li
and the code accompanying their paper "Energy-Based Concept Bottleneck Models:
Unifying Prediction, Concept Intervention, and Probabilistic Interpretations"

The data can be downloaded from: https://cvml.ista.ac.at/AwA2/

"""
import numpy as np
import os
import logging
import sklearn
import torch
import torchvision.transforms as transforms

from functools import reduce
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader

logger = logging.getLogger(__name__)

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50

# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", 'data/AwA2/')


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



class AwA2Dataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the AwA2 dataset
    """

    def __init__(
        self,
        root,
        training_augment=True,
        split='train',
        image_size=224,
        concept_transform=None,
        sample_transform=None,
        selected_concepts=None,
        seed=42,
    ):
        self.root = root
        self.training_augment = training_augment
        self.split = split
        self.concept_transform = concept_transform or (lambda x: x)
        self.name = 'AwA2'

        if not os.path.exists(self.root):
            raise ValueError(
                f'{self.root} does not exist yet. Please download the '
                f'dataset first.'
            )

        if split == 'train':
            self.transform = get_transform_awa2(
                train=True,
                augment_data=training_augment,
                image_size=image_size,
                sample_transform=sample_transform,
            )
        else:
            self.transform = get_transform_awa2(
                train=False,
                augment_data=False,
                image_size=image_size,
                sample_transform=sample_transform,
            )


        self.predicate_binary_mat = np.array(np.genfromtxt(
            os.path.join(root, 'predicate-matrix-binary.txt'),
            dtype='int',
        ))
        self.class_to_index = dict()
        # Build dictionary of indices to classes
        with open(f"{root}/classes.txt") as f:
            for line in f:
                class_name = line.split('\t')[1].strip()
                self.class_to_index[class_name] = len(self.class_to_index)

        for split_attempt in ['train', 'val', 'test']:
            split_file = os.path.join(
                self.root,
                f'{split_attempt}_split.npz',
            )
            if not os.path.exists(split_file):
                logger.info(
                    f"Split files for AWA2 could not be found. Generating new "
                    f"train, validation, and test splits with seed {seed}."
                )
                self._generate_splits(seed=seed)
                break

        # And now we can simply load the actual paths and classes to be used
        # for each split :)
        split_file = os.path.join(
            self.root,
            f'{split}_split.npz',
        )
        split_info = np.load(split_file)
        self.img_paths = split_info['paths']
        self.img_labels = split_info['labels']
        if selected_concepts is None:
            selected_concepts = list(range(len(CONCEPT_SEMANTICS)))
        self.selected_concepts = selected_concepts
        self.concept_names = self.concept_attr_names = list(
            np.array(
                CONCEPT_SEMANTICS
            )[selected_concepts]
        )
        self.task_names = self.task_attr_names = CLASS_NAMES

    def _generate_splits(self, seed, train_size=0.6, val_size=0.2):
        # First find all samples and generate a list of their paths
        image_paths = []
        image_classes = []
        img_dir = os.path.join(self.root, 'JPEGImages')
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    image_paths.append(os.path.abspath(os.path.join(root, file)))
                    parent_dir = os.path.basename(
                        os.path.dirname(image_paths[-1])
                    )
                    image_classes.append(self.class_to_index[parent_dir])

        np.random.seed(seed)
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)

        train_end = int(train_size * len(image_paths))
        val_end = train_end + int(val_size * len(image_paths))

        # Now time to generate our split matrices and saving them
        image_paths = np.array(image_paths)
        image_classes = np.array(image_classes)

        train_indices = indices[:train_end]
        train_paths = image_paths[train_indices]
        train_classes = image_classes[train_indices]
        np.savez(
            os.path.join(self.root, 'train_split.npz'),
            paths=train_paths,
            labels=train_classes,
        )

        val_indices = indices[train_end:val_end]
        val_paths = image_paths[val_indices]
        val_classes = image_classes[val_indices]
        np.savez(
            os.path.join(self.root, 'val_split.npz'),
            paths=val_paths,
            labels=val_classes,
        )

        test_indices = indices[val_end:]
        test_paths = image_paths[test_indices]
        test_classes = image_classes[test_indices]
        np.savez(
            os.path.join(self.root, 'test_split.npz'),
            paths=test_paths,
            labels=test_classes,
        )

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if img.getbands()[0] == 'L':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_idx = self.img_labels[index]
        concepts = self.predicate_binary_mat[label_idx,:]
        concepts = self.concept_transform(
            np.array(concepts)[self.selected_concepts]
        )
        return img, torch.FloatTensor(concepts), label_idx

    def __len__(self):
        return len(self.img_paths)


def get_transform_awa2(
    train,
    augment_data,
    image_size=224,
    sample_transform=None,
):
    """Helper function to get the appropiate transformation for the awa2
    data loader.

    Args:
        train (bool): Whether or not this transform is for the training fold
            of the awa2 dataset or not.
        augment_data (bool): Whether or not we want to perform standard
            augmentations (crops and flips) used for the CUB dataset.
        image_size (int, optional): Size of the width and height of each
            of the generated images. Defaults to 224.

    Returns:
        torchvision.Transform: a valid torchvision transform to be applied to
            each image of the awa2 dataset being loaded.
    """
    scale = 256.0/224.0
    sample_transform = (
        sample_transform if sample_transform is not None
        else (lambda x: x)
    )
    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((
                int(image_size*scale),
                int(image_size*scale),
            )),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            sample_transform,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            sample_transform,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform