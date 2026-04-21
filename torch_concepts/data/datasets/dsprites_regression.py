import os
import torch
import numpy as np
import logging
from typing import List, Optional, Dict
import sympy
import urllib
import sympytorch
from tqdm import tqdm

from ..base.dataset import ConceptDataset
from ...annotations import Annotations, AxisAnnotation
      
logger = logging.getLogger(__name__)

# Default available concept columns in dsprites
DSPRITES_CONCEPTS = ['color', 'shape', 'scale', 'orientation', 'x_position', 'y_position']

IDS_TO_SHAPES = {1: 'square', 2: 'circle', 3: 'heart'}

class DSpritesRegressionDataset(ConceptDataset):
    """DSprites regression dataset with sympy formula-based targets.

    Each sample is a 64x64 grayscale image of a simple shape with known
    generative factors (concepts). A per-shape sympy formula over the concept
    values produces the regression target.

    Parameters
    ----------
    root : str, optional
        Root directory for caching. Default: ``'./data/dsprites_regression'``.
    formulas : dict, optional
        Mapping from shape name ('square', 'circle', 'heart') to a sympy
        formula string using the concept column names as variables.
        Default: ``{'square': 'value_x_position + value_y_position',
        'circle': 'value_x_position * value_y_position',
        'heart': 'value_x_position - value_y_position'}``.
    num_samples : int, optional
        Number of samples to subsample. Default: None (all).
    seed : int, optional
        Random seed. Default: 42.
    concept_subset : list of str, optional
        Subset of concept names. Default: None.
    label_descriptions : dict, optional
        Optional dict mapping concept names to descriptions.
    """

    def __init__(
        self,
        root: str = None,
        formulas: Optional[Dict[str, str]] = None,
        seed: int = 42,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
    ):

        self.seed = seed
        self.label_descriptions = label_descriptions
        self.concept_subset = concept_subset

        if formulas is None:
            raise ValueError("Formulas must be provided for DSpritesRegressionDataset.")
        self.formulas = formulas

        self._concept_columns = DSPRITES_CONCEPTS

        # Check validity of formulas and subset of concepts
        self._check_concepts_and_formulas()

        # Set the sympy torch formulas for target computation
        self._torch_formulas = {}
        for shape, formula in self.formulas.items():
            torch_exp = sympytorch.SymPyModule(expressions=[sympy.sympify(formula)])
            self._torch_formulas[shape] = torch_exp

        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'dsprites_regression')
        self.root = root

        input_data, concepts_tensor, annotations, graph = self.load()

        super().__init__(
            input_data=input_data,
            concepts=concepts_tensor,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name="DSpritesRegressionDataset",
        )

    def _check_concepts_and_formulas(self):
        # Check that there is a formula for each shape
        for shape in IDS_TO_SHAPES.values():
            if shape not in self.formulas:
                raise ValueError(f"Missing formula for shape '{shape}'. Formulas must be provided for all shapes: {list(IDS_TO_SHAPES.values())}")

        # Check whether the formulas contain valide concept names
        for shape, formula in self.formulas.items():
            for var in sympy.sympify(formula).free_symbols:
                if str(var) not in self._concept_columns:
                    raise ValueError(f"Formula for shape '{shape}' contains unknown variable '{var}'. "
                                     f"Valid concept names are: {self._concept_columns}")
                
        # Check whether the subset of concepts selected by the user is valid
        if self.concept_subset is not None:
            for c in self.concept_subset:
                if c not in self._concept_columns:
                    raise ValueError(f"Selected concept '{c}' is not a valid concept name. "
                                     f"Valid concept names are: {self._concept_columns}")

    @property
    def raw_filenames(self) -> List[str]:
        return [
            "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        ]

    @property
    def processed_filenames(self) -> List[str]:
        return [
            f"images.pt",
            f"concepts.pt",
            f"annotations.pt",
        ]

    def download(self):
        """"Download the dSprites dataset from the original source and save to root directory."""

        url = "https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        filename = self.raw_filenames[0]
        filepath = os.path.join(self.root, filename)

        print(f"Downloading dSprites dataset...")
        print(f"Source: {url}")
        print(f"Destination: {filepath}")

        urllib.request.urlretrieve(url, filepath)
        print(f"\nDownload complete!")

    def build(self):
        """Extract concepts, compute formula targets, save to disk."""

        logger.info(f"Downloading DSprites regression dataset to {self.root}...")

        self.maybe_download()

        logger.info(f"Building DSprites regression dataset")

        # Load dsprties npz file
        dsprites_path = os.path.join(self.root, self.raw_filenames[0])
        dsprites_data = np.load(dsprites_path, allow_pickle=True)
        N = 10
        # NOTE uncomment N = dsprites_data['imgs'].shape[0]

        # Concept order: color, shape, scale, orientation, x_pos, y_pos
        concepts = dsprites_data['latents_values']

        concepts_list = []
        targets_list = []

        # compute the formula-based target for each sample according to its shape
        for idx in tqdm(range(N), desc="Computing targets given user-defined expressions"):
            concept_values = torch.tensor(concepts[idx], dtype=torch.float32)

            # Get shape name
            shape_id = concept_values[1].item()
            shape_name = IDS_TO_SHAPES[shape_id]

            # Compute formula target according to the shape
            var_dict = dict(zip(self._concept_columns, [concept_values[i] for i in range(len(self._concept_columns))]))
            target = self._torch_formulas[shape_name](**var_dict)

            # select the subset of concepts if specified
            if self.concept_subset is not None:
                concept_values = torch.tensor([var_dict[c] for c in self.concept_subset], dtype=torch.float32)

            concepts_list.append(concept_values.unsqueeze(0))  # (1, n_concepts)
            targets_list.append([target.item()])

        concepts_tensor = torch.cat(concepts_list, dim=0)
        targets_tensor = torch.tensor(targets_list, dtype=torch.float32)

        # Combine concepts + target
        cy = torch.cat([concepts_tensor, targets_tensor], dim=1)

        # Update concepts with the subset selected by the user, if specified
        if self.concept_subset is not None:
            cy_names = list(self.concept_subset) + ['target'] 
        else:
            cy_names = list(self._concept_columns) + ['target']
        
        concept_metadata = {name: {'type': 'continuous'} for name in cy_names}
        cardinalities = tuple([1] * len(cy_names))

        annotations = Annotations({
            1: AxisAnnotation(
                labels=cy_names,
                cardinalities=cardinalities,
                metadata=concept_metadata,
            )
        })

        # images
        images = dsprites_data['imgs']

        os.makedirs(self.root_dir, exist_ok=True)
        torch.save(torch.from_numpy(images), self.processed_paths[0])
        torch.save(cy, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])

        logger.info(f"DSprites regression dataset saved to {self.root_dir}")

    def load_raw(self):
        self.maybe_build()
        logger.info(f"Loading DSprites regression dataset from {self.root_dir}")

        input_data = torch.load(self.processed_paths[0], weights_only=False).numpy()
        concepts = torch.load(self.processed_paths[1], weights_only=False)
        annotations = torch.load(self.processed_paths[2], weights_only=False)

        return input_data, concepts, annotations, None

    def load(self):
        """Load and optionally preprocess dataset."""
        inputs, concepts, annotations, graph = self.load_raw()
        
        return inputs, concepts, annotations, graph

    def __getitem__(self, item):
        if self.embs_precomputed:
            x = self.input_data[item]
        else:
            image = torch.tensor(self.input_data[item], dtype=torch.float32)
            x = image.unsqueeze(0)  # (1, 64, 64)

        c = self.concepts[item]

        return {
            'inputs': {'x': x},
            'concepts': {'c': c},
        }

    @property
    def n_samples(self) -> int:
        return 10 # NOTE self.input_data.shape[0]

    @property
    def n_features(self) -> tuple:
        return tuple(self[0]['inputs']['x'].shape)

    @property
    def shape(self) -> tuple:
        return (self.n_samples, *self.n_features)
