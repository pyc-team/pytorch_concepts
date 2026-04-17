import os
import torch
import numpy as np
import logging
from typing import List, Optional, Dict
from datasets import load_dataset

from ..base.dataset import ConceptDataset
from ...annotations import Annotations, AxisAnnotation

logger = logging.getLogger(__name__)

# Default available concept columns in dsprites
DSPRITES_CONCEPTS = [
    'value_shape', 'value_scale', 'value_orientation',
    'value_x_position', 'value_y_position',
]

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
    concepts : list of str, optional
        Concept column names to use. Default: None (all value_* columns).
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

    Examples
    --------
    >>> from torch_concepts.data.datasets import DSpritesRegressionDataset
    >>> formulas = {
    ...     'square': 'value_x_position + value_y_position',
    ...     'circle': 'value_x_position * value_y_position',
    ...     'heart': 'value_x_position - value_y_position',
    ... }
    >>> dataset = DSpritesRegressionDataset(
    ...     concepts=['value_x_position', 'value_y_position'],
    ...     formulas=formulas,
    ...     num_samples=1000,
    ... )
    >>> sample = dataset[0]
    >>> c = sample['concepts']['c']  # [...concept values..., target]
    """

    DEFAULT_FORMULAS = {
        'square': 'value_x_position + value_y_position',
        'circle': 'value_x_position * value_y_position',
        'heart': 'value_x_position - value_y_position',
    }

    def __init__(
        self,
        root: str = None,
        concepts: Optional[List[str]] = None,
        formulas: Optional[Dict[str, str]] = None,
        num_samples: Optional[int] = None,
        seed: int = 42,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
    ):
        import sympy
        import sympytorch

        self.seed = seed
        self.num_samples = num_samples
        self.label_descriptions = label_descriptions
        self._concept_columns = concepts
        self.formulas = formulas if formulas is not None else self.DEFAULT_FORMULAS

        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'dsprites_regression')
        self.root = root

        # Load HF dataset
        self._hf_dataset = load_dataset("dpdl-benchmark/dsprites")["train"]

        # Resolve concept columns
        available = [c for c in self._hf_dataset.column_names if c.startswith('value_')]
        if self._concept_columns is None:
            self._concept_columns = available
        else:
            for c in self._concept_columns:
                if c not in available:
                    raise ValueError(f"Concept '{c}' not found. Available: {available}")
            self._concept_columns = sorted(self._concept_columns, key=lambda x: available.index(x))

        # Build sympy -> torch formula modules
        self._torch_formulas = {}
        for shape_name, formula_str in self.formulas.items():
            torch_exp = sympytorch.SymPyModule(expressions=[sympy.sympify(formula_str)])
            self._torch_formulas[shape_name] = torch_exp

        # Subsampling
        full_indices = np.arange(len(self._hf_dataset))
        if self.num_samples is not None:
            rng = np.random.default_rng(self.seed)
            self._indices = rng.choice(full_indices, size=self.num_samples, replace=False)
        else:
            self._indices = full_indices

        input_data, concepts_tensor, annotations, graph = self.load()

        super().__init__(
            input_data=input_data,
            concepts=concepts_tensor,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name="DSpritesRegressionDataset",
        )

    @property
    def raw_filenames(self) -> List[str]:
        return []

    @property
    def processed_filenames(self) -> List[str]:
        n = self.num_samples if self.num_samples is not None else "all"
        return [
            f"indices_N_{n}_seed_{self.seed}.pt",
            f"concepts_N_{n}_seed_{self.seed}.pt",
            f"annotations_N_{n}.pt",
        ]

    def download(self):
        pass

    def build(self):
        """Extract concepts, compute formula targets, save to disk."""
        import sympy
        import sympytorch

        logger.info(f"Building DSprites regression dataset (N={len(self._indices)}, seed={self.seed})")

        concepts_list = []
        targets_list = []

        for idx in self._indices:
            real_idx = int(idx)
            sample = self._hf_dataset[real_idx]
            row = [sample[c] for c in self._concept_columns]
            concept_values = torch.tensor(row, dtype=torch.float32)

            # Get shape name
            shape_id = sample['value_shape']
            shape_name = IDS_TO_SHAPES[shape_id]

            # Compute formula target
            var_dict = dict(zip(self._concept_columns, [concept_values[i] for i in range(len(self._concept_columns))]))
            target = self._torch_formulas[shape_name](**var_dict)

            concepts_list.append(row)
            targets_list.append([target.item()])

        concepts_tensor = torch.tensor(concepts_list, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_list, dtype=torch.float32)

        # Combine concepts + target
        cy = torch.cat([concepts_tensor, targets_tensor], dim=1)

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

        os.makedirs(self.root_dir, exist_ok=True)
        torch.save(torch.from_numpy(self._indices), self.processed_paths[0])
        torch.save(cy, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])

        logger.info(f"DSprites regression dataset saved to {self.root_dir}")

    def load_raw(self):
        self.maybe_build()
        logger.info(f"Loading DSprites regression dataset from {self.root_dir}")

        self._indices = torch.load(self.processed_paths[0], weights_only=False).numpy()
        concepts = torch.load(self.processed_paths[1], weights_only=False)
        annotations = torch.load(self.processed_paths[2], weights_only=False)

        input_data = torch.arange(len(self._indices))

        return input_data, concepts, annotations, None

    def load(self):
        return self.load_raw()

    def __getitem__(self, item):
        if self.embs_precomputed:
            x = self.input_data[item]
        else:
            real_idx = int(self._indices[item])
            sample = self._hf_dataset[real_idx]
            image = torch.tensor(np.array(sample['image']), dtype=torch.float32)
            x = image.unsqueeze(0)  # (1, 64, 64)

        c = self.concepts[item]

        return {
            'inputs': {'x': x},
            'concepts': {'c': c},
        }

    @property
    def n_samples(self) -> int:
        return len(self._indices)

    @property
    def n_features(self) -> tuple:
        return tuple(self[0]['inputs']['x'].shape)

    @property
    def shape(self) -> tuple:
        return (self.n_samples, *self.n_features)
