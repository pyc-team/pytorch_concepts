import os
import torch
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms import Compose
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset
import torchvision.transforms as T
from glob import glob
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class CelebADataset(ConceptDataset):
    """Dataset class for CelebA.
    
    CelebA is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.
    This class wraps torchvision's CelebA dataset to work with the ConceptDataset framework.
    The dataset can be downloaded from the official website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
    
    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: The split of the dataset to use ('train', 'valid', or 'test'). Default is 'train'.
        transform: The transformations to apply to the images. Default is None.
        download: Whether to download the dataset if it does not exist. Default is False.
        task_label: The attribute(s) to use for the task. Default is 'Attractive'.
        concept_subset: Optional subset of concept labels to use.
        label_descriptions: Optional dict mapping concept names to descriptions.
    """
    
    def __init__(
        self,
        name: str,
        root: str,
        transform: Union[Compose, torch.nn.Module] = None,
        task_label: Optional[List[str]] = None,
        class_attributes: Optional[List[str]] = None,  # Alias for task_label
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
    ):
        self.name = name
        self.transform = transform

        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', self.name)

        self.root = root

        # Support both task_label and class_attributes (class_attributes takes precedence)
        if class_attributes is not None:
            self.task_label = class_attributes if isinstance(class_attributes, list) else [class_attributes]
        elif task_label is not None:
            self.task_label = task_label if isinstance(task_label, list) else [task_label]
        else:
            self.task_label = ['Attractive']
            
        self.label_descriptions = label_descriptions
        
        # These will be set during build/load
        self.concept_attr_names = []
        self.task_attr_names = []
        
        # Load data and annotations
        input_data, concepts, annotations, graph = self.load()
        
        # Initialize parent class
        super().__init__(
            input_data=input_data,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
        )

    @property
    def raw_filenames(self) -> List[str]:
        """List of raw filenames that must be present to skip downloading."""
        # find the directory of downloaded data (if any)
        path_base_file = os.path.join(self.root, "flwrlabs___celeba", "*", "*", "*", "dataset_info.json")
        matches = glob(path_base_file)
        if len(matches)==0:
            return ["__nonexistent_file__"]
        d = os.path.dirname(matches[0])
        
        # eliminate self.root (it is added by default later).
        d = d.replace(self.root +"/", "")
        base_file = matches[0].replace(self.root +"/", "")

        n_train_files = 19
        n_valid_files = 3
        n_test_files = 3

        train_files = []
        valid_files = []
        test_files = []
 
        for i in range(n_train_files):
            if i<10:        
                train_files.append(os.path.join(d, f"celeba-train-0000{i}-of-000{n_train_files}.arrow"))
            else:
                train_files.append(os.path.join(d, f"celeba-train-000{i}-of-000{n_train_files}.arrow"))
        for i in range(n_valid_files):
            valid_files.append(os.path.join(d, f"celeba-valid-0000{i}-of-0000{n_valid_files}.arrow"))
        for i in range(n_test_files):
            test_files.append(os.path.join(d, f"celeba-test-0000{i}-of-0000{n_test_files}.arrow"))

        return [base_file] + train_files + valid_files + test_files


    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            f"images.pt",
            f"concepts.h5",
            "annotations.pt",
           f"split_mapping.h5",
        ]
  
    def download(self):
        """Download raw data files from HuggingFace and save to root directory."""
        logger.info(f"Downloading CelebA dataset from HuggingFace...")
        load_dataset(
                "flwrlabs/celeba", 
                cache_dir=self.root
            )
        logger.info(f"CelebA dataset downloaded and saved to {self.root}.")

    def build(self):
        """Build processed dataset from all splits (train, valid, test) concatenated."""
        self.maybe_download()

        # --- Load data ---
        logger.info(f"Building CelebA dataset from raw files in {self.root_dir}...")
        ds = load_dataset(path=self.root)

        # --- Construct input_data, concepts, annotations --- 
        # construct concept_names
        concept_names = list(ds['train'].features.keys())
        concept_names.remove('image')
        concept_names.remove('celeb_id')

        # process each split
        split_indices = []
        all_images = []
        all_concepts_df = pd.DataFrame()
        for split_name in ['train','validation', 'test']:
            split_dataset = ds[split_name]
            corrupted_images = []
            # extract images
            for idx in tqdm(range(len(split_dataset)), desc=f"  {split_name} images", unit="img"):
                try:
                    img = split_dataset[idx]['image']
                    arr = np.array(img)
                    all_images.append(torch.from_numpy(arr))
                except OSError as e:
                    logger.warning(f"Skipping image at index {idx} in split {split_name} due to error: {e}")
                    corrupted_images.append(idx)
          
            if len(corrupted_images)!=0:
                logger.warning(f"Skipping {len(corrupted_images)} corrupted images in split {split_name}.")
                #remove corrupted indices from split_dataset
                split_dataset = split_dataset.select([i for i in range(len(split_dataset)) if i not in corrupted_images])
            
            # extract split_indices
            split_indices.extend([split_name] * len(split_dataset))
            
            # extract concepts for this split
            split_concepts = split_dataset.to_pandas()[concept_names]
            all_concepts_df = pd.concat([all_concepts_df, split_concepts], ignore_index=True)

        
        assert all_concepts_df.columns.tolist() == concept_names, "Concept names do not match."


        # combine all data
        input_data = torch.stack(all_images)
        logger.info(f"Input data shape: {input_data.shape}")
        concepts = all_concepts_df.astype(int)
        logger.info(f"Concepts shape: {concepts.shape}")
    
        # create annotations
        cardinalities = tuple([2] * len(concept_names))
        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                cardinalities=cardinalities,
                metadata={name: {'type': 'discrete'} for name in concept_names}
            )
        })

        # --- Save processed data ---
        logger.info(f"Saving concepts, annotations and split mapping to {self.root}")
        torch.save(input_data, self.processed_paths[0])
        concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        torch.save(annotations, self.processed_paths[2])
        pd.Series(split_indices).to_hdf(self.processed_paths[4], key="split_mapping", mode="w")

    def load_raw(self):
        """Load raw processed files for the current split."""
        self.maybe_build()  # Ensures build() is called if needed
        
        logger.info(f"Loading dataset from {self.root_dir}")
        input_data = torch.load(self.processed_paths[0])
        concepts = pd.read_hdf(self.processed_paths[1], "concepts")
        annotations = torch.load(self.processed_paths[2])
        graph = None
        
        return input_data, concepts, annotations, graph

    def load(self):
        """Load and optionally preprocess dataset."""
        inputs, concepts, annotations, graph = self.load_raw()
        
        # Add any additional preprocessing here if needed
        # For most cases, just return raw data
        
        return inputs, concepts, annotations, graph
    
    def __getitem__(self, item):
        """
        Get a single sample from the dataset.

        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing 'inputs' and 'concepts' sub-dictionaries.
        """
        # Get raw input data and concepts
        x = self.input_data[item]
        x = x.permute(2,0,1).float() / 255.0 
        c = self.concepts[item]

        # TODO: handle missing values with masks

        # Create sample dictionary
        sample = {
            'inputs': {'x': x},    # input data: multiple inputs can be stored in a dict
            'concepts': {'c': c},  # concepts: multiple concepts can be stored in a dict
            # TODO: add scalers when these are set
            # also check if batch transforms work correctly inside the model training loop
            # 'transforms': {'x': self.scalers.get('input', None),
            #               'c': self.scalers.get('concepts', None)}
        }

        return sample