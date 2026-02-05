import os
import csv
import torch
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union
from tqdm import tqdm
from torchvision.transforms import Compose
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive, check_integrity
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset
import torchvision.transforms as T
from glob import glob
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# CelebA file list from torchvision - Google Drive file IDs, MD5 hashes, and filenames
CELEBA_FILE_LIST = [
    # File ID, MD5 Hash, Filename
    ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
    ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
    ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
    ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
    ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
    ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
]

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
        root: str = None, # root directory to store/load the dataset
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
    ):

        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', "celeba")

        self.root = root
            
        self.label_descriptions = label_descriptions
        
        # Load data and annotations
        filenames, concepts, annotations, graph = self.load()
        
        # Initialize parent class
        super().__init__(
            input_data=filenames,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
        )

    @property
    def raw_filenames(self) -> List[str]:
        """List of raw filenames that must be present to skip downloading."""
        return [
            "raw/img_align_celeba.zip",
            "raw/list_attr_celeba.txt",
            "raw/list_eval_partition.txt",
        ]

    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            "filenames.txt",
            "concepts.h5",
            "annotations.pt",
            "split_mapping.h5",
        ]

    def download(self):
        """Download CelebA images zip and annotation files from Google Drive.
        
        Downloads the aligned and cropped face images archive and annotation files.
        Extraction is handled separately by the extract() method.
        
        Note: Requires gdown package for Google Drive downloads.
        """
        celeba_folder = os.path.join(self.root, "raw")
        os.makedirs(celeba_folder, exist_ok=True)

        # Files to download: zip file and annotation files
        files_to_download = [
            CELEBA_FILE_LIST[0],  # img_align_celeba.zip
            CELEBA_FILE_LIST[1],  # list_attr_celeba.txt
            CELEBA_FILE_LIST[5],  # list_eval_partition.txt
        ]
        
        for file_id, md5, filename in files_to_download:
            file_path = os.path.join(celeba_folder, filename)
            if os.path.exists(file_path):
                logger.info(f"{filename} already present, skipping download")
                continue
                
            logger.info(f"Downloading {filename} from Google Drive to {celeba_folder}...")
            download_file_from_google_drive(
                file_id, 
                celeba_folder, 
                filename, 
                md5
            )
        
        logger.info(f"CelebA files downloaded to {celeba_folder}.")

    def maybe_extract(self):
        """Extract the CelebA images archive.
        
        Extracts img_align_celeba.zip to the raw celeba folder.
        """
        celeba_folder = os.path.join(self.root, "raw")
        archive_path = os.path.join(celeba_folder, "img_align_celeba.zip")
        
        if os.path.isdir(os.path.join(celeba_folder, "img_align_celeba")):
            logger.info("Images already extracted")
            return
            
        if not os.path.exists(archive_path):
            logger.warning(f"Archive not found: {archive_path}")
            return
            
        logger.info("Extracting img_align_celeba.zip...")
        extract_archive(archive_path)
        logger.info(f"CelebA images extracted to {celeba_folder}")

    def maybe_download(self):
        """Download and extract the dataset if needed."""
        super().maybe_download()

    def _load_csv(self, filename: str, header: Optional[int] = None):
        """Load a CSV file in CelebA format (torchvision style).
        
        Args:
            filename: Name of the CSV file to load.
            header: Row index containing headers, or None if no headers.
            
        Returns:
            Tuple of (headers, indices, data) where data is a torch tensor.
        """
        filepath = os.path.join(self.root, "raw", filename)
        with open(filepath) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = [h for h in data[header] if h]  # Filter out empty strings
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        # Filter out empty strings from data rows as well
        data_int = [list(map(int, [x for x in row if x])) for row in data]

        return headers, indices, torch.tensor(data_int)

    def build(self):
        """Build processed dataset: save concepts, annotations and splits metadata.
        
        Images are not saved as they are already in the downloaded folder and
        will be loaded on-the-fly in __getitem__.
        """
        self.maybe_download()

        self.maybe_extract()

        celeba_folder = os.path.join(self.root, "raw")
        logger.info(f"Building CelebA dataset from raw files in {celeba_folder}...")

        # Load annotation files (torchvision style)
        _, filenames, splits_data = self._load_csv("list_eval_partition.txt")
        attr_names, _, attr_data = self._load_csv("list_attr_celeba.txt", header=1)
        
        # Map from {-1, 1} to {0, 1} (same as torchvision)
        attr_data = torch.div(attr_data + 1, 2, rounding_mode="floor")
        
        # Create split labels
        split_map = {0: 'train', 1: 'valid', 2: 'test'}
        split_labels = [split_map[splits_data[idx].item()] for idx in range(len(filenames))]
        
        # Convert concepts to DataFrame
        concepts_df = pd.DataFrame(attr_data.numpy(), columns=attr_names)
    
        # Create annotations
        cardinalities = tuple([1] * len(attr_names))
        annotations = Annotations({
            1: AxisAnnotation(
                labels=attr_names,
                cardinalities=cardinalities,
                metadata={name: {'type': 'discrete'} for name in attr_names}
            )
        })

        # --- Save processed data ---
        logger.info(f"Saving filenames, concepts, annotations and split mapping to {self.root_dir}")
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Save filenames list
        with open(self.processed_paths[0], 'w') as f:
            f.write('\n'.join(filenames))
        
        concepts_df.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        torch.save(annotations, self.processed_paths[2])
        pd.Series(split_labels).to_hdf(self.processed_paths[3], key="split_mapping", mode="w")

    def load_raw(self):
        """Load raw processed files for the current split."""
        self.maybe_build()  # Ensures build() is called if needed
        
        logger.info(f"Loading dataset from {self.root_dir}")
        
        # Load filenames list
        with open(self.processed_paths[0], 'r') as f:
            filenames = f.read().strip().split('\n')
        
        concepts = pd.read_hdf(self.processed_paths[1], "concepts")
        annotations = torch.load(self.processed_paths[2])
        graph = None
        
        return filenames, concepts, annotations, graph

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
        # Load image on-the-fly
        if self.embs_precomputed:
            x = self.input_data[item]  # input_data contains precomputed embeddings
        else:
            filename = self.input_data[item]  # input_data contains filenames
            img_path = os.path.join(self.root, "raw", "img_align_celeba", filename)
            img = Image.open(img_path)
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        c = self.concepts[item]

        # Create sample dictionary
        sample = {
            'inputs': {'x': x},
            'concepts': {'c': c},
        }

        return sample

    # Override properties that assume input_data is a tensor
    # In CelebA, input_data is a list of filenames (images loaded on-the-fly)
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.input_data)

    @property
    def n_features(self) -> tuple:
        """Shape of features in dataset's input (excluding number of samples).
        
        CelebA images are 218x178x3 (H x W x C) reordered to (C, H, W).
        """
        return tuple(self[0]['inputs']['x'].shape)

    @property
    def shape(self) -> tuple:
        """Shape of the input tensor (n_samples, C, H, W)."""
        return (self.n_samples, *self.n_features)
    
