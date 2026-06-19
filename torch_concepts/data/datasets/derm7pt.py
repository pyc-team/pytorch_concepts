import os
import torch
import pandas as pd
import logging
from typing import List, Optional
from torchvision.datasets.utils import extract_archive
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset
from PIL import Image

logger = logging.getLogger(__name__)


########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################
CONCEPT_NAMES = ["pigment_network",
                 "streaks",
                 "dots_and_globules",
                 "blue_whitish_veil",
                 "regression_structures",
                 "vascular_structures",
                 "pigmentation"] # all the concept names except the target label

TARGET = "diagnosis"

########################################################
## MAPPINGS
########################################################

mapping_pigment_network = {
    0: "absent",
    1: "typical",
    2: "atypical"
}

mapping_streaks = {
    0: "absent",
    1: "regular",
    2: "irregular"
}

mapping_dots_and_globules = {
    0: "absent",
    1: "regular",
    2: "irregular"
}

mapping_blue_whitish_veil = {
    0: "absent",
    1: "present"
}

mapping_regression_structures = {
    0: "absent",
    1: "blue areas",
    2: "white areas",
    3: "combinations"
}

mapping_vascular_structures = {
    0: "absent",
    1: 'arborizing',
    2: 'comma',
    3: 'hairpin',
    4: 'within regression',
    5: 'wreath',
    6: 'dotted',
    7: 'linear irregular'
}

mapping_pigmentation = {
    0: "absent",
    1: "diffuse regular",
    2: "localized regular",
    3: "diffuse irregular",
    4: "localized irregular"
}

mapping_diagnosis = {
    0: ['basal cell carcinoma'],
    1: ['blue nevus'],
    2: ['clark nevus'],
    3: ['combined nevus'],
    4: ['congenital nevus'],
    5: ['dermal nevus'],
    6: ['dermatofibroma'],
    7: ['lentigo'],
    8: ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                              'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)',
                              'melanoma metastasis'],
    9:  ['melanosis'],
    10: ['miscellaneous'],
    11: ['recurrent nevus'],
    12: ['reed or spitz nevus'],
    13: ['seborrheic keratosis'],
    14: ['vascular lesion'],
}


# Mapping grouping infrequent classes together
mapping_diagnosis_ginfrequent = {
 0: ['basal cell carcinoma'],
 1: ['nevus','blue nevus', 'clark nevus', 'combined nevus', 
     'congenital nevus', 'dermal nevus', 'recurrent nevus', 'reed or spitz nevus'],
 2: ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                              'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)',
                              'melanoma metastasis'],
 3: [ 'DF/LT/MLS/MISC','dermatofibroma', 'lentigo', 'melanosis', 'miscellaneous', 'vascular lesion'],
 4: ['seborrheic keratosis']                             
}

mapping_vascular_structures_ginfrequent = {
    0: ["absent"],
    1: ['regular', 'arborizing', 'comma', 'hairpin', 'within regression', 'wreath'],
    2: ['dotted/irregular', 'dotted', 'linear irregular']
}

mapping_pigmentation_ginfrequent = {
    0: ["absent"],
    1: ['regular', 'diffuse regular', 'localized regular'],
    2: ['irregular', 'diffuse irregular', 'localized irregular']
}

mapping_regression_structures_ginfrequent = {
    0: ["absent"],
    1: ['present', 'blue areas', 'white areas', 'combinations'],
}

class Derm7ptDataset(ConceptDataset):
    """Dataset class for Derm7pt.
    
    Derm7pt is a dataset for skin lesion analysis with multiple attributes and annotations.
    This class wraps the Derm7pt dataset to work with the ConceptDataset framework.
    The preprocessing steps are adapted from: https://github.com/jeremykawahara/derm7pt.

    IMPORTANT NOTE:
    For this script to work, the dataset MUST be manually downloaded before running the code from the official website:
    https://derm.cs.sfu.ca/Download.html and placed in the 'raw' folder under the root directory.
    The dataset is not automatically downloaded due to licensing restrictions.

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        image_type: Type of images to use ('derm' for dermoscopic, 'clinic' for clinical images).
                    Dermoscopic images are captured with a dermatoscope and offer a standardized field of view 
                    and controlled acquisition (e.g., lighting and field of view). 
                    Clinical images are less standardized, taken at various fields of view, 
                    and can contain image artefacts (e.g., a ruler to measure the lesion) 
                    [Kawahara, J., et al. "7-Point checklist and skin lesion classification 
                    using multi-task multi-modal neural nets, IEEE J Biomed Health Inform].
                    Default is 'derm'.
        group_infrequent_classes: Whether to group infrequent classes together for certain concepts.
                                  Default is True, which is recommended in https://github.com/jeremykawahara/derm7pt.
        concept_subset: Optional subset of concept labels to use.
        label_descriptions: Optional dict mapping concept names to descriptions.
    """
    
    def __init__(
        self,
        root: str = None, # root directory to store/load the dataset
        image_type: str = "derm", # type of images to use: "derm" for dermoscopic, "clinic" for clinical images
        group_infrequent_classes: bool = True, # whether to group infrequent classes together
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
    ):

        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', "derm7pt")

        self.root = root
        self.concept_subset = concept_subset  
        self.label_descriptions = label_descriptions
        self.image_type = image_type
        # check that image_type is valid
        if self.image_type not in ["derm", "clinic"]:
            raise ValueError(f"Invalid image_type: {self.image_type}. Must be 'derm' or 'clinic'.")
        self.group_infrequent_classes = group_infrequent_classes
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
            "raw/release_v0.zip" # must be manually downloaded from https://derm.cs.sfu.ca/Download.html.
        ]

    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            f"filenames_{self.image_type}_{self.group_infrequent_classes}.txt",
            f"concepts_{self.image_type}_{self.group_infrequent_classes}.h5",
            f"annotations_{self.image_type}_{self.group_infrequent_classes}.pt",
            f"split_mapping_{self.image_type}_{self.group_infrequent_classes}.h5",
        ]

    def download(self):
        """This dataset MUST be manually downloaded."""
        raise RuntimeError(
            "The Derm7pt dataset must be manually downloaded from https://derm.cs.sfu.ca/Download.html "
            "and placed in the 'raw' folder under the root directory: {}.".format(self.root)
        )

    def maybe_download(self):
        """Download and extract the dataset if needed."""
        logger.info(f"Checking if raw data files are present in {self.root}...")
        super().maybe_download()

    def maybe_extract(self):
        """Extracts release_v0.zip to the raw derm7pt folder.
        """
        derm7pt_folder = os.path.join(self.root, "raw")

        # if the extracted folder already exists, skip extraction
        if os.path.exists(os.path.join(derm7pt_folder, "release_v0")):
            logger.info("Extracted folder already exists, skipping extraction.")
            return
        
        archive_path = os.path.join(derm7pt_folder, "release_v0.zip")
        logger.info("Extracting release_v0.zip...")
        extract_archive(archive_path)
        logger.info(f"Derm7pt images extracted to {derm7pt_folder}")

    def _load_metadata(self, concepts_file, split_files):
        """Loads metadata files: concepts annotations and split indexes."""
        concepts_df = pd.read_csv(concepts_file)
        concepts_df.rename(columns={self.image_type: "image_id"}, inplace=True)
        concepts_df = concepts_df[CONCEPT_NAMES + [TARGET] + ["image_id"]]

        split_indexes = {}
        for split_name, split_file in split_files.items():
            values = pd.read_csv(split_file, header=0).squeeze().tolist()
            if not isinstance(values, list):
                values = [values]
            split_indexes[split_name] = values

        return concepts_df, split_indexes


    def _validate_metadata(self,concepts_df,split_indexes, img_dir) -> bool:
        """Performs integrity checks on the raw files to ensure they are present and valid."""
        # check that all indexes are present and non-overlapping
        all_indexes = set(split_indexes['train'] + split_indexes['valid'] + split_indexes['test'])
        # replace asserts with explicit error messages for better debugging
        if len(all_indexes) != len(concepts_df):
            raise ValueError(f"Indexes in split files do not cover all samples in concepts. "
                             f"Expected {len(concepts_df)} unique indexes, but got {len(all_indexes)}.")
        if len(set(split_indexes['train']).intersection(split_indexes['valid'])) != 0:
            raise ValueError("Train and valid indexes overlap.")
        if len(set(split_indexes['train']).intersection(split_indexes['test'])) != 0:
            raise ValueError("Train and test indexes overlap.")
        if len(set(split_indexes['valid']).intersection(split_indexes['test'])) != 0:
            raise ValueError("Valid and test indexes overlap.")

        # check that all image files are present
        for filename in concepts_df['image_id']:
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file {img_path} not found. Please ensure the dataset is correctly downloaded.")

        logger.info("All metadata integrity checks passed.")
        
        return None
    

    def build(self):
        """Build processed dataset: save concepts, annotations and splits metadata.
        
        Images are not saved as they are already in the downloaded folder and
        will be loaded on-the-fly in __getitem__.
        """
        self.maybe_download()

        logger.info("Proceeding to extraction...")
        self.maybe_extract()

        derm7pt_folder = os.path.join(self.root, "raw")
        logger.info(f"Building Derm7pt dataset from raw files extracted in {derm7pt_folder}...")

        # Load img directory, annotations and split files
        img_dir = os.path.join(derm7pt_folder, "release_v0", "images")
        concepts_file = os.path.join(derm7pt_folder, "release_v0", "meta", "meta.csv")
        split_files = {
            'train': os.path.join(derm7pt_folder, "release_v0", "meta", "train_indexes.csv"),
            'valid': os.path.join(derm7pt_folder, "release_v0", "meta", "valid_indexes.csv"),
            'test': os.path.join(derm7pt_folder, "release_v0", "meta", "test_indexes.csv"),
        }

        concepts_df, split_indexes = self._load_metadata(concepts_file, split_files)

        # Validate metadata and check integrity of raw files. 
        self._validate_metadata(
                concepts_df,
                split_indexes,
                img_dir
        )
        
        # replace annotations with concept labels using the mappings defined above
        for concept in CONCEPT_NAMES+[TARGET]:
            if self.group_infrequent_classes and concept in ["diagnosis", "vascular_structures", "regression_structures", "pigmentation"]:
                mapping = globals()[f"mapping_{concept}_ginfrequent"]
            else:
                mapping = globals()[f"mapping_{concept}"]
            # create inverse mapping to map string labels back to integer codes
            inv_mapping = {
                v: k
                for k, values in mapping.items()
                for v in (values if isinstance(values, list) else [values])
            }
            concepts_df[concept] = concepts_df[concept].map(inv_mapping)

        # check if any NaN values are present after mapping
        if concepts_df[CONCEPT_NAMES + [TARGET]].isnull().any().any():
            raise ValueError("Some concept labels could not be mapped to integer codes. Please check the mappings and the raw data.")
        
        # create split labels
        split_labels = []
        for idx in concepts_df.index:
            if idx in split_indexes['train']:
                split_labels.append("train")
            elif idx in split_indexes['valid']:
                split_labels.append("valid")
            elif idx in split_indexes['test']:
                split_labels.append("test")
            else:
                raise ValueError(f"Index {idx} not found in any split")
    
        # Create annotations
        cardinalities = tuple([concepts_df[concept].nunique() for concept in CONCEPT_NAMES + [TARGET]])
        annotations = Annotations({
            1: AxisAnnotation(
                labels=CONCEPT_NAMES + [TARGET],
                cardinalities=cardinalities,
                metadata={name: {'type': 'discrete'} for name in CONCEPT_NAMES + [TARGET]}
            )
        })

        # --- Save processed data ---
        logger.info(f"Saving filenames, concepts, annotations and split mapping to {self.root_dir}")
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Save filenames list
        with open(self.processed_paths[0], 'w') as f:
            f.write('\n'.join(concepts_df['image_id'].tolist()))

        # drop image_id column from concepts_df before saving
        concepts_df = concepts_df.drop(columns=['image_id'])
        
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
            img_path = os.path.join(self.root, "raw", "release_v0", "images", filename)
            with Image.open(img_path) as img:
                # crop of 25
                img = img.crop((25, 25, img.width - 25, img.height - 25)) # suggested in https://github.com/jeremykawahara/derm7p
                #x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                x = img.convert("RGB")


        
        c = self.concepts[item]

        # Create sample dictionary
        sample = {
            'inputs': {'x': x},
            'concepts': {'c': c},
        }

        return sample

    # Override properties that assume input_data is a tensor
    # In Derm7pt, input_data is a list of filenames (images loaded on-the-fly)
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.input_data)

    @property
    def n_features(self) -> tuple:
        """Shape of features in dataset's input (excluding number of samples).
        
        Derm7pt images are 224x224x3 (H x W x C) reordered to (C, H, W).
        """
        x = self[0]["inputs"]["x"]
        if hasattr(x, "shape"):
            return tuple(x.shape)
        return (x.height, x.width, len(x.getbands()))

    @property
    def shape(self) -> tuple:
        """Shape of the input tensor (n_samples, C, H, W)."""
        return (self.n_samples, *self.n_features)
    
