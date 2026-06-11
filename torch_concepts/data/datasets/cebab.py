import ast
import os
from datasets import load_dataset, load_from_disk
from typing import List
import torch
import pandas as pd
import numpy as np
import logging
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset

logger = logging.getLogger(__name__)

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################
INPUT = 'description'
CONCEPTS = ['food', 'ambiance', 'service', 'noise']
TARGET = ['review']


class CEBaBDataset(ConceptDataset):
    """Dataset class for CEBaB.
    
    CEBaB is a benchmark dataset for evaluating concept-based explanation methods in Natural Language Processing (NLP). 
    It consists of short restaurant descriptions extended with human-written counterfactual descriptions. 
    Each description is annotated by multiple human annotators with four aspect-level label distributions: food, ambiance, service, and noise, plus one overall review rating distribution. 
    Aspect labels are distributions over Negative, unknown, and Positive; review ratings are distributions over 1 to 5 stars. 

    Example:
    - food_aspect_label_distribution: {'Negative': 2, 'unknown': 1, 'Positive': 3}
    - ambiance_aspect_label_distribution: {'Negative': 1, 'unknown': 2, 'Positive': 2}
    - service_aspect_label_distribution: {'Negative': 0, 'unknown': 1, 'Positive': 4}
    - noise_aspect_label_distribution: {'Negative': 3, 'unknown': 1, 'Positive': 1}
    - review_label_distribution: {'1': 1, '2': 0, '3': 2, '4': 1, '5': 2}
    
    The constructed dataset will contain:
    - Inputs: Original textual descriptions of restaurant descriptions.
    - Concepts: 4 restaurant-related attributes (food, ambiance, service, noise) + target (review) calculated from the original distributions.
      Specifically, if:
        - 'concepts_type': 'discrete', each attribute is represented as the category with the highest count in the original distribution.
        - 'concepts_type': 'continuous', each attribute is represented as the weighted mean (further standardized) of the original distribution.
    
    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        concepts_type: Whether to construct 'discrete' or 'continuous' concepts from raw data. Default is 'discrete'.
    """
    
    def __init__(
        self,
        root: str = None, # root directory to store/load the dataset
        concepts_type: str = 'discrete',
    ):

        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', "CEBaB")

        self.root = root
        self.concepts_type = concepts_type
        if self.concepts_type not in ['discrete', 'continuous']:
            raise ValueError(f"Invalid concepts_type: {self.concepts_type}. Must be 'discrete' or 'continuous'.")
        
        # Load data and annotations
        inputs, concepts, annotations, graph = self.load()
        
        # Initialize parent class
        super().__init__(
            input_data=inputs,
            concepts=concepts,
            annotations=annotations,
            graph=graph
        )

    @property
    def raw_filenames(self) -> List[str]:
        """List of raw filenames that must be present to skip downloading."""
        return [
            "raw/dataset_dict/dataset_dict.json",
            "raw/dataset_dict/train_exclusive/state.json",
            "raw/dataset_dict/train_inclusive/state.json",
            "raw/dataset_dict/train_observational/state.json",
            "raw/dataset_dict/validation/state.json",
            "raw/dataset_dict/test/state.json",
        ]

    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            "raw/inputs.txt",
            f"raw/concepts_{self.concepts_type}.h5",
            f"raw/annotations_{self.concepts_type}.pt",
            f"raw/split_mapping_{self.concepts_type}.h5",
        ]

    def download(self):
        """Download CEBaB dataset from Hugging Face and save it to the raw folder.
        """
        cebab_folder = os.path.join(self.root, "raw")
        os.makedirs(cebab_folder, exist_ok=True)

        ds = load_dataset("CEBaB/CEBaB")

        # save files
        ds.save_to_disk(os.path.join(cebab_folder, "dataset_dict"))

        logger.info(f"CEBaB files downloaded to {cebab_folder}.")

    def maybe_download(self):
        """Download the dataset if needed."""
        super().maybe_download()
       

    def _get_max_key(self, dictionary_str):
        """
        Given a dictionary in string format, return the key with the highest value.
        """
        if dictionary_str is None or dictionary_str == '':
            return None
        
        try:
            dictionary = ast.literal_eval(dictionary_str)
            if not dictionary:
                return None
            return max(dictionary, key=dictionary.get)
        except:
            return None
        
    def _mapping_evaluation_to_value(self, key, type='review'):
        """
        Map the evaluation given in the original dataset to values. 
        """
        if type == 'concept':
            if key=='Negative':
                return 0
            elif key=='unknown':
                return 1
            elif key=='Positive':
                return 2
        else:
            return float(key)-1 # For review, the original values are between 1 and 5, we map them to 0-4 for easier normalization later on.
        
        
    def _get_weighted_mean(self, dictionary=None, type='review'):
        """
        Given a dictionary of key-value pairs, return the weighted mean of the keys, where the weights are given by the values.
        """
        # transform the string into a dictionary
        if dictionary is None or len(dictionary) == 0:
            return np.nan
        dictionary = ast.literal_eval(dictionary)
        dictionary = {self._mapping_evaluation_to_value(key, type): value for key, value in dictionary.items()}
        try:
            total = sum(int(key) * int(value) for key, value in dictionary.items())
            count = sum([int(x) for x in dictionary.values()])
            mean = total / count if count > 0 else 0
        except:
            mean = np.nan  
        return mean

    def _preprocess_raw(self, ds):
        """Preprocess raw dataset in order to create a DataFrame with discrete or continuous concepts, as specified by the `concepts_type` parameter.""" # Get the split name from the dataset (train_observational, validation, or test)
        ds = ds.to_pandas()

        selected_columns = ['description', 
                            'review_label_distribution',
                            'food_aspect_label_distribution',
                            'ambiance_aspect_label_distribution',
                            'service_aspect_label_distribution',
                            'noise_aspect_label_distribution']
        
        mapping_columns_to_concepts = {
            'review_label_distribution': 'review',
            'food_aspect_label_distribution': 'food',
            'ambiance_aspect_label_distribution': 'ambiance',
            'service_aspect_label_distribution': 'service',
            'noise_aspect_label_distribution': 'noise',
        }

        #select only the relevant columns and drop rows with missing values in those columns
        ds = ds[selected_columns]
        ds = ds.dropna() 

        # if the concepts are continuous, we will compute the weighted mean of the values.
        # If the concepts are discrete, we will take the value with the highest count as the label for that concept. 
        for col in selected_columns[1:]:  # Skip 'description' column
            concept = mapping_columns_to_concepts[col]
            if col == 'review_label_distribution':
                mean_type = 'review'
            else:
                mean_type = 'concept'
            if self.concepts_type == "continuous":
                ds[concept] = ds.apply(lambda row: self._get_weighted_mean(row[col], mean_type), axis=1)
            else:
                ds[concept] = ds.apply(lambda row: self._get_max_key(row[col]), axis=1)
                ds[concept] = ds[concept].apply(lambda x: self._mapping_evaluation_to_value(x, mean_type))

        ds = ds.drop(columns=selected_columns[1:])  # Drop original distribution columns
        ds = ds.dropna()  # Drop rows with NaN values after preprocessing
        # fix the index
        ds = ds.reset_index(drop=True)
        return ds


    def build(self):
        """Build processed dataset: save concepts, annotations and splits metadata.
        """
        self.maybe_download()

        cebab_folder = os.path.join(self.root, "raw")
        logger.info(f"Building CEBaB dataset from raw files in {cebab_folder}...")

        ds_tot = load_from_disk(os.path.join(cebab_folder, "dataset_dict"))
        # preprocess each split
        for split in ['train_observational', 'validation', 'test']:
            ds = ds_tot[split]
            ds = self._preprocess_raw(ds)
            ds_tot[split] = ds.copy()

        # standardize columns
        if self.concepts_type == 'continuous':
            cols = CONCEPTS + TARGET
            train_mean_statistics = ds_tot['train_observational'][cols].mean()
            train_std_statistics = ds_tot['train_observational'][cols].std()
            for split in ['train_observational', 'validation', 'test']:
                ds_tot[split][cols] = (ds_tot[split][cols] - train_mean_statistics) / train_std_statistics

        # create split_labels
        split_labels = ['train'] * len(ds_tot['train_observational']) + ['valid'] * len(ds_tot['validation']) + ['test'] * len(ds_tot['test'])
        split_series = pd.Series(split_labels, name='split')

        # create input_series containing textual descriptions
        input_series = pd.concat(
            [ds_tot['train_observational'][INPUT], ds_tot['validation'][INPUT], ds_tot['test'][INPUT]],
            ignore_index=True,
        )
        # create concepts_df containing concept and target labels
        concepts_df = pd.concat(
            [
                ds_tot['train_observational'][CONCEPTS + TARGET],
                ds_tot['validation'][CONCEPTS + TARGET],
                ds_tot['test'][CONCEPTS + TARGET],
            ],
            ignore_index=True,
        )

        if len(input_series) != len(concepts_df):
            raise RuntimeError(
                "CEBaB build produced misaligned inputs/concepts: "
                f"{len(input_series)} inputs vs {len(concepts_df)} concept rows."
            )
        
        if self.concepts_type == 'discrete':
            concept_cardinalities = [3] * len(CONCEPTS) + [5] * len(TARGET)  # Each concept has 3 categories: Negative, unknown, Positive.
            metadata={
                    col: {'type': 'discrete'}
                    for col in concepts_df.columns
                }
            
        else: 
            concept_cardinalities = [1] * len(CONCEPTS + TARGET)
            metadata={
                    col: {'type': 'continuous'}
                    for col in concepts_df.columns
                }


        annotations = Annotations({
            1: AxisAnnotation(
                labels=concepts_df.columns.tolist(),
                cardinalities=concept_cardinalities,
                metadata=metadata,
            )
        })     

        # ---- save all ----
        # save text inputs
        input_series.to_csv(self.processed_paths[0], index=False, header=False)
        # save concepts
        concepts_df.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        # save concept annotations
        torch.save(annotations, self.processed_paths[2])
        # save split
        split_series.to_hdf(self.processed_paths[3], key='split_mapping')



    def load_raw(self):
        """Load raw processed files for the current split."""
        self.maybe_build()  # Ensures build() is called if needed
        
        logger.info(f"Loading dataset from {self.root_dir}")
        
        inputs = pd.read_csv(self.processed_paths[0], header=None)[0].tolist()        
        concepts = pd.read_hdf(self.processed_paths[1], "concepts")
        annotations = torch.load(self.processed_paths[2])
        graph = None
        
        return inputs, concepts, annotations, graph

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
        x = self.input_data[item]  # input_data contains raw text descriptions 
        c = self.concepts[item]

        # Create sample dictionary
        sample = {
            'inputs': {'x': x},
            'concepts': {'c': c},
        }

        return sample

    
    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.input_data)
    
