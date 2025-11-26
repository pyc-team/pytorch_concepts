import os
import json
import pandas as pd
import torch
from typing import List
from typing import Union
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose

from ..base import ConceptDataset
from ..utils import colorize_and_transform

class FashionMNISTDataset(ConceptDataset):
    """Dataset class for the FashionMNIST dataset.

    This dataset represents a small expert system that models the relationship
    between color features and various attributes in the FashionMNIST dataset.
    """

    #TODO: add url 
    # url = 

    def __init__(
            self,
            seed: int, # seed for data generation
            concept_subset: list | None = None, # subset of concept labels
            label_descriptions: dict | None = None,
            task_type: str = 'classification',
            transform: Union[Compose, torch.nn.Module] = None,
            coloring: dict | None = None,
            root: str = None
    ):
        self.seed = seed
        self.root = root
        self.task_type = task_type
        self.transform = transform
        self.coloring = coloring

        # embeddings is a torch tensor
        # concepts is a pandas dataframe
        # graph is the adjacency matrix as a pandas dataframe
        # concept_cardinality is a dict {concept_name: cardinality}
        embeddings, concepts, graph, concept_cardinality = self.load()
        concept_names = concepts.columns.tolist()

        # Initialize parent class
        super().__init__(
            input_data=embeddings,
            concepts=concepts,
            graph=graph,
            concept_cardinality=concept_cardinality,
            concept_names_all=concept_names, # all concept names
            concept_names_subset=concept_subset, # subset of concept names
            label_descriptions=label_descriptions,
        )
        
    @property
    def raw_filenames(self) -> List[str]:
        """List of raw filenames that need to be present in the raw directory
        for the dataset to be considered present."""
        return ["fashionmnist_data.pt", "fashionmnist_targets.pt"]

    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            f"embs_seed_{self.seed}.pt",
            f"concepts_seed_{self.seed}.h5",
            "graph.h5",
            "cardinality.json",
            f"coloring_mode_seed_{self.seed}.json"
        ]

    def download(self):
        train_data = FashionMNIST(root=self.root, train=True, download=True, transform=self.transform)
        test_data = FashionMNIST(root=self.root, train=False, download=True, transform=self.transform)

        data = torch.cat([train_data.data, test_data.data], dim=0)
        targets = torch.cat([train_data.targets, test_data.targets], dim=0)

        torch.save(data, self.raw_paths[0])
        torch.save(targets, self.raw_paths[1])

    def build(self):
        self.maybe_download()

        # load raw data
        data = torch.load(self.raw_paths[0])
        targets = torch.load(self.raw_paths[1])

        # color the images based on the coloring scheme
        if self.coloring is None:
            raise ValueError("coloring scheme must be provided.")
        if 'training_mode' not in self.coloring:
            raise ValueError("coloring scheme must contain 'training_mode'.")
        if 'test_mode' not in self.coloring:
            raise ValueError("coloring scheme must contain 'test_mode'.")
        if 'training_kwargs' not in self.coloring:
            raise ValueError("coloring scheme must contain 'training_kwargs'.")
        if 'test_kwargs' not in self.coloring:
            raise ValueError("coloring scheme must contain 'test_kwargs'.")

        embeddings, concepts_dict, targets, coloring_mode = colorize_and_transform(data,
                                                              targets,
                                                              training_percentage=self.coloring.get('training_percentage', 0.8),
                                                              test_percentage=self.coloring.get('test_percentage', 0.2),
                                                              training_mode=[self.coloring.get('training_mode', 'random')],
                                                              test_mode=[self.coloring.get('test_mode', 'random')],
                                                              training_kwargs=[self.coloring.get('training_kwargs', {})],
                                                              test_kwargs=[self.coloring.get('test_kwargs', {})])

        # save coloring mode
        with open(self.processed_paths[4], "w") as f:
            json.dump(coloring_mode, f)

        # construct dataframe with concepts
        concepts = pd.DataFrame()
        # add these only if they are in the concept dict
        for key in concepts_dict:
            concepts[key] = concepts_dict[key].numpy()
        concepts['clothing'] = targets.numpy()

        # construct the graph
        graph = pd.DataFrame(0, index=concepts.columns, columns=concepts.columns)
        graph = graph.astype(int)

        # get concepts cardinality
        concept_cardinality = {col: int(concepts[col].nunique()) for col in concepts.columns}
        concept_metadata = {'task': self.task_type, 
                            'cardinality': concept_cardinality}

        # save embeddings
        print(f"Saving dataset from {self.root_dir}")
        torch.save(embeddings, self.processed_paths[0])
        # save concepts
        concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        # save graph
        graph.to_hdf(self.processed_paths[2], key="graph", mode="w")
        # save cardinality
        with open(self.processed_paths[3], "w") as f:
            json.dump(concept_cardinality, f)

    def load_raw(self):
        self.maybe_build()
        print(f"Loading dataset from {self.root_dir}")
        embeddings = torch.load(self.processed_paths[0])
        concepts = pd.read_hdf(self.processed_paths[1], "concepts")
        graph = pd.read_hdf(self.processed_paths[2], "graph")
        with open(self.processed_paths[3], "r") as f:
            concept_cardinality = json.load(f)
        return embeddings, concepts, graph, concept_cardinality

    def load(self):
        embeddings, concepts, graph, concept_cardinality = self.load_raw()
        return embeddings, concepts, graph, concept_cardinality



