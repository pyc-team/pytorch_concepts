import os
import gzip
import shutil
import pandas as pd
import torch
import logging
from typing import List, Optional
import bnlearn as bn
from pgmpy.sampling import BayesianModelSampling

from ...annotations import Annotations, AxisAnnotation

logger = logging.getLogger(__name__)

from ..base import ConceptDataset
from ..preprocessing.autoencoder import extract_embs_from_autoencoder
from ..io import download_url

BUILTIN_DAGS = ['asia', 'alarm', 'andes', 'sachs', 'water']

class BnLearnDataset(ConceptDataset):
    """Dataset class for the Asia dataset from bnlearn.

    This dataset represents a small expert system that models the relationship
    between traveling to Asia, smoking habits, and various lung diseases.
    """

    def __init__(
            self,
            name: str, # name of the bnlearn DAG
            root: str = None, # root directory to store/load the dataset
            seed: int = 42, # seed for data generation
            n_gen: int = 10000,
            concept_subset: Optional[list] = None, # subset of concept labels
            label_descriptions: Optional[dict] = None,
            autoencoder_kwargs: Optional[dict] = None, # kwargs of the autoencoder used to extract latent representations
    ):
        self.name = name
        self.seed = seed

        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', self.name)
            
        self.root = root
        self.n_gen = n_gen

        self.autoencoder_kwargs = autoencoder_kwargs
        self.label_descriptions = label_descriptions

        # embeddings is a torch tensor
        # concepts is a pandas dataframe
        # annotations is an object Annotations
        # graph is the adjacency matrix as a pandas dataframe
        embeddings, concepts, annotations, graph = self.load()

        # Initialize parent class
        super().__init__(
            input_data=embeddings,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset, # subset of concept names
        )
        
    @property
    def raw_filenames(self) -> List[str]:
        """List of raw filenames that need to be present in the raw directory
        for the dataset to be considered present."""
        if self.name in BUILTIN_DAGS:
            return []  # nothing to download, these are built-in in bnlearn
        else:
            return [f"{self.name}.bif"]

    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            f"embs_N_{self.n_gen}_seed_{self.seed}.pt",
            f"concepts_N_{self.n_gen}_seed_{self.seed}.h5",
            "annotations.pt",
            "graph.h5"
        ]

    def download(self):
        if self.name in BUILTIN_DAGS:
            pass
        else:
            url = f'https://www.bnlearn.com/bnrepository/{self.name}/{self.name}.bif.gz'
            gz_path = download_url(url, self.root_dir)
            bif_path = self.raw_paths[0]
            
            # Decompress .gz file
            with gzip.open(gz_path, 'rb') as f_in:
                with open(bif_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove the .gz file after extraction
            os.unlink(gz_path)

    def build(self):
        self.maybe_download()
        if self.name in BUILTIN_DAGS:
            self.bn_model_dict = bn.import_DAG(self.name)
        else:
            self.bn_model_dict = bn.import_DAG(self.raw_paths[0])
        self.bn_model = self.bn_model_dict["model"]

        # generate data
        inference = BayesianModelSampling(self.bn_model)
        df = inference.forward_sample(size=self.n_gen, 
                                      seed=self.seed)
        
        # extract embeddings from latent autoencoder state
        concepts = df.copy()
        embeddings = extract_embs_from_autoencoder(
            df, 
            self.autoencoder_kwargs if self.autoencoder_kwargs is not None else {}
        )

        # get concept annotations
        concept_names = list(self.bn_model.nodes())
        # get concept metadata, store as many objects as you need.
        # at least store the variable 'type'! ('discrete' or 'continuous')
        concept_metadata = {
            node: {'type': 'discrete'} for node in concept_names
        }
        
        cardinalities = [int(self.bn_model.get_cardinality()[node]) for node in concept_names]
        # categorical concepts with card=2 will be treated as Bernoulli (card=1)
        cardinalities = [1 if card == 2 else card for card in cardinalities]

        annotations = Annotations({
            # 0: batch axis, do not need to annotate
            # 1: concepts axis, always annotate
            1: AxisAnnotation(labels=concept_names,
                              cardinalities=cardinalities,
                              metadata=concept_metadata)})
        
        # get the graph for the endogenous concepts
        graph = self.bn_model_dict['adjmat']
        graph = graph.astype(int)

        # ---- save all ----
        # save embeddings
        logger.info(f"Saving dataset to {self.root_dir}")
        torch.save(embeddings, self.processed_paths[0])
        # save concepts
        concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        # save concept annotations
        torch.save(annotations, self.processed_paths[2])
        # save graph
        graph.to_hdf(self.processed_paths[3], key="graph", mode="w")

    def load_raw(self):
        self.maybe_build()
        logger.info(f"Loading dataset from {self.root_dir}")
        embeddings = torch.load(self.processed_paths[0], weights_only=False)
        concepts = pd.read_hdf(self.processed_paths[1], "concepts")
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = pd.read_hdf(self.processed_paths[3], "graph")
        return embeddings, concepts, annotations, graph

    def load(self):
        embeddings, concepts, annotations, graph = self.load_raw()
        return embeddings, concepts, annotations, graph
