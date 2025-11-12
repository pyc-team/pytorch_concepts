import os
import gzip
import shutil
import pandas as pd
import torch
from typing import List
import bnlearn as bn
from pgmpy.sampling import BayesianModelSampling

from torch_concepts import Annotations, AxisAnnotation

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
            seed: int, # seed for data generation
            n_gen: int = 10000,
            concept_subset: list | None = None, # subset of concept labels
            label_descriptions: dict | None = None,
            autoencoder_kwargs: dict | None = None, # kwargs of the autoencoder used to extract latent representations
            root: str = None
    ):
        self.name = name
        self.seed = seed
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
    def files_to_download_names(self) -> List[str]:
        """List of files that need to be found in the raw directory for the dataset to be
        considered present."""
        if self.name in BUILTIN_DAGS:
            return {} # nothing to download, these are built-in in bnlearn
        else:
            return {"bif": f"{self.name}.bif"}

    @property
    def files_to_build_names(self) -> dict[str, str]:
        return {"embeddings": f"embs_N_{self.n_gen}_seed_{self.seed}.pt",
                "concepts": f"concepts_N_{self.n_gen}_seed_{self.seed}.h5",
                "annotations": "annotations.pt",
                "graph": "graph.h5"}

    def download(self):
        if self.name in BUILTIN_DAGS:
            pass
        else:
            url = f'https://www.bnlearn.com/bnrepository/{self.name}/{self.name}.bif.gz'
            gz_path = download_url(url, self.root_dir)
            bif_path = os.path.join(self.root_dir, f"{self.name}.bif")
            
            # Decompress .gz file
            with gzip.open(gz_path, 'rb') as f_in:
                with open(bif_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)  # Use copyfileobj for file objects
            
            # Remove the .gz file after extraction
            os.unlink(gz_path)

    def build(self):
        self.maybe_download()
        if self.name in BUILTIN_DAGS:
            self.bn_model_dict = bn.import_DAG(self.name)
        else:
            self.bn_model_dict = bn.import_DAG(self.files_to_download_paths["bif"])
        self.bn_model = self.bn_model_dict["model"]

        # generate data
        inference = BayesianModelSampling(self.bn_model)
        df = inference.forward_sample(size=self.n_gen, 
                                      seed=self.seed)
        
        # extract embeddings from latent autoencoder state
        concepts = df.copy()
        embeddings = extract_embs_from_autoencoder(df, self.autoencoder_kwargs)

        # get concept annotations
        concept_names = list(self.bn_model.nodes())
        # get concept metadata, store as many objects as you need.
        # at least store the 'task' and the 'type'!
        concept_metadata = {node: {'type': 'discrete',
                                   'task': 'classification',
                                   'description': self.label_descriptions.get(node, "") 
                                                  if self.label_descriptions is not None else ""}
                            for node in concept_names}
        
        cardinalities = [int(self.bn_model.get_cardinality()[node]) for node in concept_names]
        # categorical concepts with card=2 are treated as Bernoulli (card=1)
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
        print(f"Saving dataset from {self.root_dir}")
        torch.save(embeddings, self.files_to_build_paths["embeddings"])
        # save concepts
        concepts.to_hdf(self.files_to_build_paths["concepts"], key="concepts", mode="w")
        # save concept annotations
        torch.save(annotations, self.files_to_build_paths["annotations"])
        # save graph
        graph.to_hdf(self.files_to_build_paths["graph"], key="graph", mode="w")

    def load_raw(self):
        self.maybe_build()
        print(f"Loading dataset from {self.root_dir}")
        embeddings = torch.load(self.files_to_build_paths["embeddings"])
        concepts = pd.read_hdf(self.files_to_build_paths["concepts"], "concepts")
        annotations = torch.load(self.files_to_build_paths["annotations"])
        graph = pd.read_hdf(self.files_to_build_paths["graph"], "graph")
        return embeddings, concepts, annotations, graph

    def load(self):
        embeddings, concepts, annotations, graph = self.load_raw()
        return embeddings, concepts, annotations, graph

