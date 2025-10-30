from typing import Dict

import torch
import pandas as pd

from torch_concepts import AnnotatedAdjacencyMatrix, Annotations
from .graph import GraphModel
from ....nn import Propagator

class BipartiteModel(GraphModel):
    """
    Model using a bipartite graph structure between concepts and tasks.
    Assuming independent concepts and tasks.
    """
    def __init__(self,
                 task_names: list[str],
                 input_size: int,
                 annotations: Annotations,
                 encoder: Propagator,
                 predictor: Propagator,
                 predictor_in_embedding: int,
                 predictor_in_exogenous: int,
                 exogenous: Propagator = None,
                 ):

        # create bipartite graph from concepts and tasks
        concept_names = annotations.get_axis_labels(axis=1)
        assert all([t in concept_names for t in task_names]), "All tasks must be in concept names"
        graph = pd.DataFrame(0, index=concept_names, columns=concept_names)
        graph.loc[:, task_names] = 1  # concepts point to tasks
        graph.loc[task_names, task_names] = 0  # tasks do not point to themselves
        bipartite_graph = AnnotatedAdjacencyMatrix(torch.FloatTensor(graph.values), annotations)

        self.predictor_in_embedding = predictor_in_embedding
        self.predictor_in_exogenous = predictor_in_exogenous
        self.predictor_in_logits = 1

        super(BipartiteModel, self).__init__(
            input_size=input_size,
            annotations=annotations,
            encoder=encoder,
            predictor=predictor,
            model_graph=bipartite_graph,
            exogenous=exogenous
        )
