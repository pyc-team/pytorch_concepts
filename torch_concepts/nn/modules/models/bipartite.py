from typing import List, Optional, Union

import pandas as pd
import torch

from torch_concepts import Annotations, ConceptGraph
from ..propagator import Propagator
from .graph import GraphModel


class BipartiteModel(GraphModel):
    def __init__(
            self,
            task_names: Union[List[str], str, List[int]],
            input_size: int,
            annotations: Annotations,
            encoder: Propagator,
            predictor: Propagator,
            use_source_exogenous: bool = None,
            source_exogenous: Optional[Propagator] = None,
            internal_exogenous: Optional[Propagator] = None,
    ):
        # get label names
        label_names = annotations.get_axis_labels(axis=1)
        assert all([t in label_names for t in task_names]), "All tasks must be in axis label names"
        concept_names = [c for c in annotations.get_axis_annotation(1).labels if c not in task_names]

        # build bipartite graph
        graph = pd.DataFrame(0, index=label_names, columns=label_names)
        graph.loc[:, task_names] = 1  # concepts point to tasks
        graph.loc[task_names, task_names] = 0  # tasks do not point to themselves
        model_graph = ConceptGraph(torch.FloatTensor(graph.values), node_names=list(label_names))

        super(BipartiteModel, self).__init__(
            model_graph=model_graph,
            input_size=input_size,
            annotations=annotations,
            encoder=encoder,
            predictor=predictor,
            use_source_exogenous=use_source_exogenous,
            source_exogenous=source_exogenous,
            internal_exogenous=internal_exogenous,
        )
        self.label_names = label_names
        self.concept_names = concept_names
        self.task_names = task_names
