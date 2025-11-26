from typing import List, Optional, Union

import pandas as pd
import torch
from torch.nn import Module

from .....annotations import Annotations
from .concept_graph import ConceptGraph
from ...low.lazy import LazyConstructor
from .graph import GraphModel
from .....data.utils import ensure_list

class BipartiteModel(GraphModel):
    """
    Bipartite concept graph model with concepts and tasks in separate layers.

    This model implements a bipartite graph structure where concepts only connect
    to tasks (not to each other), creating a clean separation between concept
    and task layers. This is useful for multi-task learning with shared concepts.

    Attributes:
        label_names (List[str]): All node labels (concepts + tasks).
        concept_names (List[str]): Concept node labels.
        task_names (List[str]): Task node labels.

    Args:
        task_names: List of task names (must be in annotations labels).
        input_size: Size of input features.
        annotations: Annotations object with concept and task metadata.
        encoder: LazyConstructor for encoding concepts from inputs.
        predictor: LazyConstructor for predicting tasks from concepts.
        use_source_exogenous: Whether to use exogenous features for source nodes.
        source_exogenous: Optional propagator for source exogenous features.
        internal_exogenous: Optional propagator for internal exogenous features.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations, AxisAnnotation
        >>> from torch_concepts.nn import BipartiteModel, LazyConstructor, LinearCC
        >>> from torch.distributions import Bernoulli
        >>>
        >>> # Define concepts and tasks
        >>> all_labels = ('color', 'shape', 'size', 'task1', 'task2')
        >>> metadata = {'color': {'distribution': Bernoulli},
        ...             'shape': {'distribution': Bernoulli},
        ...             'size': {'distribution': Bernoulli},
        ...             'task1': {'distribution': Bernoulli},
        ...             'task2': {'distribution': Bernoulli}}
        >>> annotations = Annotations({
        ...     1: AxisAnnotation(labels=all_labels, metadata=metadata)
        ... })
        >>>
        >>> # Create bipartite model with tasks
        >>> task_names = ['task1', 'task2']
        >>>
        >>> model = BipartiteModel(
        ...     task_names=task_names,
        ...     input_size=784,
        ...     annotations=annotations,
        ...     encoder=LazyConstructor(torch.nn.Linear),
        ...     predictor=LazyConstructor(LinearCC)
        ... )
        >>>
        >>> # Generate random input
        >>> x = torch.randn(8, 784)  # batch_size=8
        >>>
        >>> # Forward pass (implementation depends on GraphModel)
        >>> # Concepts are encoded, then tasks predicted from concepts
        >>> print(model.concept_names)  # ['color', 'shape', 'size']
        >>> print(model.task_names)     # ['task1', 'task2']
        >>> print(model.probabilistic_model)
        >>>
        >>> # The bipartite structure ensures:
        >>> # - Concepts don't predict other concepts
        >>> # - Only concepts -> tasks edges exist
    """
    def __init__(
            self,
            task_names: Union[List[str], str],
            input_size: int,
            annotations: Annotations,
            encoder: Union[LazyConstructor, Module],
            predictor: Union[LazyConstructor, Module],
            use_source_exogenous: bool = None,
            source_exogenous: Optional[Union[LazyConstructor, Module]] = None,
            internal_exogenous: Optional[Union[LazyConstructor, Module]] = None,
    ):
        task_names = ensure_list(task_names)
        # get label names
        label_names = annotations.get_axis_labels(axis=1)
        assert all([t in label_names for t in task_names]), (f"All tasks must be in axis label names. "
                                                             f"Tasks {[t for t in task_names if t not in label_names]} "
                                                             f"are not in labels {label_names}")
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
