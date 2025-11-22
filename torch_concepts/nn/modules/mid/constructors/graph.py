from typing import List, Tuple, Optional
from torch.nn import Identity

from .....annotations import Annotations
from ..models.variable import Variable
from .concept_graph import ConceptGraph
from ..models.factor import Factor
from ..models.probabilistic_model import ProbabilisticModel
from .....distributions import Delta
from ..base.model import BaseConstructor
from ...low.lazy import LazyConstructor


class GraphModel(BaseConstructor):
    """
    Concept-based model with explicit graph structure between concepts and tasks.

    This model builds a probabilistic model based on a provided
    concept graph structure. It automatically constructs the necessary variables
    and factors following the graph's topological order, supporting both root
    concepts (encoded from inputs) and internal concepts (predicted from parents).

    The graph structure defines dependencies between concepts, enabling:
    - Hierarchical concept learning
    - Causal reasoning with interventions
    - Structured prediction with concept dependencies

    Attributes:
        model_graph (ConceptGraph): Directed acyclic graph defining concept relationships.
        root_nodes (List[str]): Concepts with no parents (encoded from inputs).
        internal_nodes (List[str]): Concepts with parents (predicted from other concepts).
        graph_order (List[str]): Topologically sorted concept names.
        probabilistic_model (ProbabilisticModel): Underlying PGM with variables and factors.

    Args:
        model_graph: ConceptGraph defining the structure (must be a DAG).
        input_size: Size of input features.
        annotations: Annotations object with concept metadata and distributions.
        encoder: LazyConstructor for encoding root concepts from inputs.
        predictor: LazyConstructor for predicting internal concepts from parents.
        use_source_exogenous: Whether to use source exogenous features for predictions.
        source_exogenous: Optional propagator for source exogenous features.
        internal_exogenous: Optional propagator for internal exogenous features.

    Raises:
        AssertionError: If model_graph is not a DAG.
        AssertionError: If node names don't match annotations labels.

    Example:
        >>> import torch
        >>> import pandas as pd
        >>> from torch_concepts import Annotations, AxisAnnotation, ConceptGraph
        >>> from torch_concepts.nn import GraphModel, LazyConstructor
        >>> from torch.distributions import Bernoulli
        >>>
        >>> # Define concepts and their structure
        >>> # Structure: input -> [A, B] -> C -> D
        >>> # A and B are root nodes (no parents)
        >>> # C depends on A and B
        >>> # D depends on C
        >>> concept_names = ['A', 'B', 'C', 'D']
        >>>
        >>> # Create graph structure as adjacency matrix
        >>> graph_df = pd.DataFrame(0, index=concept_names, columns=concept_names)
        >>> graph_df.loc['A', 'C'] = 1  # A -> C
        >>> graph_df.loc['B', 'C'] = 1  # B -> C
        >>> graph_df.loc['C', 'D'] = 1  # C -> D
        >>>
        >>> graph = ConceptGraph(
        ...     torch.FloatTensor(graph_df.values),
        ...     node_names=concept_names
        ... )
        >>>
        >>> # Create annotations with distributions
        >>> annotations = Annotations({
        ...     1: AxisAnnotation(
        ...         labels=tuple(concept_names),
        ...         metadata={
        ...             'A': {'distribution': Bernoulli},
        ...             'B': {'distribution': Bernoulli},
        ...             'C': {'distribution': Bernoulli},
        ...             'D': {'distribution': Bernoulli}
        ...         }
        ...     )
        ... })
        >>>
        >>> # Create GraphModel
        >>> model = GraphModel(
        ...     model_graph=graph,
        ...     input_size=784,
        ...     annotations=annotations,
        ...     encoder=LazyConstructor(torch.nn.Linear),
        ...     predictor=LazyConstructor(torch.nn.Linear),
        ... )
        >>>
        >>> # Inspect the graph structure
        >>> print(model.root_nodes)  # ['A', 'B'] - no parents
        >>> print(model.internal_nodes)  # ['C', 'D'] - have parents
        >>> print(model.graph_order)  # ['A', 'B', 'C', 'D'] - topological order
        >>>
        >>> # Check graph properties
        >>> print(model.model_graph.is_dag())  # True
        >>> print(model.model_graph.get_predecessors('C'))  # ['A', 'B']
        >>> print(model.model_graph.get_successors('C'))  # ['D']

        References
            Dominici, et al. "Causal concept graph models: Beyond causal opacity in deep learning", ICLR 2025. https://arxiv.org/abs/2405.16507.
            De Felice, et al. "Causally reliable concept bottleneck models", NeurIPS https://arxiv.org/abs/2503.04363v1.
    """
    def __init__(self,
                 model_graph: ConceptGraph,
                input_size: int,
                annotations: Annotations,
                encoder: LazyConstructor,
                predictor: LazyConstructor,
                use_source_exogenous: bool = None,
                source_exogenous: Optional[LazyConstructor] = None,
                internal_exogenous: Optional[LazyConstructor] = None,
                 ):
        super(GraphModel, self).__init__(
            input_size=input_size,
            annotations=annotations,
            encoder=encoder,
            predictor=predictor,
        )
        self._source_exogenous_class = source_exogenous
        self._target_exogenous_class = internal_exogenous
        self.use_source_exogenous = use_source_exogenous

        assert model_graph.is_directed_acyclic(), "Input model graph must be a directed acyclic graph."
        assert model_graph.node_names == list(self.labels), "concept_names must match model_graph annotations."
        self.model_graph = model_graph
        self.root_nodes = [r for r in model_graph.get_root_nodes()]
        self.graph_order = model_graph.topological_sort()  # TODO: group by graph levels?
        self.internal_nodes = [c for c in self.graph_order if c not in self.root_nodes]
        self.root_nodes_idx = [self.labels.index(r) for r in self.root_nodes]
        self.graph_order_idx = [self.labels.index(i) for i in self.graph_order]
        self.internal_node_idx = [self.labels.index(i) for i in self.internal_nodes]

        # embedding variable and factor
        embedding_var = Variable('embedding', parents=[], size=self.input_size)
        embedding_factor = Factor('embedding', module_class=Identity())

        # concepts init
        if source_exogenous is not None:
            cardinalities = [self.annotations.get_axis_annotation(1).cardinalities[self.root_nodes_idx[idx]] for idx, c in enumerate(self.root_nodes)]
            source_exogenous_vars, source_exogenous_factors = self._init_exog(source_exogenous, label_names=self.root_nodes, parent_var=embedding_var, cardinalities=cardinalities)
            encoder_vars, encoder_factors = self._init_encoder(encoder, label_names=self.root_nodes, parent_vars=source_exogenous_vars, cardinalities=cardinalities)
        else:
            source_exogenous_vars, source_exogenous_factors = [], []
            encoder_vars, encoder_factors = self._init_encoder(encoder, label_names=self.root_nodes, parent_vars=[embedding_var])

        # tasks init
        if internal_exogenous is not None:
            cardinalities = [self.annotations.get_axis_annotation(1).cardinalities[self.internal_node_idx[idx]] for idx, c in enumerate(self.internal_nodes)]
            internal_exogenous_vars, internal_exogenous_factors = self._init_exog(internal_exogenous, label_names=self.internal_nodes, parent_var=embedding_var, cardinalities=cardinalities)
            predictor_vars, predictor_factors = self._init_predictors(predictor, label_names=self.internal_nodes, available_vars=encoder_vars, self_exog_vars=internal_exogenous_vars, cardinalities=cardinalities)
        elif use_source_exogenous:
            cardinalities = [self.annotations.get_axis_annotation(1).cardinalities[self.root_nodes_idx[idx]] for idx, c in enumerate(self.root_nodes)]
            internal_exogenous_vars, internal_exogenous_factors = [], []
            predictor_vars, predictor_factors = self._init_predictors(predictor, label_names=self.internal_nodes, available_vars=encoder_vars, source_exog_vars=source_exogenous_vars, cardinalities=cardinalities)
        else:
            internal_exogenous_vars, internal_exogenous_factors = [], []
            predictor_vars, predictor_factors = self._init_predictors(predictor, label_names=self.internal_nodes, available_vars=encoder_vars)

        # ProbabilisticModel Initialization
        self.probabilistic_model = ProbabilisticModel(
            variables=[embedding_var, *source_exogenous_vars, *encoder_vars, *internal_exogenous_vars, *predictor_vars],
            factors=[embedding_factor, *source_exogenous_factors, *encoder_factors, *internal_exogenous_factors, *predictor_factors],
        )

    def _init_exog(self, layer: LazyConstructor, label_names, parent_var, cardinalities) -> Tuple[Variable, Factor]:
        """
        Initialize exogenous variables and factors.

        Args:
            layer: LazyConstructor for exogenous features.
            label_names: Names of concepts to create exogenous features for.
            parent_var: Parent variable (typically embedding).
            cardinalities: Cardinalities of each concept.

        Returns:
            Tuple of (exogenous variables, exogenous factors).
        """
        exog_names = [f"exog_{c}_state_{i}" for cix, c in enumerate(label_names) for i in range(cardinalities[cix])]
        exog_vars = Variable(exog_names,
                            parents=parent_var.concepts,
                            distribution = Delta,
                            size = layer._module_kwargs['embedding_size'])

        lazy_constructor = layer.build(
            in_features_embedding=parent_var.size,
            in_features_logits=None,
            in_features_exogenous=None,
            out_features=1,
        )

        exog_factors = Factor(exog_names, module_class=lazy_constructor)
        return exog_vars, exog_factors

    def _init_encoder(self, layer: LazyConstructor, label_names, parent_vars, cardinalities=None) -> Tuple[Variable, Factor]:
        """
        Initialize encoder variables and factors for root concepts.

        Args:
            layer: LazyConstructor for encoding.
            label_names: Names of root concepts.
            parent_vars: Parent variables (embedding or exogenous).
            cardinalities: Optional cardinalities for concepts.

        Returns:
            Tuple of (encoder variables, encoder factors).
        """
        if parent_vars[0].concepts[0] == 'embedding':
            encoder_vars = Variable(label_names,
                                parents=['embedding'],
                                distribution=[self.annotations[1].metadata[c]['distribution'] for c in label_names],
                                size=[self.annotations[1].cardinalities[self.annotations[1].get_index(c)] for c in label_names])
            # Ensure encoder_vars is always a list
            if not isinstance(encoder_vars, list):
                encoder_vars = [encoder_vars]

            lazy_constructor = layer.build(
                in_features_embedding=parent_vars[0].size,
                in_features_logits=None,
                in_features_exogenous=None,
                out_features=encoder_vars[0].size,
            )
            encoder_factors = Factor(label_names, module_class=lazy_constructor)
            # Ensure encoder_factors is always a list
            if not isinstance(encoder_factors, list):
                encoder_factors = [encoder_factors]
        else:
            assert len(parent_vars) == sum(cardinalities)
            encoder_vars = []
            encoder_factors = []
            for label_name in label_names:
                exog_vars = [v for v in parent_vars if v.concepts[0].startswith(f"exog_{label_name}")]
                exog_vars_names = [v.concepts[0] for v in exog_vars]
                encoder_var = Variable(label_name,
                                    parents=exog_vars_names,
                                    distribution=self.annotations[1].metadata[label_name]['distribution'],
                                    size=self.annotations[1].cardinalities[self.annotations[1].get_index(label_name)])
                lazy_constructor = layer.build(
                    in_features_embedding=None,
                    in_features_logits=None,
                    in_features_exogenous=exog_vars[0].size,
                    out_features=encoder_var.size,
                )
                encoder_factor = Factor(label_name, module_class=lazy_constructor)
                encoder_vars.append(encoder_var)
                encoder_factors.append(encoder_factor)
        return encoder_vars, encoder_factors

    def _init_predictors(self,
                         layer: LazyConstructor,
                         label_names: List[str],
                         available_vars,
                         cardinalities=None,
                         self_exog_vars=None,
                         source_exog_vars=None) -> Tuple[List[Variable], List[Factor]]:
        """
        Initialize predictor variables and factors for internal concepts.

        Args:
            layer: LazyConstructor for prediction.
            label_names: Names of internal concepts to predict.
            available_vars: Variables available as parents (previously created concepts).
            cardinalities: Optional cardinalities for concepts.
            self_exog_vars: Optional self-exogenous variables.
            source_exog_vars: Optional source-exogenous variables.

        Returns:
            Tuple of (predictor variables, predictor factors).
        """
        available_vars = [] + available_vars
        predictor_vars, predictor_factors = [], []
        for c_name in label_names:
            endogenous_parents_names = self.model_graph.get_predecessors(c_name)
            endogenous_parents_vars = [v for v in available_vars if v.concepts[0] in endogenous_parents_names]
            in_features_logits = sum([c.size for c in endogenous_parents_vars])

            # check exogenous
            if self_exog_vars is not None:
                assert len(self_exog_vars) == sum(cardinalities)
                used_exog_vars = [v for v in self_exog_vars if v.concepts[0].startswith(f"exog_{c_name}")]
                exog_vars_names = [v.concepts[0] for v in used_exog_vars]
                in_features_exogenous = used_exog_vars[0].size
            elif source_exog_vars is not None:
                assert len(source_exog_vars) == len(endogenous_parents_names)
                exog_vars_names = [v.concepts[0] for v in source_exog_vars]
                used_exog_vars = source_exog_vars
                in_features_exogenous = used_exog_vars[0].size
            else:
                exog_vars_names = []
                used_exog_vars = []
                in_features_exogenous = None

            predictor_var = Variable(c_name,
                                     parents=endogenous_parents_names+exog_vars_names,
                                    distribution=self.annotations[1].metadata[c_name]['distribution'],
                                    size=self.annotations[1].cardinalities[self.annotations[1].get_index(c_name)])

            # TODO: we currently assume predictors can use exogenous vars if any, but not embedding
            lazy_constructor = layer.build(
                in_features_logits=in_features_logits,
                in_features_exogenous=in_features_exogenous,
                in_features_embedding=None,
                out_features=predictor_var.size,
                cardinalities=[predictor_var.size]
            )

            predictor_factor = Factor(c_name, module_class=lazy_constructor)

            predictor_vars.append(predictor_var)
            predictor_factors.append(predictor_factor)

            available_vars.append(predictor_var)

        return predictor_vars, predictor_factors
