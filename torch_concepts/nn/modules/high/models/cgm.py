"""Causal Concept Graph Model and its model-specific helper functions."""

from __future__ import annotations

import random
from functools import cached_property, partial
from typing import Optional, Sequence, Union

import networkx as nx
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal, OneHotCategorical

from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from .....graph_generator import (
    GraphGeneratorLearnable,
    remove_weakest_cycles,
)
from .....distributions import Delta
from .....utils import ensure_list
from ...low.dense_layers import MLP
from ...low.graph_aggregator import GraphAggregator
from ...low.intervention.intervention import InterventionModule
from ...low.base.intervention import InterventionPolicy
from ...low.intervention.strategy.do import DoIntervention
from ...low.sequential import Sequential
from ...low.priors import LearnablePrior
from ...low.predictors.mix import (
    MixConceptEmbeddingToConceptEmbedding,
)
from ...utils import state_embedding_counts
from ...low.predictors.neural_structural_equations import (
    NeuralStructuralEquations,
)
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.inference.torch.independent import IndependentInference
from ...mid.variable import ConceptVariable, EmbeddingVariable
from ...outputs import ModelOutput
from ..base.graph import DirectedGraphModel


# ---------------------------------------------------------------------------
# Helper functions used only by CausalCGM
# ---------------------------------------------------------------------------

class _CGMInterventionPolicy(InterventionPolicy):
    """Helper policy that selects flattened concept columns per batch row."""

    def __init__(self, indices: torch.Tensor):
        super().__init__()
        self.register_buffer("indices", indices.long())

    def forward(self, concepts, *args, **kwargs):
        indices = self.indices.to(concepts.device)
        if indices.dim() == 1:
            indices = indices.unsqueeze(-1)
        return torch.ones_like(concepts).scatter(1, indices, 0.0)


def _select_params(output, names, prefix=""):
    """Helper function that selects variables and optionally prefixes quantities."""
    selected = {}
    for quantity, tensor in output.params.items():
        labels = {
            *tensor.annotation.label_to_index,
            *tensor.annotation.label_groups,
        }
        present = [name for name in names if name in labels]
        if present:
            selected[f"{prefix}{quantity}"] = tensor[present]
    return selected



# ---------------------------------------------------------------------------
# CausalCGM model
# ---------------------------------------------------------------------------


class CausalCGM(DirectedGraphModel):
    """Causal CGM with joint graph and mechanism learning.

    The training PGM contains three specific variable families:

    - ``U``: one exogenous context extracted from the input for each concept;
    - ``V_prime``: independent concept copies, predicted from ``U`` and set to
      the ground truth through teacher forcing;
    - ``V``: the endogenous concepts predicted from ``U`` and ``V_prime``
      through the current graph and the structural equations.

    Concepts in ``V`` with the same type and cardinality share a plate so their
    common mixer, graph aggregation, and structural computation run together.
    During training, the mechanisms and, when present, ``graph_generator`` are
    optimized jointly. A fixed DAG can instead be passed through ``graph``.
    Mixed concept types may create multiple endogenous CPDs, one per compatible
    plate.

    During evaluation, a learned adjacency is materialized as a DAG, while a
    supplied DAG is used directly. The graph is unfolded into a Bayesian network:
    the auxiliary copies ``V_prime`` are removed and their learned mechanisms are
    installed on the corresponding endogenous variables ``V``. The resulting
    network is queried with the configured evaluation inference. For learned
    graphs, the pruned DAG defines traversal order and roots; non-root equations
    retain raw edges from earlier nodes, matching the upstream implementation.
    Root variables reuse their copy-CPD parametrization directly. Non-root
    variables are evaluated through mixer, graph aggregation with the raw
    adjacency, and structural equations. Thus the evaluation network can contain
    effective dependencies absent from the materialized ``graph``.

    For details, see the `paper <https://arxiv.org/abs/2405.16507>`_.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types).
    task_names : Union[Sequence[str], str], optional
        Names of task variables. At least one task is required. Tasks are
        excluded from random training interventions and, by default, cannot
        have outgoing edges.
    embedding_size : int, default 8
        Width of each per-concept state embedding.
    shared_n_layers : int, default 0
        Hidden layers in the shared structural equations.
    shared_hidden_size : Optional[int], default None
        Shared-equation hidden width; defaults to ``embedding_size``.
    concept_n_layers : int, default 1
        Hidden layers in each concept-specific structural equation.
    concept_hidden_size : Optional[int], default None
        Concept-specific hidden width; defaults to ``embedding_size``.
    shared_activation : str, default "leaky_relu"
        Activation inside the shared structural MLP and automatically applied
        to its output.
    concept_activation : str, default "leaky_relu"
        Activation inside each concept-specific structural MLP.
    graph : Optional[ConceptGraph], default None
        A pre-learned or otherwise fixed DAG. Its topology is used during
        training and directly unfolded into the evaluation Bayesian network;
        only the CGM mechanisms are optimized. Pass either ``graph`` or
        ``graph_generator``; when both are omitted, DAGMA-CGM is created.
    graph_generator : Optional[GraphGeneratorLearnable], default None
        Learnable graph generator; ``None`` creates the paper's DAGMA-CGM
        generator with its default configuration. Graph-specific options such
        as ``no_out_task``, ``edges_to_check``, and ``initialization`` belong
        to the generator. If provided, it must be a
        ``GraphGeneratorLearnable``.
    inference : Optional[type[BaseInference]], default DeterministicInference
        Inference engine class for evaluation (see :class:`BaseInference`).
    inference_kwargs : Optional[dict], default None
        Keyword arguments for the evaluation inference engine.
    train_inference : Optional[type[BaseInference]], default IndependentInference
        Inference engine for the training Bayesian network.
    train_inference_kwargs : Optional[dict], default None
        Keyword arguments forwarded unchanged to the training inference engine.
    run_interventions : bool, default True
        If True, each training forward also evaluates the random intervention
        scenarios used by the CACE regularizer. Disable this to compute only
        the observational prior and posterior predictions.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``).
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})

    # Per-type distribution policy: how this model models each concept type.
    variable_distributions = {
        "binary": Bernoulli,
        "categorical": OneHotCategorical,
        "continuous": Normal,
    }
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Optional[Union[Sequence[str], str]] = None,
        embedding_size: int = 8,
        shared_n_layers: int = 0,
        shared_hidden_size: Optional[int] = None,
        concept_n_layers: int = 1,
        concept_hidden_size: Optional[int] = None,
        shared_activation: str = "leaky_relu",
        concept_activation: str = "leaky_relu",
        graph: Optional[ConceptGraph] = None,
        graph_generator: Optional[GraphGeneratorLearnable] = None,
        inference: Optional[type[BaseInference]] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[type[BaseInference]] = IndependentInference,
        train_inference_kwargs: Optional[dict] = None,
        run_interventions: bool = True,
        lightning: bool = False,
        **kwargs,
    ) -> None:
        task_names = ensure_list(task_names) if task_names is not None else []
        if not task_names:
            raise ValueError("CausalCGM requires at least one task in `task_names`.")
        if not set(task_names).issubset(annotations.labels):
            raise ValueError("task_names must be present in annotations.")
        if graph is not None and graph_generator is not None:
            raise ValueError("Pass either `graph` or `graph_generator`, not both.")
        if graph is None and graph_generator is None:
            graph_generator = GraphGeneratorLearnable(
                name="dagma_cgm",
                source="DAGMA_CGM",
                concept_names=list(annotations.labels),
                task_names=task_names,
                refinement=remove_weakest_cycles,
                initialization=lambda generator: None,
            )
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            plate=True,
            graph=graph,
            **kwargs,
        )
        self.embedding_size = embedding_size
        self._shared_encoder = Sequential(
            self.backbone,
            MLP(
                self.latent_size,
                self.embedding_size,
                output_size=self.embedding_size,
                n_layers=2,
                activation="leaky_relu",
            ),
        )
        self.task_names = task_names
        self.copy_names = [f"{name}__copy" for name in self.concept_names]
        self.n_concepts = len(self.concept_names) - len(self.task_names)
        if self.n_concepts == 0:
            raise ValueError("CausalCGM requires at least one intervenable concept.")
        self.run_interventions = bool(run_interventions)
        self.intervention_indices = [
            index for index, name in enumerate(self.concept_names)
            if name not in self.task_names
        ]
        self.structural_equation_kwargs = {
            "shared_n_layers": shared_n_layers,
            "shared_hidden_size": shared_hidden_size,
            "concept_n_layers": concept_n_layers,
            "concept_hidden_size": concept_hidden_size,
            "shared_activation": shared_activation,
            "concept_activation": concept_activation,
        }

        self.endogenous_types = [
            self.concept_annotations.concept(name).type
            for name in self.concept_names
        ]
        self.cardinalities = [
            self.concept_annotations.concept(name).cardinality
            for name in self.concept_names
        ]
        self.n_state_embeddings = state_embedding_counts(
            self.endogenous_types, self.cardinalities
        )
        if graph_generator is not None and not isinstance(graph_generator, GraphGeneratorLearnable):
            raise TypeError("`graph_generator` must be a GraphGeneratorLearnable.")
        if graph_generator is not None and list(graph_generator.concept_names) != list(self.concept_names):
            raise ValueError("Graph generator concept names must match annotations.")
        self.graph_layer = GraphAggregator(
            generator=graph_generator,
            adjacency=None if graph_generator is not None else self.graph.data,
        )
        configure_loss_terms = getattr(
            getattr(self, "loss", None), "configure_terms", None
        )
        if configure_loss_terms is not None:
            configure_loss_terms(self.graph_generator)
            if hasattr(self.loss, "task_names"):
                self.loss.task_names = list(self.task_names)
        if self.graph_generator is not None:
            self._validate_graph_generator_compatibility()
        self._build_model()

        # The training PGM now exists. Its inference engine can be created
        # immediately; evaluation inference is created later, after the learned
        # graph has been materialized as a DAG.
        self.setup_inference(
            inference or DeterministicInference,
            inference_kwargs,
            train_inference or IndependentInference,
            train_inference_kwargs,
        )
        self.register_load_state_dict_post_hook(
            lambda module, _incompatible_keys: module._invalidate_eval_pgm()
        )

    # ------------------------------------------------------------------
    # Graph configuration and validation
    # ------------------------------------------------------------------

    def _resolve_graph(self):
        """Use an empty DAG until the learned structure is materialized."""
        if self._given_graph is not None:
            return self._given_graph
        adjacency = torch.zeros(len(self.concept_names), len(self.concept_names))
        return ConceptGraph(adjacency, node_names=list(self.concept_names))

    @property
    def graph_generator(self):
        return self.graph_layer.generator

    def _adjacency(self) -> torch.Tensor:
        """Return the trainable or fixed graph adjacency."""
        return self.graph_layer.graph()

    def _validate_graph_generator_compatibility(
        self, adjacency: Optional[torch.Tensor] = None,
    ) -> None:
        """Validate the adjacency contract required by CausalCGM."""
        if adjacency is None:
            with torch.no_grad():
                adjacency = self.graph_generator()
        expected = (len(self.concept_names), len(self.concept_names))
        if not isinstance(adjacency, torch.Tensor) or adjacency.shape != expected:
            raise ValueError(
                "CausalCGM graph generators must return a tensor with shape "
                f"{expected}."
            )
        if not torch.isfinite(adjacency).all():
            raise ValueError("CausalCGM graph adjacencies must be finite.")
        if (adjacency < 0).any():
            raise ValueError("CausalCGM graph adjacencies must be non-negative.")
        if not torch.allclose(
            adjacency.diagonal(), torch.zeros_like(adjacency.diagonal())
        ):
            raise ValueError("CausalCGM graph adjacencies must have a zero diagonal.")
        if (
            getattr(self.graph_generator, "no_out_task", False)
            and self.task_names
            and adjacency[
                [self.concept_names.index(name) for name in self.task_names]
            ].any()
        ):
            raise ValueError(
                "With no_out_task=True, task rows in the adjacency must be zero."
            )

    # ------------------------------------------------------------------
    # Training PGM construction
    # ------------------------------------------------------------------

    @property
    def mixer(self):
        return self._mixer

    @property
    def structural_equations(self):
        return self._structural_equations

    def _aggregate_inputs(
        self, inputs, *, exogenous, endogenous, sources, root=None,
        empty_structural=False,
    ):
        """Collect parent embeddings and values for a structural equation."""
        if not sources:
            if empty_structural:
                reference = inputs[exogenous[root]]
                leading_shape = reference.shape[:-1]
                return {
                    "concept_embeddings": [
                        reference.new_zeros(
                            *leading_shape,
                            self.n_state_embeddings[root],
                            self.embedding_size,
                        )
                    ],
                    "concept_values": [
                        reference.new_zeros(
                            *leading_shape,
                            self.cardinalities[root],
                        )
                    ],
                    "source_concepts": [root],
                }
            return inputs[exogenous[root]]
        aggregated = {
            "concept_embeddings": [
                inputs[exogenous[source]].unflatten(
                    -1, (
                        self.n_state_embeddings[source],
                        self.embedding_size,
                    ),
                )
                for source in sources
            ],
            "concept_values": [
                inputs[endogenous[source]] for source in sources
            ],
        }
        if root is not None:
            aggregated["source_concepts"] = sources
        return aggregated

    def _input_latent_block(self):
        """Build the input and shared latent variables and their CPDs.

        The extra MLP after the backbone matches the original CausalCGM encoder.
        """
        input_var = EmbeddingVariable(
            "input", distribution=Delta, shape=self.input_size
        )
        shared_var = EmbeddingVariable(
            "shared_embedding", distribution=Delta, size=self.embedding_size
        )
        input_cpd = ParametricCPD(input_var, LearnablePrior(input_var.shape))
        shared_cpd = ParametricCPD(
            shared_var, self._shared_encoder, parents=[input_var]
        )
        return input_var, shared_var, input_cpd, shared_cpd

    def _build_exogenous_variables(
        self, names, embedding_size, name_fmt="{}_embedding",
    ):
        """Build one state-conditioned embedding variable per concept."""
        return [
            EmbeddingVariable(
                name_fmt.format(name), distribution=Delta,
                size=n_states * embedding_size,
            )
            for name, n_states in zip(names, self.n_state_embeddings)
        ]

    def _build_model(self) -> None:
        """Build the training PGM."""
        input_var, shared_var, self.input_cpd, self.shared_cpd = self._input_latent_block()
        exogenous = self._build_exogenous_variables(
            self.concept_names, self.embedding_size,
            name_fmt="{}__u",
        )
        endogenous = self.build_concept_variables(
            self.concept_names, "endogenous"
        )
        endogenous_copies = ConceptVariable(
            self.copy_names,
            distribution=[
                self.distribution_of(name) for name in self.concept_names
            ],
            size=self.cardinalities,
            dist_kwargs=[
                self.dist_kwargs_of(name) for name in self.concept_names
            ],
        )

        exogenous_encoders = [
            Sequential(
                MLP(
                    self.embedding_size, variable.size,
                    output_size=variable.size, n_layers=1,
                    activation="leaky_relu",
                ),
                nn.LeakyReLU(),
            )
            for variable in exogenous
        ]
        structural_equations = NeuralStructuralEquations(
            in_embeddings=self.embedding_size,
            out_concept_types=self.endogenous_types,
            cardinalities_out_concepts=self.cardinalities,
            **self.structural_equation_kwargs,
        )
        object.__setattr__(self, "_structural_equations", structural_equations)
        self._mixer = MixConceptEmbeddingToConceptEmbedding(
            in_embeddings=self.embedding_size,
            concept_types=self.endogenous_types,
            cardinalities=self.cardinalities,
            expand_binary_embeddings=False,
            complete_output=True,
        )
        self.exogenous_cpds = nn.ModuleList([
            ParametricCPD(variable, encoder, parents=[shared_var])
            for variable, encoder in zip(exogenous, exogenous_encoders)
        ])
        self.endogenous_copy_cpds = nn.ModuleList([
            ParametricCPD(
                copy_var,
                self._flexible_parametrization(
                    copy_var,
                    structural_equations.concept_structural_equations[node],
                    second="copy",
                ),
                parents=[exogenous_var],
            )
            for node, (copy_var, exogenous_var) in enumerate(zip(
                endogenous_copies, exogenous
            ))
        ])

        aggregate = partial(
            self._aggregate_inputs,
            exogenous=exogenous,
            endogenous=endogenous_copies,
            sources=tuple(range(len(self.concept_names))),
        )

        indices = {name: index for index, name in enumerate(self.concept_names)}
        self.endogenous_cpds = nn.ModuleList()
        for variable in endogenous:
            targets = [indices[name] for name in variable.members]

            # All endogenous CPDs share the same mixer, graph layer, and
            # structural-equation module. Plates only decide which target heads
            # are evaluated together.
            equations = (
                structural_equations
                if targets == list(range(len(self.concept_names)))
                else structural_equations.for_targets(targets)
            )
            cpd = ParametricCPD(
                variable,
                self._flexible_parametrization(
                    variable, equations, second="copy",
                ),
                parents=[*exogenous, *endogenous_copies],
                aggregate=aggregate,
                trunk=Sequential(self.mixer, self.graph_layer),
            )
            self.endogenous_cpds.append(cpd)

        self.pgm = BayesianNetwork(
            variables=[
                input_var, shared_var, *exogenous,
                *endogenous_copies, *endogenous,
            ],
            factors=[
                self.input_cpd, self.shared_cpd,
                *self.exogenous_cpds, *self.endogenous_copy_cpds,
                *self.endogenous_cpds,
            ],
        )

    # ------------------------------------------------------------------
    # Inference setup and evaluation PGM materialization
    # ------------------------------------------------------------------

    def setup_inference(
        self,
        inference=None,
        inference_kwargs=None,
        train_inference=None,
        train_inference_kwargs=None,
    ):
        """Configure ordinary inference engines for the two CGM graphs.

        Training-specific graph inputs and intervention scenarios are prepared
        by :meth:`_training_forward` and :meth:`_training_interventions`.
        """
        if train_inference is not None:
            super().setup_inference(
                train_inference=train_inference,
                train_inference_kwargs=train_inference_kwargs,
            )
        if inference is not None:
            self._eval_inference_cls = inference
            self._eval_inference_kwargs = dict(inference_kwargs or {})

    def materialize_graph(self):
        """Materialize the DAG defining evaluation order and root nodes."""
        if self.graph_generator is None:
            graph = ConceptGraph(
                self._adjacency(),
                node_names=self.concept_names,
            )
        else:
            graph = self.graph_generator.construct_graph()
        self._set_graph(graph)
        return graph

    def materialize_bayesian_network(self):
        """Match upstream traversal while retaining its raw aggregation weights.

        The pruned DAG determines order and roots. For non-roots, upstream
        aggregates raw edges from already visited nodes; later nodes still
        have zero embeddings. Roots bypass mixer/graph/structural equations and
        reuse the copy CPD. Non-roots keep the CGM trunk with a fixed raw
        adjacency. Encode those effective dependencies explicitly so the
        evaluation Bayesian network remains acyclic.
        """
        raw_adjacency = self._adjacency().detach().clone()
        graph = self.materialize_graph()
        exogenous = [cpd.variable for cpd in self.exogenous_cpds]
        endogenous = [
            self._make_concept_variable(name) for name in self.concept_names
        ]
        dag = nx.from_numpy_array(
            graph.data.detach().cpu().numpy(), create_using=nx.DiGraph,
        )
        parent_indices = {
            node: list(dag.predecessors(node))
            for node in list(nx.topological_sort(dag))
        }

        factors = []
        visited = []
        for node, parents in parent_indices.items():
            is_root = len(parents) == 0
            # The projected DAG defines traversal/root status. The original CGM
            # still aggregates with the raw learned adjacency; at a given node,
            # only already visited embeddings can contribute.
            sources = [] if is_root else [
                source for source in visited
                if raw_adjacency[source, node] != 0
            ]

            if not is_root and not sources:
                aggregate = partial(
                    self._aggregate_inputs,
                    exogenous=exogenous,
                    endogenous=endogenous,
                    sources=(),
                    root=node,
                    empty_structural=True,
                )
                factor_parents = [exogenous[node]]
            else:
                aggregate = partial(
                    self._aggregate_inputs,
                    exogenous=exogenous,
                    endogenous=endogenous,
                    sources=sources,
                    root=node,
                )
                factor_parents = (
                    [exogenous[node]] if is_root else
                    [*(exogenous[i] for i in sources),
                     *(endogenous[i] for i in sources)]
                )

            parametrization = (
                dict(self.endogenous_copy_cpds[node].parametrization)
                if is_root else
                self._flexible_parametrization(
                    endogenous[node],
                    self.structural_equations.for_targets([node]),
                    second="copy",
                )
            )
            factors.append(ParametricCPD(
                endogenous[node], parametrization,
                parents=factor_parents,
                aggregate=aggregate,
                trunk=None if is_root else Sequential(
                    self.mixer,
                    GraphAggregator(adjacency=raw_adjacency),
                ),
            ))
            visited.append(node)

        prefix_cpds = [self.input_cpd, self.shared_cpd, *self.exogenous_cpds]
        eval_pgm = BayesianNetwork(
            variables=[*(cpd.variable for cpd in prefix_cpds), *endogenous],
            factors=[*prefix_cpds, *factors],
        )
        eval_inference = self._eval_inference_cls(
            eval_pgm, **self._eval_inference_kwargs
        )
        self._modules.pop("eval_pgm", None)
        self._modules.pop("eval_inference", None)
        object.__setattr__(self, "eval_pgm", eval_pgm)
        object.__setattr__(self, "eval_inference", eval_inference)
        self._eval_pgm_stale = False
        return eval_pgm

    def _invalidate_eval_pgm(self):
        """Mark the materialized evaluation graph as derived from stale weights."""
        self._eval_pgm_stale = True
        if hasattr(self, "graph_layer"):
            self.graph_layer.clear()

    def _apply(self, fn):
        """Invalidate the evaluation PGM after a device or dtype change."""
        result = super()._apply(fn)
        if hasattr(self, "eval_pgm"):
            self._invalidate_eval_pgm()
        return result

    # ------------------------------------------------------------------
    # Training query and intervention helpers
    # ------------------------------------------------------------------

    @cached_property
    def _query_plan(self):
        """Map target columns to the copy variables used during training."""
        axis = self.concept_annotations
        return [
            (
                copy_name,
                [(axis.get_index(name), axis.concept(name).cardinality)],
            )
            for copy_name, name in zip(self.copy_names, self.concept_names)
        ]

    def _training_interventions(self, observed):
        """Helper that builds the random training intervention scenarios.

        One non-task concept is selected independently in every batch row.
        Binary concepts generate low/high scenarios; homogeneous categorical
        concepts generate one scenario per state. Other variable mixtures do
        not currently define a common intervention grid.
        """
        offsets = [0]
        for cardinality in self.cardinalities:
            offsets.append(offsets[-1] + cardinality)
        eligible_columns = [
            column
            for node in self.intervention_indices
            for column in range(offsets[node], offsets[node + 1])
        ]

        def intervene(constants, selected_columns):
            values = torch.cat(observed, dim=-1)
            module = InterventionModule(
                nn.Identity(), DoIntervention(constants),
                _CGMInterventionPolicy(selected_columns),
                out_concepts_to_intervene_on=eligible_columns,
                quantile=0.0,
            )
            return list(module(values).split(self.cardinalities, dim=-1))

        if all(type_name == "binary" for type_name in self.endogenous_types):
            leading_shape = observed[0].shape[:-1]
            selected = torch.tensor(
                [
                    random.sample(range(len(self.intervention_indices)), 1)[0]
                    for _ in range(leading_shape.numel())
                ],
                device=observed[0].device,
            ).reshape(leading_shape)
            selected_columns = torch.tensor(
                [offsets[self.intervention_indices[choice]] for choice in selected],
                device=observed[0].device,
            ).reshape(leading_shape)
            return {
                "low": intervene(0.0, selected_columns),
                "high": intervene(1.0, selected_columns),
            }
        if (
            len(set(self.endogenous_types)) == 1
            and len(set(self.cardinalities)) == 1
            and self.endogenous_types[0] == "categorical"
        ):
            leading_shape = observed[0].shape[:-1]
            selected = torch.tensor(
                [
                    random.sample(range(len(self.intervention_indices)), 1)[0]
                    for _ in range(leading_shape.numel())
                ],
                device=observed[0].device,
            ).reshape(leading_shape)
            selected_columns = torch.stack([
                torch.arange(
                    offsets[self.intervention_indices[choice]],
                    offsets[self.intervention_indices[choice] + 1],
                    device=observed[0].device,
                )
                for choice in selected.flatten()
            ]).reshape(*leading_shape, self.cardinalities[0])
            interventions = {}
            for category in range(self.cardinalities[0]):
                constants = torch.zeros_like(torch.cat(observed, dim=-1))
                constants.scatter_(
                    1, selected_columns[:, category:category + 1], 1.0
                )
                interventions[f"category_{category}"] = intervene(
                    constants, selected_columns
                )
            return interventions
        return {}

    def _training_forward(self, query, evidence, **kwargs):
        """Run training inference and assemble loss terms.

        This invalidates the materialized evaluation network, clears the graph
        cache, executes the observational training query, optionally executes
        the intervention queries, and stores the single adjacency generated by
        the graph layer under ``params["adjacency"]`` for graph losses.
        """
        self._invalidate_eval_pgm()
        output = self.inference.query(
            query=query,
            evidence=evidence,
            **kwargs,
        )
        adjacency = self.graph_layer.adjacency
        params = _select_params(output, self.concept_names)
        params.update(_select_params(output, self.copy_names, "prior_"))
        if self.run_interventions:
            observed = [query[name] for name in self.copy_names]
            for label, values in self._training_interventions(observed).items():
                intervened = self.inference.query(
                    query={
                        **dict(zip(self.copy_names, values)),
                        **dict.fromkeys(self.concept_names),
                    },
                    evidence=evidence,
                    **kwargs,
                )
                params.update(_select_params(
                    intervened, self.concept_names, f"{label}_",
                ))
        params["adjacency"] = adjacency
        return ModelOutput(
            params=params,
            extra={"task_names": tuple(self.task_names)},
        )

    # ------------------------------------------------------------------
    # Public forward and default query
    # ------------------------------------------------------------------

    def forward(
        self, query=None, evidence=None, input=None, target=None, **inference_kwargs
    ):
        """Run CGM inference without exposing its internal query layout.

        Standard PyTorch training passes ``target``; evaluation needs only the
        input. Lightning may continue to pass the query prepared by its shared
        step. An explicit ``query`` remains available for advanced inference.
        """
        if not self.training and (
            not hasattr(self, "eval_inference")
            or getattr(self, "_eval_pgm_stale", False)
        ):
            self.materialize_bayesian_network()
        if query is not None and target is not None:
            raise ValueError("Pass either `query` or `target`, not both.")
        if query is None:
            if self.training and target is None:
                raise ValueError("CausalCGM training requires `target`.")
            query = self.default_query(target, "train" if self.training else "eval")
        if self.training:
            evidence = dict(evidence or {})
            if input is not None:
                evidence["input"] = input
            return self._training_forward(
                query, evidence, **inference_kwargs
            )
        return super().forward(
            query=query, evidence=evidence, input=input, **inference_kwargs
        )

    def default_query(self, ground_truth, step="train"):
        """Build the CGM training/evaluation query.

        During training, :class:`CausalCGM` teacher-forces the copy variables
        from ``ground_truth`` through the model-specific ``_query_plan`` and
        also inserts the original concept variables with ``None`` values so the
        inference engine predicts them. During validation, testing, and direct
        evaluation, only the original concept variables are queried unobserved.
        """
        if step == "train":
            query = super().fully_observed_query(ground_truth)
            query.update(dict.fromkeys(self.concept_names))
            return query
        return dict.fromkeys(self.concept_names)


__all__ = ["CausalCGM"]
