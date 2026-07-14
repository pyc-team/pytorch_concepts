"""Hybrid Concept Bottleneck Model (Hybrid CBM).

A Concept Bottleneck Model (CBM) whose bottleneck is extended with an
unsupervised set of neurons (which will not be aligned with any known concepts).
This allows for a more flexible representation of the latent space, where some
dimensions can be dedicated to concept prediction and others to task prediction.

This architecture is a standard baseline, particularly for evaluating the
performance of concept bottleneck models when the concept set is incomplete or
noisy.
"""
from typing import Dict, List, Optional, Union

import torch

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from .....distributions import Delta
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.models.bayesian_network import BayesianNetwork
from ...mid.models.cpd import ParametricCPD
from ...mid.models.variable import ConceptVariable, EmbeddingVariable, \
    _DEFAULT_DIST_KWARGS
from .cbm import ConceptBottleneckModel


def _merge_bottleneck_parents(
    concepts: Dict,
    embeddings: Dict,
) -> Dict[str, torch.Tensor]:
    """Aggregate for the task CPDs: fuse the whole bottleneck into one input.

    The task predictors take both the supervised concepts (``ConceptVariable``
    parents) and the unsupervised dimensions (``EmbeddingVariable`` parents);
    this concatenates them (concepts first, then unsupervised dimensions,
    each group in parent order) into the single ``concepts`` input of
    type `LinearConceptToConcept`.
    """
    values = [
        v.float() if not v.is_floating_point() else v
        for v in [*concepts.values(), *embeddings.values()]
    ]
    return {"concepts": torch.cat(values, dim=-1)}


class HybridConceptBottleneckModel(ConceptBottleneckModel):
    """Hybrid Concept Bottleneck Model.

    Linear ``latent → concepts + unsupervised dimensions → tasks`` bottleneck
    with unsupervised latent dimensions in the bottleneck (which can be thought
    of as a form of a shared embedding across concepts).

    The unsupervised dimensions enter the probabilistic model as
    *non-interpretable* `EmbeddingVariable` nodes (of dimension 1), so they are
    never supervised or included in concept losses/metrics. Continuous
    dimensions are modelled as deterministic (``Delta``) neurons; binary
    dimensions as ``Bernoulli`` logits (see `unsupervised_distributions`).

    Works as a pure PyTorch module by default, or as a Lightning module when
    ``lightning=True``.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types).
    additional_dims : int
        Dimensionality of additional latent dimensions in the bottleneck. If this
            is set to 0, the model reduces to a standard ConceptBottleneckModel.
    task_names : Union[List[str], str]
        Names of the task variables (a subset of the annotation labels).
    additional_dim_types: Union[List[str], str], optional
        Types of the additional latent dimensions. If a list is provided, each
        element corresponds to the type of the respective dimension. If a string
        is provided, it is used for all additional dimensions. Must be one of
        ``"binary"`` or ``"continuous"`` (``"categorical"`` is not supported,
        as each unsupervised dimension occupies a single bottleneck neuron).
        Defaults to ``"continuous"`` for all dimensions. If ``additional_dims``
        is 0, this argument is ignored.
    inference : BaseInference, optional
        Evaluation inference engine class. Defaults to ``DeterministicInference``.
    inference_kwargs : dict, optional
        Keyword arguments forwarded to the evaluation inference engine.
    train_inference : BaseInference, optional
        Training inference engine class (defaults to ``inference``).
    train_inference_kwargs : dict, optional
        Keyword arguments forwarded to the training inference engine.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default None
        Controls which building path is used.  ``None`` (default) auto-detects:
        uses plates only when **all** graph levels are plate-compatible (see
        :meth:`plate_compatible_levels` — for this model the supervised
        concepts and the unsupervised dimensions are checked as separate
        groups, since each is built as its own plate), otherwise falls back to
        individual variables.  Pass ``True`` to force plates or ``False`` to
        force individual variables.
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``, and
        the Lightning training arguments).
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    param_for_discrete_var = "logits"

    # Per-type distribution policy: how this model models each concept type.
    variable_distributions = {
        'binary': Bernoulli,
        'categorical': OneHotCategorical,
        'continuous': Normal,
    }
    variable_dist_kwargs = dict(_DEFAULT_DIST_KWARGS)

    # Per-type distribution policy for the *unsupervised* bottleneck dimensions.
    # Continuous dimensions are deterministic (``Delta``) neurons while binary
    # dimensions are modelled as ``Bernoulli`` logits.
    unsupervised_distributions = {
        'binary': Bernoulli,
        'continuous': Delta,
    }

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        additional_dims: int,
        task_names: Union[List[str], str],
        additional_dim_types: Optional[Union[List[str], str]] = "continuous",
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs,
    ):
        additional_dims = max(0, additional_dims)

        # First determine the types of the additional latent dimensions. If a
        # single type is provided, we will use it for all additional dimensions.
        # If a list is provided, we will use the corresponding type for each
        # dimension.
        if isinstance(additional_dim_types, str):
            additional_dim_types = [additional_dim_types] * additional_dims
        elif not isinstance(additional_dim_types, list):
            raise ValueError(
                f"additional_dim_types must be a string or a list of strings, "
                f"got {type(additional_dim_types)}."
            )
        if additional_dims and (len(additional_dim_types) != additional_dims):
            raise ValueError(
                f"Length of additional_dim_types ({len(additional_dim_types)}) "
                f"must match additional_dims ({additional_dims})."
            )
        if additional_dims:
            unsupported_types = sorted(
                set(additional_dim_types) - set(self.unsupervised_distributions)
            )
            if unsupported_types:
                raise ValueError(
                    f"additional_dim_types must be among "
                    f"{sorted(self.unsupervised_distributions)}, got "
                    f"{unsupported_types}. In particular, 'categorical' "
                    f"unsupervised dimensions are not supported as each "
                    f"unsupervised dimension occupies a single bottleneck "
                    f"neuron."
                )

        # Extend the annotations with a set of dummy concepts for the additional
        # latent dimensions. These will be treated as unsupervised latent
        # variables in the bottleneck.
        # However, to avoid potential name conflicts, we will first figure out
        # how many underscores we will prepend "unsup_" with, to avoid conflicts
        # with existing concept names.
        self.additional_dims = additional_dims
        self.unsup_names = []
        self.unsup_plate_name = None
        if additional_dims > 0:
            # Find the number of underscores to prepend to "unsup_" to avoid
            # conflicts with existing concept names.
            used_names = set(annotations.labels)
            prefix = "__unsup_"
            while any(name.startswith(prefix) for name in used_names):
                prefix = "_" + prefix
            self.unsup_names = [f"{prefix}{i}" for i in range(additional_dims)]
            # Name used for the unsupervised plate variable in the plate
            # building path (collision-free by construction of ``prefix``).
            self.unsup_plate_name = f"{prefix}plate"
            metadata = None
            if annotations.metadata:
                metadata = {
                    **annotations.metadata,
                    **{name: {} for name in self.unsup_names},
                }
            used_annotations = Annotations(
                labels=(annotations.labels + self.unsup_names),
                states=(annotations.states + [['0']] * additional_dims),
                cardinalities=(
                    annotations.cardinalities + [1] * additional_dims
                ),
                types=(annotations.types + additional_dim_types),
                metadata=metadata,
                concept_space=annotations.concept_space,
            )
        else:
            # Otherwise, this will be the same as a standard CBM
            used_annotations = annotations

        # The parent constructor dispatches to the (overridden) building paths
        # below and wires the inference engines around the resulting PGM.
        super().__init__(
            input_size=input_size,
            annotations=used_annotations,
            task_names=task_names,
            inference=inference,
            inference_kwargs=inference_kwargs,
            train_inference=train_inference,
            train_inference_kwargs=train_inference_kwargs,
            lightning=lightning,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @property
    def supervised_concept_names(self) -> List[str]:
        """Intermediate concept labels excluding the unsupervised dimensions."""
        unsup = set(self.unsup_names)
        return [n for n in self.intermediate_concept_names if n not in unsup]

    def _unsup_distribution_of(self, name: str) -> type:
        """Distribution class used for unsupervised dimension ``name``
        (by type).
        """
        return self.unsupervised_distributions[
            self.concept_annotations.concept(name).type
        ]

    def _unsup_dist_kwargs_of(self, name: str) -> dict:
        """Distribution keyword arguments for unsupervised dimension
        ``name``.
        """
        return dict(
            self.variable_dist_kwargs.get(self._unsup_distribution_of(name), {})
        )

    def plate_compatible_levels(
        self,
        axis_annotation: Annotations,
        graph: ConceptGraph,
    ) -> List[bool]:
        """Flag, per graph level, whether its concepts can share plates.

        Overrides the base check: the supervised concepts and the unsupervised
        dimensions are built as *separate* plate variables, so a level is
        plate-compatible when each of the two groups is internally homogeneous
        (same type and cardinality) (however notice that the level as a whole
        may mix, e.g., binary concepts with continuous unsupervised dimensions).
        """
        unsup = set(self.unsup_names)

        def type_and_size(name: str):
            idx = axis_annotation.get_index(name)
            return (
                axis_annotation.types[idx],
                int(axis_annotation.cardinalities[idx]),
            )

        def homogeneous(names: List[str]) -> bool:
            return len({type_and_size(name) for name in names}) <= 1

        return [
            homogeneous([n for n in level if n not in unsup])
            and homogeneous([n for n in level if n in unsup])
            for level in graph.get_levels()
        ]

    # ------------------------------------------------------------------
    # Building paths
    # ------------------------------------------------------------------
    def _build_plate_model(self) -> BayesianNetwork:
        """Build using one plate variable per bipartite
        level (concepts, tasks), plus one plate for the unsupervised dimensions.
        """
        if not self.additional_dims:
            # Then this is simply the standard CBM!
            return super()._build_plate_model()

        # Otherwise we are in Hybrid CBM territory
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        variables = [input_var, latent_var]
        factors = [input_cpd, latent_cpd]
        task_parents = []

        supervised = self.supervised_concept_names
        if supervised:
            concept0 = axis.concept(supervised[0])
            concepts = ConceptVariable(
                names="concepts",
                members=supervised,
                distribution=self.distribution_of(concept0.name),
                dist_kwargs=self.dist_kwargs_of(concept0.name),
                size=concept0.cardinality,
            )
            encoders = ParametricCPD(
                variable=concepts,
                parents=[latent_var],
                parametrization=self._flexible_parametrization(
                    variable=concepts,
                    first=LinearEmbeddingToConcept(
                        in_embeddings=self.latent_size,
                        out_concepts=concepts.size,
                    ),
                    second=None,
                )
            )
            variables.append(concepts)
            factors.append(encoders)
            task_parents.append(concepts)

        # The unsupervised dimensions form their own (non-interpretable) plate:
        # they receive no supervision, so they must not be concept variables.
        unsup0 = axis.concept(self.unsup_names[0])
        unsup = EmbeddingVariable(
            names=self.unsup_plate_name,
            members=self.unsup_names,
            distribution=self._unsup_distribution_of(unsup0.name),
            dist_kwargs=self._unsup_dist_kwargs_of(unsup0.name),
            size=unsup0.cardinality, # Notice that this will be 1 for all
                                     # unsupervised dimensions, as they are
                                     # single neurons
        )
        unsup_encoders = ParametricCPD(
            variable=unsup,
            parents=[latent_var],
            parametrization=self._flexible_parametrization(
                variable=unsup,
                first=LinearEmbeddingToConcept(
                    in_embeddings=self.latent_size,
                    out_concepts=unsup.size,
                ),
                second=None,
            )
        )
        variables.append(unsup)
        factors.append(unsup_encoders)
        task_parents.append(unsup)

        task0 = axis.concept(self.task_names[0])
        tasks = ConceptVariable(
            names="tasks",
            members=self.task_names,
            distribution=self.distribution_of(task0.name),
            dist_kwargs=self.dist_kwargs_of(task0.name),
            size=task0.cardinality,
        )
        predictors = ParametricCPD(
            variable=tasks,
            parents=task_parents,
            parametrization=self._flexible_parametrization(
                variable=tasks,
                first=LinearConceptToConcept(
                    in_concepts=sum(p.size for p in task_parents),
                    out_concepts=tasks.size,
                ),
                second=None,
            ),
            aggregate=_merge_bottleneck_parents,
        )
        variables.append(tasks)
        factors.append(predictors)

        return BayesianNetwork(variables=variables, factors=factors)

    def _build_individual_model(self) -> BayesianNetwork:
        """Build with one variable per concept, one per unsupervised dimension,
        and one per task."""
        if not self.additional_dims:
            # Then this is simply the standard CBM!
            return super()._build_individual_model()

        # Otherwise we are in Hybrid CBM territory
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        supervised = [
            axis.concept(name)
            for name in self.supervised_concept_names
        ]
        unsup_axis = [axis.concept(name) for name in self.unsup_names]
        task_concepts = [axis.concept(name) for name in self.task_names]
        concepts = ConceptVariable(
            names=[c.name for c in supervised],
            distribution=[self.distribution_of(c.name) for c in supervised],
            dist_kwargs=[self.dist_kwargs_of(c.name) for c in supervised],
            size=[c.cardinality for c in supervised],
        )
        # The unsupervised dimensions are non-interpretable (embedding)
        # variables: they receive no supervision.
        unsups = EmbeddingVariable(
            names=self.unsup_names,
            distribution=[
                self._unsup_distribution_of(u.name)
                for u in unsup_axis
            ],
            dist_kwargs=[
                self._unsup_dist_kwargs_of(u.name)
                for u in unsup_axis
            ],
            # Notice that, as above, this will be 1 for all unsupervised
            # dimensions, as they are single neurons by construction.
            size=[u.cardinality for u in unsup_axis],
        )
        tasks = ConceptVariable(
            names=self.task_names,
            distribution=[self.distribution_of(t.name) for t in task_concepts],
            dist_kwargs=[self.dist_kwargs_of(t.name) for t in task_concepts],
            size=[t.cardinality for t in task_concepts],
        )

        # For clarity, separate the encoders for the supervised concepts and the
        # unsupervised parts of the bottleneck (notice this can be done in
        # theory using a single encoder, but we want to keep the two groups
        # separate for clarity).
        encoders = ParametricCPD(
            variable=concepts,
            parents=[latent_var],
            parametrization=[
                self._flexible_parametrization(
                    variable=concept,
                    first=LinearEmbeddingToConcept(
                        in_embeddings=self.latent_size,
                        out_concepts=concept.size,
                    ),
                    second=None,
                )
                for concept in concepts
            ],
        )
        unsup_encoders = ParametricCPD(
            variable=unsups,
            parents=[latent_var],
            parametrization=[
                self._flexible_parametrization(
                    variable=u,
                    first=LinearEmbeddingToConcept(
                        in_embeddings=self.latent_size,
                        out_concepts=u.size,
                    ),
                    second=None,
                )
                for u in unsups
            ],
        )
        bottleneck_size = (
            sum(c.size for c in concepts) + sum(u.size for u in unsups)
        )
        predictors = ParametricCPD(
            variable=tasks,
            parents=[*concepts, *unsups],
            parametrization=[
                self._flexible_parametrization(
                    variable=task,
                    first=LinearConceptToConcept(
                        in_concepts=bottleneck_size,
                        out_concepts=task.size,
                    ),
                    second=None,
                )
                for task in tasks
            ],
            aggregate=_merge_bottleneck_parents,
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *concepts, *unsups, *tasks],
            factors=[
                input_cpd,
                latent_cpd,
                *encoders,
                *unsup_encoders,
                *predictors,
            ],
        )
