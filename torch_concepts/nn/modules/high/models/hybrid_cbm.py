"""Hybrid Concept Bottleneck Model (Hybrid CBM).

A Concept Bottleneck Model (CBM) whose bottleneck is extended with an
unsupervised set of neurons (which will not be aligned with any known concepts).
This allows for a more flexible representation of the latent space, where some
dimensions can be dedicated to concept prediction and others to task prediction.

This architecture is a standard baseline, particularly for evaluating the
performance of concept bottleneck models when the concept set is incomplete or
noisy (Mahinpei et al., "Promises and Pitfalls of Black-Box Concept Learning
Models", 2021; the ``hybrid``/``joint`` bottleneck of Koh et al., ICML 2020).
"""
from typing import Dict, List, Optional, Union

import torch

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....distributions import Delta
from ...low.lazy import LazyConstructor
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.factors.cpd import ParametricCPD
from ...mid.variable import EmbeddingVariable
from ...mid.distributions import DEFAULT_DIST_KWARGS
from .cbm import ConceptBottleneckModel


def _merge_bottleneck_parents(
    concepts: Dict,
    embeddings: Dict,
) -> Dict[str, torch.Tensor]:
    """Aggregate for the task CPDs: fuse the whole bottleneck into one input.

    The task predictors take both the supervised concepts (``ConceptVariable``
    parents) and the unsupervised dimensions (``EmbeddingVariable`` parents);
    this concatenates them (concepts first, then unsupervised dimensions,
    each group in parent order) into the single ``concepts`` input of a
    :class:`~torch_concepts.nn.LinearConceptToConcept`.
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
    *non-interpretable* :class:`~torch_concepts.nn.EmbeddingVariable` nodes (of
    dimension 1), so they are never supervised or included in concept
    losses/metrics. Continuous dimensions are modelled as deterministic
    (``Delta``) neurons; binary dimensions as ``Bernoulli`` logits (see
    :attr:`unsupervised_distributions`).

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
        is set to 0, the model reduces to a standard ``ConceptBottleneckModel``.
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
        Per-level plate preference (forwarded to :class:`BaseModel`). ``None``
        (default) / ``True`` group homogeneous concepts (and, separately, the
        homogeneous unsupervised dimensions) into the minimum number of plates;
        ``False`` uses one individual variable per concept / dimension.
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
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

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
        # variables in the bottleneck. To avoid name conflicts, we first figure
        # out how many underscores to prepend to "unsup_".
        self.additional_dims = additional_dims
        self.unsup_names = []
        self.unsup_plate_name = None
        if additional_dims > 0:
            used_names = set(annotations.labels)
            prefix = "__unsup_"
            while any(name.startswith(prefix) for name in used_names):
                prefix = "_" + prefix
            self.unsup_names = [f"{prefix}{i}" for i in range(additional_dims)]
            # Name used for the (single) unsupervised plate variable (collision-free
            # by construction of ``prefix``).
            self.unsup_plate_name = f"{prefix}plate"
            used_annotations = Annotations(
                labels=(list(annotations.labels) + self.unsup_names),
                states=(list(annotations.states) + [['0']] * additional_dims),
                cardinalities=(
                    list(annotations.cardinalities) + [1] * additional_dims
                ),
                types=(list(annotations.types) + additional_dim_types),
                concept_space=annotations.concept_space,
            )
        else:
            # Otherwise, this is just a standard CBM.
            used_annotations = annotations

        # The parent constructor dispatches to the (overridden) ``_build_model``
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
        """Distribution class used for unsupervised dimension ``name`` (by type)."""
        return self.unsupervised_distributions[
            self.concept_annotations.concept(name).type
        ]

    def _unsup_dist_kwargs_of(self, name: str) -> dict:
        """Distribution keyword arguments for unsupervised dimension ``name``."""
        return dict(
            self.variable_dist_kwargs.get(self._unsup_distribution_of(name), {})
        )

    def _build_unsupervised_variables(self) -> List[EmbeddingVariable]:
        """Build the unsupervised bottleneck dimensions as embedding variable(s).

        Reuses the shared plate layout (:meth:`_plate_layout`) — so the ``plate``
        preference is honoured exactly as for the supervised concepts — but emits
        non-interpretable :class:`EmbeddingVariable` nodes whose distribution comes
        from :attr:`unsupervised_distributions` (continuous → ``Delta``, binary →
        ``Bernoulli``). Homogeneous dimensions collapse to a single plate; mixing
        types (or ``plate=False``) splits them into one variable per group /
        dimension.
        """
        out: List[EmbeddingVariable] = []
        for kind, name, members in self._plate_layout(
            self.unsup_names, self.unsup_plate_name
        ):
            u0 = self.concept_annotations.concept(members[0])
            dist = self._unsup_distribution_of(u0.name)
            dkw = self._unsup_dist_kwargs_of(u0.name)
            if kind == "plate":
                out.append(EmbeddingVariable(
                    names=name, members=members,
                    distribution=dist, dist_kwargs=dkw, size=u0.cardinality,
                ))
            else:
                out.append(EmbeddingVariable(
                    names=name,
                    distribution=dist, dist_kwargs=dkw, size=u0.cardinality,
                ))
        return out

    # ------------------------------------------------------------------
    # Model assembly (written once for both layouts)
    # ------------------------------------------------------------------
    def _build_model(self) -> BayesianNetwork:
        """Assemble ``input → latent → {concepts, unsupervised dims} → tasks``.

        The supervised concepts and the tasks are grouped into the minimum
        number of plates by :meth:`build_concept_variables`; the unsupervised
        dimensions are grouped the same way but as embedding variables (see
        :meth:`_build_unsupervised_variables`). Each concept/dimension is
        encoded from the latent with a linear layer; every task consumes the
        *whole* bottleneck (supervised concepts + unsupervised dimensions) via a
        single linear head sized over the concatenation.
        """
        # With no additional dimensions the model is a plain CBM.
        if not self.additional_dims:
            return super()._build_model()

        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        concepts = self.build_concept_variables(
            self.supervised_concept_names, plate_name="concepts"
        )
        unsup = self._build_unsupervised_variables()
        tasks = self.build_concept_variables(
            self.task_names, plate_name="tasks"
        )

        # latent → supervised concepts: one linear encoder per concept group.
        encoders = ParametricCPD(
            variable=concepts,
            parents=[latent_var],
            parametrization=[
                self._flexible_parametrization(
                    variable=c,
                    first=LazyConstructor(LinearEmbeddingToConcept),
                    second=LazyConstructor(LinearEmbeddingToConcept),
                )
                for c in concepts
            ],
        )
        # latent → unsupervised dimensions: kept as a separate encoder group for
        # clarity (they are never supervised).
        unsup_encoders = ParametricCPD(
            variable=unsup,
            parents=[latent_var],
            parametrization=[
                self._flexible_parametrization(
                    variable=u,
                    first=LazyConstructor(LinearEmbeddingToConcept),
                    second='auto',
                )
                for u in unsup
            ],
        )
        # concepts + unsupervised dimensions → tasks. The unsupervised dimensions
        # are embedding parents, so a lazily-sized head would miss them; size the
        # head explicitly over the whole bottleneck and fuse the parents with
        # ``_merge_bottleneck_parents``.
        bottleneck_size = (
            sum(c.size for c in concepts) + sum(u.size for u in unsup)
        )
        predictors = ParametricCPD(
            variable=tasks,
            parents=[*concepts, *unsup],
            parametrization=[
                self._flexible_parametrization(
                    variable=t,
                    first=LinearConceptToConcept(
                        in_concepts=bottleneck_size,
                        out_concepts=t.size,
                    ),
                    second='auto',
                )
                for t in tasks
            ],
            aggregate=_merge_bottleneck_parents,
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *concepts, *unsup, *tasks],
            factors=[
                input_cpd, latent_cpd, *encoders, *unsup_encoders, *predictors,
            ],
        )
