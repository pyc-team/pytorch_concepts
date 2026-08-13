"""Hybrid Concept Bottleneck Model (Hybrid CBM).

A Concept Bottleneck Model (CBM) whose bottleneck is extended with an
unsupervised set of neurons (not necessarily aligned with any known concepts).
This allows for a more flexible representation of the latent space, where the
bottleneck may more easily encode task-relevant information that is not already
in the training concept (with the obvious caveat that these extra unsupervised
dimensions are not necessarily interepretable and may be prone to leakage).

This architecture is a standard baseline in works exploring datasets with
potentially incomplete or noisy concept sets.

To the best of our knowledge, Mahinpei et al. (2021) first used them as part of
their experiments on exploring alignment of unsupervised dimensions. Then,
Espinosa Zarlenga et al. (2022) named it the *Hybrid CBM* and formalise the
bottleneck as ``c_hat in R^(k + gamma)``, the concatenation of ``k`` supervised
concept dimensions and ``gamma`` unsupervised ones.

References
----------
Mahinpei et al. "Promises and Pitfalls of Black-Box Concept Learning Models",
ICML 2021 Workshop. https://arxiv.org/abs/2106.13314

Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
Accuracy-Explainability Trade-Off", NeurIPS 2022. https://arxiv.org/abs/2209.09056
"""
from typing import Dict, List, Optional, Union

import torch

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....distributions import Delta

from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.lazy import LazyConstructor
from ...low.predictors.linear import LinearConceptToConcept
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.variable import EmbeddingVariable

from .cbm import ConceptBottleneckModel


def _merge_parents(nodes: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    This method concatenates the values of the set of nodes into a single
    tensor. Used to merge, for example, the supervised concepts of a task node
    and the unsupervised dimensions of the bottleneck into a single tensor to
    be used as the input of the task head.
    """
    all_vals = []
    for node in nodes:
        all_vals.extend(node.values())
    values = [
        # A binary unsupervised dimension propagates a Bernoulli sample, which
        # may arrive as a non-float tensor under the sampling engines.
        v.float() if not v.is_floating_point() else v
        for v in all_vals
    ]
    return {"concepts": torch.cat(values, dim=-1)}


class HybridConceptBottleneckModel(ConceptBottleneckModel):
    """
    A Hybrid Concept Bottleneck Model as described in Mahinpei et al. (2021) and
    Espinosa Zarlenga et al. (2022).

    This is a model that follows the following structure:
    ``input -> latent -> (concepts + unsupervised dimensions) -> tasks``, where,
    contrary to a standard CBM where the bottleneck contains only neurons
    aligned with known concepts, the bottleneck in this model (i.e., the layer
    between `latent` and the `tasks`) is extended with unsupervised latent
    dimensions. This unsupervised bottleneck representation can be
    thought of as a form of a shared embedding across concepts.

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
        Type of each additional latent dimension, either one per dimension or a
        single string used for all of them. Must be ``"binary"`` or
        ``"continuous"``. Notice that ``"categorical"`` is not supported, since
        each unsupervised dimension occupies a single bottleneck neuron.
        Defaults to ``"continuous"``; ignored when ``additional_dims`` is 0.
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
        or ``True`` groups homogeneous concepts and, separately, the
        unsupervised dimensions into the minimum number of plates; ``False``
        uses one variable per concept / dimension (less efficient).
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
        additional_dim_types: Optional[Union[List[str], str]]="continuous",
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs,
    ):
        # If these are negative, we will just ignore them and build a standard
        # CBM.
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
        if additional_dims:
            if len(additional_dim_types) != additional_dims:
                raise ValueError(
                    f"Length of additional_dim_types "
                    f"({len(additional_dim_types)}) must match additional_dims "
                    f"({additional_dims})."
                )
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

        # The unsupervised dimensions will be included in the annotations as
        # dummy cardinality-1 concepts. For this, we have to name them and,
        # therefore, we need to find a prefix which does not collide with
        # concepts the caller already has in the list of supervised concepts (
        # unlikely but one never knows).
        self.additional_dims = additional_dims
        self.unsup_names = []  # We will save these names to easily identify them
        self.unsup_plate_name = None
        if additional_dims > 0:
            used_names = set(annotations.labels)
            prefix = "__unsup_"
            while any(name.startswith(prefix) for name in used_names):
                prefix = "_" + prefix
            self.unsup_names = [f"{prefix}{i}" for i in range(additional_dims)]

            # Name used for the (single) unsupervised plate variable
            # (collision-free by construction).
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
        """
        Intermediate concept labels excluding the unsupervised dimensions.
        """
        unsup = set(self.unsup_names)
        return [n for n in self.intermediate_concept_names if n not in unsup]

    def _build_unsupervised_variables(self) -> List[EmbeddingVariable]:
        """
        Build the unsupervised bottleneck dimensions as embedding variable(s).

        Reuses the shared plate layout but emits non-interpretable
        :class:`EmbeddingVariable` nodes whose distribution comes from
        :attr:`unsupervised_distributions`. Homogeneous dimensions collapse to
        a single plate; mixing types (or ``plate=False``) splits them one per
        group / dimension (as in the standard CBM class).
        """
        result = []
        for kind, name, members in self._plate_layout(
            self.unsup_names,
            self.unsup_plate_name,
        ):
            # Note for self: variables are group by type, so we can just look at
            # the first member to determine the distribution.
            first = self.concept_annotations.concept(members[0])
            distribution = self.unsupervised_distributions[first.type]
            result.append(EmbeddingVariable(
                names=name,
                members=members if kind == "plate" else None,
                distribution=distribution,
                dist_kwargs=dict(
                    self.variable_dist_kwargs.get(distribution, {})
                ),
                size=first.cardinality,
            ))
        return result

    # ------------------------------------------------------------------
    # Model assembly (written once for both layouts)
    # ------------------------------------------------------------------
    def _build_model(self) -> BayesianNetwork:
        """
        The key method for building the graph for the Hybrid CBM. This method is
        called by the parent constructor and returns a :class:`BayesianNetwork`
        object that describes the structure of the model.
        The graph is built  to satisfy the following structure:
        ``input -> latent -> {concepts, unsupervised dims} -> tasks``.

        Each concept/dimension is encoded from the latent with a linear layer;
        every task consumes the *whole* bottleneck (supervised concepts +
        unsupervised dimensions) via a single linear head sized over the
        concatenation.
        """
        # With no additional dimensions the model is a plain CBM so let's just
        # reuise the parent's building method
        if not self.additional_dims:
            return super()._build_model()

        # Otherwise, let's first build the input and latent variables and CDPs
        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        # Let's get the supervised concepts and the unsupervised dimensions
        # variables.
        concepts = self.build_concept_variables(
            self.supervised_concept_names,
            plate_name="concepts",
        )
        unsup = self._build_unsupervised_variables()

        # And the downstream end tasks
        tasks = self.build_concept_variables(
            self.task_names,
            plate_name="tasks",
        )

        # We now build the latent to bottleneck maps (we use linear maps)
        # latent -> supervised concepts
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

        # latent -> unsupervised dimensions
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

        # Now time to build the label predictor: concepts + unsupervised -> tasks
        # For this, we will first aggregate both the supervised and unsupervised
        # nodes into a single set of parents for the task nodes.
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
            aggregate=lambda x, y: _merge_parents([x, y]),
        )

        # Aaaaand that's it
        return BayesianNetwork(
            variables=[input_var, latent_var, *concepts, *unsup, *tasks],
            factors=[
                input_cpd,
                latent_cpd,
                *encoders,
                *unsup_encoders,
                *predictors,
            ],
        )
