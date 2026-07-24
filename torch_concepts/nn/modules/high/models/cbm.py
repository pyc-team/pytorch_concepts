"""Concept Bottleneck Model (CBM).

A bipartite model where the input is mapped to a set of concepts and the tasks
are predicted from those concepts (Koh et al., ICML 2020): a linear encoder maps
the latent representation to the concepts, and a linear predictor maps the
concepts to the tasks (no concept embeddings).

The model is assembled by a single builder, :meth:`_build_model`. Each bipartite
level (concepts, tasks) is built by the shared factory
:meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.build_concept_variables`,
which groups homogeneous concepts into the minimum number of plates: a fully
homogeneous level collapses to a single plate, a heterogeneous level splits into
one plate per ``(type, cardinality)`` family, and even a lone concept is a
single-member plate. Because each level is always a list of variables, the CPD
wiring and the :class:`BayesianNetwork` assembly are written once and work for
every layout.

References
----------
Koh et al. "Concept Bottleneck Models", ICML 2020. https://proceedings.mlr.press/v119/koh20a.html
"""
from typing import List, Optional, Union

import torch

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....distributions import Delta
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.lazy import LazyConstructor
from ...low.predictors.linear import LinearConceptToConcept
from ...low.priors import LearnablePrior
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.factors.cpd import ParametricCPD
from ...mid.variable import EmbeddingVariable
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ..base.bipartite import BipartiteModel


class ConceptBottleneckModel(BipartiteModel):
    """Concept Bottleneck Model.

    Linear ``latent → concepts → tasks`` bottleneck with no concept embeddings.
    Works as a pure PyTorch module by default, or as a Lightning module when
    ``lightning=True``.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types).
    task_names : Union[List[str], str]
        Names of the task variables (a subset of the annotation labels).
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
        Per-level plate preference (forwarded to :class:`BaseModel` and consumed by
        the level factories). ``None`` (default) and ``True`` group homogeneous
        concepts into the minimum number of plates — even a lone concept becomes a
        single-member plate. ``False`` uses one individual variable per concept.
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

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        plate: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            lightning=lightning,
            plate=plate,
            **kwargs,
        )
        # One builder for both layouts (plate / individual, decided per level).
        self.pgm = self._build_model()

        # once self.pgm is built, we can set up the inference engines (train and eval)
        self.setup_inference(
            inference,
            inference_kwargs,
            train_inference,
            train_inference_kwargs,
        )

    def _input_latent_block(self):
        """Raw input → latent block.

        Returns ``(input_var, latent_var, input_cpd, latent_cpd)``: the raw
        ``input`` enters the PGM as evidence and the backbone runs *inside* the
        PGM as the ``latent | input`` CPD.
        """
        input_var = EmbeddingVariable("input", distribution=Delta, shape=self.input_size)
        latent_var = EmbeddingVariable("latent", distribution=Delta, size=self.latent_size)
        input_cpd = ParametricCPD(
            input_var,
            parents=[],
            parametrization=LearnablePrior(input_var.shape),
        )
        latent_cpd = ParametricCPD(
            latent_var,
            parents=[input_var],
            parametrization=self.backbone,
        )
        return input_var, latent_var, input_cpd, latent_cpd
    
    def _build_model(self) -> BayesianNetwork:
        """Assemble the CBM Bayesian network: ``input → latent → concepts → tasks``.

        The backbone runs as the ``latent | input`` CPD; a linear encoder maps
        ``latent`` to the concepts and a linear predictor maps the concepts to the
        tasks. Each concept-side level is materialised by
        :meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.build_concept_variables`
        as a list of concept variables — homogeneous concepts share a plate, the
        rest are individual variables. ``ParametricCPD`` broadcasts over each list
        to build one CPD per variable, with the parametrization built per variable
        so every distribution family gets its correct parameter (e.g. ``logits``);
        ``LazyConstructor`` then sizes each layer from the variable's ``param_sizes``
        and its parents' sizes.
        """
        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        concepts = self.build_concept_variables(self.intermediate_concept_names, plate_name="concepts")
        tasks = self.build_concept_variables(self.task_names, plate_name="tasks")

        # latent → concepts: one encoder per concept variable (per group).
        encoders = ParametricCPD(
            variable=concepts,
            parents=[latent_var],
            parametrization=[
                self._flexible_parametrization(
                    variable=c,
                    first=LazyConstructor(LinearEmbeddingToConcept), # parameterization for the first parameter
                    second=LazyConstructor(LinearEmbeddingToConcept), # parameterization for the second parameter
                    # nn.Softplus() or ScaleTrilActivation will be 
                    # attached automatically to the second head for continuous variables.
                )
                for c in concepts
            ],
        )
        # concepts → tasks: every concept group is a parent of every task.
        predictors = ParametricCPD(
            variable=tasks,
            parents=concepts,
            parametrization=[
                self._flexible_parametrization(
                    variable=t,
                    first=LazyConstructor(LinearConceptToConcept),
                    second=LazyConstructor(LinearConceptToConcept),
                )
                for t in tasks
            ],
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *concepts, *tasks],
            factors=[input_cpd, latent_cpd, *encoders, *predictors],
        )
