"""Concept Bottleneck Model (CBM).

A bipartite concept model: a linear encoder maps the latent representation to
concepts, and a linear predictor maps concepts to tasks. Two building paths are
provided: :meth:`_build_plate_model` (``plate=True``, default) groups each
bipartite level into a single plate variable; :meth:`_build_individual_model`
(``plate=False``) creates one variable per concept/task.
"""
from typing import List, Optional, Union

import torch

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....distributions import Delta
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...low.priors import LearnablePrior
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.models.bayesian_network import BayesianNetwork
from ...mid.models.cpd import ParametricCPD
from ...mid.models.variable import ConceptVariable, EmbeddingVariable, _DEFAULT_DIST_KWARGS
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
        Controls which building path is used.  ``None`` (default) auto-detects:
        uses plates only when **all** graph levels are plate-compatible (see
        :meth:`~torch_concepts.nn.modules.high.base.graph.DirectedGraphModel.plate_compatible_levels`),
        otherwise falls back to individual variables.  Pass ``True`` to force
        plates or ``False`` to force individual variables.
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
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            lightning=lightning,
            **kwargs,
        )
        if all(self.plate):
            # if all graph levels are plate-compatible
            # build the model with one plate variable per bipartite level (concepts, tasks)
            self.pgm = self._build_plate_model()
        else:
            # build the model with one variable per concept and one per task
            self.pgm = self._build_individual_model()

        # once self.pgm is built, we can set up the inference engines (train and eval)
        self.setup_inference(
            inference,
            inference_kwargs,
            train_inference,
            train_inference_kwargs,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _input_latent_block(self):
        """Raw input → latent block shared by both building paths.

        Returns ``(input_var, latent_var, [input_cpd, latent_cpd])``: the raw
        ``input`` enters the PGM as evidence and the backbone runs *inside* the
        PGM as the ``latent | input`` CPD.
        """
        input_var = EmbeddingVariable("input", distribution=Delta, size=self.input_size)
        latent_var = EmbeddingVariable("latent", distribution=Delta, size=self.latent_size)
        input_cpd = ParametricCPD(
            input_var, 
            parents=[],
            parametrization=LearnablePrior(self.input_size),
        )
        latent_cpd = ParametricCPD(
            latent_var, 
            parents=[input_var],
            parametrization=self.backbone,
        )
        return input_var, latent_var, input_cpd, latent_cpd
    
    def _build_plate_model(self) -> BayesianNetwork:
        """Build using one plate variable per bipartite level (concepts, tasks)."""
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        concept0 = axis.concept(self.intermediate_concept_names[0])
        task0 = axis.concept(self.task_names[0])
        concepts = ConceptVariable(
            names="concepts",
            members=self.intermediate_concept_names,
            distribution=self.distribution_of(concept0.name),
            dist_kwargs=self.dist_kwargs_of(concept0.name),
            size=concept0.cardinality,
        )
        tasks = ConceptVariable(
            names="tasks",
            members=self.task_names,
            distribution=self.distribution_of(task0.name),
            dist_kwargs=self.dist_kwargs_of(task0.name),
            size=task0.cardinality,
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
                second=None # will be partial(...)
            )
        )
        predictors = ParametricCPD(
            variable=tasks,
            parents=[concepts],
            parametrization=self._flexible_parametrization(
                variable=tasks,
                first=LinearConceptToConcept(
                    in_concepts=concepts.size,
                    out_concepts=tasks.size,
                ),
                second=None # will be partial(...)
            )
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, concepts, tasks],
            factors=[input_cpd, latent_cpd, encoders, predictors],
        )

    def _build_individual_model(self) -> BayesianNetwork:
        """Build with one variable per concept and one per task."""
        axis = self.concept_annotations
        
        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        intermediate = [axis.concept(name) for name in self.intermediate_concept_names]
        task_concepts = [axis.concept(name) for name in self.task_names]
        concepts = ConceptVariable(
            names=self.intermediate_concept_names,
            distribution=[self.distribution_of(c.name) for c in intermediate],
            dist_kwargs=[self.dist_kwargs_of(c.name) for c in intermediate],
            size=[c.cardinality for c in intermediate],
        )
        tasks = ConceptVariable(
            names=self.task_names,
            distribution=[self.distribution_of(t.name) for t in task_concepts],
            dist_kwargs=[self.dist_kwargs_of(t.name) for t in task_concepts],
            size=[t.cardinality for t in task_concepts],
        )

        encoders = ParametricCPD(
            variable=concepts,
            parents=[latent_var],
            parametrization=[self._flexible_parametrization(
                variable=concept,
                first=LinearEmbeddingToConcept(
                    in_embeddings=self.latent_size,
                    out_concepts=concept.size,
                ),
                second=None  # will be partial(...)
            ) for concept in concepts],
        )
        predictors = ParametricCPD(
            variable=tasks,
            parents=[*concepts],
            parametrization=[self._flexible_parametrization(
                variable=task,
                first=LinearConceptToConcept(
                    in_concepts=sum(c.size for c in concepts),
                    out_concepts=task.size,
                ),
                second=None  # will be partial(...)
            ) for task in tasks],
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *concepts, *tasks],
            factors=[input_cpd, latent_cpd, *encoders, *predictors],
        )
