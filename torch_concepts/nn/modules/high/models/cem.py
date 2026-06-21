"""Concept Embedding Model (CEM).

A bipartite model where each concept is represented by learned per-state
embeddings (Espinosa Zarlenga et al., NeurIPS 2022): the input is mapped to
per-concept embeddings, each concept is decoded from its embedding, and tasks
are predicted by *mixing* concept activations with their embeddings. Two building
paths are provided (mirroring :class:`ConceptBottleneckModel`):
:meth:`_build_plate_model` (``plate=True``, default when all graph levels are
homogeneous) groups each bipartite level into a single plate variable and encodes
all of a level's embeddings in one batched layer; :meth:`_build_individual_model`
(``plate=False``) creates one embedding/concept variable per concept. The
graph/inference lifecycle is inherited from
:class:`~torch_concepts.nn.modules.high.base.bipartite.BipartiteModel`.

References
----------
Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
Accuracy-Explainability Trade-Off", NeurIPS 2022. https://arxiv.org/abs/2209.09056
"""
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .....annotations import Annotations
from .....distributions import Delta
from ...low.dense_layers import LinearEmbeddingEncoder
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.mix import MixConceptEmbeddingToConcept
from ...low.priors import LearnablePrior
from ...low.sequential import Sequential
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.models.bayesian_network import BayesianNetwork
from ...mid.models.cpd import ParametricCPD
from ...mid.models.variable import ConceptVariable, EmbeddingVariable
from ..base.bipartite import BipartiteModel


class ConceptEmbeddingModel(BipartiteModel):
    """Concept Embedding Model.

    Root concepts are decoded from per-concept embeddings (produced from the
    latent representation); tasks are predicted by mixing the parent concepts'
    activations with their embeddings.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, distributions).
    task_names : Union[List[str], str]
        Names of the task variables (a subset of the annotation labels).
    embedding_size : int, default 16
        Width of each per-state concept embedding.
    plate : bool or None, default None
        Controls which building path is used.  ``None`` (default) auto-detects:
        uses plates only when **all** graph levels are plate-compatible (see
        :meth:`~torch_concepts.nn.modules.high.base.graph.DirectedGraphModel.plate_compatible_levels`),
        otherwise falls back to individual variables.  Pass ``True`` to force
        plates or ``False`` to force individual variables.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration (see :class:`ConceptBottleneckModel`).
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    **kwargs
        Forwarded to :class:`BaseModel`.
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    param_for_discrete_var = "logits"

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        embedding_size: int = 8,
        plate: Optional[bool] = None,
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
        self.embedding_size = embedding_size
        # Split the concept annotations into intermediate-concept and task views
        # (both axes of the bipartite model live in self.concept_annotations).
        self.axis_concepts = self.concept_annotations.subset(self.intermediate_concept_names)
        self.axis_tasks = self.concept_annotations.subset(self.task_names)

        if plate is None:
            plate = all(self.plate_compatible_levels(self.concept_annotations, self.graph))
        self.plate = plate
        if self.plate:
            self.pgm = self._build_plate_model()
        else:
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

    # ------------------------------------------------------------------
    # Building paths
    # ------------------------------------------------------------------
    def _build_plate_model(self) -> BayesianNetwork:
        """Optimised path for homogeneous levels: one plate variable per level.

        All intermediate concepts share a single plate concept variable and a
        single batched embedding variable (``n_concepts * card`` state embeddings
        produced in one layer); likewise all tasks share a single plate. Requires
        every level to be homogeneous (same type and cardinality) — enforced by
        the ``plate`` auto-detection.
        """

        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        n_concepts = len(self.intermediate_concept_names)
        n_tasks = len(self.task_names)
        concept0 = self.axis_concepts.concept(self.intermediate_concept_names[0])
        task0 = self.axis_tasks.concept(self.task_names[0])
        concept_card = concept0.cardinality
        task_card = task0.cardinality

        # All concepts' state embeddings in one batched variable: (n_concepts * card, emb).
        embedding = EmbeddingVariable(
            "embeddings",
            distribution=Delta,
            shape=(n_concepts * concept_card, self.embedding_size),
        )
        # Single plate concept variable; decode all members in one shot.
        concepts = ConceptVariable(
            names="concepts",
            members=self.intermediate_concept_names,
            distribution=concept0.distribution,
            size=concept_card,
        )
        tasks = ConceptVariable(
            names="tasks",
            members=self.task_names,
            distribution=task0.distribution,
            size=task_card,
        )
        
        emb_cpd = ParametricCPD(
            variable=embedding,
            parents=[latent_var],
            parametrization={
                "value": LinearEmbeddingEncoder(
                    in_features=self.latent_size, 
                    out_features=self.embedding_size,
                    n_embeddings=n_concepts * concept_card,
                )
            }
        )
        concept_cpd = ParametricCPD(
            variable=concepts, 
            parents=[embedding],
            parametrization=self._flexible_parametrization(
                variable=concepts,
                first=Sequential(
                    LinearEmbeddingToConcept(
                        in_embeddings=self.embedding_size, 
                        out_concepts=1
                    ),
                    nn.Flatten(start_dim=1),
                ),
                # flexible_parametrization will add a second CPD for variance, if needed
                # TODO: to be updated once a layer producing variance is implemented
                second=None # will be partial(...)
            )
        )
        task_cpd = ParametricCPD(
            variable=tasks, 
            parents=[concepts, embedding],
            parametrization=self._flexible_parametrization(
                variable=tasks,
                first=MixConceptEmbeddingToConcept(
                    in_concepts=self.axis_concepts,
                    in_embeddings=self.embedding_size,
                    out_concepts=n_tasks * task_card,
                ),
                # flexible_parametrization will add a second CPD for variance, if needed
                # TODO: to be updated once a layer producing variance is implemented
                second=None # will be partial(...)
            ),
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, embedding, concepts, tasks],
            factors=[input_cpd, latent_cpd, emb_cpd, concept_cpd, task_cpd],
        )

    def _build_individual_model(self) -> BayesianNetwork:
        """Assemble the CEM Bayesian network: input → embeddings → concepts → tasks.

        Each concept gets ``cardinality`` per-state embeddings (produced from the
        latent input); the concept is decoded with one score per state embedding,
        and tasks mix the parent concepts' activations with their embeddings.
        """

        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        intermediate = [self.axis_concepts.concept(name) for name in self.intermediate_concept_names]
        task_concepts = [self.axis_tasks.concept(name) for name in self.task_names]

        # One embedding variable per concept (its per-state embeddings, shape
        # (card, emb)), one concept variable, and one task variable each.
        embeddings = EmbeddingVariable(
            names=[f"emb_{c.name}" for c in intermediate],
            distribution=Delta,
            shape=[(c.cardinality, self.embedding_size) for c in intermediate]
        )
        concepts = ConceptVariable(
            names=self.intermediate_concept_names,
            distribution=[c.distribution for c in intermediate],
            size=[c.cardinality for c in intermediate]
        )
        tasks = ConceptVariable(
            names=self.task_names,
            distribution=[t.distribution for t in task_concepts],
            size=[t.cardinality for t in task_concepts]
        )

        # Aggregate the parents for the mixer: concept activations concatenated on
        # the feature axis, embeddings stacked on the concept axis.
        def mix_parents(concepts, embeddings):
            return {
                "concepts": torch.cat(list(concepts.values()), dim=-1),
                "embeddings": torch.cat(list(embeddings.values()), dim=1),
            }

        emb_encoders = ParametricCPD(
            variable=embeddings,
            parents=[latent_var],
            parametrization=[{
                "value": LinearEmbeddingEncoder(  # (batch, latent) -> (batch, card, emb_size)
                    in_features=self.latent_size,
                    out_features=self.embedding_size,
                    n_embeddings=c.cardinality,
                )
            } for c in intermediate],
        )
        # One CPD per concept: each concept is decoded from its *own* embedding
        # (batch, card, emb_size) -> (batch, card).
        c_encoders = [
            ParametricCPD(
                variable=concept,
                parents=[embedding],
                parametrization=self._flexible_parametrization(
                    variable=concept,
                    first=Sequential(
                        LinearEmbeddingToConcept(
                            in_embeddings=self.embedding_size,
                            out_concepts=1
                        ),
                        nn.Flatten(start_dim=1),
                    ),
                    # flexible_parametrization will add a second CPD for variance, if needed
                    # TODO: to be updated once a layer producing variance is implemented
                    second=None  # will be partial(...)
                ),
            )
            for concept, embedding in zip(concepts, embeddings)
        ]
        predictors = ParametricCPD(
            variable=tasks,
            parents=[*concepts, *embeddings],
            parametrization=[self._flexible_parametrization(
                variable=task,
                first=MixConceptEmbeddingToConcept(  # (batch, sum(card)) & (batch, sum(card), emb_size) -> (batch, card)
                    in_concepts=self.axis_concepts,
                    in_embeddings=self.embedding_size,
                    out_concepts=task.size,
                ),
                # flexible_parametrization will add a second CPD for variance, if needed
                # TODO: to be updated once a layer producing variance is implemented
                second=None  # will be partial(...)
            ) for task in tasks],
            aggregate=mix_parents,
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *embeddings, *concepts, *tasks],
            factors=[input_cpd, latent_cpd, *emb_encoders, *c_encoders, *predictors],
        )
