"""Concept Embedding Model (CEM).

A bipartite model where each concept is represented by learned per-state
embeddings (Espinosa Zarlenga et al., NeurIPS 2022): the input is mapped to
per-concept embeddings, each concept is decoded from its embedding, and tasks
are predicted by *mixing* concept activations with their embeddings.

The model is assembled by a single builder, :meth:`_build_model`. Both the
concepts and their embeddings are grouped into the minimum number of plates by
the shared factories
:meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.build_concept_variables`
and
:meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.build_concept_embedding_variables`.
Because both use the same grouping, the two lists align element-by-element:
group ``i``'s embedding matrix produces group ``i``'s concept scores. A fully
homogeneous level collapses to a single plate (one batched embedding matrix); a
heterogeneous level splits into one plate per ``(type, cardinality)`` family.

References
----------
Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
Accuracy-Explainability Trade-Off", NeurIPS 2022. https://arxiv.org/abs/2209.09056
"""
from typing import List, Optional, Union

import torch
import torch.nn as nn

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....distributions import Delta
from ...low.dense_layers import LinearEmbeddingEncoder
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.mix import MixConceptEmbeddingToConcept
from ...low.priors import LearnablePrior
from ...low.sequential import Sequential
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.factors.cpd import ParametricCPD
from ...mid.variable import EmbeddingVariable
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ..base.bipartite import BipartiteModel


class ConceptEmbeddingModel(BipartiteModel):
    """Concept Embedding Model.

    Root concepts are decoded from per-concept embeddings (produced from the
    latent representation); tasks are predicted by mixing the parent concepts'
    activations with their embeddings. Works as a pure PyTorch module by default,
    or as a Lightning module when ``lightning=True``.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types).
    task_names : Union[List[str], str]
        Names of the task variables (a subset of the annotation labels).
    embedding_size : int, default 8
        Width of each per-state concept embedding.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration (see :class:`ConceptBottleneckModel`).
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
        embedding_size: int = 8,
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
        self.embedding_size = embedding_size
        # Split the concept annotations into intermediate-concept and task views
        # (both axes of the bipartite model live in self.concept_annotations).
        self.axis_concepts = self.concept_annotations.subset(self.intermediate_concept_names)
        self.axis_tasks = self.concept_annotations.subset(self.task_names)

        # One builder for both layouts (plate / individual, decided per group).
        self.pgm = self._build_model()

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

    # ------------------------------------------------------------------
    # Model assembly (written once for both layouts)
    # ------------------------------------------------------------------
    def _build_model(self) -> BayesianNetwork:
        """Assemble the CEM Bayesian network: ``input → latent → embeddings → concepts → tasks``.

        The concepts and their state embeddings are grouped identically (same
        minimum-plate layout), so ``embeddings[i]`` holds ``concepts[i]``'s state
        embeddings. Per group: ``latent`` produces the embedding matrix in one
        batched layer; each concept is decoded with one score per state embedding;
        tasks mix every concept's activation with its embedding. The mixer reads
        concept cardinalities/types positionally, so it is given the concept axis in
        the concatenation (group-member) order rather than annotation order.
        """
        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        # Concepts and their embeddings share the grouping, hence align 1:1.
        concepts = self.build_concept_variables(
            self.intermediate_concept_names, 
            plate_name="concepts"
        )
        embeddings = self.build_concept_embedding_variables(
            self.intermediate_concept_names, 
            self.embedding_size, 
            plate_name="embeddings"
        )
        tasks = self.build_concept_variables(
            self.task_names, 
            plate_name="tasks"
        )

        # latent → embeddings: one batched encoder per group, producing that
        # group's (n_states, embedding_size) matrix.
        emb_encoders = ParametricCPD(
            variable=embeddings,
            parents=[latent_var],
            parametrization=[
                {"value": LinearEmbeddingEncoder(
                    in_features=self.latent_size,
                    out_features=self.embedding_size,
                    n_embeddings=e.shape[0],
                )}
                for e in embeddings
            ],
        )
        # embeddings → concepts: decode one score per state embedding (per group).
        c_encoders = [
            ParametricCPD(
                variable=cvar,
                parents=[evar],
                parametrization=self._flexible_parametrization(
                    variable=cvar,
                    first=Sequential(
                        LinearEmbeddingToConcept(
                            in_embeddings=self.embedding_size,
                            out_concepts=1,
                        ),
                        # Collapse the (n_concepts, 1) score dims -> n_concepts
                        nn.Flatten(start_dim=-2),
                    ),
                    second=None,  # will be partial(...)
                ),
            )
            for cvar, evar in zip(concepts, embeddings)
        ]

        # concepts + embeddings → tasks: mix each concept activation with its
        # embedding. The mixer indexes concepts positionally, so its axis must
        # follow the concatenation (group-member) order, not the annotation order.
        ordered_names = [m for cvar in concepts for m in cvar.members]
        mix_axis = self.axis_concepts.subset(ordered_names)

        # by default, the concatenation of concepts and embeddings is done along the last axis, 
        # but the mixer expects the embeddings to be the second-last axis (the last axis is the concept states). 
        # So we define a custom aggregation function to mix them correctly.
        def mix_parents(concepts, embeddings):
            return {
                "concepts": torch.cat(list(concepts.values()), dim=-1),
                "embeddings": torch.cat(list(embeddings.values()), dim=-2),
            }

        predictors = ParametricCPD(
            variable=tasks,
            parents=[*concepts, *embeddings],
            parametrization=[
                self._flexible_parametrization(
                    variable=tvar,
                    first=MixConceptEmbeddingToConcept(
                        in_concepts=mix_axis,
                        in_embeddings=self.embedding_size,
                        out_concepts=tvar.size,
                    ),
                    second=None,  # will be partial(...)
                )
                for tvar in tasks
            ],
            aggregate=mix_parents,
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *embeddings, *concepts, *tasks],
            factors=[input_cpd, latent_cpd, *emb_encoders, *c_encoders, *predictors],
        )
