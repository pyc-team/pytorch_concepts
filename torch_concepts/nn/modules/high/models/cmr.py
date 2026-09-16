"""Concept-based Memory Reasoner (CMR)."""
from typing import List, Optional, Union

import torch
from torch import nn
from torch.distributions import Bernoulli, OneHotCategorical

from .....annotations import Annotations
from .....distributions import Delta
from ...low.dense_layers import MLP
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.rule import (
    ReconstructionRuleConceptEmbeddingToConcept,
    RuleConceptEmbeddingToConcept,
    RuleMemory,
)
from ...low.priors import LearnablePrior
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.variable import EmbeddingVariable
from ..base.bipartite import BipartiteModel


class ConceptMemoryReasoner(BipartiteModel):
    """A concept-based model with learned rules and instance-wise selection.

    The model keeps the standard bipartite concept-task structure but introduces
    three latent objects inside the PGM:

    - ``rule_selector``: per-task categorical rule weights predicted from the
      latent representation;
    - ``rule_roles``: decoded concept role probabilities from the learned
      memory, representing propositional logic rules;
    - ``tasks_with_rec``: an auxiliary reconstruction-aware task output.

    Args:
        input_size: Number of input features.
        annotations: Dataset annotations containing binary concepts and tasks.
        task_names: Name or names of the task labels.
        n_rules: Number of rules stored per task.
        memory_latent_size: Size of each learned task memory embedding.
        memory_decoder_hidden_layers: Number of hidden decoder layers in
            ``RuleMemory``.
        selector_hidden_layers: Number of hidden layers in the rule selector.
        rec_weight: Non-negative exponent applied to each rule reconstruction
            probability.
        hard_roles_at_eval: If true, use the argmax one-hot role assignment in
            evaluation mode.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024.
        https://arxiv.org/abs/2407.15527
    """

    supported_concept_types = frozenset({"binary"})
    param_for_discrete_var = "probs"
    variable_distributions = {"binary": Bernoulli}
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        n_rules: int = 10,
        memory_latent_size: int = 100,
        memory_decoder_hidden_layers: int = 1,
        selector_hidden_layers: int = 1,
        rec_weight: float = 0.1,
        hard_roles_at_eval: bool = True,
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
        if any(
            self.concept_annotations.concept(n).cardinality != 1
            for n in self.intermediate_concept_names
        ):
            raise ValueError(
                "ConceptMemoryReasoner requires binary scalar concepts."
            )
        if n_rules <= 0:
            raise ValueError("n_rules must be positive.")
        if memory_decoder_hidden_layers < 0:
            raise ValueError(
                "memory_decoder_hidden_layers must be non-negative."
            )
        if selector_hidden_layers < 0:
            raise ValueError("selector_hidden_layers must be non-negative.")
        if rec_weight < 0:
            raise ValueError("rec_weight must be non-negative.")

        self.n_rules = n_rules
        self.memory_latent_size = memory_latent_size
        self.memory_decoder_hidden_layers = memory_decoder_hidden_layers
        self.selector_hidden_layers = selector_hidden_layers
        self.rec_weight = rec_weight
        self.hard_roles_at_eval = hard_roles_at_eval
        self.pgm = self._build_model()
        self.setup_inference(
            inference,
            inference_kwargs,
            train_inference,
            train_inference_kwargs,
        )

    def default_query(self, ground_truth):
        """Train both CMR task paths in one inference query."""
        query = super().default_query(ground_truth)
        query["tasks_with_rec"] = None
        return query

    def _input_latent_block(self):
        """Build the standard raw-input to latent block used by CBM/CEM."""
        input_var = EmbeddingVariable(
            "input", distribution=Delta, shape=self.input_size
        )
        latent_var = EmbeddingVariable(
            "latent", distribution=Delta, size=self.latent_size
        )
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
        input_var, latent_var, input_cpd, latent_cpd = (
            self._input_latent_block()
        )

        concepts = self.build_concept_variables(
            self.intermediate_concept_names, "concepts"
        )
        concept_cpds = ParametricCPD(
            concepts,
            parents=[latent_var],
            parametrization=[
                {"probs": nn.Sequential(
                    LinearEmbeddingToConcept(self.latent_size, c.size),
                    nn.Sigmoid(),
                )}
                for c in concepts
            ],
        )
        n_concepts = sum(c.size for c in concepts)

        tasks = self.build_concept_variables(self.task_names, "tasks")
        if len(tasks) != 1:
            raise ValueError("CMR requires homogeneous binary task variables.")
        n_tasks = tasks[0].size

        selector = EmbeddingVariable(
            "rule_selector",
            distribution=OneHotCategorical,
            shape=(n_tasks, self.n_rules),
        )
        selector_network = (
            nn.Linear(self.latent_size, n_tasks * self.n_rules)
            if self.selector_hidden_layers == 0
            else MLP(
                input_size=self.latent_size,
                hidden_size=self.latent_size,
                output_size=n_tasks * self.n_rules,
                n_layers=self.selector_hidden_layers,
                activation="relu",
            )
        )
        selector_cpd = ParametricCPD(
            selector,
            parents=[latent_var],
            parametrization={
                "logits": nn.Sequential(
                    selector_network,
                    nn.Unflatten(-1, (n_tasks, self.n_rules)),
                )
            },
        )

        roles = EmbeddingVariable(
            "rule_roles",
            distribution=OneHotCategorical,
            shape=(n_tasks, self.n_rules, n_concepts, 3),
        )
        roles_cpd = ParametricCPD(
            roles,
            parents=[],
            parametrization={
                "probs": RuleMemory(
                    n_tasks,
                    self.n_rules,
                    n_concepts,
                    self.memory_latent_size,
                    self.memory_decoder_hidden_layers,
                    hard_at_eval=self.hard_roles_at_eval,
                )
            },
        )

        def aggregate_rule_inputs(concept_values, embedding_values):
            selector_value = embedding_values[selector].flatten(start_dim=-2)
            roles_value = embedding_values[roles].flatten(start_dim=-4)
            return {
                "concepts": torch.cat(list(concept_values.values()), dim=-1),
                "embeddings": torch.cat(
                    [selector_value, roles_value], dim=-1
                ),
            }

        rule_embedding_size = selector.size + roles.size
        task_cpd = ParametricCPD(
            tasks[0],
            parents=[*concepts, selector, roles],
            parametrization={
                "probs": RuleConceptEmbeddingToConcept(
                    out_concepts=n_tasks,
                    in_concepts=n_concepts,
                    in_embeddings=rule_embedding_size,
                    n_rules=self.n_rules,
                )
            },
            aggregate=aggregate_rule_inputs,
        )

        rec_tasks = EmbeddingVariable(
            "tasks_with_rec",
            distribution=Delta,
            shape=tasks[0].shape,
        )
        rec_cpd = ParametricCPD(
            rec_tasks,
            parents=[*concepts, selector, roles],
            parametrization={
                "value": ReconstructionRuleConceptEmbeddingToConcept(
                    out_concepts=n_tasks,
                    in_concepts=n_concepts,
                    in_embeddings=rule_embedding_size,
                    n_rules=self.n_rules,
                    rec_weight=self.rec_weight,
                )
            },
            aggregate=aggregate_rule_inputs,
        )

        return BayesianNetwork(
            variables=[
                input_var,
                latent_var,
                *concepts,
                selector,
                roles,
                *tasks,
                rec_tasks,
            ],
            factors=[
                input_cpd,
                latent_cpd,
                *concept_cpds,
                selector_cpd,
                roles_cpd,
                task_cpd,
                rec_cpd,
            ],
        )
