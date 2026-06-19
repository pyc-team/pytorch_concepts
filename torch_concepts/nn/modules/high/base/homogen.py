"""Homogeneous graph models: the graph → BayesianNetwork assembler.

``HomogenGraphModel`` is the abstract base where a generic concept *graph* is
assembled into a :class:`~torch_concepts.nn.BayesianNetwork`. It is *homogeneous*
in the sense that every node is parametrized the same way: a single ``encoder``
builds root concepts from the input and a single ``predictor`` builds internal
concepts from their parents. Graph models like ``C2BM`` extend this and implement
the :meth:`build_encoder` / :meth:`build_predictor` hooks (and, optionally,
embedding behaviour) — they do not re-implement the assembly or the inference
wiring.

This is the *heavy* assembler, kept only to simplify building arbitrary-DAG
models. The simple bipartite models (CBM, CEM) do **not** use it: they extend
:class:`~torch_concepts.nn.modules.high.base.bipartite.BipartiteModel` and build
their probabilistic model explicitly, so reading those classes shows exactly the
mid-level elements they assemble.

Embeddings
----------
Two class-level switches control concept embeddings (Concept Embedding Model
style):

* ``source_embeddings`` — root concepts are encoded from a per-concept embedding
  (produced from the input) rather than directly from the input.
* ``internal_embeddings`` — internal concepts get their *own* embedding that the
  predictor consumes (hypernetwork-style, used by the causal models).

Activations are derived from each variable's distribution (not from
annotations): Bernoulli-family → sigmoid, Categorical-family → softmax. A model
that wants different activations overrides :meth:`build_encoder` /
:meth:`build_predictor` to supply its own parametrization.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributions as dist
import torch.nn as nn

from .....annotations import AxisAnnotation
from .....distributions import Delta
from ...low.dense_layers import LinearEmbeddingEncoder
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.priors import LearnablePrior
from ...low.sequential import Sequential
from ..base.graph import DirectedGraphModel
from ...mid.models.variable import ConceptVariable, EmbeddingVariable
from ...mid.models.cpd import ParametricCPD
from ...mid.models.bayesian_network import BayesianNetwork


# Distribution families and how their natural parameter is produced. Annotations
# carry only the distribution; the activation is fixed here per family.
_SIGMOID_DISTS = {dist.Bernoulli, dist.RelaxedBernoulli}
_SOFTMAX_DISTS = {
    dist.Categorical,
    dist.OneHotCategorical,
    dist.RelaxedOneHotCategorical,
}

# Annotations may declare a *relaxed* (Concrete / Gumbel-Softmax) family to signal
# that the variable should be relaxed for gradient flow. The PGM ``Variable`` is
# declared with the corresponding *base* family: the torch inference backend
# relaxes it internally (with its own temperature) for sampling, and the
# deterministic path reads the base family's primary parameter. This mirrors the
# canonical mid-level examples, which declare ``Bernoulli`` / ``OneHotCategorical``.
_RELAXED_TO_BASE = {
    dist.RelaxedBernoulli: dist.Bernoulli,
    dist.RelaxedOneHotCategorical: dist.OneHotCategorical,
}


def _param_name(distribution) -> str:
    """Natural parameter key for ``distribution`` (the constructors emit this)."""
    if distribution in _SIGMOID_DISTS or distribution in _SOFTMAX_DISTS:
        return "probs"
    if distribution is Delta:
        return "value"
    if distribution is dist.Normal:
        return "loc"
    return "probs"


def _activation_for(distribution) -> nn.Module:
    """Activation module mapping a layer's output into ``distribution``'s param domain."""
    if distribution in _SIGMOID_DISTS:
        return nn.Sigmoid()
    if distribution in _SOFTMAX_DISTS:
        return nn.Softmax(dim=-1)
    return nn.Identity()


def _mix_aggregate(concepts, embeddings):
    """Aggregate parents for a concept+embedding predictor (CEM/hypernet style).

    Concept values are concatenated along the feature axis; per-concept
    embeddings are stacked along a concept axis (dim=1), yielding tensors shaped
    ``(batch, sum_cardinalities)`` and ``(batch, n_concepts, embedding_size)``.
    """
    return {
        "concepts": torch.cat(list(concepts.values()), dim=-1),
        "embeddings": torch.cat(list(embeddings.values()), dim=1),
    }


class HomogenGraphModel(DirectedGraphModel, ABC):
    """Abstract directed model that assembles a homogeneous Bayesian network.

    Subclasses implement :meth:`build_encoder` and :meth:`build_predictor` and,
    optionally, set the embedding switches / :attr:`embedding_size`. The
    :meth:`_assemble` template (called by the concrete model's ``__init__`` once
    config and ``latent_size`` are known) resolves the graph, builds the
    probabilistic model and wires the inference engines.
    """

    #: Root concepts are encoded from a per-concept embedding (not the raw input).
    source_embeddings: bool = False
    #: Internal concepts get their own embedding consumed by the predictor (hypernet).
    internal_embeddings: bool = False
    #: Per-concept embedding width (set by subclasses that use embeddings).
    embedding_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def build_encoder(self, out_concepts: int, in_embeddings: int) -> nn.Module:
        """Build the layer encoding a root concept from its input/embedding.

        ``in_embeddings`` is the latent size when ``source_embeddings`` is False,
        otherwise the per-concept ``embedding_size``.
        """

    @abstractmethod
    def build_predictor(
        self,
        out_concepts: int,
        in_concepts: AxisAnnotation,
        in_embeddings: Optional[int],
    ) -> nn.Module:
        """Build the layer predicting an internal concept from its parents.

        ``in_concepts`` is the annotation of the parent concepts (so layers that
        need cardinalities/types, e.g. the CEM mixer, can read them).
        ``in_embeddings`` is the embedding width when the predictor consumes
        embeddings, else ``None``.
        """

    def build_embedding_encoder(self, n_embeddings: int) -> nn.Module:
        """Build the input → embeddings layer (``n_embeddings`` of ``embedding_size``)."""
        return LinearEmbeddingEncoder(
            in_features=self.latent_size,
            out_features=self.embedding_size,
            n_embeddings=n_embeddings,
        )

    # ------------------------------------------------------------------
    # Assembly (lifecycle inherited from DirectedGraphModel._assemble)
    # ------------------------------------------------------------------
    def _build_probabilistic_model(self) -> BayesianNetwork:
        """Build the Bayesian network from the graph + the build hooks."""
        axis = self.concept_annotations

        input_var = EmbeddingVariable("input", distribution=Delta, size=self.latent_size)
        input_cpd = ParametricCPD(
            input_var,
            parametrization=LearnablePrior(self.latent_size),
            parents=[],
        )
        variables = [input_var]
        factors = [input_cpd]

        concept_vars = {}      # name -> ConceptVariable

        for name in self.graph_order:
            card = int(axis.cardinalities[axis.get_index(name)])
            distribution = axis.metadata[name]["distribution"]
            param = _param_name(distribution)
            activation = _activation_for(distribution)
            parents_names = list(self.graph.get_predecessors(name))
            is_root = len(parents_names) == 0

            # Declare the PGM variable with the base family; the relaxed-only
            # ``temperature`` kwarg (if any) is dropped since the inference backend
            # supplies its own temperature when relaxing for sampling.
            base_distribution = _RELAXED_TO_BASE.get(distribution, distribution)
            var_dist_kwargs = (
                None if base_distribution is not distribution
                else axis.metadata[name].get("dist_kwargs")
            )
            concept_var = ConceptVariable(
                name, distribution=base_distribution, size=card, dist_kwargs=var_dist_kwargs,
            )

            if is_root:
                if self.source_embeddings:
                    # One embedding per concept *state* (CEM-style), shaped
                    # (cardinality, embedding_size).
                    emb_var = EmbeddingVariable(
                        f"{name}__emb", distribution=Delta, shape=(card, self.embedding_size),
                    )
                    emb_cpd = ParametricCPD(
                        emb_var,
                        parametrization={"value": self.build_embedding_encoder(card)},
                        parents=[input_var],
                    )
                    variables.append(emb_var)
                    factors.append(emb_cpd)

                    # Encode one score per state embedding: (batch, card, emb) ->
                    # (batch, card, 1) -> (batch, card), then apply the activation
                    # over the cardinality axis.
                    encoder = self.build_encoder(out_concepts=1, in_embeddings=self.embedding_size)
                    param_module = Sequential(encoder, nn.Flatten(start_dim=1), activation)
                    cpd = ParametricCPD(
                        concept_var, parametrization={param: param_module}, parents=[emb_var],
                    )
                else:
                    encoder = self.build_encoder(out_concepts=card, in_embeddings=self.latent_size)
                    param_module = Sequential(encoder, activation)
                    cpd = ParametricCPD(
                        concept_var, parametrization={param: param_module}, parents=[input_var],
                    )
            else:
                parent_cvars = [concept_vars[p] for p in parents_names]
                # Build the parents' annotation explicitly with the first-class
                # ``types`` field populated from metadata: predictors such as the
                # CEM mixer read ``in_concepts.types`` / ``.cardinalities``, and
                # ``subset`` does not derive ``types`` from per-concept metadata.
                parents_ann = AxisAnnotation(
                    labels=list(parents_names),
                    cardinalities=[int(axis.cardinalities[axis.get_index(p)]) for p in parents_names],
                    types=[axis.metadata[p].get("type", "discrete") for p in parents_names],
                )

                if self.internal_embeddings:
                    emb_var = EmbeddingVariable(
                        f"{name}__emb", distribution=Delta, shape=(card, self.embedding_size),
                    )
                    emb_cpd = ParametricCPD(
                        emb_var,
                        parametrization={"value": self.build_embedding_encoder(card)},
                        parents=[input_var],
                    )
                    variables.append(emb_var)
                    factors.append(emb_cpd)

                    predictor = self.build_predictor(
                        out_concepts=card, in_concepts=parents_ann, in_embeddings=self.embedding_size,
                    )
                    param_module = Sequential(predictor, activation)
                    cpd = ParametricCPD(
                        concept_var,
                        parametrization={param: param_module},
                        parents=[*parent_cvars, emb_var],
                        aggregate=_mix_aggregate,
                    )
                else:
                    predictor = self.build_predictor(
                        out_concepts=card, in_concepts=parents_ann, in_embeddings=None,
                    )
                    param_module = Sequential(predictor, activation)
                    cpd = ParametricCPD(
                        concept_var, parametrization={param: param_module}, parents=parent_cvars,
                    )

            concept_vars[name] = concept_var
            variables.append(concept_var)
            factors.append(cpd)

        return BayesianNetwork(variables=variables, factors=factors)
