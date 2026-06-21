"""Homogeneous graph models: the graph → BayesianNetwork assembler.

``HomogenGraphModel`` is the abstract base where a generic concept *graph* is
assembled into a :class:`~torch_concepts.nn.BayesianNetwork`. It is *homogeneous*
in the sense that every node is parametrized the same way: a single ``encoder``
builds root concepts and a single ``predictor`` builds internal concepts from
their parents. Concrete graph models (e.g. ``C2BM``, a graph CBM) extend this and
implement only :meth:`build_encoder` / :meth:`build_predictor` (and, optionally,
the embedding switches) — they do not re-implement the assembly or inference
wiring.

This is the *heavy* assembler, kept only to simplify building arbitrary-DAG
models. The bipartite models (CBM, CEM) do **not** use it: they extend
:class:`~torch_concepts.nn.modules.high.base.bipartite.BipartiteModel` and build
their probabilistic model explicitly.

Embeddings
----------
Two class-level switches control concept embeddings (Concept Embedding Model
style):

* ``source_embeddings`` — root concepts are decoded from a per-concept embedding
  (produced from the latent) rather than directly from the latent.
* ``internal_embeddings`` — internal concepts get their *own* embedding that the
  predictor consumes (hypernetwork-style, used by the causal models).

Each concept variable is declared with its annotation distribution; the encoders
/ predictors emit raw scores stored as ``logits`` (see ``param_for_discrete_var``)
and the distribution activates them downstream — exactly as in CBM/CEM.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .....annotations import AxisAnnotation
from .....distributions import Delta
from ...low.dense_layers import LinearEmbeddingEncoder
from ...low.priors import LearnablePrior
from ...low.sequential import Sequential
from ..base.graph import DirectedGraphModel
from ...mid.models.variable import ConceptVariable, EmbeddingVariable
from ...mid.models.cpd import ParametricCPD
from ...mid.models.bayesian_network import BayesianNetwork


class HomogenGraphModel(DirectedGraphModel, ABC):
    """Abstract directed model that assembles a homogeneous Bayesian network.

    Subclasses implement :meth:`build_encoder` and :meth:`build_predictor` and,
    optionally, set the embedding switches / :attr:`embedding_size`. The concrete
    model's ``__init__`` then mirrors CBM/CEM: build ``self.pgm`` via
    :meth:`_build_individual_model` and call ``setup_inference``.
    """

    #: Root concepts are decoded from a per-concept embedding (not the raw latent).
    source_embeddings: bool = False
    #: Internal concepts get their own embedding consumed by the predictor.
    internal_embeddings: bool = False
    #: Per-concept embedding width (set by subclasses that use embeddings).
    embedding_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def build_encoder(
        self, 
        in_embeddings: int, 
        out_concepts: int
    ) -> nn.Module:
        """Build the layer encoding a root concept from its latent/embedding.

        ``in_embeddings`` is the latent size when ``source_embeddings`` is False,
        otherwise the per-concept ``embedding_size`` (with ``out_concepts=1``: one
        score per state embedding).
        """

    @abstractmethod
    def build_predictor(
        self,
        in_concepts: AxisAnnotation,
        in_embeddings: Optional[int],
        out_concepts: Optional[int],
    ) -> nn.Module:
        """Build the layer predicting an internal concept from its parents.

        ``in_concepts`` is the annotation of the parent concepts (so layers that
        need cardinalities/types can read them). ``in_embeddings`` is the embedding
        width when the predictor consumes embeddings, else ``None``.
        """

    def build_embedding_encoder(self, n_embeddings: int) -> nn.Module:
        """Build the latent → embeddings layer (``n_embeddings`` of ``embedding_size``)."""
        return LinearEmbeddingEncoder(
            in_features=self.latent_size,
            out_features=self.embedding_size,
            n_embeddings=n_embeddings,
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
    # Building path
    # ------------------------------------------------------------------
    def _build_individual_model(self) -> BayesianNetwork:
        """Build one concept variable per node, wired along the DAG via the hooks.

        Root concepts are encoded from the latent (or a per-concept embedding when
        ``source_embeddings``); internal concepts are predicted from their parents
        (mixed with an embedding when ``internal_embeddings``). The two layer
        choices are deferred to :meth:`build_encoder` / :meth:`build_predictor`.
        """
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()
        variables = [input_var, latent_var]
        factors = [input_cpd, latent_cpd]

        # Aggregate parents for an embedding-conditioned predictor: parent
        # activations concatenated on the feature axis, embedding stacked on dim 1.
        def mix_parents(concepts, embeddings):
            return {
                "concepts": torch.cat(list(concepts.values()), dim=-1),
                "embeddings": torch.cat(list(embeddings.values()), dim=1),
            }

        concept_vars = {}  # name -> ConceptVariable (topological order: parents exist first)
        for name in self.graph_order:
            concept = axis.concept(name)
            parents = list(self.graph.get_predecessors(name))
            is_root = not parents
            concept_var = ConceptVariable(
                names=name, 
                distribution=concept.distribution, 
                size=concept.cardinality
            )

            # Optional per-concept embedding (produced from the latent).
            embedding = None
            if self.source_embeddings if is_root else self.internal_embeddings:
                embedding = EmbeddingVariable(
                    f"{name}__embedding", 
                    distribution=Delta, 
                    shape=(concept.cardinality, self.embedding_size),
                )
                factors.append(ParametricCPD(
                    variable=embedding,
                    parents=[latent_var],
                    parametrization={"value": self.build_embedding_encoder(concept.cardinality)},
                ))
                variables.append(embedding)

            if is_root and embedding is not None:
                # Decode one score per state embedding -> (batch, card).
                concept_cpd = ParametricCPD(
                    variable=concept_var,
                    parents=[embedding],
                    parametrization=self._flexible_parametrization(
                        variable=concept_var, 
                        first=Sequential(
                            self.build_encoder(
                                in_embeddings=self.embedding_size,
                                out_concepts=1
                            ),
                            nn.Flatten(start_dim=1),
                        ), 
                        second=None
                    ),
                )
            elif is_root:
                concept_cpd = ParametricCPD(
                    variable=concept_var,
                    parents=[latent_var],
                    parametrization=self._flexible_parametrization(
                        variable=concept_var, 
                        first=self.build_encoder(
                            in_embeddings=self.latent_size,
                            out_concepts=concept.cardinality
                        ), 
                        second=None
                    ),
                )
            else:
                parent_vars = [concept_vars[p] for p in parents]
                in_concepts = axis.subset(parents)
                concept_cpd = ParametricCPD(
                    variable=concept_var,
                    parents=[*parent_vars, embedding] if embedding is not None else parent_vars,
                    parametrization=self._flexible_parametrization(
                        variable=concept_var, 
                        first=self.build_predictor(
                            in_concepts=in_concepts,
                            in_embeddings=self.embedding_size if embedding is not None else None,
                            out_concepts=concept.cardinality
                        ), 
                        second=None
                    ),
                    aggregate=mix_parents if embedding is not None else None,
                )

            concept_vars[name] = concept_var
            variables += [concept_var]
            factors += [concept_cpd]

        return BayesianNetwork(variables=variables, factors=factors)
