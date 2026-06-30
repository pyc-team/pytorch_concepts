"""Bipartite-graph mixin and abstract base for concept→task models.

A *bipartite* concept model splits its variables into two layers — concepts and
tasks — with edges only from concepts to tasks (no concept→concept edges). This
module provides:

* :class:`BipartiteMixin` — derives the bipartite :class:`ConceptGraph` from a
  list of ``task_names`` (the rest of the annotation labels become the concept
  layer). Shared by every bipartite model so the graph-building logic lives in
  one place.
* :class:`BipartiteModel` — abstract bipartite model (a ``DirectedGraphModel``
  using the mixin) and the parent of the concrete bipartite models (CBM, CEM).
  It builds the bipartite graph but leaves ``_build_probabilistic_model``
  abstract, so each concrete model assembles its own probabilistic model
  explicitly.
"""
from abc import ABC
from typing import List, Union

import pandas as pd
import torch

from .....concept_graph import ConceptGraph
from .....data.utils import ensure_list
from .graph import DirectedGraphModel


class BipartiteMixin:
    """Mixin that derives a bipartite concept→task graph from ``task_names``.

    Consumes a ``task_names`` keyword at construction, stores it, and overrides
    :meth:`_resolve_graph` to build the adjacency where every concept points to
    every task and tasks have no outgoing edges. Mix this in *before* a
    :class:`~torch_concepts.nn.modules.high.base.graph.GraphModel` subclass so
    its ``_resolve_graph`` takes precedence in the MRO.
    """

    def __init__(self, *args, task_names: Union[List[str], str], **kwargs):
        self.task_names: List[str] = ensure_list(task_names)
        super().__init__(*args, **kwargs)

    @property
    def intermediate_concept_names(self) -> List[str]:
        """Concept (non-task) labels, in annotation order."""
        return [c for c in self.concept_names if c not in self.task_names]

    def _resolve_graph(self) -> ConceptGraph:
        """Build the bipartite graph: every concept points to every task."""
        labels = list(self.concept_names)
        missing = [t for t in self.task_names if t not in labels]
        assert not missing, (
            f"All task_names must be annotation labels; {missing} are not in {labels}."
        )
        adjacency = pd.DataFrame(0, index=labels, columns=labels)
        adjacency.loc[:, self.task_names] = 1            # concepts -> tasks
        adjacency.loc[self.task_names, self.task_names] = 0  # tasks do not self-loop
        return ConceptGraph(
            torch.FloatTensor(adjacency.values),
            node_names=labels,
        )


class BipartiteModel(BipartiteMixin, DirectedGraphModel, ABC):
    """Abstract base for bipartite concept→task models (parent of CBM and CEM).

    Combines the bipartite graph assumption (:class:`BipartiteMixin`, which derives
    the concept→task graph from ``task_names``) with the directed-graph
    (Bayesian network) lifecycle (:class:`DirectedGraphModel`).

    It deliberately does **not** assemble the probabilistic model: concrete
    subclasses implement :meth:`_build_probabilistic_model` themselves so that the
    mid-level elements they construct (variables, CPDs, the Bayesian network) are
    visible directly in the model class. The shared :meth:`setup_inference` wiring is inherited.
    """
