"""Learn a graph as a PyTorch model component.

Call generator(concept_values, concept_names, concept_descriptions). Training
returns a fresh ConceptGraph with gradients, without refinements or DAG checking.
Eval caches a refined, validated graph without gradients until inputs, weights or
configuration change. Each result is an independent copy. DAGMA-CGM
ignores concept values.

Node count is a method option, while names and descriptions belong to the call::

    generator = GraphGeneratorLearnable("dagma_cgm", n_concepts=3)
    graph = generator(values, names, descriptions)  # Training graph.
    generator.eval()
    graph = generator(values, names, descriptions)  # Final graph; reused in eval.

Assign the generator to a model attribute to train its parameters with the model.
Use graph.data in model computations. After training, call generator in eval
mode to obtain the final graph.
Save any graph with graph.save(path); save state_dict() separately to resume training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import torch
from torch import nn

from torch_concepts.concept_graph import ConceptGraph
from .base import GraphGenerator, GraphGeneratorSpec


class GraphGeneratorLearnable(GraphGenerator, nn.Module):
    """Trainable PyTorch graph module with explicit final graph export.

    Assign the generator to a model attribute so parameters, buffers, device,
    dtype and mode follow that model. See the module example for direct
    computation during training and eval.

    Parameters
    ----------
    name : str
        Registered method name, for example dagma_cgm.
    source : str, optional
        Implementation family, inferred from a unique method registration.
    refinement : callable or list/tuple of callable, optional
        Graph-to-graph operations applied during eval.
    require_dag : bool, default True
        Validate acyclicity after all export refinements.
    initialization : callable, optional
        Applied once after source state is created. DAGMA passes fc1.weight
        to the callable. None uses the source default.
    **kwargs
        Source arguments, such as n_concepts, n_tasks and threshold for
        DAGMA_CGM. The training objective and optimizer are supplied externally.
    """

    trainable = True
    _source_loaders: dict[str, Callable] = {}

    def __init__(
        self,
        name: str,
        source: Optional[str] = None,
        refinement: Optional[Callable[[ConceptGraph], ConceptGraph] | Sequence[Callable[[ConceptGraph], ConceptGraph]]] = None,
        require_dag: bool = True,
        initialization: Optional[Callable] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            source=source,
            refinement=refinement,
            require_dag=require_dag,
            **kwargs,
        )
        initialization = initialization if initialization is not None else self._spec.initialization
        if initialization is not None and not callable(initialization):
            raise TypeError("`initialization` must be callable or None.")
        self._spec = replace(self._spec, initialization=initialization)
        if initialization is not None:
            if self._spec.weights is None:
                raise ValueError("The source must provide weights for initialization.")
            initialization(self._spec.weights)
        self._freeze_configuration()

    @property
    def initialization(self):
        """Initialization callable used at construction; reading does not rerun it."""
        return self._spec.initialization

    def forward(
        self, concept_values: Optional[torch.Tensor] = None,
        concept_names: Optional[Sequence[str]] = None,
        concept_descriptions: Optional[dict[str, str]] = None,
    ) -> ConceptGraph:
        """Return a training graph or a cached final graph in eval mode.

        concept_values has shape (samples, concepts). DAGMA-CGM ignores it and
        uses its learned weights. Training computes with gradients and skips
        refinements and DAG validation. Eval builds a refined, validated graph
        without gradients and returns independent copies while the cache key is unchanged.
        Omitting observations remains supported for sources that do not need them.
        Names follow column order and must match the number of nodes. Descriptions
        are shared with refinements; descriptions are supplied at call time.
        """
        if self.training:
            self._invalidate_cache()
            return self._construct_graph(concept_values, concept_names, concept_descriptions, finalize=False)
        names, descriptions = self._validate_inputs(concept_values, concept_names, concept_descriptions)
        self._prepare_context(concept_names=names, concept_descriptions=descriptions)
        key = self._build_cache_key(concept_values=concept_values)
        if self._trained_cached_graph is None or self._cache_key != key:
            with torch.no_grad():
                graph = self._construct_graph(concept_values, names, descriptions)
            self._trained_cached_graph = graph.clone()
            self._cache_key = key
        return self._trained_cached_graph.clone()

    def _tensor_state(self) -> tuple:
        """Track parameter/buffer identity, version, device and dtype for caching."""
        state = []
        for tensors in (self.parameters(), self.buffers()):
            for tensor in tensors:
                state.append((id(tensor), tensor._version, str(tensor.device), str(tensor.dtype)))
        return tuple(state)



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorLearnableSpec(GraphGeneratorSpec):
    """Forward callback, weight tensor and optional tensor initializer."""

    forward: Callable
    weights: Optional[torch.Tensor] = None
    initialization: Optional[Callable] = None
