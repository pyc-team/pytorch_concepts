"""Black-box baseline: ``input → latent → outputs``, with no concept bottleneck.

The same Bayesian-network machinery as the concept models, minus the concept
layer: every output is read straight off the latent. ``forward``, the inference
engines, teacher forcing and the output contract (``logits`` / ``loc`` +
``scale``) are therefore inherited unchanged, so a black box is a drop-in
replacement for any other model.
"""
from typing import List, Optional, Union

import torch

from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from .....utils import ensure_list
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.lazy import LazyConstructor
from ...metrics import ConceptMetrics
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ..base.graph import DirectedGraphModel


class BlackBoxModel(DirectedGraphModel):
    """Black-box model: a backbone and one linear head per output, no concepts in between.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types).
    task_names : Union[List[str], str], optional
        Labels the head predicts. Defaults to every annotation label (a
        multi-output black box); pass the task labels for a task-only baseline.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engines, as in :class:`~torch_concepts.nn.ConceptBottleneckModel`.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``, the
        Lightning training arguments). A ``graph`` is accepted and ignored: a
        black box has no concept edges.

    Example
    -------
    >>> from torch_concepts.annotations import Annotations
    >>> ann = Annotations(labels=['c1', 'task'], cardinalities=[1, 1])
    >>> model = BlackBoxModel(input_size=8, annotations=ann, task_names=['task'])
    >>> out = model(query=['task'], input=torch.randn(2, 8))
    >>> out.logits.shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Optional[Union[List[str], str]] = None,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs,
    ):
        self.task_names: List[str] = (
            list(annotations.labels) if task_names is None else ensure_list(task_names)
        )
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            **kwargs,
        )
        self.pgm = self._build_model()
        self.setup_inference(
            inference,
            inference_kwargs,
            train_inference,
            train_inference_kwargs,
        )

    def _resolve_graph(self) -> ConceptGraph:
        """No concept edges: every output depends on the latent alone."""
        n = len(self.concept_names)
        return ConceptGraph(torch.zeros(n, n), node_names=list(self.concept_names))

    def _build_model(self) -> BayesianNetwork:
        """``input → latent → tasks``: one linear head per task variable (plate)."""
        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()
        tasks = self.build_concept_variables(self.task_names, plate_name="tasks")
        heads = ParametricCPD(
            variable=tasks,
            parents=[latent_var],
            parametrization=[
                self._flexible_parametrization(
                    variable=t,
                    first=LazyConstructor(LinearEmbeddingToConcept),
                    second=LazyConstructor(LinearEmbeddingToConcept),
                )
                for t in tasks
            ],
        )
        return BayesianNetwork(
            variables=[input_var, latent_var, *tasks],
            factors=[input_cpd, latent_cpd, *heads],
        )

    # ------------------------------------------------------------------
    # Learner hooks: score only the labels the head predicts
    # ------------------------------------------------------------------
    def prepare_target(self, batch) -> dict:
        """The default target with ``'c'`` sliced to :attr:`task_names`."""
        return {**batch['inputs'], 'c': batch['concepts']['c'][self.task_names]}

    def setup_metrics(self, metrics: ConceptMetrics):
        """Rebuild ``metrics`` on the task labels, so per-concept metrics never
        look up a label the head does not predict."""
        super().setup_metrics(ConceptMetrics(
            annotations=metrics.concept_annotations.subset(self.task_names),
            binary=metrics.fn_collection.get('binary'),
            categorical=metrics.fn_collection.get('categorical'),
            continuous=metrics.fn_collection.get('continuous'),
            summary=metrics.summary,
            per_concept=metrics.per_concept,
        ))

    def unscale_output(self, out, transforms, labels):
        """Refuse concept scaling when the head predicts a strict subset of the
        labels: :meth:`BaseLearner.unscale_output` inverse-transforms the full
        set of columns the scaler was fit on.

        Raises
        ------
        NotImplementedError
            If concept scaling is active and :attr:`task_names` is a strict
            subset of the annotation labels.
        """
        subset = set(self.task_names) != set(self.concept_names)
        if subset and self.scale_concepts and transforms.get('concepts') is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support concept scaling: its "
                "task-only prediction is a strict subset of the fitted concepts. "
                "Pass a datamodule without a 'concepts' scaler, or construct "
                "this model with scale_concepts=False."
            )
        return super().unscale_output(out, transforms, labels)
