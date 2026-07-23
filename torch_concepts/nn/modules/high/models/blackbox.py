import torch
from torch import nn
from typing import List, Optional, Union

from .....utils import ensure_list
from .....annotations import Annotations
from ...metrics import ConceptMetrics
from ...outputs import ModelOutput
from .....tensor import AnnotatedTensor

from ...low.dense_layers import MLP
from ..base.model import BaseModel


#: Arguments that only mean something to a model owning a probabilistic
#: graphical model: its concept ``graph`` and its inference engines. A blackbox
#: has neither — it maps the latent straight to a linear head — but the shared
#: model config sets the engines for *every* model and the experiment runner
#: passes ``graph`` to whatever model it builds, so a blackbox swapped into a
#: sweep must tolerate rather than reject them.
_PGM_ONLY_ARGS = (
    "graph",
    "inference",
    "inference_kwargs",
    "train_inference",
    "train_inference_kwargs",
)


def _drop_pgm_args(kwargs: dict) -> dict:
    """Drop the PGM-only arguments (see :data:`_PGM_ONLY_ARGS`) from ``kwargs``.

    Without this they reach ``LightningModule.__init__`` and raise
    ``TypeError: ... got an unexpected keyword argument 'inference'``.
    """
    return {key: value for key, value in kwargs.items() if key not in _PGM_ONLY_ARGS}


def _report_by_type(predictions: AnnotatedTensor) -> ModelOutput:
    """Wrap a head's output in a :class:`ModelOutput`, keyed by concept type.

    A blackbox head is a single flat tensor, but losses and metrics read a
    discrete concept from ``logits`` and a continuous one from ``loc`` — the
    quantity the PGM models report for a ``Normal``. Splitting the columns the
    same way keeps the output contract identical across model families, so one
    :class:`~torch_concepts.nn.ConceptLoss` / :class:`~torch_concepts.nn.ConceptMetrics`
    configuration works for both. No ``scale`` is reported: a blackbox head
    predicts a point estimate, not a distribution.
    """
    out = ModelOutput()
    continuous = set(predictions.annotation.labels_by_type.get("continuous", []))
    if not continuous:
        out.logits = predictions  # all-discrete: the common case, sliced by nobody
        return out
    labels = list(predictions.annotation.labels)
    discrete = [name for name in labels if name not in continuous]
    if discrete:
        out.logits = predictions[discrete]
    out.loc = predictions[[name for name in labels if name in continuous]]
    return out


class BlackBox(BaseModel):
    """
    BlackBox model.

    This model implements a standard neural network architecture for concept-based tasks,
    without explicit concept bottleneck or interpretable intermediate representations.
    It uses a backbone mapping the raw input to the latent representation, then a linear head.

    Args:
        input_size (int): Dimensionality of input features.
        annotations (Annotations): Annotation object for output variables.
        lightning (bool, optional): Enable Lightning training. Default False.
        **kwargs: Additional arguments for BaseModel. The PGM-only inference
            arguments (:data:`_PGM_ONLY_ARGS`) are accepted and ignored, so this
            model can be swapped into a sweep that configures them.

    Example:
        >>> from torch_concepts.annotations import Annotations
        >>> ann = Annotations(labels=['c1', 'task'], cardinalities=[1, 1])
        >>> model = BlackBox(input_size=8, annotations=ann)
        >>> out = model(torch.randn(2, 8))
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        lightning: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            **_drop_pgm_args(kwargs)
        )
        output_size = sum(self.concept_annotations.cardinalities)
        self.linear = nn.Linear(self.latent_size, output_size)

    def fully_observed_query(self, ground_truth) -> dict:
        """Build query dict mapping each concept name to its ground-truth column.

        Parameters
        ----------
        ground_truth : torch.Tensor
            Full concept-level ground truth, shape ``(batch, n_concepts)``.

        Returns
        -------
        dict
            ``{concept_name: tensor(batch, cardinality)}`` for every concept.
        """
        if ground_truth is None:
            return {name: None for name in self.concept_names}
        axis = self.concept_annotations
        query = {}
        for i, name in enumerate(axis.labels):
            card = axis.concept(name).cardinality
            if card == 1:
                query[name] = ground_truth[:, i].float().unsqueeze(-1)
            else:
                import torch.nn.functional as F
                query[name] = F.one_hot(ground_truth[:, i].long(), card).float()
        return query

    def forward(
        self,
        x: torch.Tensor = None,
        query=None,
        evidence: torch.Tensor = None,
        **kwargs
    ) -> ModelOutput:
        """Forward pass through the BlackBox model.

        Parameters
        ----------
        x : torch.Tensor, optional
            Input tensor. When ``None``, the tensor is extracted from
            ``evidence['input']`` (used by :meth:`BaseLearner.shared_step`).
        query : list of str or dict, optional
            Concept names to return. Defaults to all concepts.  When a dict
            is supplied (from ``fully_observed_query``), the keys are used as names.
        evidence : dict or torch.Tensor, optional
            Evidence dict (``{'input': x}`` from shared_step) or raw tensor
            (ignored for BlackBox).
        **kwargs
            Additional arguments (ignored).

        Returns
        -------
        ModelOutput
            ``logits`` for the queried discrete concepts and ``loc`` for the
            continuous ones — the same quantities the PGM-based models report.
        """
        # Resolve the raw input tensor
        if x is None and isinstance(evidence, dict):
            x = evidence.get('input', None)

        output = self.linear(self.backbone(x))

        axis = self.concept_annotations
        # query may be a list of strings, a dict (from fully_observed_query), or None
        if isinstance(query, dict):
            names = list(query.keys()) if query else axis.labels
        else:
            names = list(query) if query is not None else axis.labels

        # The head spans the whole concept annotation; label-slice it down to the
        # queried concepts (a no-op, and no copy, when everything is queried).
        predictions = AnnotatedTensor(output, axis, axis=-1)
        if names != axis.labels:
            predictions = predictions[names]
        return _report_by_type(predictions)


class BlackBoxTaskOnly(BaseModel):
    """
    BlackBox model.

    This model implements a standard neural network architecture for predicting tasks only,
    without explicit concept bottleneck or interpretable intermediate representations.
    It uses a backbone mapping the raw input to the latent representation, then a linear head.

    Args:
        input_size (int): Dimensionality of input features.
        annotations (Annotations): Annotation object for output variables.
        task_names (Union[List[str], str]): Task names to predict.
        lightning (bool, optional): Enable Lightning training. Default False.
        **kwargs: Additional arguments for BaseModel. The PGM-only inference
            arguments (:data:`_PGM_ONLY_ARGS`) are accepted and ignored, so this
            model can be swapped into a sweep that configures them.

    Attributes:
        task_annotations (Annotations): Sub-annotation restricted to task
            concepts only; used to size the linear head.
        task_concept_idx (List[int]): Concept-level column indices used to
            slice the ground-truth target tensor to match the task-only output.

    Example:
        >>> from torch_concepts.annotations import Annotations
        >>> ann = Annotations(labels=['c1', 'task'], cardinalities=[1, 1])
        >>> model = BlackBoxTaskOnly(input_size=8, annotations=ann, task_names=['task'])
        >>> out = model(torch.randn(2, 8))
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        lightning: bool = False,
        **kwargs
    ) -> None:
        self.task_names = ensure_list(task_names)

        # Task sub-annotation sizes the linear head; the concept-level indices
        # slice the ground-truth target to match the task-only output.
        self.task_annotations = annotations.subset(self.task_names)
        self.task_concept_idx = [
            annotations.get_index(name)
            for name in self.task_names
        ]

        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            **_drop_pgm_args(kwargs)
        )

        # Logit-level output size from the task sub-annotation
        output_size = sum(self.task_annotations.cardinalities)
        self.linear = nn.Linear(self.latent_size, output_size)

    def fully_observed_query(self, ground_truth) -> dict:
        """Build query dict mapping each *task* name to its ground-truth column.

        Parameters
        ----------
        ground_truth : torch.Tensor
            Full concept-level ground truth, shape ``(batch, n_all_concepts)``.

        Returns
        -------
        dict
            ``{task_name: tensor(batch, cardinality)}`` for every task.
        """
        if ground_truth is None:
            return {name: None for name in self.task_names}
        axis = self.concept_annotations
        query = {}
        for idx, name in zip(self.task_concept_idx, self.task_names):
            card = axis.concept(name).cardinality
            if card == 1:
                query[name] = ground_truth[:, idx].float().unsqueeze(-1)
            else:
                import torch.nn.functional as F
                query[name] = F.one_hot(ground_truth[:, idx].long(), card).float()
        return query

    def forward(self,
                x: torch.Tensor = None,
                query=None,
                evidence=None,
                **kwargs
        ) -> ModelOutput:
        """Forward pass through the BlackBoxTaskOnly model.

        Parameters
        ----------
        x : torch.Tensor, optional
            Input tensor. When ``None``, the tensor is extracted from
            ``evidence['input']`` (used by :meth:`BaseLearner.shared_step`).
        query : list of str or dict, optional
            Ignored; predictions are always returned for ``task_names``.
        evidence : dict or torch.Tensor, optional
            Evidence dict (``{'input': x}`` from shared_step) or raw tensor
            (ignored).
        **kwargs
            Additional arguments (ignored).

        Returns
        -------
        ModelOutput
            ``logits`` for the discrete tasks and ``loc`` for the continuous
            ones — the same quantities the PGM-based models report.
        """
        # Resolve the raw input tensor
        if x is None and isinstance(evidence, dict):
            x = evidence.get('input', None)

        output = self.linear(self.backbone(x))

        # The linear head spans exactly the task sub-annotation.
        return _report_by_type(
            AnnotatedTensor(output, self.task_annotations, axis=-1)
        )

    def prepare_target(self, target: torch.Tensor) -> torch.Tensor:
        """Slice the target to task-only columns and annotate it in task
        concept-space, matching the task-only output.

        Parameters
        ----------
        target : torch.Tensor
            Full concept-level ground-truth labels.

        Returns
        -------
        AnnotatedTensor
            Task-only concept-space annotated target.
        """
        sliced = target[:, self.task_concept_idx].as_subclass(torch.Tensor)
        return AnnotatedTensor(sliced, self.task_annotations.to_concept_space(), axis=-1)

    def unscale_output(self, out, transforms):
        """Not supported: :meth:`BaseLearner.unscale_output` assumes a
        prediction covers exactly the concepts the scaler was fit on, but this
        model's ``task_names`` is a strict subset of them.

        Raises
        ------
        NotImplementedError
            If concept scaling is active (:attr:`scale_concepts` and a
            'concepts' scaler were both supplied).
        """
        if self.scale_concepts and transforms.get('concepts') is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support concept scaling: its "
                "task-only prediction is a strict subset of the fitted concepts. "
                "Pass a datamodule without a 'concepts' scaler, or construct "
                "this model with scale_concepts=False."
            )
        return super().unscale_output(out, transforms)

    def setup_metrics(self, metrics: ConceptMetrics):
        """Rebuild metrics with task-only annotations.

        The base ``setup_metrics`` clones the original ``ConceptMetrics``, which
        was constructed against the *full* concept annotations and so has a
        metric submodule for every concept. Because ``BlackBoxTaskOnly`` only
        ever produces task logits, this override reconstructs the metrics
        scoped to ``task_annotations`` so only task concepts are tracked.
        """
        task_metrics = ConceptMetrics(
            annotations=self.task_annotations,
            binary=metrics.fn_collection.get('binary'),
            categorical=metrics.fn_collection.get('categorical'),
            continuous=metrics.fn_collection.get('continuous'),
            summary=metrics.summary,
            per_concept=metrics.per_concept,
        )
        super().setup_metrics(task_metrics)