"""Label-Free Concept Bottleneck Models (Oikarinen et al., ICLR 2023).

The model learns its vocabulary and two linear heads from raw concept
similarities, using the inherited ConceptBottleneckModel graph for inference.
Reference: https://arxiv.org/abs/2304.06129 and the authors' implementation at
https://github.com/Trustworthy-ML-Lab/Label-free-CBM (fa2869d).
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch_concepts import AnnotatedTensor, Annotations
from torch_concepts.distributions import Delta
from ...low.encoders.aligned import AlignedEmbeddingToConcept
from ...low.predictors.sparse import SparseLinearConceptToConcept
from ...outputs import InferenceOutput
from .cbm import ConceptBottleneckModel

if TYPE_CHECKING:
    from torch_concepts.data.base import ConceptDataModule
    from torch_concepts.data.generation import ConceptGenerationPipeline


class LabelFreeCBM(ConceptBottleneckModel):
    """A CBM whose bottleneck is discovered and fitted by a generation pipeline.

    ``fit(datamodule, task_name=...)`` discovers concepts from a prepared datamodule,
    fits an ``AlignedEmbeddingToConcept`` projection and a sparse linear task
    head, then installs the inherited CBM graph. The backbone is copied and
    frozen. Failed refits preserve the previous predictor and diagnostics.
    Fitting needs the optional ``lfcbm`` extra; inference does not.

    If the dataset is initially completely unlabelled, still choose a
    ``task_name`` and configure a generation pipeline to generate labels for
    that task. Run task-label generation during external data preparation and
    expose its results as native categorical task labels before calling ``fit``.
    The pipeline called inside ``fit`` generates bottleneck concepts; it does
    not supply missing task labels.

    Args:
        pipeline: Raw concept generation/annotation pipeline, or None for loading.
        backbone: Module mapping raw inputs to feature vectors.
        clip_cutoff: Strict training top-five activation cutoff (default .25).
        interpretability_cutoff: Strict validation alignment cutoff (default .45).
        projection_kwargs: Settings for ``AlignedEmbeddingToConcept.fit``.
        classifier_kwargs: Settings for ``SparseLinearConceptToConcept.fit``.
            Both stages use the paper defaults when omitted.
        annotation_source: Pipeline result name (default CLIPAnnotator).
        device: Training/inference device, also updated by ``to(...)``.
        seed: Projection and classifier random seed.
    """

    variable_distributions = {**ConceptBottleneckModel.variable_distributions, "continuous": Delta}

    def __init__(
        self, pipeline: Optional[ConceptGenerationPipeline], backbone: nn.Module,
        *,
        clip_cutoff: float = .25, interpretability_cutoff: float = .45,
        projection_kwargs: Optional[Mapping] = None,
        classifier_kwargs: Optional[Mapping] = None,
        annotation_source: str = "CLIPAnnotator", device: str = "cpu", seed: int = 0,
    ):
        # The vocabulary is unknown until fit; initialize module storage only.
        nn.Module.__init__(self)
        self.pipeline = pipeline
        self._backbone = deepcopy(backbone).to(device).eval().requires_grad_(False)
        for name in ("mean_", "std_", "projection_weight_"):
            self.register_buffer(name, torch.empty(0, device=device))
        self.pgm = None
        self.concept_annotations = None
        self.concept_names = []
        self.task_names = []
        self.clip_cutoff = clip_cutoff
        self.interpretability_cutoff = interpretability_cutoff
        self.projection_kwargs = dict(projection_kwargs or {})
        self.classifier_kwargs = dict(classifier_kwargs or {})
        self.annotation_source = annotation_source
        self.seed = seed

    @property
    def device(self) -> torch.device:
        return self.mean_.device

    @property
    def concept_names_(self) -> list[str]:
        self._check_fitted()
        return list(self.intermediate_concept_names)

    @property
    def task_name(self) -> str:
        self._check_fitted()
        return self.task_names[0]

    @property
    def class_names(self) -> list[str]:
        self._check_fitted()
        return list(self.concept_annotations.concept(self.task_name).states)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def fit(self, datamodule: ConceptDataModule, *, task_name: str) -> "LabelFreeCBM":
        """Fit a named categorical task on prepared splits.

        Call ``datamodule.setup('fit')`` beforehand. Task labels are required;
        ordered class names are read from the native task annotation. Concept
        names are discovered by the pipeline during fitting.

        For an initially completely unlabelled dataset, ``task_name`` is still
        required. Configure the generation pipeline used for data preparation
        to generate labels for that named task, and provide them as native
        categorical task labels before fitting. Missing task labels raise an
        error before this method invokes its bottleneck-generation pipeline.

        Args:
            datamodule: Prepared datamodule containing inputs and task labels.
            task_name: Name of the native categorical task to predict, including
                when its labels were generated during data preparation.
        """
        from torch_concepts.data.base import ConceptDataModule
        from torch_concepts.data.scalers import StandardScaler

        if not isinstance(datamodule, ConceptDataModule):
            raise TypeError("LabelFreeCBM.fit requires a ConceptDataModule.")
        train, val = datamodule.trainset, datamodule.valset
        if train is None or val is None or min(len(train), len(val)) < 2:
            raise ValueError(
                "Prepare train/validation splits with at least two observations each using "
                "datamodule.setup('fit') before calling fit()."
            )
        if datamodule.dataset.embs_precomputed:
            raise ValueError("LF-CBM requires raw dataset inputs for concept annotation.")
        schema = datamodule.dataset.native_annotations
        if schema is None or task_name not in schema.labels or schema.concept(task_name).type != "categorical":
            raise ValueError("The datamodule must provide native categorical task labels.")
        task = schema.subset([task_name])
        class_names = list(task.states[0])
        if len(set(class_names)) != len(class_names):
            raise ValueError("The task must declare distinct ordered class states.")
        train_scores, val_scores = self._generate_scores(train, val, task)
        activation = AnnotatedTensor(
            train_scores.tensor.topk(min(5, len(train)), dim=0).values.mean(0, keepdim=True),
            train_scores.annotations, axis=1,
        )
        active = [name for name, keep in zip(activation.annotations.labels,
                                           activation.tensor[0] > self.clip_cutoff) if keep]
        if not active:
            raise ValueError("No concepts survived the training CLIP activation cutoff.")
        train_scores, val_scores = train_scores[active], val_scores[active]
        backbone = deepcopy(self.backbone).eval().requires_grad_(False)
        loader_kwargs = dict(
            batch_size=datamodule.batch_size, num_workers=datamodule.workers,
            collate_fn=datamodule.dataset.collate, shuffle=False, drop_last=False,
        )
        train_h, train_y = self._features_and_targets(
            DataLoader(train, pin_memory=datamodule.pin_memory, **loader_kwargs), backbone, task,
        )
        val_h, val_y = self._features_and_targets(
            DataLoader(val, **loader_kwargs), backbone, task,
        )
        projection = AlignedEmbeddingToConcept(train_h.shape[1], train_scores.shape[1]).to(self.device).fit(
            train_h, train_scores.tensor, val_h, val_scores.tensor,
            seed=self.seed, **self.projection_kwargs,
        )
        alignment = projection.alignment_scores_
        alignment_scores = AnnotatedTensor(alignment.unsqueeze(0), train_scores.annotations, axis=1)
        retained = [name for name, keep in zip(active, alignment > self.interpretability_cutoff) if keep]
        if not retained:
            raise ValueError("No concepts survived the validation interpretability cutoff.")
        annotations = Annotations(
            labels=retained, cardinalities=[1] * len(retained), types=["continuous"] * len(retained),
            groups=train_scores[retained].annotations.groups,
        )
        weight = projection.encoder.weight[alignment > self.interpretability_cutoff].detach()
        with torch.no_grad():
            train_c = F.linear(train_h.to(self.device), weight).cpu()
            val_c = F.linear(val_h.to(self.device), weight).cpu()
        scaler = StandardScaler().fit(train_c)
        classifier = SparseLinearConceptToConcept(len(retained), len(class_names)).to(self.device).fit(
            scaler.transform(train_c), train_y, scaler.transform(val_c), val_y,
            seed=self.seed, **self.classifier_kwargs,
        )
        mean, std = scaler.mean[0].to(self.device), scaler.std[0].to(self.device)
        model = self._build_predictor(
            backbone, tuple(train[0]["inputs"]["x"].shape), annotations,
            weight, classifier, mean, std, task,
        )
        diagnostics = {
            "activation_scores": activation, "alignment_scores": alignment_scores,
            "projection_history": projection.history_, "projection_best_step": projection.best_step_,
            "classifier": classifier.diagnostics_,
        }
        # Publish only complete state; failed fitting leaves the old model intact.
        self.projection_weight_ = weight.clone()
        self.mean_, self.std_ = mean, std
        self.diagnostics_ = diagnostics
        self._install_predictor(model)
        self.eval()
        return self

    def _generate_scores(self, train, val, task):
        if self.pipeline is None:
            raise ValueError("A generation pipeline is required for fit().")
        if any(getattr(self.pipeline, name) is not None for name in (
            "calibrator", "raw_annotation_filter", "calibrated_annotation_filter", "aggregator",
        )):
            raise ValueError("LF-CBM requires raw similarities: no calibration, annotation filters, or aggregation.")
        outputs = self.pipeline(
            train, class_names=list(task.states[0]), generation_indices=list(range(len(train))),
            annotation_datasets={"train": train, "val": val},
        )
        required = [f"{split}_{self.annotation_source}" for split in ("train", "val")]
        if any(name not in outputs for name in required):
            raise ValueError(f"annotation_source {self.annotation_source!r} is unavailable; outputs: {list(outputs)}.")
        train_scores = outputs[f"train_{self.annotation_source}"]
        val_scores = outputs[f"val_{self.annotation_source}"]
        # The pipeline validates tensor shapes, rows, and routed metadata.
        for scores in (train_scores, val_scores):
            if (any(t != "binary" for t in scores.annotations.types)
                    or any(c != 1 for c in scores.annotations.cardinalities)
                    or not torch.isfinite(scores.tensor).all()):
                raise ValueError("LF-CBM requires one finite raw similarity per binary candidate concept.")
            if task.labels[0] in scores.annotations.labels:
                raise ValueError("A candidate concept cannot have the task variable's name.")
        if set(train_scores.annotations.labels) != set(val_scores.annotations.labels):
            raise ValueError("Train and validation concept identities differ.")
        val_scores = val_scores[train_scores.annotations.labels]
        if val_scores.annotations.to_dict() != train_scores.annotations.to_dict():
            raise ValueError("Train and validation concept definitions differ.")
        train_scores = AnnotatedTensor(train_scores.tensor.detach().float().cpu(), train_scores.annotations, axis=1)
        val_scores = AnnotatedTensor(val_scores.tensor.detach().float().cpu(), val_scores.annotations, axis=1)
        return train_scores, val_scores

    @torch.no_grad()
    def _features_and_targets(self, loader, backbone, task):
        features, targets = [], []
        for batch in loader:
            h = backbone(batch["inputs"]["x"].to(self.device))
            if not isinstance(h, Tensor) or h.ndim != 2:
                raise ValueError("backbone must return a (batch, features) tensor.")
            features.append(h.detach().float().cpu())
            targets.append(batch["concepts"]["native"][task.labels[0]].tensor[:, 0].cpu())
        h, y = torch.cat(features), torch.cat(targets).float()
        if (not torch.isfinite(y).all() or not torch.equal(y, y.round())
                or (y < 0).any() or (y >= len(task.states[0])).any()):
            raise ValueError("Task labels must be finite class indices into class_names.")
        if len(h) != len(loader.dataset) or not torch.isfinite(h).all():
            raise ValueError("Backbone features must be finite (samples, features) tensors.")
        return h, y.long()

    def _new_predictor(self, backbone, input_shape, annotations, feature_size, dtype, task_name):
        return ConceptBottleneckModel(
            input_size=input_shape, latent_size=feature_size, backbone=backbone,
            annotations=annotations, task_names=[task_name],
            variable_distributions=self.variable_distributions,
            variable_dist_kwargs=self.variable_dist_kwargs,
        ).to(device=self.device, dtype=dtype)

    def _build_predictor(self, backbone, input_shape, annotations, weight, classifier, mean, std, task):
        model = self._new_predictor(
            backbone, input_shape, annotations.union_with(task), weight.shape[1], weight.dtype, task.labels[0],
        )
        model.set_linear_heads(weight / std[:, None], -mean / std,
                               classifier.predictor.weight, classifier.predictor.bias)
        return model.eval().requires_grad_(False)

    def _install_predictor(self, model):
        """Install a complete graph while preserving this module's user hooks."""
        for name in (
            "input_size", "latent_size", "concept_annotations", "concept_names", "task_names",
            "plate", "variable_distributions", "variable_dist_kwargs", "_given_graph",
            "graph", "graph_order", "root_nodes", "internal_nodes",
            "_backbone", "pgm", "eval_inference", "train_inference",
        ):
            setattr(self, name, getattr(model, name))
        self.__dict__.pop("_query_plan", None)
        self.__dict__.pop("_query_segments", None)

    def _check_fitted(self):
        if self.pgm is None:
            raise RuntimeError("LabelFreeCBM is not fitted; call fit() first.")

    def forward(self, query=None, evidence=None, input: Optional[Tensor] = None,
                **inference_kwargs) -> InferenceOutput:
        """Query learned concepts/tasks through the standard CBM interface."""
        self._check_fitted()
        return super().forward(query=query, evidence=evidence, input=input, **inference_kwargs)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Return task logits in class_names order."""
        task = self.task_name
        return self(query=[task], input=x.to(self.device)).logits[task].tensor

    @torch.no_grad()
    def predict_concepts(self, x: Tensor) -> AnnotatedTensor:
        """Return standardized, real-valued concept activations."""
        return self(query=self.concept_names_, input=x.to(self.device)).value[self.concept_names_]

    def save(self, path) -> None:
        """Save JSON metadata and a tensor state dict, without generation components."""
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 3, "task_name": self.task_names[0],
            "annotations": self.concept_annotations.to_dict(),
            "input_shape": self.input_size, "feature_size": self.latent_size,
            "identity_backbone": type(self.backbone) is nn.Identity,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        torch.save(self.state_dict(), path / "state_dict.pt")

    @classmethod
    def load(cls, path, *, backbone: Optional[nn.Module] = None, device: str = "cpu") -> "LabelFreeCBM":
        """Load inference; provide the architecture for a non-Identity backbone."""
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["format_version"] not in (2, 3):
            raise ValueError("Unsupported LF-CBM checkpoint format.")
        if backbone is None:
            if not metadata["identity_backbone"]:
                raise ValueError("Supply a compatible backbone module.")
            backbone = nn.Identity()
        annotations = Annotations.from_dict(metadata["annotations"])
        obj = cls(None, backbone, device=device)
        state = torch.load(path / "state_dict.pt", map_location=device, weights_only=True)
        if metadata["format_version"] == 2:
            state = {name.removeprefix("model_."): value for name, value in state.items()}
        obj.mean_, obj.std_, obj.projection_weight_ = state["mean_"], state["std_"], state["projection_weight_"]
        n_concepts = len(annotations.labels) - 1
        if (obj.mean_.shape != (n_concepts,) or obj.std_.shape != obj.mean_.shape
                or not torch.isfinite(obj.mean_).all() or not torch.isfinite(obj.std_).all()
                or (obj.std_ <= 0).any()):
            raise ValueError("Invalid LF-CBM normalization statistics.")
        model = obj._new_predictor(
            obj._backbone, tuple(metadata["input_shape"]), annotations,
            metadata["feature_size"], obj.mean_.dtype, metadata["task_name"],
        )
        obj._install_predictor(model)
        obj.load_state_dict(state)
        return obj.eval().requires_grad_(False)
