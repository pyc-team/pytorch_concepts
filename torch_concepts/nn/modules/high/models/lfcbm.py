"""Label-Free Concept Bottleneck Models (Oikarinen et al., ICLR 2023).

The model learns its vocabulary and two linear heads from raw concept
similarities, then uses an ordinary ConceptBottleneckModel for inference.
Reference: https://arxiv.org/abs/2304.06129 and the authors' implementation at
https://github.com/Trustworthy-ML-Lab/Label-free-CBM (fa2869d).
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from torch_concepts import AnnotatedTensor, Annotations
from torch_concepts.distributions import Delta
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...outputs import InferenceOutput
from .cbm import ConceptBottleneckModel

if TYPE_CHECKING:
    from torch_concepts.data.generation import ConceptGenerationPipeline


def _cosine_cubed(scores: Tensor, projected: Tensor) -> Tensor:
    """Centered, cubed, columnwise cosine; zero-norm columns have score zero."""
    centered_scores = (scores.float() - scores.float().mean(0, keepdim=True)).pow(3)
    centered_projection = (projected - projected.mean(0, keepdim=True)).pow(3)
    score_norm = centered_scores.norm(dim=0, keepdim=True)
    projection_norm = centered_projection.norm(dim=0, keepdim=True)
    scores_unit = centered_scores / torch.where(score_norm == 0, 1., score_norm)
    projection_unit = centered_projection / torch.where(projection_norm == 0, 1., projection_norm)
    return (scores_unit * projection_unit).sum(0)


def _sample_list(samples):
    return samples


@torch.no_grad()
def _fit_saga(linear, x, y, *, penalty, alpha, epochs, batch_size,
              learning_rate, tolerance, seed):
    """Fit one elastic-net penalty using MadryLab's published SAGA solver."""
    from glm_saga.elasticnet import IndexedTensorDataset, train_saga

    linear.weight.zero_()
    linear.bias.zero_()
    loader = DataLoader(
        IndexedTensorDataset(x.cpu(), y.cpu()), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    train_saga(
        linear, loader, learning_rate, epochs, penalty, alpha, group=False,
        n_ex=len(x), n_classes=linear.out_features, tol=tolerance,
        logger=lambda message: None,
    )
    return {"solver": "glm_saga", "regularization": penalty, "alpha": alpha}


class LabelFreeCBM(nn.Module):
    """High-level LF-CBM without manually annotated bottleneck concepts.

    ``fit(datamodule)`` uses its existing train/validation sets, or calls
    ``setup('fit')`` if needed. Alternatively use ``fit(train, val)`` or
    ``fit(dataset, train_indices=..., val_indices=...)``. Test samples are never
    annotated or used for fitting. Task labels default to the named native
    concept; ``target_getter(sample)`` supports other dataset label formats.

    The pipeline must return finite raw similarity matrices, one column per
    binary candidate. Calibrators, annotation filters, and aggregators are
    rejected. ``annotation_source`` names the raw output (usually CLIPAnnotator).
    Generators receive only the training dataset and its local row indices.

    Defaults follow train_cbm.py: training top-five cutoff .25; bias-free Adam
    projection, lr .001, 1000 steps, batch 50000, validation every 50 steps with
    stopping at the first non-improvement; alignment cutoff .45; multinomial
    GLM-SAGA with batch 256, lr .1, 1000 epochs, tolerance 1e-4, alpha .99.
    ``classifier_lambda`` follows the authors' CLI ``lam`` convention: the
    effective penalty is lambda / max(.001, alpha).

    ``predict`` returns task logits, not class indices. ``predict_concepts``
    returns the standardized learned continuous activations as AnnotatedTensor.
    ``forward(query=..., input=...)`` returns the same InferenceOutput as other
    high-level CBMs, including support for evidence and inference arguments.
    Inference uses only the fitted ``model_``, never CLIP or the pipeline.
    The vocabulary and prediction graph are built during ``fit``. Training uses
    the method-specific Adam/SAGA stages rather than the Lightning training loop.

    Args:
        pipeline: Existing raw concept generation/annotation pipeline.
        backbone: Feature-vector module; the model owns a frozen copy.
        task_name: Native task label and final model task variable name.
        class_names: Ordered classifier states (at least two).
        annotation_source: Unprefixed pipeline output name to consume.
        clip_cutoff: Strict lower bound on training mean top-five similarity.
        interpretability_cutoff: Strict lower bound on validation alignment.
        projection_steps: Maximum Adam updates.
        projection_batch_size: Training observations per projection update.
        projection_lr: Adam learning rate.
        projection_validation_interval: Validation frequency, also at first/last step.
        classifier_alpha: Elastic-net L1 mixing fraction.
        classifier_lambda: Reference CLI regularization parameter.
        classifier_epochs: Maximum SAGA epochs.
        classifier_batch_size: Indexed SAGA minibatch size.
        classifier_lr: SAGA step size.
        classifier_tolerance: Parameter-change stopping tolerance.
        batch_size: Frozen backbone extraction batch size.
        target_getter: Optional task-label extractor for a raw dataset sample.
        device: Training and inference device.
        seed: Local projection sampling, initialization, and classifier seed.
    """

    def __init__(
        self, pipeline: Optional[ConceptGenerationPipeline], backbone: nn.Module,
        task_name: str, class_names: Sequence[str], annotation_source: str = "CLIPAnnotator",
        *, clip_cutoff: float = .25, interpretability_cutoff: float = .45,
        projection_steps: int = 1000, projection_batch_size: int = 50000,
        projection_lr: float = .001, projection_validation_interval: int = 50,
        classifier_alpha: float = .99, classifier_lambda: float = .0007,
        classifier_epochs: int = 1000, classifier_batch_size: int = 256,
        classifier_lr: float = .1, classifier_tolerance: float = 1e-4,
        batch_size: int = 512, target_getter: Optional[Callable] = None,
        device: str = "cpu", seed: int = 0,
    ):
        super().__init__()
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("class_names must contain at least two distinct ordered states.")
        if not 0 <= classifier_alpha <= 1 or classifier_lambda < 0:
            raise ValueError("classifier_alpha must be in [0, 1] and lambda nonnegative.")
        if min(projection_steps, projection_batch_size, projection_validation_interval,
               classifier_epochs, classifier_batch_size, batch_size) < 1:
            raise ValueError("Step counts, validation interval, and batch sizes must be positive.")
        if min(projection_lr, classifier_lr) <= 0 or classifier_tolerance < 0:
            raise ValueError("Learning rates must be positive and tolerance nonnegative.")
        self.pipeline = pipeline
        self.backbone = deepcopy(backbone).to(device).eval().requires_grad_(False)
        self.register_buffer("_device_anchor", torch.empty(0, device=device), persistent=False)
        self.register_buffer("mean_", torch.empty(0, device=device))
        self.register_buffer("std_", torch.empty(0, device=device))
        self.task_name = task_name
        self.class_names = list(class_names)
        self.annotation_source = annotation_source
        self.clip_cutoff = clip_cutoff
        self.interpretability_cutoff = interpretability_cutoff
        self.projection_steps = projection_steps
        self.projection_batch_size = projection_batch_size
        self.projection_lr = projection_lr
        self.projection_validation_interval = projection_validation_interval
        self.classifier_alpha = classifier_alpha
        self.classifier_lambda = classifier_lambda
        self.classifier_epochs = classifier_epochs
        self.classifier_batch_size = classifier_batch_size
        self.classifier_lr = classifier_lr
        self.classifier_tolerance = classifier_tolerance
        self.batch_size = batch_size
        self.target_getter = target_getter
        self.seed = seed

    @property
    def device(self) -> torch.device:
        """Current module device, including after ``to(...)``."""
        return self._device_anchor.device

    def train(self, mode: bool = True):
        """Set module mode while keeping the frozen backbone in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def fit(
        self, data, val_dataset: Optional[Dataset] = None, *,
        train_indices: Optional[Sequence[int]] = None,
        val_indices: Optional[Sequence[int]] = None,
        embeddings: Optional[Mapping[str, Tensor]] = None,
    ) -> "LabelFreeCBM":
        """Fit using train/val sources; optional embeddings are split-aligned.

        ``embeddings={'train': ..., 'val': ...}`` bypasses feature extraction,
        while retaining the original backbone and raw datasets for annotation.
        Each embedding row must correspond to that split's observation order.
        Datasets whose inputs have already been replaced with embeddings are
        rejected: CLIP must still see the original inputs.
        """
        try:
            from glm_saga.elasticnet import elastic_loss
        except ImportError as error:
            raise ImportError(
                "LF-CBM fitting requires glm-saga; install 'pytorch_concepts[lfcbm]'."
            ) from error
        if self.pipeline is None:
            raise ValueError("A generation pipeline is required for fit().")
        if any(getattr(self.pipeline, name) is not None for name in (
            "calibrator", "raw_annotation_filter", "calibrated_annotation_filter", "aggregator",
        )):
            raise ValueError("LF-CBM requires raw similarities: no calibration, annotation filters, or aggregation.")
        train, val = self._splits(data, val_dataset, train_indices, val_indices)
        if min(len(train), len(val)) < 2:
            raise ValueError("LF-CBM needs at least two training and validation observations.")
        for dataset in (train, val):
            source = dataset
            while isinstance(source, Subset):
                source = source.dataset
            if getattr(source, "embs_precomputed", False):
                raise ValueError("Keep raw dataset inputs; pass precomputed features via embeddings instead.")
        outputs = self.pipeline(
            train, class_names=self.class_names, generation_indices=list(range(len(train))),
            annotation_datasets={"train": train, "val": val},
        )
        required = [f"{split}_{self.annotation_source}" for split in ("train", "val")]
        if any(name not in outputs for name in required):
            raise ValueError(f"annotation_source {self.annotation_source!r} is unavailable; outputs: {list(outputs)}.")
        train_scores = outputs[f"train_{self.annotation_source}"]
        val_scores = outputs[f"val_{self.annotation_source}"]
        self._validate_scores(train_scores, len(train))
        self._validate_scores(val_scores, len(val))
        if set(train_scores.annotations.labels) != set(val_scores.annotations.labels):
            raise ValueError("Train and validation concept identities differ.")
        val_scores = val_scores[train_scores.annotations.labels]
        if val_scores.annotations.to_dict() != train_scores.annotations.to_dict():
            raise ValueError("Train and validation concept definitions differ.")
        train_scores = AnnotatedTensor(train_scores.tensor.detach().float().cpu(), train_scores.annotations, axis=1)
        val_scores = AnnotatedTensor(val_scores.tensor.detach().float().cpu(), val_scores.annotations, axis=1)
        self.candidate_annotations_ = train_scores.annotations
        self.activation_scores_ = AnnotatedTensor(
            train_scores.tensor.topk(min(5, len(train)), dim=0).values.mean(0, keepdim=True),
            self.candidate_annotations_, axis=1,
        )
        active = [
            label for label, keep in zip(self.candidate_annotations_.labels,
                                        self.activation_scores_.tensor[0] > self.clip_cutoff) if keep
        ]
        if not active:
            raise ValueError("No concepts survived the training CLIP activation cutoff.")
        train_scores, val_scores = train_scores[active], val_scores[active]
        self.backbone_ = deepcopy(self.backbone).to(self.device).eval().requires_grad_(False)
        self.backbone = self.backbone_
        self.input_shape_ = tuple(train[0]["inputs"]["x"].shape)
        train_features, train_y = self._features_and_targets(train, None if embeddings is None else embeddings["train"])
        val_features, val_y = self._features_and_targets(val, None if embeddings is None else embeddings["val"])
        if train_features.shape[1] != val_features.shape[1]:
            raise ValueError("Training and validation embedding dimensions differ.")
        self.feature_size_ = train_features.shape[1]
        projection, alignment, history = self._fit_projection(
            train_features, val_features, train_scores, val_scores,
        )
        self.alignment_scores_ = AnnotatedTensor(alignment.unsqueeze(0).cpu(), train_scores.annotations, axis=1)
        self.projection_history_ = history
        retained = [
            label for label, keep in zip(active, alignment > self.interpretability_cutoff) if keep
        ]
        if not retained:
            raise ValueError("No concepts survived the validation interpretability cutoff.")
        self.concept_annotations_ = Annotations(
            labels=retained, cardinalities=[1] * len(retained), types=["continuous"] * len(retained),
            groups=train_scores[retained].annotations.groups,
        )
        self.generated_annotations_ = {"train": train_scores[retained], "val": val_scores[retained]}
        row_indices = [train_scores.annotations.get_index(label) for label in retained]
        self.projection_ = LinearEmbeddingToConcept(self.feature_size_, len(retained), bias=False).to(self.device)
        with torch.no_grad():
            self.projection_.encoder.weight.copy_(projection.encoder.weight[row_indices])
        self.projection_.requires_grad_(False).eval()
        with torch.no_grad():
            train_c = self.projection_(train_features.to(self.device)).cpu()
            val_c = self.projection_(val_features.to(self.device)).cpu()
        self.mean_ = train_c.mean(0)
        self.std_ = train_c.std(0)
        if not torch.isfinite(self.std_).all() or (self.std_ <= 0).any():
            raise ValueError("Retained projected concepts must have positive training standard deviation.")
        train_z, val_z = (train_c - self.mean_) / self.std_, (val_c - self.mean_) / self.std_
        self.classifier_ = LinearConceptToConcept(len(retained), len(self.class_names)).to(self.device)
        self.classifier_regularization_ = self.classifier_lambda / max(.001, self.classifier_alpha)
        self.classifier_diagnostics_ = _fit_saga(
            self.classifier_.predictor, train_z, train_y,
            penalty=self.classifier_regularization_, alpha=self.classifier_alpha,
            epochs=self.classifier_epochs, batch_size=self.classifier_batch_size,
            learning_rate=self.classifier_lr, tolerance=self.classifier_tolerance, seed=self.seed,
        )
        self.classifier_.requires_grad_(False).eval()
        with torch.no_grad():
            for name, x, y in (("train", train_z, train_y), ("val", val_z, val_y)):
                self.classifier_diagnostics_[f"{name}_loss"] = float(elastic_loss(
                    self.classifier_.predictor, x.to(self.device), y.to(self.device),
                    self.classifier_regularization_, self.classifier_alpha,
                ))
        self._build_predictor()
        return self

    @property
    def concept_names_(self) -> list[str]:
        self._check_fitted()
        return list(self.concept_annotations_.labels)

    @staticmethod
    def _splits(data, val, train_indices, val_indices):
        from torch_concepts.data.base import ConceptDataModule

        if isinstance(data, ConceptDataModule):
            if val is not None or train_indices is not None or val_indices is not None:
                raise ValueError("Use the datamodule's existing splits without additional datasets or indices.")
            if data.trainset is None or data.valset is None:
                data.setup("fit")
            if data.trainset is None or data.valset is None:
                raise ValueError("The datamodule must provide nonempty train and validation splits.")
            return data.trainset, data.valset
        if train_indices is not None or val_indices is not None:
            if val is not None or train_indices is None or val_indices is None:
                raise ValueError("Provide both train_indices and val_indices, without val_dataset.")
            if set(train_indices) & set(val_indices):
                raise ValueError("Training and validation indices must be disjoint.")
            return Subset(data, list(train_indices)), Subset(data, list(val_indices))
        if not isinstance(data, Dataset) or not isinstance(val, Dataset):
            raise ValueError("Provide a datamodule, train/val datasets, or train/val indices.")
        return data, val

    def _validate_scores(self, values, n_samples):
        if not isinstance(values, AnnotatedTensor) or values.axis not in (1, -1) or values.ndim != 2:
            raise ValueError("LF-CBM annotations must be sample-by-concept AnnotatedTensors on axis 1.")
        if values.shape[0] != n_samples or values.shape[1] != len(values.annotations.labels):
            raise ValueError("Annotation rows and concept columns must align with each dataset split.")
        if any(t != "binary" for t in values.annotations.types):
            raise ValueError("LF-CBM requires one raw similarity column per binary candidate concept.")
        if self.task_name in values.annotations.labels:
            raise ValueError("A candidate concept cannot have the task variable's name.")
        if len(set(values.annotations.labels)) != values.shape[1] or not torch.isfinite(values.tensor).all():
            raise ValueError("Candidate labels must be unique and raw similarities finite.")

    def _features_and_targets(self, dataset, embeddings):
        source = dataset
        while isinstance(source, Subset):
            source = source.dataset
        if self.target_getter is None:
            native = getattr(source, "native_concepts", None)
            if native is None or self.task_name not in native.annotations.labels:
                raise ValueError("Supply native task labels or target_getter(sample).")
            target_index = native.annotations.get_index(self.task_name)
            schema = source.annotations.concept(self.task_name)
            if schema.type == "categorical" and list(schema.states) != self.class_names:
                raise ValueError("Native task state ordering must match class_names.")
        features, targets = [], []
        with torch.no_grad():
            for samples in DataLoader(dataset, batch_size=self.batch_size, collate_fn=_sample_list):
                targets.extend([
                    self.target_getter(sample) if self.target_getter is not None
                    else sample["concepts"]["native"][target_index] for sample in samples
                ])
                if embeddings is None:
                    x = torch.stack([sample["inputs"]["x"] for sample in samples]).to(self.device)
                    h = self.backbone_(x)
                    if not isinstance(h, Tensor) or h.ndim != 2:
                        raise ValueError("backbone must return a (batch, features) tensor.")
                    features.append(h.detach().float().cpu())
        targets = torch.as_tensor(targets).float()
        if not torch.isfinite(targets).all() or not torch.equal(targets, targets.round()):
            raise ValueError("Task labels must be finite class indices.")
        y = targets.long()
        if y.ndim != 1 or y.min() < 0 or y.max() >= len(self.class_names):
            raise ValueError("Task labels must index class_names in its declared order.")
        h = torch.cat(features) if embeddings is None else embeddings.detach().float().cpu()
        if h.ndim != 2 or len(h) != len(dataset) or not torch.isfinite(h).all():
            raise ValueError("Embeddings must be finite, split-aligned (samples, features) tensors.")
        return h, y

    def _fit_projection(self, train_h, val_h, train_scores, val_scores):
        projection = LinearEmbeddingToConcept(train_h.shape[1], train_scores.shape[1], bias=False)
        generator = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            projection.encoder.weight.uniform_(
                -1 / math.sqrt(train_h.shape[1]), 1 / math.sqrt(train_h.shape[1]), generator=generator,
            )
        projection.to(self.device)
        optimizer = torch.optim.Adam(projection.parameters(), lr=self.projection_lr)
        rng = random.Random(self.seed)
        best_loss, best_weight, best_step = float("inf"), None, None
        history = []
        for step in range(self.projection_steps):
            indices = rng.sample(range(len(train_h)), min(self.projection_batch_size, len(train_h)))
            optimizer.zero_grad()
            loss = -_cosine_cubed(
                train_scores.tensor[indices].to(self.device), projection(train_h[indices].to(self.device)),
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite projection loss; check feature and annotation variation.")
            loss.backward()
            optimizer.step()
            if step % self.projection_validation_interval == 0 or step == self.projection_steps - 1:
                with torch.no_grad():
                    alignment = _cosine_cubed(val_scores.tensor.to(self.device), projection(val_h.to(self.device)))
                    val_loss = float(-alignment.mean())
                if not math.isfinite(val_loss):
                    raise ValueError("Nonfinite validation alignment; check feature variation.")
                history.append({"step": step, "train_loss": float(loss.detach()), "val_loss": val_loss})
                if val_loss < best_loss:
                    best_loss, best_step = val_loss, step
                    best_weight = projection.encoder.weight.detach().clone()
                else:
                    break
        with torch.no_grad():
            projection.encoder.weight.copy_(best_weight)
        self.projection_best_step_ = best_step
        with torch.no_grad():
            alignment = _cosine_cubed(val_scores.tensor.to(self.device), projection(val_h.to(self.device)))
        return projection, alignment.cpu(), history

    def _build_predictor(self):
        self.mean_, self.std_ = self.mean_.to(self.device), self.std_.to(self.device)
        task = Annotations(labels=[self.task_name], states=[self.class_names], types=["categorical"])
        annotations = self.concept_annotations_.union_with(task)
        self.model_ = ConceptBottleneckModel(
            input_size=self.input_shape_, latent_size=self.feature_size_,
            backbone=self.backbone_, annotations=annotations, task_names=[self.task_name],
            variable_distributions={"continuous": Delta},
        ).to(self.device).eval().requires_grad_(False)
        encoder = self.model_.pgm.factors["concepts"].parametrization["value"].encoder
        classifier = self.model_.pgm.factors["tasks"].parametrization["logits"]
        with torch.no_grad():
            std, mean = self.std_.to(self.device), self.mean_.to(self.device)
            encoder.weight.copy_(self.projection_.encoder.weight / std[:, None])
            encoder.bias.copy_(-mean / std)
        classifier.load_state_dict(self.classifier_.state_dict())
        self.eval()

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("LabelFreeCBM is not fitted; call fit() first.")

    def forward(self, query, evidence=None, input: Optional[Tensor] = None,
                **inference_kwargs) -> InferenceOutput:
        """Query learned concepts or tasks using the standard high-level API.

        The retained vocabulary is available after ``fit``. Results contain
        standardized concept activations under ``value`` and task ``logits``.
        """
        self._check_fitted()
        return self.model_(query=query, evidence=evidence, input=input, **inference_kwargs)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Return class logits in class_names order using the fitted CBM."""
        self._check_fitted()
        self.model_.eval()
        return self(query=[self.task_name], input=x.to(self.device)).logits[self.task_name].tensor

    @torch.no_grad()
    def predict_concepts(self, x: Tensor) -> AnnotatedTensor:
        """Return standardized learned concept activations, accessible by name."""
        self._check_fitted()
        self.model_.eval()
        return self(query=self.concept_names_, input=x.to(self.device)).value[self.concept_names_]

    def save(self, path) -> None:
        """Save explicit metadata and state dicts; never pickle the pipeline."""
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1, "task_name": self.task_name, "class_names": self.class_names,
            "concept_annotations": self.concept_annotations_.to_dict(),
            "feature_size": self.feature_size_, "input_shape": self.input_shape_,
            "backbone_type": f"{type(self.backbone_).__module__}.{type(self.backbone_).__qualname__}",
            "identity_backbone": type(self.backbone_) is nn.Identity,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        torch.save({
            "backbone": self.backbone_.state_dict(), "projection": self.projection_.state_dict(),
            "classifier": self.classifier_.state_dict(), "mean": self.mean_, "std": self.std_,
        }, path / "state_dict.pt")

    @classmethod
    def load(cls, path, *, backbone: Optional[nn.Module] = None, device: str = "cpu") -> "LabelFreeCBM":
        """Load inference without CLIP/LLM; supply a matching custom backbone.

        Identity backbones reconstruct automatically. For any other backbone,
        pass an untrained compatible module; its weights are restored strictly
        from the checkpoint. No import or arbitrary code is loaded from metadata.
        """
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported LF-CBM checkpoint format.")
        if backbone is None:
            if not metadata["identity_backbone"]:
                raise ValueError(f"Supply a compatible backbone module: {metadata['backbone_type']}.")
            backbone = nn.Identity()
        state = torch.load(path / "state_dict.pt", map_location=device, weights_only=True)
        obj = cls(None, backbone, metadata["task_name"], metadata["class_names"], device=device)
        obj.backbone_ = deepcopy(backbone).to(device).eval().requires_grad_(False)
        obj.backbone = obj.backbone_
        obj.backbone_.load_state_dict(state["backbone"])
        obj.concept_annotations_ = Annotations.from_dict(metadata["concept_annotations"])
        obj.feature_size_, obj.input_shape_ = metadata["feature_size"], tuple(metadata["input_shape"])
        obj.projection_ = LinearEmbeddingToConcept(obj.feature_size_, len(obj.concept_annotations_.labels), bias=False).to(device)
        obj.projection_.load_state_dict(state["projection"])
        obj.projection_.requires_grad_(False).eval()
        obj.classifier_ = LinearConceptToConcept(len(obj.concept_annotations_.labels), len(obj.class_names)).to(device)
        obj.classifier_.load_state_dict(state["classifier"])
        obj.classifier_.requires_grad_(False).eval()
        obj.mean_, obj.std_ = state["mean"].cpu(), state["std"].cpu()
        obj._build_predictor()
        return obj
