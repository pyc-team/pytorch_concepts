"""Label-Free Concept Bottleneck Models (Oikarinen et al., ICLR 2023).

The model learns its vocabulary and two linear heads from raw concept
similarities, using the inherited ConceptBottleneckModel graph for inference.
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
    )
    return {"solver": "glm_saga", "regularization": penalty, "alpha": alpha}


class LabelFreeCBM(ConceptBottleneckModel):
    """A CBM whose bottleneck is discovered and fitted by a generation pipeline.

    The pipeline discovers binary concept definitions and raw similarities from
    training data. The backbone is copied and frozen. Native categorical task
    labels must follow ``class_names``; ``target_getter(sample)`` supports custom
    label sources. Fitting needs the optional ``lfcbm`` extra; inference does not.

    Use ``fit(datamodule)`` or ``fit(train_dataset, val_dataset)``. For indexed
    splits, pass ordinary PyTorch ``Subset`` objects. The pipeline is called
    internally during ``fit``; the resulting graph belongs directly to this
    model. ``concept_names_`` lists the retained vocabulary;
    ``diagnostics_`` holds filtering scores and training history. Failed refits
    preserve the previous model and diagnostics.

    Args:
        pipeline: Raw concept generation/annotation pipeline, or None for loading.
        backbone: Module mapping raw inputs to feature vectors.
        task_name: Name of the categorical task.
        class_names: Ordered task states (at least two).
        clip_cutoff: Strict training top-five activation cutoff (default .25).
        interpretability_cutoff: Strict validation alignment cutoff (default .45).
        projection_kwargs: Adam settings: steps=1000, batch_size=50000, lr=.001,
            validation_interval=50. Stops at the first validation non-improvement.
        classifier_kwargs: SAGA settings: epochs=1000, batch_size=256, lr=.1,
            alpha=.99, lam=.0007, tolerance=1e-4. The reference penalty is
            lam / max(.001, alpha); the intercept is unpenalized.
        annotation_source: Pipeline result name (default CLIPAnnotator).
        batch_size: Frozen backbone extraction batch size (default 512).
        target_getter: Optional task-label extractor for each sample.
        device: Training/inference device, also updated by ``to(...)``.
        seed: Projection and classifier random seed.
    """

    variable_distributions = {**ConceptBottleneckModel.variable_distributions, "continuous": Delta}

    def __init__(
        self, pipeline: Optional[ConceptGenerationPipeline], backbone: nn.Module,
        task_name: str, class_names: Sequence[str], *,
        clip_cutoff: float = .25, interpretability_cutoff: float = .45,
        projection_kwargs: Optional[Mapping] = None,
        classifier_kwargs: Optional[Mapping] = None,
        annotation_source: str = "CLIPAnnotator", batch_size: int = 512,
        target_getter: Optional[Callable] = None, device: str = "cpu", seed: int = 0,
    ):
        # The vocabulary is unknown until fit; initialize module storage only.
        nn.Module.__init__(self)
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("class_names must contain at least two distinct ordered states.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.pipeline = pipeline
        self._backbone = deepcopy(backbone).to(device).eval().requires_grad_(False)
        for name in ("mean_", "std_", "projection_weight_"):
            self.register_buffer(name, torch.empty(0, device=device))
        self.pgm = None
        self.concept_annotations = None
        self.concept_names = []
        self.task_names = [task_name]
        self.task_name = task_name
        self.class_names = list(class_names)
        self.clip_cutoff = clip_cutoff
        self.interpretability_cutoff = interpretability_cutoff
        self.projection_kwargs = dict(projection_kwargs or {})
        self.classifier_kwargs = dict(classifier_kwargs or {})
        self.annotation_source = annotation_source
        self.batch_size = batch_size
        self.target_getter = target_getter
        self.seed = seed

    @property
    def device(self) -> torch.device:
        return self.mean_.device

    @property
    def concept_names_(self) -> list[str]:
        self._check_fitted()
        return list(self.intermediate_concept_names)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def fit(self, data, val_dataset: Optional[Dataset] = None, *,
            embeddings: Optional[Mapping[str, Tensor]] = None) -> "LabelFreeCBM":
        """Fit on a datamodule's splits or explicit train/validation datasets."""
        try:
            import glm_saga.elasticnet
        except ImportError as error:
            raise ImportError(
                "LF-CBM fitting requires glm-saga; install 'pytorch_concepts[lfcbm]'."
            ) from error
        train, val = self._splits(data, val_dataset)
        train_scores, val_scores = self._generate_scores(train, val)
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
        train_h, train_y = self._features_and_targets(train, backbone, None if embeddings is None else embeddings["train"])
        val_h, val_y = self._features_and_targets(val, backbone, None if embeddings is None else embeddings["val"])
        if train_h.shape[1] != val_h.shape[1]:
            raise ValueError("Training and validation feature dimensions differ.")
        projection, alignment, history, best_step = self._fit_projection(
            train_h, val_h, train_scores, val_scores, **self.projection_kwargs,
        )
        alignment_scores = AnnotatedTensor(alignment.unsqueeze(0), train_scores.annotations, axis=1)
        retained = [name for name, keep in zip(active, alignment > self.interpretability_cutoff) if keep]
        if not retained:
            raise ValueError("No concepts survived the validation interpretability cutoff.")
        annotations = Annotations(
            labels=retained, cardinalities=[1] * len(retained), types=["continuous"] * len(retained),
            groups=train_scores[retained].annotations.groups,
        )
        rows = [train_scores.annotations.get_index(name) for name in retained]
        selected = LinearEmbeddingToConcept(train_h.shape[1], len(retained), bias=False).to(self.device)
        with torch.no_grad():
            selected.encoder.weight.copy_(projection.encoder.weight[rows])
        selected.requires_grad_(False)
        classifier, mean, std, classifier_diagnostics = self._fit_classifier(
            selected, train_h, val_h, train_y, val_y, **self.classifier_kwargs,
        )
        model = self._build_predictor(
            backbone, tuple(train[0]["inputs"]["x"].shape), annotations,
            selected.encoder.weight, classifier, mean, std,
        )
        diagnostics = {
            "activation_scores": activation, "alignment_scores": alignment_scores,
            "projection_history": history, "projection_best_step": best_step,
            "classifier": classifier_diagnostics,
        }
        # Publish only complete state; failed fitting leaves the old model intact.
        self.projection_weight_ = selected.encoder.weight.detach().clone()
        self.mean_, self.std_ = mean, std
        self.diagnostics_ = diagnostics
        self._install_predictor(model)
        self.eval()
        return self

    @staticmethod
    def _splits(data, val):
        from torch_concepts.data.base import ConceptDataModule

        if isinstance(data, ConceptDataModule):
            if val is not None:
                raise ValueError("Use the datamodule's existing splits without val_dataset.")
            if data.trainset is None or data.valset is None:
                data.setup("fit")
            data, val = data.trainset, data.valset
        if not isinstance(data, Dataset) or not isinstance(val, Dataset):
            raise ValueError("Provide a datamodule or train/validation datasets.")
        if min(len(data), len(val)) < 2:
            raise ValueError("LF-CBM needs at least two training and validation observations.")
        return data, val

    def _generate_scores(self, train, val):
        if self.pipeline is None:
            raise ValueError("A generation pipeline is required for fit().")
        if any(getattr(self.pipeline, name) is not None for name in (
            "calibrator", "raw_annotation_filter", "calibrated_annotation_filter", "aggregator",
        )):
            raise ValueError("LF-CBM requires raw similarities: no calibration, annotation filters, or aggregation.")
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
        return train_scores, val_scores

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

    def _features_and_targets(self, dataset, backbone, embeddings=None):
        source = dataset
        while isinstance(source, Subset):
            source = source.dataset
        if self.target_getter is None:
            native = getattr(source, "native_concepts", None)
            if native is None or self.task_name not in native.annotations.labels:
                raise ValueError("Supply native task labels or target_getter(sample).")
            target_index = native.annotations.get_index(self.task_name)
            schema = getattr(source, "native_annotations", source.annotations).concept(self.task_name)
            if schema.type == "categorical" and list(schema.states) != self.class_names:
                raise ValueError("Native task state ordering must match class_names.")
        features, targets = [], []
        with torch.no_grad():
            for samples in DataLoader(dataset, batch_size=self.batch_size, collate_fn=list):
                targets.extend([
                    self.target_getter(sample) if self.target_getter is not None
                    else sample["concepts"]["native"][target_index] for sample in samples
                ])
                if embeddings is None:
                    x = torch.stack([sample["inputs"]["x"] for sample in samples]).to(self.device)
                    h = backbone(x)
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
            raise ValueError("Backbone features must be finite (samples, features) tensors.")
        return h, y

    def _fit_projection(self, train_h, val_h, train_scores, val_scores, *,
                        steps=1000, batch_size=50000, lr=.001, validation_interval=50):
        if min(steps, batch_size, validation_interval) < 1 or lr <= 0:
            raise ValueError("Projection steps, batch sizes, validation interval, and learning rate must be positive.")
        projection = LinearEmbeddingToConcept(train_h.shape[1], train_scores.shape[1], bias=False)
        generator = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            projection.encoder.weight.uniform_(
                -1 / math.sqrt(train_h.shape[1]), 1 / math.sqrt(train_h.shape[1]), generator=generator,
            )
        projection.to(self.device)
        optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
        rng = random.Random(self.seed)
        best_loss, best_weight, best_step = float("inf"), None, None
        history = []
        for step in range(steps):
            indices = rng.sample(range(len(train_h)), min(batch_size, len(train_h)))
            optimizer.zero_grad()
            loss = -_cosine_cubed(
                train_scores.tensor[indices].to(self.device), projection(train_h[indices].to(self.device)),
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite projection loss; check feature and annotation variation.")
            loss.backward()
            optimizer.step()
            if step % validation_interval == 0 or step == steps - 1:
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
        with torch.no_grad():
            alignment = _cosine_cubed(val_scores.tensor.to(self.device), projection(val_h.to(self.device)))
        return projection, alignment.cpu(), history, best_step

    def _fit_classifier(self, projection, train_features, val_features, train_y, val_y, *,
                        epochs=1000, batch_size=256, lr=.1, alpha=.99,
                        lam=.0007, tolerance=1e-4):
        from glm_saga.elasticnet import elastic_loss

        if not 0 <= alpha <= 1 or lam < 0:
            raise ValueError("Classifier alpha must be in [0, 1] and lam nonnegative.")
        if epochs < 1 or batch_size < 1 or lr <= 0 or tolerance < 0:
            raise ValueError("Classifier steps, batch size, and learning rate must be positive; tolerance nonnegative.")
        with torch.no_grad():
            train_c = projection(train_features.to(self.device)).cpu()
            val_c = projection(val_features.to(self.device)).cpu()
        mean = train_c.mean(0)
        std = train_c.std(0)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Retained projected concepts must have positive training standard deviation.")
        train_z, val_z = (train_c - mean) / std, (val_c - mean) / std
        classifier = LinearConceptToConcept(projection.encoder.out_features, len(self.class_names)).to(self.device)
        classifier_regularization = lam / max(.001, alpha)
        classifier_diagnostics = _fit_saga(
            classifier.predictor, train_z, train_y,
            penalty=classifier_regularization, alpha=alpha,
            epochs=epochs, batch_size=batch_size,
            learning_rate=lr, tolerance=tolerance, seed=self.seed,
        )
        classifier.requires_grad_(False).eval()
        with torch.no_grad():
            for name, x, y in (("train", train_z, train_y), ("val", val_z, val_y)):
                classifier_diagnostics[f"{name}_loss"] = float(elastic_loss(
                    classifier.predictor, x.to(self.device), y.to(self.device),
                    classifier_regularization, alpha,
                ))
        mean, std = mean.to(self.device), std.to(self.device)
        return classifier, mean, std, classifier_diagnostics

    def _new_predictor(self, backbone, input_shape, annotations, feature_size, dtype):
        return ConceptBottleneckModel(
            input_size=input_shape, latent_size=feature_size, backbone=backbone,
            annotations=annotations, task_names=[self.task_name],
            variable_distributions=self.variable_distributions,
            variable_dist_kwargs=self.variable_dist_kwargs,
        ).to(device=self.device, dtype=dtype)

    def _build_predictor(self, backbone, input_shape, annotations, weight, classifier, mean, std):
        task = Annotations(labels=[self.task_name], states=[self.class_names], types=["categorical"])
        model = self._new_predictor(backbone, input_shape, annotations.union_with(task), weight.shape[1], weight.dtype)
        model.set_linear_heads(weight / std[:, None], -mean / std,
                               classifier.predictor.weight, classifier.predictor.bias)
        return model.eval().requires_grad_(False)

    def _install_predictor(self, model):
        """Install an already validated CBM without reinitializing this module.

        Staging construction preserves the old graph on failure. Keeping this
        module's hook registries intact also preserves user forward/backward hooks.
        """
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
        self._check_fitted()
        task = self.task_names[0]
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
        obj = cls(None, backbone, metadata["task_name"], annotations.concept(metadata["task_name"]).states, device=device)
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
            metadata["feature_size"], obj.mean_.dtype,
        )
        obj._install_predictor(model)
        obj.load_state_dict(state)
        return obj.eval().requires_grad_(False)
