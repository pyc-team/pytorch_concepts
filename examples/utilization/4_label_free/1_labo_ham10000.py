"""LaBo-style diagnosis classification on HAM10000.

This example follows Yang et al.'s Language in a Bottle (LaBo) workflow:
class-conditioned candidates are selected with CLIP and a submodular objective,
then frozen CLIP concept scores feed a learned diagnosis-association matrix.

Two intentional differences make the original 2023 setup usable today:
Gemini 2.5 Flash replaces retired GPT-3 ``text-davinci-002``, and Gemini is
asked for short atomic visual concepts directly instead of using LaBo's
fine-tuned T5-large sentence-to-concept extractor.

Usage:

    export GEMINI_API_KEY="..."
    python -m examples.utilization.4_label_free.1_labo_ham10000 \\
        --root ./data/ham10000 --shots full --clip-device cuda --train-device cuda
"""

import argparse
import copy
import json
import math
import os
import random
import re
import time
import warnings
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Subset, TensorDataset

from torch_concepts import Annotations
from torch_concepts.data import HAM10000Dataset
from torch_concepts.data.generation import (
    ConceptGenerationPipeline,
    FilterGenerator,
    Generator,
)
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.generators import LiteLLMBackend, LLMConceptGenerator
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept


DIAGNOSIS_DESCRIPTIONS = {
    "akiec": "actinic keratosis and intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesion",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}
LABO_PROMPT_THEMES = (
    "describe what the {class_name} looks like",
    "describe the appearance of the {class_name}",
    "describe the color of the {class_name}",
    "describe the pattern of the {class_name}",
    "describe the shape of the {class_name}",
)
LABO_HAM10000_CONFIG = {
    "1": {"num_concepts": 350, "alpha": 1e7, "beta": 0.1, "lr": 1e-3, "batch_size": 4},
    "2": {"num_concepts": 350, "alpha": 1e7, "beta": 0.1, "lr": 1e-3, "batch_size": 4},
    "4": {"num_concepts": 350, "alpha": 1e7, "beta": 1.0, "lr": 1e-4, "batch_size": 8},
    "8": {"num_concepts": 350, "alpha": 1e7, "beta": 10.0, "lr": 1e-3, "batch_size": 8},
    "16": {"num_concepts": 350, "alpha": 1e7, "beta": 15.0, "lr": 1e-3, "batch_size": 16},
    "full": {"num_concepts": 350, "alpha": 1e7, "beta": 0.1, "lr": 5e-4, "batch_size": 256},
}
LABO_PAPER_TEST_ACCURACY = {
    "1": 36.62,
    "2": 45.17,
    "4": 45.87,
    "8": 52.04,
    "16": 55.72,
    "full": 81.39,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def diagnosis_targets(dataset: HAM10000Dataset) -> tuple[torch.Tensor, list[str]]:
    """Return diagnosis indices and the declared categorical state order."""
    states = list(dataset.annotations.get_label_states("diagnosis"))
    if any(state not in DIAGNOSIS_DESCRIPTIONS for state in states):
        raise ValueError(f"Unsupported HAM10000 diagnosis states: {states}.")
    native = dataset.native_concepts["diagnosis"].tensor.squeeze(-1)
    if torch.isnan(native).any():
        raise ValueError("HAM10000 diagnosis values must not be missing for this experiment.")
    return native.long(), states


def labo_split(targets: torch.Tensor, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create LaBo's intentional image-level 8010/1000/1005 split."""
    if len(targets) != 10_015:
        raise ValueError(f"LaBo's HAM10000 split requires 10,015 images, got {len(targets)}.")
    labels = targets.cpu().numpy()
    all_indices = np.arange(len(labels))
    first_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=8010,
        test_size=2005,
        random_state=seed,
    )
    train_indices, held_out_indices = next(first_split.split(all_indices, labels))
    second_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=1000,
        test_size=1005,
        random_state=seed,
    )
    dev_local, test_local = next(
        second_split.split(held_out_indices, labels[held_out_indices])
    )
    dev_indices = held_out_indices[dev_local]
    test_indices = held_out_indices[test_local]
    return train_indices, dev_indices, test_indices


def few_shot_indices(
    train_indices: Sequence[int],
    targets: torch.Tensor,
    num_classes: int,
    shots: str,
    seed: int,
) -> np.ndarray:
    if shots == "full":
        return np.asarray(train_indices, dtype=int)

    examples_per_class = int(shots)
    random_state = np.random.default_rng(seed)
    train_indices = np.asarray(train_indices, dtype=int)
    selected = []
    targets_np = targets.cpu().numpy()
    for class_index in range(num_classes):
        class_indices = train_indices[targets_np[train_indices] == class_index]
        if len(class_indices) < examples_per_class:
            raise ValueError(
                f"Class {class_index} has only {len(class_indices)} training images; "
                f"cannot select {examples_per_class} shots."
            )
        selected.extend(random_state.choice(class_indices, examples_per_class, replace=False))
    return np.sort(np.asarray(selected, dtype=int))


def clean_class_leakage(concept: str, class_name: str, superclass: str = "lesion") -> str | None:
    """Apply LaBo-style human-readable class-name removal to one candidate."""
    concept = re.sub(r"\s+", " ", concept.strip().rstrip(".")).lower()
    if not concept:
        return None

    name_pattern = re.compile(rf"\b{re.escape(class_name.lower())}\b")
    concept = name_pattern.sub(superclass, concept)
    meaningful_tokens = [
        token
        for token in re.findall(r"[a-z]+", class_name.lower())
        if token not in {"and", "of", "the", "a", "an"}
    ]
    concept_tokens = set(re.findall(r"[a-z]+", concept))
    if len(meaningful_tokens) > 1 and all(token in concept_tokens for token in meaningful_tokens):
        return None
    if not (concept_tokens - {superclass}):
        return None
    return concept


class LaBoConceptGenerator(Generator):
    """Generate class-conditioned candidates and retain class provenance groups."""

    def __init__(
        self,
        llm: LiteLLMBackend,
        class_codes: Sequence[str],
        candidates_per_class: int,
        requests_per_class: int,
        service_retries: int,
        retry_delay: float,
        concepts_output: str,
    ):
        self.llm_generator = LLMConceptGenerator(llm=llm)
        self.class_codes = list(class_codes)
        self.candidates_per_class = candidates_per_class
        self.requests_per_class = requests_per_class
        self.service_retries = service_retries
        self.retry_delay = retry_delay
        self.concepts_output = concepts_output
        self.candidate_concepts: Annotations | None = None

    def generate(self, dataset=None, class_names=None, **kwargs) -> Annotations:
        del class_names, kwargs
        labels: list[str] = []
        labels_by_key: dict[str, str] = {}
        source_groups: dict[str, list[str]] = {}
        if not 1 <= self.requests_per_class <= len(LABO_PROMPT_THEMES):
            raise ValueError(
                "requests_per_class must be between 1 and "
                f"{len(LABO_PROMPT_THEMES)}."
            )
        theme_groups = np.array_split(LABO_PROMPT_THEMES, self.requests_per_class)

        for class_index, code in enumerate(self.class_codes, start=1):
            print(f"Generating candidates for {code} ({class_index}/{len(self.class_codes)})...")
            description = DIAGNOSIS_DESCRIPTIONS[code]
            class_name = (
                description
                if description.endswith(" lesion")
                else f"{description} lesion"
            )
            source_labels: list[str] = []
            for themes in theme_groups:
                themes = list(themes)
                candidate_count = math.ceil(
                    self.candidates_per_class * len(themes) / len(LABO_PROMPT_THEMES)
                )
                theme_text = "; ".join(
                    theme.format(class_name=class_name) for theme in themes
                )
                prompt = (
                    f"For a {class_name}, cover these visual aspects: {theme_text}. "
                    f"Return about {candidate_count} short atomic visual concepts useful for "
                    "recognizing this lesion. Blend all requested aspects in one flat list. "
                    f"Never include the diagnosis name '{description}' in a concept; describe "
                    "only visible properties. "
                    "Return one concept per line, with no headings, numbering, explanation, "
                    "diagnosis prediction, or non-visual epidemiological facts."
                )
                generated = self._generate_with_retry(dataset, class_name, prompt)
                for candidate in generated.labels:
                    cleaned = clean_class_leakage(candidate, description)
                    if cleaned is None:
                        continue
                    key = cleaned.casefold()
                    label = labels_by_key.get(key)
                    if label is None:
                        labels_by_key[key] = cleaned
                        labels.append(cleaned)
                        label = cleaned
                    if label not in source_labels:
                        source_labels.append(label)
                    if len(source_labels) >= self.candidates_per_class:
                        break
                if len(source_labels) >= self.candidates_per_class:
                    break
            if source_labels:
                source_groups[f"source:{code}"] = source_labels
                self.candidate_concepts = self._candidate_annotations(
                    labels, source_groups
                )
                save_concepts(self.concepts_output, self.candidate_concepts)
            else:
                warnings.warn(f"Gemini returned no usable candidates for {code}.")

        if self.candidate_concepts is None:
            raise ValueError("Gemini returned no usable HAM10000 visual concepts.")
        print(f"Generated {len(self.candidate_concepts.labels)} unique candidate concepts.")
        return self.candidate_concepts

    @staticmethod
    def _candidate_annotations(
        labels: Sequence[str], source_groups: dict[str, list[str]]
    ) -> Annotations:
        annotations = Annotations(
            labels=labels,
            states=[["0"] for _ in labels],
            cardinalities=[1] * len(labels),
            types=["binary"] * len(labels),
        )
        for source, members in source_groups.items():
            annotations.register_group(source, members)
        return annotations

    def _generate_with_retry(self, dataset, class_name: str, prompt: str) -> Annotations:
        for attempt in range(self.service_retries + 1):
            try:
                return self.llm_generator.generate(
                    dataset=dataset,
                    class_names=[class_name],
                    prompt=prompt,
                )
            except Exception as error:
                error_text = str(error).casefold()
                is_rate_limited = "429" in error_text or "resource_exhausted" in error_text
                is_transient = any(
                    marker in error_text
                    for marker in (
                        "503",
                        "unavailable",
                        "timeout",
                        "timed out",
                        "connection reset",
                        "peer closed connection",
                        "incomplete chunked read",
                    )
                )
                if is_rate_limited or not is_transient or attempt == self.service_retries:
                    raise
                delay = self.retry_delay * (2 ** attempt)
                print(f"LLM request failed transiently; retrying in {delay:.0f}s...")
                time.sleep(delay)
        raise RuntimeError("Unreachable retry loop termination.")


def save_concepts(
    path: str,
    candidates: Annotations,
    selected: Annotations | None = None,
) -> None:
    """Write generated concepts and provenance for later inspection."""
    payload = {
        "candidate_concepts": {
            "labels": candidates.labels,
            "source_groups": candidates.groups or {},
        },
    }
    if selected is not None:
        payload["selected_concepts"] = {
            "labels": selected.labels,
            "source_groups": selected.groups or {},
        }
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved generated concepts to {path}")


def print_concepts(concepts: Annotations, class_codes: Sequence[str]) -> None:
    """Print selected concepts grouped by the diagnosis that generated them."""
    groups = concepts.groups or {}
    for code in class_codes:
        labels = groups.get(f"source:{code}", [])
        print(f"\n{code} ({len(labels)} selected concepts)")
        for label in labels:
            print(f"  - {label}")


class LaBoConceptSelector(FilterGenerator):
    """LaBo discriminability plus facility-location selection on training rows."""

    def __init__(
        self,
        dataset: HAM10000Dataset,
        train_indices: Sequence[int],
        diagnosis_targets: torch.Tensor,
        class_codes: Sequence[str],
        clip_annotator: CLIPAnnotator,
        num_concepts: int,
        discriminability_weight: float,
        coverage_weight: float,
        image_batch_size: int,
        text_batch_size: int,
    ):
        self.dataset = dataset
        self.train_indices = list(train_indices)
        self.diagnosis_targets = diagnosis_targets.cpu()
        self.class_codes = list(class_codes)
        self.clip_annotator = clip_annotator
        self.num_concepts = num_concepts
        self.discriminability_weight = discriminability_weight
        self.coverage_weight = coverage_weight
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self._image_features: torch.Tensor | None = None
        self.selected_by_source: dict[str, list[str]] = {}

    def filter(self, concepts: Annotations) -> Annotations:
        if not concepts.groups:
            raise ValueError("LaBo concept selection requires source:* provenance groups.")
        print("Selecting concepts with CLIP features...")
        image_features = self._training_image_features()
        text_features = self._text_features(concepts.labels)
        alignment = self._class_concept_alignment(image_features, text_features)
        discriminability = self._mi_score(alignment)
        labels_to_index = concepts.label_to_index
        num_per_class = math.ceil(self.num_concepts / len(self.class_codes))
        selected_labels: list[str] = []
        seen = set()
        self.selected_by_source = {}

        for class_index, code in enumerate(self.class_codes):
            pool_labels = list(concepts.groups.get(f"source:{code}", []))
            if len(pool_labels) < num_per_class:
                warnings.warn(
                    f"{code} has {len(pool_labels)} candidates; selecting all instead of "
                    f"the requested {num_per_class}."
                )
            pool_indices = [labels_to_index[label] for label in pool_labels]
            chosen = self._greedy_select(
                pool_indices=pool_indices,
                text_features=text_features,
                discriminability=discriminability,
                count=min(num_per_class, len(pool_indices)),
            )
            source_selection = [concepts.labels[index] for index in chosen]
            self.selected_by_source[code] = source_selection
            for label in source_selection:
                if label not in seen:
                    seen.add(label)
                    selected_labels.append(label)

        if not selected_labels:
            raise ValueError("LaBo selection found no concepts.")
        return concepts.subset(selected_labels)

    def _training_image_features(self) -> torch.Tensor:
        if self._image_features is not None:
            return self._image_features

        loader = DataLoader(
            Subset(self.dataset, self.train_indices),
            batch_size=self.image_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        features = []
        for batch in loader:
            images = [sample["inputs"]["x"] for sample in batch]
            with torch.no_grad():
                features.append(self.clip_annotator.encode_images(images).cpu())
        self._image_features = torch.cat(features, dim=0)
        return self._image_features

    def _text_features(self, labels: Sequence[str]) -> torch.Tensor:
        features = []
        for start in range(0, len(labels), self.text_batch_size):
            with torch.no_grad():
                features.append(
                    self.clip_annotator.encode_texts(labels[start:start + self.text_batch_size]).cpu()
                )
        return torch.cat(features, dim=0)

    def _class_concept_alignment(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        train_targets = self.diagnosis_targets[self.train_indices]
        alignment = torch.empty((len(text_features), len(self.class_codes)))
        for class_index in range(len(self.class_codes)):
            class_features = image_features[train_targets == class_index]
            if not len(class_features):
                raise ValueError(f"No training images for class {class_index}.")
            alignment[:, class_index] = (class_features @ text_features.T).mean(dim=0)
        return alignment

    @staticmethod
    def _mi_score(scores_mean: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
        """LaBo's released discriminability calculation, with numerical guards."""
        num_classes = scores_mean.shape[1]
        denominator = scores_mean.sum(dim=0, keepdim=True) * num_classes
        denominator = torch.where(
            denominator.abs() < epsilon,
            torch.full_like(denominator, epsilon),
            denominator,
        )
        normalized_scores = scores_mean / denominator
        margin_x = normalized_scores.sum(dim=1, keepdim=True)
        ratio_denominator = margin_x * (1.0 / num_classes)
        ratio_denominator = torch.where(
            ratio_denominator.abs() < epsilon,
            torch.full_like(ratio_denominator, epsilon),
            ratio_denominator,
        )
        pmi = torch.log((normalized_scores / ratio_denominator).clamp_min(epsilon))
        return (normalized_scores * pmi).sum(dim=1)

    def _greedy_select(
        self,
        pool_indices: Sequence[int],
        text_features: torch.Tensor,
        discriminability: torch.Tensor,
        count: int,
    ) -> list[int]:
        if not pool_indices or count == 0:
            return []
        pool = torch.tensor(pool_indices, dtype=torch.long)
        pool_features = text_features[pool]
        similarity = pool_features @ pool_features.T
        coverage = torch.zeros(len(pool))
        remaining = list(range(len(pool)))
        selected: list[int] = []
        for _ in range(count):
            gains = []
            for candidate in remaining:
                coverage_gain = (torch.maximum(coverage, similarity[:, candidate]) - coverage).sum()
                score = (
                    self.discriminability_weight * discriminability[pool[candidate]]
                    + self.coverage_weight * coverage_gain
                )
                gains.append(float(score))
            best_position = max(range(len(remaining)), key=lambda index: gains[index])
            candidate = remaining.pop(best_position)
            coverage = torch.maximum(coverage, similarity[:, candidate])
            selected.append(int(pool[candidate]))
        return selected


class RowSoftmax(nn.Module):
    """Constrain each diagnosis row to a distribution over concepts."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.softmax(weight, dim=-1)


class LaBoCBM(nn.Module):
    """LaBo's frozen-score concept-to-diagnosis association head."""

    def __init__(self, concepts: Annotations, class_codes: Sequence[str]):
        super().__init__()
        self.linear = LinearConceptToConcept(
            in_concepts=len(concepts.labels),
            out_concepts=len(class_codes),
            bias=False,
        )
        provenance = concepts.groups or {}
        with torch.no_grad():
            self.linear.predictor.weight.zero_()
            for class_index, code in enumerate(class_codes):
                for label in provenance.get(f"source:{code}", []):
                    self.linear.predictor.weight[class_index, concepts.get_index(label)] = 1.0
        parametrize.register_parametrization(self.linear.predictor, "weight", RowSoftmax())

    def forward(self, concept_scores: torch.Tensor) -> torch.Tensor:
        return 100.0 * self.linear(concept_scores)


def accuracy(model: nn.Module, scores: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in DataLoader(TensorDataset(scores, targets), batch_size=1024):
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
    return correct / len(targets)


def train_labo_head(
    model: LaBoCBM,
    scores: torch.Tensor,
    targets: torch.Tensor,
    train_indices: Sequence[int],
    dev_indices: Sequence[int],
    batch_size: int,
    learning_rate: float,
    epochs: int,
    val_every: int,
    seed: int,
    device: torch.device,
) -> float:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_scores = scores[train_indices]
    train_targets = targets[train_indices]
    dev_scores = scores[dev_indices]
    dev_targets = targets[dev_indices]
    loader = DataLoader(
        TensorDataset(train_scores, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        if epoch % val_every == 0 or epoch == epochs:
            dev_accuracy = accuracy(model, dev_scores, dev_targets, device)
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--shots", choices=("1", "2", "4", "8", "16", "full"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm-model", default="gemini/gemini-2.5-flash")
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-service-retries", type=int, default=3)
    parser.add_argument("--llm-retry-delay", type=float, default=10.0)
    parser.add_argument("--candidates-per-class", type=int, default=500)
    parser.add_argument(
        "--concepts-output",
        default=None,
        help="JSON path for generated and selected concepts (default: <root>/labo_ham10000_concepts.json).",
    )
    parser.add_argument(
        "--print-concepts",
        action="store_true",
        help="Print the selected concepts, grouped by source diagnosis.",
    )
    parser.add_argument(
        "--generation-requests-per-class",
        type=int,
        default=2,
        help="Batch the five LaBo themes into this many Gemini calls per class.",
    )
    parser.add_argument("--clip-device", default=None)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--clip-text-batch-size", type=int, default=256)
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--train-epochs", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    config = LABO_HAM10000_CONFIG[args.shots]
    dataset = HAM10000Dataset(root=args.root)
    concepts_output = args.concepts_output or os.path.join(
        dataset.root_dir, "labo_ham10000_concepts.json"
    )
    targets, class_codes = diagnosis_targets(dataset)
    train_indices, dev_indices, test_indices = labo_split(targets, args.seed)
    # Image-level splitting intentionally reproduces LaBo rather than clinical lesion-level evaluation.
    effective_train_indices = few_shot_indices(
        train_indices, targets, len(class_codes), args.shots, args.seed
    )

    llm = LiteLLMBackend(
        model=args.llm_model,
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
    )
    generator = LaBoConceptGenerator(
        llm=llm,
        class_codes=class_codes,
        candidates_per_class=args.candidates_per_class,
        requests_per_class=args.generation_requests_per_class,
        service_retries=args.llm_service_retries,
        retry_delay=args.llm_retry_delay,
        concepts_output=concepts_output,
    )
    print("Loading CLIP ViT-L/14 on the selected device...")
    clip_annotator = CLIPAnnotator(
        model_name="openai/clip-vit-large-patch14",
        prompt_template="{}",
        batch_size=args.clip_batch_size,
        device=args.clip_device,
        show_progress=True,
    )
    print("CLIP is ready; generating candidate concepts...")
    selector = LaBoConceptSelector(
        dataset=dataset,
        train_indices=effective_train_indices,
        diagnosis_targets=targets,
        class_codes=class_codes,
        clip_annotator=clip_annotator,
        num_concepts=config["num_concepts"],
        discriminability_weight=config["alpha"],
        coverage_weight=config["beta"],
        image_batch_size=args.clip_batch_size,
        text_batch_size=args.clip_text_batch_size,
    )
    # One route is equivalent to merged routing here; cartesian retains provenance groups.
    pipeline = ConceptGenerationPipeline(
        generators=generator,
        annotators=clip_annotator,
        generator_filter=selector,
        routing="cartesian",
    )
    generated = pipeline(
        dataset,
        class_names=[DIAGNOSIS_DESCRIPTIONS[code] for code in class_codes],
        generation_indices=effective_train_indices,
    )
    if len(generated) != 1:
        raise RuntimeError(f"Expected one CLIP output, received {list(generated)}.")
    concept_scores = next(iter(generated.values()))
    selected_concepts = concept_scores.annotation
    if generator.candidate_concepts is None:
        raise RuntimeError("The candidate concept vocabulary was not generated.")
    save_concepts(concepts_output, generator.candidate_concepts, selected_concepts)
    if args.print_concepts:
        print_concepts(selected_concepts, class_codes)
    model = LaBoCBM(selected_concepts, class_codes)
    train_device = resolve_device(args.train_device)
    best_dev_accuracy = train_labo_head(
        model=model,
        scores=concept_scores.tensor,
        targets=targets,
        train_indices=effective_train_indices,
        dev_indices=dev_indices,
        batch_size=config["batch_size"],
        learning_rate=config["lr"],
        epochs=args.train_epochs,
        val_every=args.val_every,
        seed=args.seed,
        device=train_device,
    )
    test_accuracy = accuracy(
        model,
        concept_scores.tensor[test_indices],
        targets[test_indices],
        train_device,
    )
    source_counts = {
        code: len(selected_concepts.groups.get(f"source:{code}", []))
        for code in class_codes
    }
    print(f"Best validation accuracy: {best_dev_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Selected concepts: {len(selected_concepts.labels)}")
    print(f"Selected concepts by source class: {source_counts}")
    print(
        f"LaBo paper reference for {args.shots}-shot HAM10000: "
        f"{LABO_PAPER_TEST_ACCURACY[args.shots]:.2f}%"
    )


if __name__ == "__main__":
    main()
