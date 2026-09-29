"""End-to-end label-free concept bottleneck learning on HAM10000.

Language in a Bottle (LaBo) uses an LLM to propose class-conditioned visual
concepts, selects a compact vocabulary with frozen CLIP features, and learns a
concept-to-class association matrix. "Label-free" refers to the concept axis:
diagnosis labels are still used to train and evaluate the final classifier.

This example makes the PyC workflow explicit:

``HAM10000Dataset -> LLMConceptGenerator -> LaBoConceptSelector ->``
``CLIPAnnotator -> LinearConceptToConcept -> training/evaluation``.

It follows LaBo's released HAM10000 experiment: the original split, original
RGB images passed directly to CLIP ViT-L/14 preprocessing, raw unnormalized
CLIP dot products, 350 selected concepts, and the published association-head
training settings. The language stage is the intentional difference: Gemini
2.5 Flash replaces LaBo's GPT-3 generation plus fine-tuned T5 extractor.

Download LaBo's three ``class2images_{train,val,test}.p`` files into
``<root>/labo_splits`` from:

https://github.com/YueYANG1996/LaBo/tree/main/datasets/HAM10000/splits

Generate and select concepts end to end:

.. code-block:: bash

    export GEMINI_API_KEY="..."
    python -m examples.utilization.4_label_free.1_labo_ham10000 \
        --root ./data/ham10000

Reuse generated candidates while repeating selection:

.. code-block:: bash

    python -m examples.utilization.4_label_free.1_labo_ham10000 \
        --root ./data/ham10000 \
        --candidate-vocabulary ./data/ham10000/labo_concepts.json

Or download LaBo's published HAM10000 vocabulary from
https://github.com/YueYANG1996/LaBo/tree/main/datasets/HAM10000, save it as
``./data/ham10000/original_labo_vocabulary.json``, and run:

.. code-block:: bash

    python -m examples.utilization.4_label_free.1_labo_ham10000 \
        --root ./data/ham10000 \
        --selected-vocabulary ./data/ham10000/original_labo_vocabulary.json
"""

import argparse
import copy
import json
import os
import pickle
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torch_concepts import Annotations, seed_everything
from torch_concepts.data import HAM10000Dataset
from torch_concepts.data.generation import FilterGenerator
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.generators import (
    LLMConceptGenerator,
    LiteLLMBackend,
)
from torch_concepts.nn import LinearConceptToConcept


CLIP_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_LLM_MODEL = "gemini/gemini-2.5-flash"
SPLIT_FILENAMES = {
    "train": "class2images_train.p",
    "validation": "class2images_val.p",
    "test": "class2images_test.p",
}
EXPECTED_SPLIT_SIZES = {"train": 8010, "validation": 1000, "test": 1005}
SPLIT_DOWNLOAD_URL = (
    "https://github.com/YueYANG1996/LaBo/tree/main/datasets/HAM10000/splits"
)
DIAGNOSIS_NAMES = {
    "akiec": "actinic keratoses",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevi",
    "vasc": "vascular lesions",
}
PROMPT_THEMES = (
    "what {diagnosis} looks like",
    "the appearance of {diagnosis}",
    "the color of {diagnosis}",
    "the pattern of {diagnosis}",
    "the shape of {diagnosis}",
)
CONCEPTS_PER_CLASS = 50  # Seven classes give LaBo's 350-concept budget.
CANDIDATES_PER_THEME = 100
MI_SCALE = 1e7
FACILITY_WEIGHT = 0.1
LEARNING_RATE = 5e-4
TRAIN_BATCH_SIZE = 256
VALIDATE_EVERY = 10
PUBLISHED_TEST_ACCURACY = 81.39


def diagnosis_targets(dataset: HAM10000Dataset) -> tuple[torch.Tensor, list[str]]:
    class_codes = list(dataset.annotations.get_label_states("diagnosis"))
    if set(class_codes) != set(DIAGNOSIS_NAMES):
        raise ValueError(f"Unexpected HAM10000 diagnoses: {class_codes}.")
    targets = dataset.native_concepts["diagnosis"].tensor.squeeze(-1)
    return targets.long(), class_codes


def load_labo_splits(
    dataset: HAM10000Dataset,
    splits_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve LaBo's published image filenames to PyC row indices."""
    index_by_id = {
        image_id: index
        for index, image_id in enumerate(dataset.metadata["image_id"].astype(str))
    }
    splits = {}
    for name, filename in SPLIT_FILENAMES.items():
        path = Path(splits_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Download the split files from "
                f"{SPLIT_DOWNLOAD_URL}."
            )
        with open(path, "rb") as file:
            class_to_images = pickle.load(file)
        image_ids = [
            Path(image_name).stem
            for class_images in class_to_images.values()
            for image_name in class_images
        ]
        missing = sorted(set(image_ids) - set(index_by_id))
        if missing:
            raise ValueError(f"{path} contains unknown image IDs: {missing[:3]}.")
        splits[name] = np.asarray([index_by_id[item] for item in image_ids])
        if len(splits[name]) != EXPECTED_SPLIT_SIZES[name]:
            raise ValueError(f"Unexpected {name} split size: {len(splits[name])}.")
    return splits["train"], splits["validation"], splits["test"]


class OriginalImageDataset(Dataset):
    """Return native RGB images so CLIP owns resizing and normalization."""

    def __init__(self, dataset: HAM10000Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Image.Image:
        path = os.path.join(self.dataset.root_dir, self.dataset.input_data[index])
        with Image.open(path) as image:
            return image.convert("RGB").copy()


def annotations_from_classes(
    concepts_by_class: dict[str, list[str]],
    class_codes: Sequence[str],
) -> Annotations:
    """Globally deduplicate strings, retaining the first source class."""
    labels = [
        label for code in class_codes for label in concepts_by_class.get(code, [])
    ]
    origins = [
        code for code in class_codes for _ in concepts_by_class.get(code, [])
    ]
    if not labels:
        raise ValueError("The concept vocabulary is empty.")
    unique_labels, first = np.unique(np.asarray(labels, dtype=str), return_index=True)
    unique_origins = np.asarray(origins, dtype=str)[first]
    concepts = Annotations(
        labels=unique_labels.tolist(),
        cardinalities=[1] * len(unique_labels),
    )
    for code in class_codes:
        members = unique_labels[unique_origins == code].tolist()
        if members:
            concepts.register_group(f"source:{code}", members)
    return concepts


def load_vocabulary(
    path: str,
    section: str,
    class_codes: Sequence[str],
) -> Annotations:
    with open(path) as file:
        payload = json.load(file)
    if section in payload:
        return Annotations.from_dict(payload[section])
    if section != "selected_concepts":
        raise ValueError(f"{path} does not contain {section!r}.")
    try:
        by_class = {
            code: list(payload[DIAGNOSIS_NAMES[code]]) for code in class_codes
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Expected a saved PyC vocabulary or LaBo's class-to-list JSON."
        ) from error
    return annotations_from_classes(by_class, class_codes)


def save_vocabulary(
    path: str | Path,
    candidates: Annotations,
    selected: Annotations | None = None,
) -> None:
    payload = {"candidate_concepts": candidates.to_dict()}
    if selected is not None:
        payload["selected_concepts"] = selected.to_dict()
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def clean_generated_concept(concept: str, code: str) -> str | None:
    concept = re.sub(r"\s+", " ", concept.strip().rstrip(".")).lower()
    diagnosis_tokens = set(re.findall(r"[a-z]+", DIAGNOSIS_NAMES[code])) - {
        "cell",
        "lesions",
        "like",
    }
    diagnosis_tokens.add(code)
    if not concept or set(re.findall(r"[a-z]+", concept)) & diagnosis_tokens:
        return None
    return concept


def generate_candidates(
    model: str,
    class_codes: Sequence[str],
    requests_per_class: int,
) -> Annotations:
    """Generate candidates through PyC using grouped LaBo prompt themes."""
    llm = LLMConceptGenerator(
        llm=LiteLLMBackend(
            model=model,
            temperature=0.0,
            max_tokens=8192,
            timeout=120.0,
            retry_on_rate_limit=True,
        )
    )
    theme_groups = np.array_split(PROMPT_THEMES, requests_per_class)
    concepts_by_class = {code: [] for code in class_codes}
    for class_number, code in enumerate(class_codes, start=1):
        diagnosis = DIAGNOSIS_NAMES[code]
        print(f"Generating concepts for {code} ({class_number}/{len(class_codes)})...")
        for themes in theme_groups:
            rendered = [theme.format(diagnosis=diagnosis) for theme in themes]
            requested = CANDIDATES_PER_THEME * len(themes)
            prompt = (
                "Create a CLIP-ready visual concept vocabulary for dermoscopic "
                f"images of {diagnosis}. Cover these aspects:\n- "
                + "\n- ".join(rendered)
                + f"\nReturn exactly {requested} distinct short, atomic visual "
                "phrases, one per line. Do not use the diagnosis name, headings, "
                "numbering, explanations, treatments, or non-visual facts."
            )
            try:
                generated = llm.generate(class_names=[diagnosis], prompt=prompt)
            except Exception as error:
                raise RuntimeError(
                    f"LLM generation failed for {code}; rerun the command."
                ) from error
            concepts_by_class[code].extend(
                cleaned
                for label in generated.labels[:requested]
                if (cleaned := clean_generated_concept(label, code)) is not None
            )
    return annotations_from_classes(concepts_by_class, class_codes)


def labo_greedy_select(features: np.ndarray, count: int) -> list[int]:
    """Reproduce patched Apricot 0.6.1's naive mixture ranking."""
    features = np.asarray(features, dtype=np.float64)
    coverage = np.zeros(features.shape[1], dtype=np.float64)
    remaining = list(range(len(features)))
    selected = []
    for _ in range(min(count, len(remaining))):
        candidates = np.asarray(remaining)
        facility_gain = (
            np.maximum(features[candidates], coverage).sum(axis=1)
            - coverage.sum()
        )
        gains = features[candidates, 0] + FACILITY_WEIGHT * facility_gain
        best = remaining.pop(int(np.argmax(gains)))
        coverage = np.maximum(coverage, features[best])
        selected.append(best)
    return selected


class LaBoConceptSelector(FilterGenerator):
    """Reproduce LaBo's released Apricot 0.6.1 selector.

    Apricot's mixture marks its components as precomputed, so facility coverage
    is applied directly to ``[scaled MI, CLIP feature]`` rather than to cosine
    similarities.
    """

    def __init__(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        train_indices: Sequence[int],
        targets: torch.Tensor,
        class_codes: Sequence[str],
    ):
        self.image_features = image_features[train_indices]
        self.text_features = text_features
        self.targets = targets[train_indices]
        self.class_codes = list(class_codes)

    def filter(self, concepts: Annotations) -> Annotations:
        mi_scores = self._mi_scores()
        selected_labels = []
        for code in self.class_codes:
            labels = list((concepts.groups or {}).get(f"source:{code}", []))
            if not labels:
                raise ValueError(f"No candidates remain for {code}.")
            indices = [concepts.get_index(label) for label in labels]
            if len(indices) < CONCEPTS_PER_CLASS:
                warnings.warn(f"{code} has only {len(indices)} unique candidates.")
            pool = torch.tensor(indices, dtype=torch.long)
            if len(indices) <= CONCEPTS_PER_CLASS:
                chosen = range(len(indices))
            else:
                # Scaled MI is both the modular term and a coverage coordinate.
                augmented = np.column_stack(
                    (
                        (mi_scores[pool] * MI_SCALE).numpy(),
                        self.text_features[pool].numpy(),
                    )
                )
                chosen = labo_greedy_select(augmented, CONCEPTS_PER_CLASS)
            selected_labels.extend(labels[index] for index in chosen)
        return concepts.subset(selected_labels)

    def _mi_scores(self) -> torch.Tensor:
        similarities = torch.empty(
            (len(self.text_features), len(self.class_codes))
        )
        for class_index in range(len(self.class_codes)):
            class_images = self.image_features[self.targets == class_index]
            similarities[:, class_index] = (
                self.text_features @ class_images.T
            ).mean(dim=1)

        num_classes = len(self.class_codes)
        normalized = similarities / (similarities.sum(dim=0) * num_classes)
        margins = normalized.sum(dim=1, keepdim=True)
        mi_scores = (normalized * torch.log(normalized / (margins / num_classes))).sum(
            dim=1
        )
        if not torch.isfinite(mi_scores).all():
            raise RuntimeError("LaBo's released MI formula produced non-finite values.")
        return mi_scores


class RowSoftmax(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.softmax(weight, dim=-1)


class LaBoHead(nn.Module):
    """Class-origin initialization followed by LaBo's row-wise softmax."""

    def __init__(self, concepts: Annotations, class_codes: Sequence[str]):
        super().__init__()
        self.association = LinearConceptToConcept(
            in_concepts=len(concepts.labels),
            out_concepts=len(class_codes),
            bias=False,
        )
        with torch.no_grad():
            self.association.predictor.weight.zero_()
            for class_index, code in enumerate(class_codes):
                for label in (concepts.groups or {}).get(f"source:{code}", []):
                    self.association.predictor.weight[
                        class_index, concepts.get_index(label)
                    ] = 1.0
        parametrize.register_parametrization(
            self.association.predictor, "weight", RowSoftmax()
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return 100.0 * self.association(scores)


def evaluate(
    model: nn.Module,
    scores: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(scores.to(device)).argmax(dim=1).cpu()
    return (predictions == targets).float().mean().item()


def train_head(
    model: LaBoHead,
    scores: torch.Tensor,
    targets: torch.Tensor,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[float, int]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(
        TensorDataset(scores[train_indices], targets[train_indices]),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        if epoch % VALIDATE_EVERY == 0 or epoch == epochs:
            accuracy = evaluate(
                model,
                scores[validation_indices],
                targets[validation_indices],
                device,
            )
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_accuracy, best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full-shot LaBo HAM10000 concept-bottleneck example."
    )
    parser.add_argument("--root", default="./data/ham10000")
    parser.add_argument("--labo-splits-dir", default=None)
    vocabulary = parser.add_mutually_exclusive_group()
    vocabulary.add_argument("--candidate-vocabulary")
    vocabulary.add_argument("--selected-vocabulary")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument(
        "--generation-requests-per-class",
        type=int,
        default=len(PROMPT_THEMES),
        help="Split LaBo's five prompt themes across 1 to 5 requests per class.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.generation_requests_per_class <= len(PROMPT_THEMES):
        raise ValueError("--generation-requests-per-class must be between 1 and 5.")
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive.")
    seed_everything(args.seed)

    dataset = HAM10000Dataset(root=args.root)
    images = OriginalImageDataset(dataset)
    targets, class_codes = diagnosis_targets(dataset)
    splits_dir = args.labo_splits_dir or Path(dataset.root_dir) / "labo_splits"
    train_indices, validation_indices, test_indices = load_labo_splits(
        dataset, splits_dir
    )
    concepts_path = Path(dataset.root_dir) / "labo_concepts.json"

    if args.selected_vocabulary:
        selected = load_vocabulary(
            args.selected_vocabulary, "selected_concepts", class_codes
        )
        candidates = None
        vocabulary_source = args.selected_vocabulary
    else:
        if args.candidate_vocabulary:
            candidates = load_vocabulary(
                args.candidate_vocabulary, "candidate_concepts", class_codes
            )
            vocabulary_source = args.candidate_vocabulary
        else:
            candidates = generate_candidates(
                args.llm_model, class_codes, args.generation_requests_per_class
            )
            save_vocabulary(concepts_path, candidates)
            vocabulary_source = args.llm_model

    print(f"Loading {CLIP_MODEL}...")
    clip = CLIPAnnotator(
        model_name=CLIP_MODEL,
        prompt_template="{}",
        batch_size=args.clip_batch_size,
        device=args.device,
        show_progress=True,
        normalize=False,  # Released LaBo uses raw image and text embeddings.
    )
    device = clip.device
    print(f"Using {device} for CLIP and association-head training.")
    image_features = clip.encode_dataset(images)

    if candidates is not None:
        candidate_features = clip.encode_concepts(candidates).cpu()
        selected = LaBoConceptSelector(
            image_features,
            candidate_features,
            train_indices,
            targets,
            class_codes,
        ).filter(candidates)
        selected_features = candidate_features[
            [candidates.get_index(label) for label in selected.labels]
        ]
        save_vocabulary(concepts_path, candidates, selected)
    else:
        selected_features = clip.encode_concepts(selected).cpu()

    concept_scores = clip.annotate(
        images,
        selected,
        image_features=image_features,
        concept_features=selected_features,
    ).tensor
    print(f"Training the association head for {args.train_epochs} epochs...")
    model = LaBoHead(selected, class_codes)
    best_validation, best_epoch = train_head(
        model,
        concept_scores,
        targets,
        train_indices,
        validation_indices,
        args.train_epochs,
        args.seed,
        device,
    )
    test_accuracy = evaluate(
        model, concept_scores[test_indices], targets[test_indices], device
    )

    print("\nLaBo HAM10000 summary")
    print(f"Vocabulary source: {vocabulary_source}")
    print(f"Selected concepts: {len(selected.labels)}")
    print("Split sizes: train=8010, validation=1000, test=1005")
    print(
        f"Best validation accuracy: {100 * best_validation:.2f}% "
        f"at epoch {best_epoch}"
    )
    print(f"Test accuracy: {100 * test_accuracy:.2f}%")
    print(f"Published LaBo test accuracy: {PUBLISHED_TEST_ACCURACY:.2f}%")


if __name__ == "__main__":
    main()
