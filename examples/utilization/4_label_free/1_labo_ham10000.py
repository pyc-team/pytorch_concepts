"""LaBo-style label-free concept bottleneck learning on HAM10000.

Language in a Bottle (LaBo) builds an interpretable classifier without manual
concept annotations. A language model proposes visual concepts for every
class, a frozen CLIP model selects and annotates those concepts, and a small
association matrix learns how concepts predict classes. The bottleneck is
"label-free" because its concept vocabulary and image-level concept scores are
created automatically; diagnosis labels are still used to train and evaluate
the final classifier.

This example expresses that workflow with PyC components:

1. :class:`HAM10000Dataset` provides images, diagnoses, and metadata.
2. :class:`LLMConceptGenerator` with :class:`LiteLLMBackend` generates short,
   class-conditioned visual concepts and stores class provenance in
   :class:`Annotations` groups.
3. :class:`LaBoConceptSelector`, a :class:`FilterGenerator`, reproduces LaBo's
   released MI/facility-location objective and chooses 50 concepts per class.
4. A local :class:`CLIPAnnotator` specialization assigns frozen CLIP scores to
   every image through :class:`ConceptGenerationPipeline`.
5. :class:`LinearConceptToConcept` learns LaBo's row-softmax association matrix.

The split, CLIP ViT-L/14 model, native CLIP image preprocessing, raw
unnormalized embeddings, selector objective, association head, and full-shot
training settings follow the released HAM10000 experiment.

Selector overview
-----------------
For concept ``c`` and diagnosis ``y``, the selector averages raw CLIP dot
products over training images of ``y`` and applies LaBo's released MI score.
For each originating class it then augments every text embedding with its MI
score scaled by ``1e7``. Naive greedy selection maximizes

``scaled_MI(S) + 0.1 * sum_i max_{j in S} similarity(i, j)``.

The classifier learns one association row per diagnosis. Each row is
initialized to favor concepts generated for that diagnosis, transformed with
a softmax, and applied to the CLIP concept scores; the resulting logits are
scaled by 100. With LaBo's published vocabulary, split, preprocessing, and
training schedule, our established regression reference is 81.40% validation
and 80.60% test accuracy. We get 81% validation and 79.90% test accuracy with
this example's PyC implementation of the same workflow.

Setup and usage
---------------
Place LaBo's ``class2images_{train,val,test}.p`` files in
``<root>/labo_splits``. They are available from:

https://github.com/YueYANG1996/LaBo/tree/main/datasets/HAM10000/splits

For the complete label-free pipeline, export a Google AI Studio
``GEMINI_API_KEY`` and run:

.. code-block:: bash

    python -m examples.utilization.4_label_free.1_labo_ham10000 \
        --root ./data/ham10000

Generated candidates are checkpointed to ``<root>/labo_concepts.json`` before
selection. Each successful class/theme request is recorded, so rerunning the
same model and request grouping resumes an interrupted generation. Reuse a
completed candidate file explicitly with:

.. code-block:: bash

    python -m examples.utilization.4_label_free.1_labo_ham10000 \
        --root ./data/ham10000 \
        --candidate-vocabulary ./data/ham10000/labo_concepts.json

To reproduce only the classifier with LaBo's published 350-concept vocabulary,
download its HAM10000 vocabulary from the original LaBo repository:

https://github.com/YueYANG1996/LaBo/tree/main/datasets/HAM10000

Save it as ``./data/ham10000/original_labo_vocabulary.json``, then skip
generation and selection with:

.. code-block:: bash

    python -m examples.utilization.4_label_free.1_labo_ham10000 \
        --root ./data/ham10000 \
        --selected-vocabulary ./data/ham10000/original_labo_vocabulary.json

The published vocabulary may use LaBo's class-to-list JSON format. Candidate
and selected inputs may also use the JSON written by this example.

``--generation-requests-per-class`` divides the ordered prompt themes into
that many disjoint, consecutive groups. The default of five makes one request
per theme; two requests create groups of three and two themes.
"""

import argparse
import copy
import json
import math
import os
import pickle
import random
import re
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torch_concepts import AnnotatedTensor, Annotations
from torch_concepts.data import HAM10000Dataset
from torch_concepts.data.generation import (
    ConceptGenerationPipeline,
    FilterGenerator,
    Generator,
)
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.generators import (
    LLMConceptGenerator,
    LiteLLMBackend,
)
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept


CLIP_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_LLM_MODEL = "gemini/gemini-2.5-flash"
SPLIT_FILENAMES = {
    "train": "class2images_train.p",
    "validation": "class2images_val.p",
    "test": "class2images_test.p",
}
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
NUM_CONCEPTS = 350
CONCEPTS_PER_CLASS = math.ceil(NUM_CONCEPTS / len(DIAGNOSIS_NAMES))
CANDIDATES_PER_THEME = 100
GENERATION_PROMPT_VERSION = 2
MAX_GENERATED_CONCEPT_WORDS = 8
MI_SCALE = 1e7
FACILITY_WEIGHT = 0.1
LEARNING_RATE = 5e-4
TRAIN_BATCH_SIZE = 256
VALIDATE_EVERY = 10
PUBLISHED_TEST_ACCURACY = 81.39


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
    """Read targets in the diagnosis order declared by the PyC dataset."""
    class_codes = list(dataset.annotations.get_label_states("diagnosis"))
    if set(class_codes) != set(DIAGNOSIS_NAMES):
        raise ValueError(f"Unexpected HAM10000 diagnosis states: {class_codes}.")
    targets = dataset.native_concepts["diagnosis"].tensor.squeeze(-1)
    return targets.long(), class_codes


def load_labo_splits(
    dataset: HAM10000Dataset,
    splits_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map LaBo's published image filenames to PyC dataset row indices."""
    metadata_ids = dataset.metadata["image_id"].astype(str).tolist()
    index_by_id = {image_id: index for index, image_id in enumerate(metadata_ids)}
    split_indices: dict[str, np.ndarray] = {}

    for split_name, filename in SPLIT_FILENAMES.items():
        path = Path(splits_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing LaBo split file: {path}. Download the three split files "
                f"from {SPLIT_DOWNLOAD_URL} into {splits_dir}."
            )
        with open(path, "rb") as file:
            class_to_images = pickle.load(file)

        image_names = [
            image_name
            for class_images in class_to_images.values()
            for image_name in class_images
        ]
        image_ids = [Path(image_name).stem for image_name in image_names]
        unresolved = sorted(set(image_ids) - set(index_by_id))
        if unresolved:
            raise ValueError(
                f"{path} contains unknown HAM10000 images, "
                f"for example {unresolved[:3]}."
            )
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"{path} contains duplicate image IDs.")
        split_indices[split_name] = np.asarray(
            [index_by_id[image_id] for image_id in image_ids], dtype=int
        )

    split_sets = {name: set(indices) for name, indices in split_indices.items()}
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise ValueError("LaBo train, validation, and test splits overlap.")
    if set.union(*split_sets.values()) != set(range(len(dataset))):
        raise ValueError("LaBo splits must cover all 10,015 HAM10000 images.")

    sizes = {name: len(indices) for name, indices in split_indices.items()}
    if sizes != {"train": 8010, "validation": 1000, "test": 1005}:
        raise ValueError(f"Unexpected LaBo split sizes: {sizes}.")
    return (
        split_indices["train"],
        split_indices["validation"],
        split_indices["test"],
    )


class OriginalImageDataset(Dataset):
    """Bypass PyC's resize so CLIP owns resizing, cropping, and normalization."""

    def __init__(self, dataset: HAM10000Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Image.Image:
        path = os.path.join(self.dataset.root_dir, self.dataset.input_data[index])
        with Image.open(path) as image:
            return image.convert("RGB").copy()


def annotations_from_class_concepts(
    concepts_by_class: dict[str, list[str]],
    class_codes: Sequence[str],
) -> Annotations:
    """Globally deduplicate concepts and retain their first originating class.

    LaBo calls ``numpy.unique(..., return_index=True)`` during preprocessing.
    This sorts unique strings and assigns a duplicate to its first class in the
    flattened class order.
    """
    labels = []
    origins = []
    for code in class_codes:
        for label in concepts_by_class.get(code, []):
            if not isinstance(label, str) or not label:
                raise ValueError(f"Invalid concept for diagnosis {code!r}: {label!r}.")
            labels.append(label)
            origins.append(code)
    if not labels:
        raise ValueError("The concept vocabulary is empty.")

    unique_labels, first_indices = np.unique(
        np.asarray(labels, dtype=str), return_index=True
    )
    unique_origins = np.asarray(origins, dtype=str)[first_indices]
    annotations = Annotations(
        labels=unique_labels.tolist(),
        states=[["0"] for _ in unique_labels],
        cardinalities=[1] * len(unique_labels),
        types=["binary"] * len(unique_labels),
    )
    for code in class_codes:
        members = unique_labels[unique_origins == code].tolist()
        if members:
            annotations.register_group(f"source:{code}", members)
    return annotations


def _class_concepts_from_json(
    payload: dict[str, Any],
    section: str,
    class_codes: Sequence[str],
) -> dict[str, list[str]]:
    """Accept LaBo's class-to-list JSON or this example's saved JSON."""
    if section in payload:
        saved = payload[section]
        labels = saved.get("labels")
        groups = saved.get("source_groups")
        if not isinstance(labels, list) or not isinstance(groups, dict):
            raise ValueError(f"Malformed {section} in vocabulary JSON.")
        concepts_by_class = {
            code: list(groups.get(f"source:{code}", [])) for code in class_codes
        }
        if any(
            label not in labels
            for values in concepts_by_class.values()
            for label in values
        ):
            raise ValueError(f"{section} provenance refers to an unknown concept.")
        grouped_labels = {
            label for values in concepts_by_class.values() for label in values
        }
        if grouped_labels != set(labels):
            raise ValueError(f"Every {section} label must have class provenance.")
        return concepts_by_class

    if not all(isinstance(value, list) for value in payload.values()):
        raise ValueError(
            f"Expected {section} or a LaBo class-to-concept-list JSON object."
        )
    concepts_by_class = {}
    for code in class_codes:
        class_name = DIAGNOSIS_NAMES[code]
        if class_name in payload:
            concepts_by_class[code] = list(payload[class_name])
        elif code in payload:
            concepts_by_class[code] = list(payload[code])
        else:
            raise ValueError(f"Vocabulary is missing concepts for {class_name!r}.")
    return concepts_by_class


def load_vocabulary(
    path: str,
    section: str,
    class_codes: Sequence[str],
) -> Annotations:
    with open(path) as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Vocabulary JSON must contain an object.")
    concepts_by_class = _class_concepts_from_json(payload, section, class_codes)
    return annotations_from_class_concepts(concepts_by_class, class_codes)


def save_vocabulary(
    path: str,
    candidates: Annotations,
    selected: Annotations | None = None,
    generation: dict[str, Any] | None = None,
) -> None:
    payload = {
        "candidate_concepts": {
            "labels": candidates.labels,
            "source_groups": candidates.groups or {},
        }
    }
    if selected is not None:
        payload["selected_concepts"] = {
            "labels": selected.labels,
            "source_groups": selected.groups or {},
        }
    if generation is not None:
        payload["generation"] = generation
    output_path = Path(path)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with open(temporary_path, "w") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary_path, output_path)


def clean_generated_concept(concept: str, code: str) -> str | None:
    """Normalize one concept and reject generation artifacts or leakage."""
    concept = re.sub(r"^\s*\d+[.)-]?\s+", "", concept)
    concept = re.sub(r"\s+", " ", concept.strip().rstrip(".")).lower()
    if not concept:
        return None
    if len(concept.split()) > MAX_GENERATED_CONCEPT_WORDS:
        return None
    if any(mark in concept for mark in ":;!?"):
        return None
    if re.search(
        r"\b(concepts?|phrases?|headings?|numbering|explanations?|"
        r"demographics?|treatments?|user|requirement|distinctness|enumerate|"
        r"generate|generating|listed visual aspects)\b",
        concept,
    ):
        return None
    if re.match(
        r"^(we|i|let(?:'s| us)|here|these|those|they|you|but|so|now|first|"
        r"second|third|aspect|category|spread)\b",
        concept,
    ):
        return None
    diagnosis_tokens = set(re.findall(r"[a-z]+", DIAGNOSIS_NAMES[code])) - {
        "cell",
        "lesions",
        "like",
    }
    if set(re.findall(r"[a-z]+", concept)) & diagnosis_tokens:
        return None
    return concept


class LaBoCandidateGenerator(Generator):
    """Generate resumable visual candidates with LaBo's five prompt themes."""

    def __init__(
        self,
        llm: LiteLLMBackend,
        class_codes: Sequence[str],
        output_path: str,
        requests_per_class: int,
    ):
        if not 1 <= requests_per_class <= len(PROMPT_THEMES):
            raise ValueError(
                "--generation-requests-per-class must be between 1 and "
                f"{len(PROMPT_THEMES)}."
            )
        self.generator = LLMConceptGenerator(llm=llm)
        self.model = llm.model
        self.class_codes = list(class_codes)
        self.output_path = output_path
        self.theme_groups = [
            tuple(group.tolist())
            for group in np.array_split(PROMPT_THEMES, requests_per_class)
        ]
        self.concepts: Annotations | None = None
        self.completed_requests: set[tuple[str, tuple[str, ...]]] = set()

    @property
    def generation_metadata(self) -> dict[str, Any]:
        return {
            "prompt_version": GENERATION_PROMPT_VERSION,
            "model": self.model,
            "candidates_per_theme": CANDIDATES_PER_THEME,
            "requests_per_class": len(self.theme_groups),
            "completed_requests": [
                {"class_code": code, "themes": list(themes)}
                for code in self.class_codes
                for themes in self.theme_groups
                if (code, themes) in self.completed_requests
            ],
        }

    def generate(self, dataset=None, class_names=None, **kwargs) -> Annotations:
        del dataset, class_names, kwargs
        concepts_by_class = self._load_checkpoint()
        for class_index, code in enumerate(self.class_codes, start=1):
            diagnosis = DIAGNOSIS_NAMES[code]
            class_concepts = concepts_by_class[code]
            print(
                f"Generating concepts for {code} "
                f"({class_index}/{len(self.class_codes)})..."
            )
            for group_index, themes in enumerate(self.theme_groups, start=1):
                if (code, themes) in self.completed_requests:
                    continue
                rendered_themes = [
                    theme.format(diagnosis=diagnosis) for theme in themes
                ]
                requested_concepts = CANDIDATES_PER_THEME * len(themes)
                prompt = (
                    "Create a CLIP-ready visual concept vocabulary for dermoscopic "
                    "images.\n"
                    f"Diagnosis: {diagnosis}\n"
                    "Visual aspects:\n- "
                    + "\n- ".join(rendered_themes)
                    + "\n"
                    f"Return exactly {requested_concepts} distinct concepts, spread "
                    "across the listed visual aspects. Each "
                    "concept must be one short, atomic phrase describing a single "
                    "visible property. Write one concept per line and nothing else. "
                    "Do not use the diagnosis name, headings, numbering, explanations, "
                    "treatments, demographics, or non-visual facts."
                )
                print(
                    f"  Requesting theme group {group_index}/{len(self.theme_groups)} "
                    f"({len(themes)} themes)..."
                )
                theme_concepts, returned_count = self._generate_with_retry(
                    code,
                    diagnosis,
                    rendered_themes,
                    requested_concepts,
                    prompt,
                )
                class_concepts.extend(theme_concepts)
                self.completed_requests.add((code, themes))
                self.concepts = annotations_from_class_concepts(
                    concepts_by_class, self.class_codes
                )
                save_vocabulary(
                    self.output_path,
                    self.concepts,
                    generation=self.generation_metadata,
                )
                print(
                    f"  Theme group {group_index}: {returned_count} returned, "
                    f"{len(set(class_concepts))} unique for {code} so far"
                )
            if len(set(class_concepts)) < CONCEPTS_PER_CLASS:
                warnings.warn(
                    f"{code} produced fewer than {CONCEPTS_PER_CLASS} unique concepts."
                )

        if self.concepts is None:
            raise RuntimeError("Candidate generation produced no concepts.")
        save_vocabulary(
            self.output_path,
            self.concepts,
            generation=self.generation_metadata,
        )
        print(f"Saved generated candidates to {self.output_path}")
        return self.concepts

    def _load_checkpoint(self) -> dict[str, list[str]]:
        concepts_by_class = {code: [] for code in self.class_codes}
        path = Path(self.output_path)
        if not path.is_file():
            return concepts_by_class

        with open(path) as file:
            payload = json.load(file)
        expected = {
            "prompt_version": GENERATION_PROMPT_VERSION,
            "model": self.model,
            "candidates_per_theme": CANDIDATES_PER_THEME,
            "requests_per_class": len(self.theme_groups),
        }
        metadata = payload.get("generation")
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            print(f"Ignoring incompatible generation checkpoint: {path}")
            return concepts_by_class

        concepts = _class_concepts_from_json(
            payload,
            "candidate_concepts",
            self.class_codes,
        )
        loaded_count = sum(len(values) for values in concepts.values())
        concepts = {
            code: [
                cleaned
                for candidate in candidates
                if (cleaned := clean_generated_concept(candidate, code)) is not None
            ]
            for code, candidates in concepts.items()
        }
        rejected_count = loaded_count - sum(
            len(values) for values in concepts.values()
        )
        if rejected_count:
            print(
                f"Ignoring {rejected_count} malformed generated concepts from {path}"
            )
        completed = metadata.get("completed_requests", [])
        self.completed_requests = {
            (item["class_code"], tuple(item["themes"]))
            for item in completed
            if isinstance(item, dict)
            and item.get("class_code") in self.class_codes
            and tuple(item.get("themes", [])) in self.theme_groups
        }
        print(
            f"Resuming {len(self.completed_requests)}/"
            f"{len(self.class_codes) * len(self.theme_groups)} "
            f"completed LLM requests from {path}"
        )
        if any(concepts.values()):
            self.concepts = annotations_from_class_concepts(
                concepts,
                self.class_codes,
            )
        return concepts

    def _generate_with_retry(
        self,
        code: str,
        diagnosis: str,
        themes: Sequence[str],
        requested_concepts: int,
        prompt: str,
    ) -> tuple[list[str], int]:
        for attempt in range(2):
            try:
                generated = self.generator.generate(
                    class_names=[diagnosis],
                    prompt=prompt,
                )
                cleaned = []
                for candidate in generated.labels[:requested_concepts]:
                    concept = clean_generated_concept(candidate, code)
                    if concept is not None:
                        cleaned.append(concept)
                if not cleaned:
                    raise RuntimeError("LLM returned no usable visual concepts.")
                return cleaned, len(generated.labels)
            except Exception as error:
                if attempt == 1:
                    raise RuntimeError(
                        f"LLM generation failed for {code} ({', '.join(themes)}). "
                        "Rerun the "
                        "same command to resume from the last checkpoint."
                    ) from error
                print("LLM request failed; retrying in 10 seconds...")
                time.sleep(10)
        raise RuntimeError("Unreachable retry loop termination.")


class FixedConceptGenerator(Generator):
    """Expose a loaded vocabulary through PyC's generator interface."""

    def __init__(self, concepts: Annotations):
        self.concepts = concepts

    def generate(self, dataset=None, class_names=None, **kwargs) -> Annotations:
        del dataset, class_names, kwargs
        return self.concepts


class RawCLIPAnnotator(CLIPAnnotator):
    """Use frozen raw embeddings instead of CLIPAnnotator's normalized ones."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_cache: dict[str, torch.Tensor] = {}
        self._cached_dataset: Dataset | None = None
        self._cached_image_features: torch.Tensor | None = None

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        texts = list(texts)
        missing = list(
            dict.fromkeys(text for text in texts if text not in self._text_cache)
        )
        if missing:
            max_length = self.model.config.text_config.max_position_embeddings
            inputs = self.processor(
                text=missing,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.no_grad():
                features = self.model.get_text_features(**inputs).cpu()
            self._text_cache.update(zip(missing, features))
        return torch.stack([self._text_cache[text] for text in texts]).to(self.device)

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=pixel_values)

    def encode_dataset(self, dataset: Dataset) -> torch.Tensor:
        if self._cached_dataset is dataset and self._cached_image_features is not None:
            return self._cached_image_features

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )
        batches = self._progress(
            loader,
            desc="CLIP image encoding",
            total=len(loader),
        )
        features = []
        for batch in batches:
            images = [self.input_getter(sample) for sample in batch]
            features.append(self.encode_images(images).cpu())
        self._cached_dataset = dataset
        self._cached_image_features = torch.cat(features)
        return self._cached_image_features

    def annotate(
        self,
        dataset: Dataset,
        concepts: Annotations,
        **kwargs,
    ) -> AnnotatedTensor:
        del kwargs
        text_prompts = self._flatten_concept_prompts(concepts)
        text_features = self.encode_texts(text_prompts)
        image_features = self.encode_dataset(dataset)
        score_batches = []
        for batch in image_features.split(self.batch_size):
            with torch.no_grad():
                score_batches.append((batch.to(self.device) @ text_features.T).cpu())
        return AnnotatedTensor(torch.cat(score_batches), concepts, axis=1)


def apricot_cosine_similarity(features: np.ndarray) -> np.ndarray:
    """Reproduce Apricot's cosine-distance-to-similarity transformation.

    Apricot converts cosine distance to a non-negative graph by applying its
    two transformations in sequence; algebraically the result is ``cosine²``.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(
        features,
        norms,
        out=np.zeros_like(features, dtype=np.float64),
        where=norms != 0,
    )
    return np.square(normalized @ normalized.T)


def naive_mixture_select(
    augmented_features: np.ndarray,
    count: int,
) -> list[int]:
    """Naively maximize LaBo's modular-MI plus facility-location objective."""
    similarity = apricot_cosine_similarity(augmented_features)
    coverage = np.zeros(len(augmented_features), dtype=np.float64)
    remaining = list(range(len(augmented_features)))
    selected = []
    for _ in range(min(count, len(remaining))):
        candidates = np.asarray(remaining)
        facility_gains = (
            np.maximum(similarity[candidates], coverage).sum(axis=1)
            - coverage.sum()
        )
        gains = augmented_features[candidates, 0] + FACILITY_WEIGHT * facility_gains
        best_position = int(np.argmax(gains))
        best = remaining.pop(best_position)
        coverage = np.maximum(coverage, similarity[best])
        selected.append(best)
    return selected


def check_selector_fixture() -> None:
    """Check the Apricot transform and deterministic naive-greedy behavior."""
    facility_features = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    expected_similarity = np.asarray(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
    )
    if not np.allclose(
        apricot_cosine_similarity(facility_features), expected_similarity
    ):
        raise RuntimeError("Apricot cosine-similarity fixture failed.")
    if naive_mixture_select(facility_features, 2) != [2, 0]:
        raise RuntimeError("LaBo naive-greedy fixture failed.")

    modular_features = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    if naive_mixture_select(modular_features, 1) != [2]:
        raise RuntimeError("LaBo modular-MI fixture failed.")


class LaBoConceptSelector(FilterGenerator):
    """Faithful PyC implementation of LaBo's released ``submodular_select``."""

    def __init__(
        self,
        dataset: Dataset,
        train_indices: Sequence[int],
        targets: torch.Tensor,
        class_codes: Sequence[str],
        clip: RawCLIPAnnotator,
        text_batch_size: int = 256,
    ):
        self.dataset = dataset
        self.train_indices = np.asarray(train_indices)
        self.targets = targets.cpu()
        self.class_codes = list(class_codes)
        self.clip = clip
        self.text_batch_size = text_batch_size

    def filter(self, concepts: Annotations) -> Annotations:
        image_features = self.clip.encode_dataset(self.dataset)[self.train_indices]
        text_features = self._encode_texts(concepts.labels)
        mi_scores = self._mi_scores(image_features, text_features)
        label_to_index = concepts.label_to_index
        selected_labels = []

        for code in self.class_codes:
            pool_labels = list((concepts.groups or {}).get(f"source:{code}", []))
            if not pool_labels:
                raise ValueError(f"No generated candidates remain for {code}.")
            pool_indices = [label_to_index[label] for label in pool_labels]
            if len(pool_indices) < CONCEPTS_PER_CLASS:
                warnings.warn(
                    f"{code} has only {len(pool_indices)} unique candidates; "
                    "selecting all."
                )
            pool = torch.tensor(pool_indices, dtype=torch.long)
            if len(pool_indices) <= CONCEPTS_PER_CLASS:
                chosen = list(range(len(pool_indices)))
            else:
                # LaBo puts scaled MI inside the vectors used for facility location.
                augmented = torch.column_stack(
                    (mi_scores[pool] * MI_SCALE, text_features[pool])
                ).double().numpy()
                chosen = naive_mixture_select(augmented, CONCEPTS_PER_CLASS)
            selected_labels.extend(
                concepts.labels[pool_indices[index]] for index in chosen
            )

        return concepts.subset(selected_labels)

    def _encode_texts(self, labels: Sequence[str]) -> torch.Tensor:
        batches = []
        for start in range(0, len(labels), self.text_batch_size):
            batches.append(
                self.clip.encode_texts(
                    labels[start : start + self.text_batch_size]
                ).cpu()
            )
        return torch.cat(batches)

    def _mi_scores(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        train_targets = self.targets[self.train_indices]
        scores_mean = torch.empty((len(text_features), len(self.class_codes)))
        for class_index in range(len(self.class_codes)):
            class_images = image_features[train_targets == class_index]
            scores_mean[:, class_index] = (
                text_features @ class_images.T
            ).mean(dim=1)

        num_classes = len(self.class_codes)
        normalized_scores = scores_mean / (scores_mean.sum(dim=0) * num_classes)
        margins = normalized_scores.sum(dim=1, keepdim=True)
        pmi = torch.log(normalized_scores / (margins / num_classes))
        mi_scores = (normalized_scores * pmi).sum(dim=1)
        if not torch.isfinite(mi_scores).all():
            raise RuntimeError(
                "LaBo's original MI formula produced non-finite values. The formula is "
                "left unclamped to preserve the released objective."
            )
        return mi_scores


class RowSoftmax(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.softmax(weight, dim=-1)


class LaBoCBM(nn.Module):
    """LaBo's class-origin-initialized, row-softmax association head."""

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
                    concept_index = concepts.get_index(label)
                    self.association.predictor.weight[class_index, concept_index] = 1.0
        parametrize.register_parametrization(
            self.association.predictor,
            "weight",
            RowSoftmax(),
        )

    def forward(self, concept_scores: torch.Tensor) -> torch.Tensor:
        return 100.0 * self.association(concept_scores)


def accuracy(
    model: nn.Module,
    scores: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        loader = DataLoader(TensorDataset(scores, targets), batch_size=1024)
        for inputs, labels in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
    return correct / len(targets)


def train_association_head(
    model: LaBoCBM,
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
    validation_scores = scores[validation_indices]
    validation_targets = targets[validation_indices]
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        if epoch % VALIDATE_EVERY == 0 or epoch == epochs:
            validation_accuracy = accuracy(
                model,
                validation_scores,
                validation_targets,
                device,
            )
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_accuracy, best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full-shot LaBo HAM10000 concept-bottleneck example."
    )
    parser.add_argument("--root", default="./data/ham10000")
    parser.add_argument(
        "--labo-splits-dir",
        default=None,
        help="Defaults to <root>/labo_splits.",
    )
    vocabulary = parser.add_mutually_exclusive_group()
    vocabulary.add_argument(
        "--candidate-vocabulary",
        help="Candidate JSON to select on the LaBo training split; skips LLM calls.",
    )
    vocabulary.add_argument(
        "--selected-vocabulary",
        help=(
            "Preselected JSON, such as LaBo's published 350 concepts; "
            "skips selection."
        ),
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument(
        "--generation-requests-per-class",
        type=int,
        default=len(PROMPT_THEMES),
        help=(
            "Divide the five LaBo themes into this many LLM requests per class "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CLIP and training device; defaults to CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive.")
    if not 1 <= args.generation_requests_per_class <= len(PROMPT_THEMES):
        raise ValueError(
            "--generation-requests-per-class must be between 1 and "
            f"{len(PROMPT_THEMES)}."
        )
    seed_everything(args.seed)
    check_selector_fixture()

    dataset = HAM10000Dataset(root=args.root)
    image_dataset = OriginalImageDataset(dataset)
    targets, class_codes = diagnosis_targets(dataset)
    splits_dir = args.labo_splits_dir or os.path.join(dataset.root_dir, "labo_splits")
    train_indices, validation_indices, test_indices = load_labo_splits(
        dataset, splits_dir
    )
    concepts_path = os.path.join(dataset.root_dir, "labo_concepts.json")

    preselected = args.selected_vocabulary is not None
    if args.selected_vocabulary:
        loaded = load_vocabulary(
            args.selected_vocabulary,
            "selected_concepts",
            class_codes,
        )
        generator: Generator = FixedConceptGenerator(loaded)
        vocabulary_source = f"preselected vocabulary: {args.selected_vocabulary}"
    elif args.candidate_vocabulary:
        loaded = load_vocabulary(
            args.candidate_vocabulary,
            "candidate_concepts",
            class_codes,
        )
        generator = FixedConceptGenerator(loaded)
        vocabulary_source = f"candidate vocabulary: {args.candidate_vocabulary}"
    else:
        llm = LiteLLMBackend(
            model=args.llm_model,
            temperature=0.0,
            max_tokens=8192,
            timeout=120.0,
            retry_on_rate_limit=True,
            reasoning_effort="none"
        )
        generator = LaBoCandidateGenerator(
            llm,
            class_codes,
            concepts_path,
            args.generation_requests_per_class,
        )
        vocabulary_source = f"LLM: {args.llm_model}"

    device = resolve_device(args.device)
    print(f"Loading {CLIP_MODEL} on {device}...")
    clip = RawCLIPAnnotator(
        model_name=CLIP_MODEL,
        prompt_template="{}",
        batch_size=args.clip_batch_size,
        device=device,
        show_progress=True,
    )
    selector = None
    if not preselected:
        selector = LaBoConceptSelector(
            dataset=image_dataset,
            train_indices=train_indices,
            targets=targets,
            class_codes=class_codes,
            clip=clip,
        )
        print("Selecting concepts on the training split, then annotating all images...")
    else:
        print("Annotating all images with the preselected vocabulary...")

    # Cartesian routing keeps the generator's source:* provenance groups.
    pipeline = ConceptGenerationPipeline(
        generators=generator,
        annotators=clip,
        generator_filter=selector,
        routing="cartesian",
    )
    generated = pipeline(
        image_dataset,
        class_names=[DIAGNOSIS_NAMES[code] for code in class_codes],
        generation_indices=train_indices,
    )
    if len(generated) != 1:
        raise RuntimeError(f"Expected one CLIP output, received {list(generated)}.")
    concept_scores = next(iter(generated.values()))
    selected_concepts = concept_scores.annotation
    candidates = generator.concepts
    if candidates is None:
        raise RuntimeError("Candidate generation did not produce a vocabulary.")
    if not preselected:
        generation_metadata = (
            generator.generation_metadata
            if isinstance(generator, LaBoCandidateGenerator)
            else None
        )
        save_vocabulary(
            concepts_path,
            candidates,
            selected_concepts,
            generation=generation_metadata,
        )

    model = LaBoCBM(selected_concepts, class_codes)
    print(f"Training the association head for {args.train_epochs} epochs...")
    best_validation_accuracy, best_epoch = train_association_head(
        model=model,
        scores=concept_scores.tensor,
        targets=targets,
        train_indices=train_indices,
        validation_indices=validation_indices,
        epochs=args.train_epochs,
        seed=args.seed,
        device=device,
    )
    test_accuracy = accuracy(
        model,
        concept_scores.tensor[test_indices],
        targets[test_indices],
        device,
    )

    print("\nLaBo HAM10000 summary")
    print(f"Vocabulary source: {vocabulary_source}")
    print(f"Candidate concepts: {'skipped' if preselected else len(candidates.labels)}")
    print(f"Selected unique concepts: {len(selected_concepts.labels)}")
    print(
        "Split sizes: "
        f"train={len(train_indices)}, validation={len(validation_indices)}, "
        f"test={len(test_indices)}"
    )
    print(f"CLIP: {CLIP_MODEL}, original images, raw unnormalized dot products")
    print(
        f"Best validation accuracy: {100 * best_validation_accuracy:.2f}% "
        f"at epoch {best_epoch}"
    )
    print(f"Test accuracy: {100 * test_accuracy:.2f}%")
    print(f"Published LaBo full-shot test accuracy: {PUBLISHED_TEST_ACCURACY:.2f}%")


if __name__ == "__main__":
    main()
