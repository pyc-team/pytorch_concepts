"""
CEBaB (Causal Estimation of Sentiment Polarity) Dataset

CEBaB is a benchmark for concept-based NLP that pairs restaurant reviews with
fine-grained aspect-level annotations (food, ambiance, service, noise) and an
overall 5-star rating.

**Data source**

The dataset is downloaded automatically from the HuggingFace Hub:
https://huggingface.co/datasets/CEBaB/CEBaB

**Concept vector layout**

- columns 0-3: 4 aspect concepts, each with 3 states
  (0 = Negative, 1 = unknown, 2 = Positive)
- column 4: overall review rating remapped to 0-4
  (original 1-5 stars -> 0-4)

Use ``default_task_names: [review_majority]`` in the Conceptarium config to
treat the review rating as the downstream task.

**Input representation**

Texts are tokenised with a configurable pretrained tokeniser (default:
``bert-base-uncased``) and stored as ``input_ids``.  The attention mask and
token-type IDs are stored as additional attributes and are returned by
``__getitem__``.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Mapping, Optional

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset

logger = logging.getLogger(__name__)

ASPECT_NAMES = [
    "food_aspect_majority",
    "ambiance_aspect_majority",
    "service_aspect_majority",
    "noise_aspect_majority",
]
TASK_NAME = "review_majority"

# Aspect states: Negative=0, unknown=1, Positive=2
ASPECT_STATES = ["Negative", "unknown", "Positive"]

# Review-majority states: original 1-star to 5-star remapped to 0-4
REVIEW_STATES = ["1-star", "2-star", "3-star", "4-star", "5-star"]

ASPECT_MAPPING = {"Positive": 2, "unknown": 1, "Negative": 0}


class CEBaBDataset(ConceptDataset):
    """Dataset class for CEBaB (Causal Estimation of Sentiment Polarity).

    Restaurant reviews from the CEBaB benchmark annotated with four
    aspect-level concepts (food, ambiance, service, noise quality) and an
    overall 5-star review rating.

    The data is downloaded automatically from the HuggingFace Hub on first use.

    Parameters
    ----------
    root : str, optional
        Root directory for caching processed artefacts.
        Defaults to ``./data/cebab``.
    pre_trained_model_name : str, optional
        HuggingFace tokeniser model name.  Default: ``bert-base-uncased``.
    max_length : int, optional
        Maximum token sequence length for truncation / padding.  Default: 512.
    concept_subset : list of str, optional
        Subset of concept names to retain.  ``None`` keeps all 5.
    label_descriptions : dict, optional
        Mapping from concept name to human-readable description.
    """

    def __init__(
        self,
        root: str = None,
        pre_trained_model_name: str = "bert-base-uncased",
        max_length: int = 512,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[Mapping] = None,
    ):
        if root is None:
            root = os.path.join(os.getcwd(), "data", "cebab")
        self.root = root
        self.pre_trained_model_name = pre_trained_model_name
        self.max_length = max_length
        self.label_descriptions = label_descriptions

        input_ids, attention_mask, token_type_ids, concepts, annotations, graph = self.load()

        # Store attention mask and token-type IDs as extra attributes
        # (they will be indexed per-sample in __getitem__)
        self._attention_mask = attention_mask
        self._token_type_ids = token_type_ids

        # input_data = input_ids tensor  (n_samples, max_length)
        super().__init__(
            input_data=input_ids,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name="CEBaBDataset",
        )

    # ------------------------------------------------------------------
    # ConceptDataset interface
    # ------------------------------------------------------------------

    @property
    def raw_filenames(self) -> List[str]:
        # HuggingFace manages its own download cache; nothing to check here.
        return []

    @property
    def processed_filenames(self) -> List[str]:
        model_str = self.pre_trained_model_name.replace("/", "-")
        return [
            f"input_ids_{model_str}_maxlen_{self.max_length}.pt",
            f"attention_mask_{model_str}_maxlen_{self.max_length}.pt",
            f"token_type_ids_{model_str}_maxlen_{self.max_length}.pt",
            "concepts.pt",
            "annotations.pt",
            "split_mapping.h5",
        ]

    def download(self):
        pass  # HuggingFace datasets handles download internally in build()

    def build(self):
        """Download from HuggingFace, process, tokenise, and save artefacts."""
        from datasets import load_dataset as hf_load_dataset
        from transformers import AutoTokenizer

        logger.info("Downloading CEBaB dataset from HuggingFace Hub ...")
        ds = hf_load_dataset("CEBaB/CEBaB")

        ds_train = ds["train_observational"].to_pandas()
        ds_val = ds["validation"].to_pandas()
        ds_test = ds["test"].to_pandas()

        keep_cols = ["description", TASK_NAME] + ASPECT_NAMES

        def _preprocess(df):
            df = df[keep_cols].copy()
            df = df.dropna()
            # Map aspect strings to integers; mark unrecognised values for removal
            for col in ASPECT_NAMES:
                df[col] = df[col].map(lambda v: ASPECT_MAPPING.get(v, None))
            # Map review rating to 0-indexed
            df[TASK_NAME] = df[TASK_NAME].map(lambda v: int(v) - 1)
            # Drop rows where any aspect could not be mapped
            df = df.dropna()
            df[ASPECT_NAMES] = df[ASPECT_NAMES].astype(int)
            return df.reset_index(drop=True)

        ds_train = _preprocess(ds_train)
        ds_val = _preprocess(ds_val)
        ds_test = _preprocess(ds_test)

        n_train = len(ds_train)
        n_val = len(ds_val)
        n_test = len(ds_test)

        all_df = pd.concat([ds_train, ds_val, ds_test], ignore_index=True)
        split_labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

        logger.info(
            f"Tokenising with {self.pre_trained_model_name!r} "
            f"(max_length={self.max_length}) ..."
        )
        tokenizer = AutoTokenizer.from_pretrained(self.pre_trained_model_name)
        encoding = tokenizer(
            all_df["description"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].long()
        attention_mask = encoding["attention_mask"].long()
        token_type_ids = encoding.get(
            "token_type_ids",
            torch.zeros_like(input_ids),
        ).long()

        # Build concept tensor: 4 aspects + review_majority
        concept_cols = ASPECT_NAMES + [TASK_NAME]
        concepts_np = all_df[concept_cols].to_numpy(dtype=np.float32)
        concepts_tensor = torch.tensor(concepts_np, dtype=torch.float32)

        # Build Annotations
        aspect_states = [ASPECT_STATES] * len(ASPECT_NAMES)
        review_states = [REVIEW_STATES]
        states = aspect_states + review_states
        cardinalities = [len(ASPECT_STATES)] * len(ASPECT_NAMES) + [len(REVIEW_STATES)]
        concept_names = ASPECT_NAMES + [TASK_NAME]
        concept_metadata = {name: {"type": "discrete"} for name in concept_names}

        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                states=states,
                cardinalities=cardinalities,
                metadata=concept_metadata,
            )
        })

        # Save artefacts
        os.makedirs(self.root_dir, exist_ok=True)
        logger.info(f"Saving CEBaB dataset to {self.root_dir}")

        torch.save(input_ids, self.processed_paths[0])
        torch.save(attention_mask, self.processed_paths[1])
        torch.save(token_type_ids, self.processed_paths[2])
        torch.save(concepts_tensor, self.processed_paths[3])
        torch.save(annotations, self.processed_paths[4])
        pd.Series(split_labels).to_hdf(
            self.processed_paths[5], key="split_mapping", mode="w"
        )

        logger.info(
            f"CEBaB dataset saved "
            f"(train={n_train}, val={n_val}, test={n_test})"
        )

    def load_raw(self):
        """Load tokenised artefacts from disk."""
        self.maybe_build()

        logger.info(f"Loading CEBaB dataset from {self.root_dir}")

        input_ids = torch.load(self.processed_paths[0], weights_only=False)
        attention_mask = torch.load(self.processed_paths[1], weights_only=False)
        token_type_ids = torch.load(self.processed_paths[2], weights_only=False)
        concepts = torch.load(self.processed_paths[3], weights_only=False)
        annotations = torch.load(self.processed_paths[4], weights_only=False)
        graph = None

        return input_ids, attention_mask, token_type_ids, concepts, annotations, graph

    def load(self):
        return self.load_raw()

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------

    def __getitem__(self, item: int) -> dict:
        if self.embs_precomputed:
            x = self.input_data[item]
            return {"inputs": {"x": x}, "concepts": {"c": self.concepts[item]}}

        return {
            "inputs": {
                "input_ids": self.input_data[item],
                "attention_mask": self._attention_mask[item],
                "token_type_ids": self._token_type_ids[item],
            },
            "concepts": {"c": self.concepts[item]},
        }

    # ------------------------------------------------------------------
    # Overridden properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.input_data.shape[0]

    @property
    def n_features(self) -> tuple:
        return tuple(self.input_data.shape[1:])

    @property
    def shape(self) -> tuple:
        return tuple(self.input_data.shape)
