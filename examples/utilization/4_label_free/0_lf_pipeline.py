"""Compact LF-CBM-style pipeline on CUB with LiteLLM and CLIP.

The example follows the main ingredients of Oikarinen et al.,
"Label-Free Concept Bottleneck Models":

1. ask an LLM for concepts from class names;
2. annotate images with CLIP concept similarities;
3. prune class-like, duplicate, and weak concepts;
4. fit the LF-CBM projection layer and sparse final classifier.

Minimal run:

    python -m examples.utilization.4_label_free.0_lf_pipeline \
      --llm-model gemini/gemini-3.5-flash \
      --llm-temperature 1.0 \
      --max-prompt-classes 1 \
      --num-samples 20 \
      --proj-steps 50 \
      --clip-device mps

Larger run:

    python -m examples.utilization.4_label_free.0_lf_pipeline \
      --llm-model gemini/gemini-3.5-flash \
      --llm-temperature 1.0 \
      --max-prompt-classes 10 \
      --num-samples 1000 \
      --clip-device mps

Important CLI flags:
- ``--max-prompt-classes`` controls how many CUB class names are sent to the
  LLM. Use ``0`` for all 200 classes.
- ``--num-concepts`` controls concepts requested per class.
- ``--num-samples`` controls how many CUB training images are annotated.
- ``--semantic-filter`` controls class-name and redundancy pruning. Use
  ``clip`` or ``none`` if ``sentence-transformers`` is not installed.
- ``--interpretability-cutoff`` defaults to a lenient value for small example
  runs. Increase it toward ``0.45`` for larger, paper-like runs.
- ``--clip-device mps`` is useful on Apple Silicon; omit it for CUDA/CPU auto
  selection inside ``CLIPAnnotator``.
- ``--llm-cache`` overrides the JSON cache path. By default the example caches
  LLM outputs under ``<data-root>/lf_cbm_llm_cache.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from torch_concepts import AxisAnnotation
from torch_concepts.data.annotators import CLIPAnnotator
from torch_concepts.data.base import ConceptSupervisionPipeline
from torch_concepts.data.concept_generators import (
    LLMConceptGenerator,
    LiteLLMBackend,
    default_concept_postprocessor,
)
from torch_concepts.data.datasets import CUBDataset
from torch_concepts.data.datasets.cub import CLASS_NAMES as CUB_CLASS_NAMES


CUB_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz?download=1"
)

API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHERAI_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "bird",
    "birds",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "with",
}

TAG_RE = re.compile(r"^\s*\[?(visual|superclass|context)\]?\s*[:\-–—]?\s*", re.I)
MPNET_MODEL = None


CLASS_PROMPT = """
List the most important features for recognizing something as a "goldfish":
-[visual] bright orange color
-[visual] a small round body
-[visual] a long flowing tail
-[visual] translucent fins
-[visual] small mouth

Give superclasses for the word "goldfish":
-[superclass] fish
-[superclass] freshwater fish

List the things most commonly seen around a "goldfish":
-[context] a pond
-[context] water
-[context] aquarium plants

List the most important features for recognizing something as a "{class_name}":
Give superclasses for the word "{class_name}":
List the things most commonly seen around a "{class_name}":

Return exactly {num_concepts} concepts total.
Rules:
- one concept per line
- prefix every line with [visual], [superclass], or [context]
- prefer visual attributes; use superclass/context only when specific
- each concept should be short, preferably under 45 characters
- do not include headings, numbering, explanations, or markdown fences
- do not include the class name itself
"""

BATCH_PROMPT = """
Generate LF-CBM concepts for each class below using the same structure as the
examples. Return exactly {num_concepts} concepts per class and output only
tagged concept lines.

Prefer concrete visual concepts over broad labels such as bird or animal.

{class_prompts}
"""


class CachedLLMBackend:
    """Tiny file-backed cache keyed by rendered prompt and LLM kwargs."""

    def __init__(self, llm, path, namespace):
        self.llm = llm
        self.path = Path(path).expanduser()
        self.namespace = namespace
        self.cache = None

    def __call__(self, prompt, **kwargs):
        if self.cache is None:
            self.cache = json.loads(self.path.read_text()) if self.path.is_file() else {}
        key = hashlib.sha256(
            json.dumps(
                {"namespace": self.namespace, "prompt": prompt, "kwargs": kwargs},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if key not in self.cache:
            print(f"LLM cache miss: {self.path}")
            self.cache[key] = {"output": self.llm(prompt, **kwargs)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, indent=2, sort_keys=True))
        else:
            print(f"LLM cache hit: {self.path}")
        return self.cache[key]["output"]


def parse_args():
    """Parse CLI flags used to trade off cost, speed, and LF-CBM fidelity."""
    parser = argparse.ArgumentParser(
        description="Run a compact LF-CBM-style concept generation pipeline on CUB."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CUB_DATA_ROOT", "./data"),
        help="Directory containing or receiving CUB_200_2011.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of CUB training images to annotate, sampled class-balanced.",
    )
    parser.add_argument(
        "--num-concepts",
        type=int,
        default=10,
        help="Number of LLM concepts requested per prompted class.",
    )
    parser.add_argument("--num-preview", type=int, default=4)
    parser.add_argument("--num-preview-concepts", type=int, default=5)
    parser.add_argument(
        "--max-prompt-classes",
        type=int,
        default=3,
        help="Number of CUB classes sent in the single LLM prompt. Use 0 for all.",
    )
    parser.add_argument(
        "--max-concept-chars",
        type=int,
        default=50,
        help="Drop very long generated concepts before CLIP annotation.",
    )
    parser.add_argument(
        "--semantic-filter",
        choices=("mpnet-clip", "clip", "none"),
        default="mpnet-clip",
        help="Similarity backend for class-name and redundancy pruning.",
    )
    parser.add_argument(
        "--class-sim-cutoff",
        type=float,
        default=0.85,
        help="Remove concepts too semantically close to any class name.",
    )
    parser.add_argument(
        "--redundancy-sim-cutoff",
        type=float,
        default=0.90,
        help="Remove one concept from pairs above this similarity.",
    )
    parser.add_argument(
        "--clip-cutoff",
        type=float,
        default=0.26,
        help="Remove concepts whose mean top-5 CLIP activation is too low.",
    )
    parser.add_argument(
        "--proj-steps",
        type=int,
        default=1000,
        help="Optimization steps for the LF-CBM projection layer.",
    )
    parser.add_argument(
        "--proj-batch-size",
        type=int,
        default=50000,
        help="Mini-batch size used while fitting the projection layer.",
    )
    parser.add_argument(
        "--interpretability-cutoff",
        type=float,
        default=0.05,
        help="Remove concepts poorly reconstructed by the projection layer.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples held out for projection/classifier validation.",
    )
    parser.add_argument(
        "--saga-c",
        type=float,
        default=1.0,
        help="Inverse L1 regularization strength for the sparse final layer.",
    )
    parser.add_argument(
        "--saga-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for sklearn's SAGA logistic regression solver.",
    )
    parser.add_argument(
        "--llm-model",
        default="gemini/gemini-2.5-flash",
        help="LiteLLM model identifier, e.g. gemini/... or openai/...",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.5)
    parser.add_argument(
        "--llm-repeats",
        type=int,
        default=1,
        help="Repeat the same prompt to increase concept diversity.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=600,
        help="LiteLLM provider request timeout in seconds.",
    )
    parser.add_argument("--llm-rate-limit-wait", type=float, default=120)
    parser.add_argument(
        "--llm-api-key-env",
        help="Environment variable for the provider key if it cannot be inferred.",
    )
    parser.add_argument("--llm-cache", help="Path to the LLM JSON cache file.")
    parser.add_argument("--no-llm-cache", action="store_true")
    parser.add_argument("--clip-device", help="CLIP device, e.g. cuda, mps, or cpu.")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--preview-output", default="cub_concept_annotations.png")
    return parser.parse_args()


def concept_prompt(dataset=None, class_names=None, num_concepts=10, max_prompt_classes=3):
    """Render one Oikarinen-style prompt containing all selected classes."""
    del dataset
    names = list(class_names or CUB_CLASS_NAMES)
    if max_prompt_classes:
        names = names[:max_prompt_classes]
    class_prompts = "\n\n".join(
        CLASS_PROMPT.format(
            class_name=name.replace("_", " "),
            num_concepts=num_concepts,
        )
        for name in names
    )
    return BATCH_PROMPT.format(
        class_prompts=class_prompts,
        num_concepts=num_concepts,
    )


def normalized(text):
    """Normalize text for cheap duplicate and class-name checks."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower().replace("_", " "))
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def clean_concept(text):
    """Strip LLM formatting, source tags, bullets, and trailing punctuation."""
    text = TAG_RE.sub("", str(text))
    text = re.sub(r"^(?:[-*]|\d+[.)-])\s*", "", text)
    return re.sub(r"\s+", " ", text.strip().rstrip("."))


def make_postprocessor(class_names, max_concepts, max_chars):
    """Clean LLM concepts before the expensive CLIP annotation step."""
    class_tokens = [set(normalized(name).split()) for name in class_names]

    def postprocess(concepts):
        output, seen = [], set()
        for spec in default_concept_postprocessor(concepts):
            name = clean_concept(spec["name"])
            key = normalized(name)
            if not key or len(name) > max_chars or key in seen:
                continue
            tokens = set(key.split())
            if any(tokens and len(tokens & cls) / min(len(tokens), len(cls)) >= 0.8 for cls in class_tokens if cls):
                continue
            seen.add(key)
            output.append({"name": name, "type": "binary"})
            if len(output) == max_concepts:
                break
        return output

    return postprocess


def read_indexed(path, parser=str):
    """Read CUB files formatted as '<integer id> <value>'."""
    values = {}
    with Path(path).open() as file:
        for line in file:
            index, value = line.strip().split(maxsplit=1)
            values[int(index)] = parser(value)
    return values


def prepare_cub_dataset(data_root, num_samples):
    """Download/extract CUB and build the small pickle expected by CUBDataset.

    The example chooses images class-balanced so small runs show different bird
    classes instead of the first images from one class.
    """
    data_root = Path(data_root).expanduser()
    cub_root = data_root / "CUB_200_2011"
    if not (cub_root / "images.txt").is_file():
        archive = data_root / "CUB_200_2011.tgz"
        data_root.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            print(f"Downloading CUB-200-2011 to {archive}...")
            urllib.request.urlretrieve(CUB_URL, archive)
        with tarfile.open(archive, "r:gz") as file:
            file.extractall(data_root)

    images = read_indexed(cub_root / "images.txt")
    labels = read_indexed(cub_root / "image_class_labels.txt", int)
    train_flags = read_indexed(cub_root / "train_test_split.txt", int)
    by_class = {}
    for image_id in images:
        if train_flags[image_id]:
            by_class.setdefault(labels[image_id], []).append(image_id)

    image_ids = []
    round_index = 0
    while len(image_ids) < num_samples:
        before = len(image_ids)
        for class_id in sorted(by_class):
            if round_index < len(by_class[class_id]):
                image_ids.append(by_class[class_id][round_index])
            if len(image_ids) == num_samples:
                break
        if len(image_ids) == before:
            break
        round_index += 1

    records = {
        image_id: {
            "img_path": str(cub_root / "images" / images[image_id]),
            "class_label": labels[image_id] - 1,
            "attribute_label": [0] * 312,
            "uncertain_attribute_label": [0.0] * 312,
            "attribute_certainty": [0] * 312,
        }
        for image_id in image_ids
    }
    with (cub_root / "attributes" / "image_attribute_labels.txt").open() as file:
        for line in file:
            image_id, attribute_id, present, certainty = line.split()[:4]
            image_id = int(image_id)
            if image_id in records:
                index = int(attribute_id) - 1
                records[image_id]["attribute_label"][index] = int(present)
                records[image_id]["uncertain_attribute_label"][index] = int(present)
                records[image_id]["attribute_certainty"][index] = int(certainty)

    output_dir = cub_root / "class_attr_data_10"
    output_dir.mkdir(exist_ok=True)
    with (output_dir / "train.pkl").open("wb") as file:
        pickle.dump([records[image_id] for image_id in image_ids], file)
    return str(cub_root)


def prepare_cub_image(image):
    """Convert CUB tensors to the normalized 224x224 CLIP input format."""
    image = image * 2 + 0.5
    image = TF.resize(image, [224, 224], antialias=True)
    return TF.normalize(
        image,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )


def make_axis(names):
    """Build a binary concept axis for the filtered generated concepts."""
    return AxisAnnotation(
        labels=list(names),
        states=[["0"] for _ in names],
        cardinalities=[1] * len(names),
        metadata={name: {"type": "discrete"} for name in names},
    )


def clip_text_features(texts, annotator, batch_size=512):
    """Encode raw text with the CLIP model already owned by CLIPAnnotator."""
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            tokens = annotator.tokenizer(texts[start:start + batch_size]).to(annotator.device)
            features = annotator.model.encode_text(tokens)
            batches.append(F.normalize(features, dim=-1).cpu())
    return torch.cat(batches, dim=0)


def semantic_similarity(left, right, annotator, mode):
    """Compute text similarity for LF-CBM pruning.

    The LF-CBM code combines MPNet and CLIP similarities. This example uses the
    same mixture when sentence-transformers is installed and falls back to CLIP.
    """
    if mode == "none":
        return None

    clip_scores = clip_text_features(left, annotator) @ clip_text_features(right, annotator).T
    if mode == "clip":
        return clip_scores

    try:
        from sentence_transformers import SentenceTransformer

        global MPNET_MODEL
        if MPNET_MODEL is None:
            MPNET_MODEL = SentenceTransformer("all-mpnet-base-v2")
        left_mpnet = torch.as_tensor(
            MPNET_MODEL.encode(left, normalize_embeddings=True),
            dtype=torch.float32,
        )
        right_mpnet = torch.as_tensor(
            MPNET_MODEL.encode(right, normalize_embeddings=True),
            dtype=torch.float32,
        )
        return (left_mpnet @ right_mpnet.T + 3 * clip_scores) / 4
    except Exception as error:
        print(f"Using CLIP-only semantic filtering: {error}")
        return clip_scores


def prune_against_classes(concepts, values, class_names, annotator, cutoff, mode):
    """Remove concepts that are the class names or near-synonyms of them."""
    readable_classes = [name.replace("_", " ") for name in class_names]
    exact = {name.casefold() for name in readable_classes}
    initial_count = len(concepts)
    keep = [i for i, concept in enumerate(concepts) if concept.casefold() not in exact]
    concepts, values = [concepts[i] for i in keep], values[:, keep]
    removed = initial_count - len(concepts)
    if not concepts or cutoff <= 0:
        return concepts, values, removed

    scores = semantic_similarity(readable_classes, concepts, annotator, mode)
    if scores is None:
        return concepts, values, removed
    keep = (scores.max(dim=0).values < cutoff).nonzero(as_tuple=False).flatten().tolist()
    removed += len(concepts) - len(keep)
    return [concepts[i] for i in keep], values[:, keep], removed


def prune_redundant(concepts, values, annotator, cutoff, mode):
    """Remove one concept from highly similar concept pairs."""
    if len(concepts) < 2 or cutoff <= 0 or mode == "none":
        return concepts, values, 0

    scores = semantic_similarity(concepts, concepts, annotator, mode)
    mass = scores.sum(dim=1)
    removed = set()
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            if scores[i, j] >= cutoff and i not in removed and j not in removed:
                removed.add(i if mass[i] < mass[j] else j)
    keep = [i for i in range(len(concepts)) if i not in removed]
    return [concepts[i] for i in keep], values[:, keep], len(removed)


def prune_low_clip(concepts, values, cutoff):
    """Remove concepts that CLIP never activates strongly on this dataset."""
    if cutoff <= 0 or not concepts or values.shape[0] == 0:
        return concepts, values, 0
    top5 = torch.topk(values, k=min(5, values.shape[0]), dim=0).values.mean(dim=0)
    keep = (top5 > cutoff).nonzero(as_tuple=False).flatten().tolist()
    return [concepts[i] for i in keep], values[:, keep], len(concepts) - len(keep)


def clip_image_features(dataset, annotator):
    """Reuse CLIPAnnotator's CLIP model to get image features for projection."""
    loader = DataLoader(
        dataset,
        batch_size=annotator.batch_size,
        shuffle=False,
        num_workers=annotator.num_workers,
        collate_fn=lambda batch: batch,
    )
    batches = []
    for batch in tqdm(loader, desc="CLIP image features"):
        images = [annotator.input_getter(sample) for sample in batch]
        with torch.no_grad():
            features = annotator.model.encode_image(annotator._prepare_clip_images(images))
        batches.append(F.normalize(features, dim=-1).cpu())
    return torch.cat(batches)


def train_val_split(n, val_fraction):
    """Create a deterministic holdout split for projection and classifier metrics."""
    if n < 4:
        indices = torch.arange(n)
        return indices, indices
    shuffled = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    val_count = min(max(1, round(n * val_fraction)), n - 1)
    return shuffled[val_count:], shuffled[:val_count]


def cubed_cosine(target, prediction, eps=1e-8):
    """LF-CBM projection objective: cubed per-concept cosine similarity."""
    target = target - target.mean(dim=0, keepdim=True)
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    return (
        (target * prediction).sum(dim=0)
        / (target.norm(dim=0) * prediction.norm(dim=0)).clamp_min(eps)
    ).clamp(-1, 1) ** 3


def train_projection(image_features, concept_values, train_idx, val_idx, args):
    """Fit the projection from CLIP image features to CLIP concept activations."""
    layer = torch.nn.Linear(image_features.shape[1], concept_values.shape[1], bias=False)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    best_weight = layer.weight.detach().clone()
    best_loss = float("inf")
    batch_size = min(args.proj_batch_size, len(train_idx))

    progress = tqdm(range(args.proj_steps), desc="Projection layer")
    for step in progress:
        batch = train_idx[torch.randperm(len(train_idx))[:batch_size]]
        loss = -cubed_cosine(concept_values[batch], layer(image_features[batch])).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.proj_steps - 1:
            with torch.no_grad():
                val_loss = -cubed_cosine(concept_values[val_idx], layer(image_features[val_idx])).mean().item()
            progress.set_postfix(val_loss=f"{val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_weight = layer.weight.detach().clone()

    layer.load_state_dict({"weight": best_weight})
    with torch.no_grad():
        interpretability = cubed_cosine(concept_values[val_idx], layer(image_features[val_idx]))
    return layer, interpretability


def save_preview(dataset, values, concepts, output_path, num_images, num_concepts):
    """Save a quick qualitative preview of strongest/weakest concept scores."""
    count = min(num_images, len(dataset))
    figure, axes = plt.subplots(count, 2, figsize=(13, 4 * count))
    if count == 1:
        axes = [axes]

    for index, (image_axis, text_axis) in enumerate(axes):
        image = Image.open(dataset.data[index]["img_path"]).convert("RGB")
        scores = values[index]
        order = scores.argsort(descending=True).tolist()
        top = order[:num_concepts]
        bottom = list(reversed(order[-max(0, min(num_concepts, len(order) - len(top))):]))
        lines = ["HIGHEST CONCEPT ACTIVATIONS:"]
        lines += [f"+ {concepts[i]}: {scores[i]:.3f}" for i in top]
        lines += ["", "LOWEST CONCEPT ACTIVATIONS:"]
        lines += [f"- {concepts[i]}: {scores[i]:.3f}" for i in bottom]

        class_name = dataset.task_names[int(dataset.data[index]["class_label"])].replace("_", " ")
        image_axis.imshow(image)
        image_axis.set_title(class_name)
        image_axis.axis("off")
        text_axis.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
        text_axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    prompt_classes = len(CUB_CLASS_NAMES) if args.max_prompt_classes == 0 else args.max_prompt_classes
    print("LLM requests to generate concepts:", args.llm_repeats)
    print("Classes in concept-generation prompt:", min(prompt_classes, len(CUB_CLASS_NAMES)))

    # Resolve credentials early so missing keys fail before CLIP/CUB setup work.
    provider = args.llm_model.split("/", 1)[0]
    api_key_env = args.llm_api_key_env or API_KEY_ENV.get(provider)
    api_key = os.environ.get(api_key_env) if api_key_env else None
    cache_path = None if args.no_llm_cache else Path(args.llm_cache or Path(args.data_root) / "lf_cbm_llm_cache.json")
    if api_key_env and not api_key and cache_path is None:
        raise RuntimeError(f"{api_key_env} is not set.")

    llm = LiteLLMBackend(
        model=args.llm_model,
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
        api_key=api_key,
        retry_on_rate_limit=args.llm_rate_limit_wait > 0,
        max_rate_limit_wait=args.llm_rate_limit_wait,
    )
    if cache_path is not None:
        print("LLM cache:", cache_path)
        llm = CachedLLMBackend(
            llm,
            cache_path,
            namespace=f"{args.llm_model}:{args.llm_temperature}:compact-v2",
        )

    # ConceptSupervisionPipeline performs: LLMConceptGenerator -> CLIPAnnotator.
    cub_root = prepare_cub_dataset(args.data_root, args.num_samples)
    max_concepts = (
        min(prompt_classes, len(CUB_CLASS_NAMES))
        * args.llm_repeats
        * args.num_concepts
    )
    generator = LLMConceptGenerator(
        llm=llm,
        prompt=concept_prompt,
        postprocessor=make_postprocessor(
            CUB_CLASS_NAMES,
            max_concepts,
            args.max_concept_chars,
        ),
        llm_kwargs={"repeats": args.llm_repeats},
    )
    annotator = CLIPAnnotator(
        model_name="ViT-B-32",
        pretrained="openai",
        batch_size=args.clip_batch_size,
        device=args.clip_device,
        output="similarity",
        prompt_template=["a photo of {}", "a close-up photo of {}"],
        input_transform=prepare_cub_image,
        show_progress=True,
    )
    pipeline = ConceptSupervisionPipeline(generator, annotator, routing="merged")
    dataset = CUBDataset(
        root=cub_root,
        split="train",
        concept_pipeline=pipeline,
        use_as_gt=False,
        concept_pipeline_kwargs={
            "num_concepts": args.num_concepts,
            "max_prompt_classes": args.max_prompt_classes,
        },
    )

    name = next(iter(dataset.generated_concepts))
    concepts = list(dataset.generated_concepts[name].labels)
    values = dataset.generated_annotations[name].float()
    print("Initial generated concepts:", len(concepts))

    # LF-CBM concept bank pruning.
    concepts, values, removed = prune_against_classes(
        concepts,
        values,
        dataset.task_names,
        annotator,
        args.class_sim_cutoff,
        args.semantic_filter,
    )
    print("Removed class-name-like concepts:", removed)

    concepts, values, removed = prune_redundant(
        concepts,
        values,
        annotator,
        args.redundancy_sim_cutoff,
        args.semantic_filter,
    )
    print("Removed redundant concepts:", removed)

    concepts, values, removed = prune_low_clip(concepts, values, args.clip_cutoff)
    print("Removed low top-5 CLIP concepts:", removed)
    if not concepts:
        raise RuntimeError("All concepts were pruned. Lower the pruning cutoffs.")

    # LF-CBM learns a projection from CLIP image embeddings to CLIP concept
    # activations, then keeps concepts whose projected activations remain
    # interpretable.
    image_features = clip_image_features(dataset, annotator)
    train_idx, val_idx = train_val_split(len(dataset), args.val_fraction)
    projection, interpretability = train_projection(
        image_features,
        values,
        train_idx,
        val_idx,
        args,
    )
    keep = (
        interpretability > args.interpretability_cutoff
    ).nonzero(as_tuple=False).flatten().tolist()
    print("Removed low-interpretability concepts:", len(concepts) - len(keep))
    if not keep:
        raise RuntimeError("All concepts failed interpretability pruning.")

    concepts = [concepts[i] for i in keep]
    projected = image_features @ projection.weight.detach()[keep].T
    mean = projected[train_idx].mean(dim=0, keepdim=True)
    std = projected[train_idx].std(
        dim=0,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-6)
    projected = (projected - mean) / std

    # Store the final bottleneck back on the dataset, then fit the sparse
    # concept-to-class layer used by LF-CBM.
    dataset.set_generated_concepts({name: make_axis(concepts)}, {name: projected.cpu()})
    labels = torch.tensor([int(sample["class_label"]) for sample in dataset.data])
    y_train = labels[train_idx].numpy()
    if len(set(y_train.tolist())) >= 2:
        classifier = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=args.saga_c,
            max_iter=args.saga_max_iter,
            multi_class="multinomial",
            random_state=0,
        )
        classifier.fit(projected[train_idx].numpy(), y_train)
        train_acc = accuracy_score(
            y_train,
            classifier.predict(projected[train_idx].numpy()),
        )
        val_acc = accuracy_score(
            labels[val_idx].numpy(),
            classifier.predict(projected[val_idx].numpy()),
        )
        nonzero = int((torch.as_tensor(classifier.coef_) != 0).sum().item())
        print(
            "Sparse final layer:",
            f"train_acc={train_acc:.3f}",
            f"val_acc={val_acc:.3f}",
            f"nonzero={nonzero}/{classifier.coef_.size}",
        )

    sample = dataset[0]
    print("Image shape:", tuple(sample["inputs"]["x"].shape))
    print("Native concepts shape:", tuple(sample["concepts"]["native"].shape))
    print("Generated annotations shape:", tuple(sample["concepts"]["generated"][name].shape))
    print("Ground-truth concepts shape:", tuple(sample["concepts"]["ground_truth"].shape))
    print("Number of generated concepts:", len(concepts))
    print("Generated concept names:", concepts)

    save_preview(
        dataset,
        dataset.generated_annotations[name],
        concepts,
        args.preview_output,
        args.num_preview,
        args.num_preview_concepts,
    )
    print("Annotation preview saved to:", args.preview_output)


if __name__ == "__main__":
    main()
