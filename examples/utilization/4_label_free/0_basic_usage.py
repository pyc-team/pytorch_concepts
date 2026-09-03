"""Label-free concept supervision on ColorMNIST with flexible dataset layouts.

This example uses:
- LLMConceptGenerator with LiteLLMBackend to produce a concept vocabulary.
- CLIPAnnotator to produce raw image-concept similarity scores.
- Calibrator and FilterAnnotator stages to turn similarities into
  probabilities and filter uncertain sample-level annotations.
- ConceptSupervisionPipeline, whose concept-discovery and annotation targets
  can be chosen independently.
- A tiny concept bottleneck classifier trained on the generated concepts.

ColorMNIST supplies native ``digit`` and ``color`` annotations for a small set
of training examples used in the LLM prompt. This demonstrates partial concept
supervision: sparse known concepts provide in-context evidence from which the
model expands to a broader generated vocabulary. The subsequent CLIP
annotation stage applies that generated vocabulary automatically across the
chosen dataset rows. This vocabulary-generation step is therefore not purely
concept-label-free, and the native and generated vocabularies remain separate.

The ``--data-mode`` option demonstrates four equivalent ways to describe a
dataset layout to the pipeline. Their difference is not the concept model; it
is where split information lives:

- ``full``: one dataset is used for both concept discovery and annotation. It
  is the simplest unsplit baseline.
- ``train-to-full``: one complete dataset is kept, but training-row indices
  limit concept discovery. Annotation still covers every row. This is the
  default because it avoids validation/test data influencing the vocabulary
  while preserving a single annotation tensor aligned with the complete
  dataset for later train/validation slicing.
- ``indexed-splits``: one complete dataset is accompanied by named index
  sequences. Use this when splits exist as indices rather  than as dataset objects.
- ``separate-datasets``: named dataset objects are passed directly for
  annotation, and the training dataset itself is used for concept discovery.
  This fits DataModules or workflows that already expose split datasets.

This flexibility lets a generator retain access to the metadata of its source
dataset while annotations can either remain globally aligned or be kept as
split-specific tensors.

Usage:

    export GEMINI_API_KEY="your_google_ai_studio_key"

    python -m examples.utilization.4_label_free.0_basic_usage \
      --llm-model gemini/gemini-3.5-flash \
      --llm-temperature 1.0 \
      --data-mode train-to-full

``--data-mode`` selects exactly one layout. Only ``train-to-full`` continues
to the CBM training demonstration.

For another LiteLLM provider, set the API key expected by that provider, e.g.
``OPENAI_API_KEY`` for ``--llm-model openai/gpt-4o``.
"""

import argparse
import base64
from io import BytesIO

import torch
from torch import nn
from torch.utils.data import Subset
from PIL import Image
from tqdm import tqdm

from torch_concepts.data.generation import ConceptSupervisionPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.filters import (
    ThresholdAnnotationFilter,
)
from torch_concepts.data.generation.generators import LiteLLMBackend, LLMConceptGenerator
from torch_concepts.data import ColorMNISTDataModule


def _image_data_url(image: torch.Tensor) -> str:
    """Encode a CHW tensor for a multimodal LiteLLM prompt."""
    array = (
        image.detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def dataset_aware_prompt(
    dataset,
    class_names=None,
    indices=None,
    num_examples=4,
    **kwargs,
):
    """Build a prompt from the rows selected for concept discovery.

    ``indices`` is supplied by ``generation_indices`` when the pipeline uses a
    complete dataset but restricts generation to a split. Sampling those rows
    here prevents validation/test images from influencing the LLM vocabulary.
    Their known native digit and color concepts provide sparse in-context
    evidence that the LLM expands into a broader visual vocabulary.
    When ``dataset`` is already a :class:`~torch.utils.data.Subset`, no indices
    are needed: its local rows are sampled while metadata is read from the
    underlying dataset.
    """
    del kwargs
    candidate_indices = range(len(dataset)) if indices is None else indices
    num_candidates = len(candidate_indices)
    if num_candidates:
        positions = torch.linspace(
            0,
            num_candidates - 1,
            steps=min(num_examples, num_candidates),
        ).long()
        example_indices = [
            int(candidate_indices[position]) for position in positions.tolist()
        ]
    else:
        example_indices = []

    metadata_dataset = dataset
    while isinstance(metadata_dataset, Subset):
        metadata_dataset = metadata_dataset.dataset

    content = [{
        "type": "text",
        "text": (
            "Expand the partial digit and color concept evidence in the "
            "labeled images below into 12 short visual concepts useful for "
            f"classifying ColorMNIST images as {class_names}. Include digit "
            "identity, color, and broader simple shape properties. Return one "
            "concept per line and no explanations."
        ),
    }]
    for index in example_indices:
        sample = dataset[index]
        native = sample["concepts"]["native"]
        native_names = metadata_dataset.native_concepts.annotation.labels
        digit = int(native[native_names.index("digit")])
        color_id = int(native[native_names.index("color")])
        color = ("red", "green")[color_id]
        content.extend([
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(sample["inputs"]["x"])},
            },
            {
                "type": "text",
                "text": f"Example label: {color} digit {digit}.",
            },
        ])
    return [{"role": "user", "content": content}]


def _print_generated(generated):
    """Print generated output names and tensor shapes."""
    for name, values in generated.items():
        print(f"{name}: {tuple(values.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model", default="gemini/gemini-3.5-flash")
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument(
        "--data-mode",
        choices=(
            "full",
            "train-to-full",
            "indexed-splits",
            "separate-datasets",
        ),
        default="train-to-full",
        help=(
            "Dataset layout demonstrated by this invocation. The default "
            "discovers concepts from training rows and annotates all rows."
        ),
    )
    parser.add_argument(
        "--clip-device",
        default=None,
        help="Explicit device; by default prefers CUDA, then MPS, then CPU.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    llm = LiteLLMBackend(
        model=args.llm_model,
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
    )
    # The generator proposes one shared concept vocabulary for the annotator.
    generator = LLMConceptGenerator(
        llm=llm,
        prompt=dataset_aware_prompt,
    )
    # The annotation target is selected by the requested data mode below.
    annotator = CLIPAnnotator(
        model_name="openai/clip-vit-base-patch32",
        prompt_template="a photo of a {}",
        batch_size=128,
        device=args.clip_device,
        show_progress=True,
    )
    # The pipeline wires concept generation and annotation together. With
    # routing="merged", all generated concepts are passed to the annotator as a
    # single concept axis.
    pipeline = ConceptSupervisionPipeline(
        generators=generator,
        annotators=annotator,
        calibrator=SigmoidCalibrator(scale=10.0),
        calibrated_annotation_filter=ThresholdAnnotationFilter(0.5),
        routing="merged",
    )

    datamodule = ColorMNISTDataModule(
        root="./data",
        coloring={"red": range(6), "green": range(6, 10)},
        batch_size=128,
        max_samples=20000,
        seed=0,
    )
    datamodule.setup("fit")
    dataset = datamodule.dataset
    train_indices = datamodule.trainset.indices
    val_indices = datamodule.valset.indices

    pipeline_kwargs = {"class_names": ["even", "odd"]}
    if args.data_mode == "full":
        generated = pipeline(dataset, **pipeline_kwargs)
        _print_generated(generated)
        return

    if args.data_mode == "indexed-splits":
        generated = pipeline(
            dataset,
            generation_indices=train_indices,
            annotation_indices={
                "train": train_indices,
                "val": val_indices,
            },
            **pipeline_kwargs,
        )
        _print_generated(generated)
        return

    if args.data_mode == "separate-datasets":
        generated = pipeline(
            datamodule.trainset,
            annotation_datasets={
                "train": datamodule.trainset,
                "val": datamodule.valset,
            },
            **pipeline_kwargs,
        )
        _print_generated(generated)
        return

    # Generated concepts are the classifier inputs; parity remains the
    # supervised downstream task target in the persistent native source.
    parity_index = dataset.native_concepts.annotation.labels.index("parity")
    train_labels = dataset.native_concepts[
        train_indices,
        parity_index,
    ].long()
    val_labels = dataset.native_concepts[
        val_indices,
        parity_index,
    ].long()

    generated_name = "CLIPAnnotator"
    datamodule.generate_concepts(
        pipeline,
        generation_indices=train_indices,
        use_as_gt=True,
        generated_gt_name=generated_name,
        **pipeline_kwargs,
    )

    generated = dataset.generated_concepts[generated_name]
    concept_axis = generated.annotation
    train_concepts = generated[train_indices].float()
    val_concepts = generated[val_indices].float()

    # These calibrated values are soft probabilities; filtering has already
    # set below-threshold (absent) concepts to zero.
    mean = train_concepts.mean(dim=0, keepdim=True)
    std = train_concepts.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-6)
    train_concepts = (train_concepts - mean) / std
    val_concepts = (val_concepts - mean) / std

    model = nn.Linear(train_concepts.shape[1], 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    progress = tqdm(range(200), desc="Training CBM")
    for _ in progress:
        optimizer.zero_grad()
        loss = loss_fn(model(train_concepts), train_labels)
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    with torch.no_grad():
        train_acc = (
            model(train_concepts).argmax(1) == train_labels
        ).float().mean().item()
        val_acc = (
            model(val_concepts).argmax(1) == val_labels
        ).float().mean().item()

    print("Generated concepts:", concept_axis.labels)
    print("Train annotation tensor shape:", tuple(train_concepts.shape))
    print("Validation annotation tensor shape:", tuple(val_concepts.shape))
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    main()
