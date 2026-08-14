"""Minimal label-free concept supervision example on ColorMNIST.

This example uses:
- LLMConceptGenerator with LiteLLMBackend to produce a concept vocabulary.
- CLIPAnnotator to produce raw image-concept similarity scores.
- Calibrator and FilterAnnotator stages to turn similarities into
  probabilities and filter uncertain sample-level annotations.
- ConceptSupervisionPipeline to generate concepts from train and annotate
  both train and validation partitions.
- A tiny concept bottleneck classifier trained on the generated concepts.

Usage:

    export GEMINI_API_KEY="your_google_ai_studio_key"

    python -m examples.utilization.4_label_free.0_basic_usage \
      --llm-model gemini/gemini-3.5-flash \
      --llm-temperature 1.0

For another LiteLLM provider, set the API key expected by that provider, e.g.
``OPENAI_API_KEY`` for ``--llm-model openai/gpt-4o``.
"""

# TODO: Crea nuovi esempi per testare la pipeline con varie combinazioni di split e non split.

import argparse
import base64
from io import BytesIO

import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

from torch_concepts.data.generation import ConceptSupervisionPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.filters import (
    ThresholdAnnotationFilter,
)
from torch_concepts.data.generation.generators import LiteLLMBackend, LLMConceptGenerator
from torch_concepts.data import ColorMNISTDataset


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


def dataset_aware_prompt(dataset, class_names=None, num_examples=4, **kwargs):
    """Build an in-context prompt using images from the generation dataset."""
    del kwargs
    example_indices = torch.linspace(
        0,
        len(dataset) - 1,
        steps=min(num_examples, len(dataset)),
    ).long()
    content = [{
        "type": "text",
        "text": (
            "Generate 12 short visual concepts useful for classifying "
            f"ColorMNIST images as {class_names}. Use the labeled images below "
            "as in-context examples. Include digit identity, color, and simple "
            "shape concepts. Return one concept per line and no explanations."
        ),
    }]
    for index in example_indices.tolist():
        sample = dataset[index]
        native = sample["concepts"]["native"]
        digit = int(native[dataset.concept_names.index("digit")])
        color_id = int(native[dataset.concept_names.index("color")])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model", default="gemini/gemini-3.5-flash")
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
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
    # The generator sees only the training dataset and proposes a shared
    # concept vocabulary for the downstream annotator.
    generator = LLMConceptGenerator(
        llm=llm,
        prompt=dataset_aware_prompt,
    )
    # The annotator scores each image against the generated concept vocabulary.
    # In this example we ask it to annotate both train and validation splits.
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

    # TODO: Use the datamodule to generate and annnotate the datasets (you have the concatenated dataset in the datamodule, so you can use that to generate concepts and annotate both train and validation splits). This implies modifiyng the generate_concepts method in dataset.
    train_name = "train_CLIPAnnotator"
    val_name = "val_CLIPAnnotator"

    # Build validation first so the training dataset can ask the pipeline to
    # annotate it in the same call that generates concepts from train.
    val_dataset = ColorMNISTDataset(
        root="./data",
        train=False,
        coloring={"red": range(6), "green": range(6, 10)},
        indices=range(10000),
    )
    train_dataset = ColorMNISTDataset(
        root="./data",
        train=True,
        coloring={"red": range(6), "green": range(6, 10)},
        indices=range(10000),
    )

    # Save the native task labels before generated concepts are selected as
    # ground truth below. That selection changes ``concept_names`` to the
    # generated vocabulary.
    train_labels = train_dataset.concepts[
        :, train_dataset.concept_names.index("parity")
    ].long()
    val_labels = val_dataset.concepts[
        :, val_dataset.concept_names.index("parity")
    ].long()

    train_dataset.generate_concepts( # TODO: use the datamodule method to call the generate_concepts() on dataset
        pipeline,
        class_names=["even", "odd"],
        self_annotation_name="train",
        datasets_to_annotate={"val": val_dataset},
        use_as_gt=True,
        generated_gt_name=train_name,
    )

    train_generated = train_dataset.generated_concepts[train_name]
    concept_axis = train_generated.annotation
    train_concepts = train_generated.float()
    # Validation was annotated in the same pipeline call, but it is not this
    # dataset's ground truth, so read it from the other generated concept output.
    val_concepts = train_dataset.generated_concepts[val_name].float()

    # These calibrated values are probabilities, so a filtered annotation
    # means that the concept is absent from that sample.
    train_concepts = torch.nan_to_num(train_concepts, nan=0.0)
    val_concepts = torch.nan_to_num(val_concepts, nan=0.0)
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
