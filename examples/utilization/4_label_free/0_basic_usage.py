"""Minimal label-free concept supervision example on ColorMNIST.

This example uses:
- LLMConceptGenerator with LiteLLMBackend to produce a concept vocabulary.
- CLIPAnnotator to score each image against the generated concepts.
- ConceptSupervisionPipeline to connect generation and annotation.
- A tiny concept bottleneck classifier trained on the generated concepts.

Usage:

    export GEMINI_API_KEY="your_google_ai_studio_key"

    python -m examples.utilization.4_label_free.0_basic_usage \
      --llm-model gemini/gemini-3.5-flash \
      --llm-temperature 1.0

For another LiteLLM provider, set the API key expected by that provider, e.g.
``OPENAI_API_KEY`` for ``--llm-model openai/gpt-4o``.
"""

import argparse

import torch
from torch import nn
from torch.utils.data import Subset
import torchvision.transforms.functional as TF
from tqdm import tqdm

from torch_concepts.data.annotators import CLIPAnnotator
from torch_concepts.data.base import ConceptSupervisionPipeline
from torch_concepts.data.concept_generators import LiteLLMBackend, LLMConceptGenerator
from torch_concepts.data.datasets.mnist import ColorMNISTDataset


def prepare_clip_image(image):
    image = image.float() / 255
    image = TF.resize(image, [224, 224], antialias=True)
    return TF.normalize(
        image,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model", default="gemini/gemini-3.5-flash")
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    args = parser.parse_args()

    torch.manual_seed(0)

    train_data = ColorMNISTDataset(
        root="./data",
        train=True,
        download=True,
        random=False,
    )
    val_data = ColorMNISTDataset(
        root="./data",
        train=False,
        download=True,
        random=False,
    )
    train_dataset = Subset(train_data, range(60000))
    val_dataset = Subset(val_data, range(10000))

    llm = LiteLLMBackend(
        model=args.llm_model,
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
    )
    generator = LLMConceptGenerator(
        llm=llm,
        prompt=(
            "Generate 12 short visual concepts useful for classifying "
            "ColorMNIST images as even or odd. Include digit identity, color, "
            "and simple shape concepts. The classes are: {class_names}. "
            "Return one concept per line and no explanations."
        ),
    )
    annotator = CLIPAnnotator(
        model_name="ViT-B-32",
        pretrained="openai",
        output="similarity",
        prompt_template="a photo of a {}",
        input_transform=prepare_clip_image,
        batch_size=128,
        device="mps",
        show_progress=True,
    )
    pipeline = ConceptSupervisionPipeline(
        generators=generator,
        annotators=annotator,
        routing="merged",
    )

    train_values, concepts = pipeline(
        train_dataset,
        class_names=["red digit", "green digit"],
    )

    name = next(iter(concepts))
    concept_axis = concepts[name]
    train_concepts = train_values[name].float()
    val_concepts = annotator.annotate(val_dataset, concept_axis).float()

    mean = train_concepts.mean(dim=0, keepdim=True)
    std = train_concepts.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_concepts = (train_concepts - mean) / std
    val_concepts = (val_concepts - mean) / std

    train_labels = torch.stack(
        [train_dataset[index][2] for index in range(len(train_dataset))]
    ).argmax(1)
    val_labels = torch.stack(
        [val_dataset[index][2] for index in range(len(val_dataset))]
    ).argmax(1)

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
