"""Minimal label-free concept supervision example on ColorMNIST.

This example uses:
- LLMConceptGenerator with LiteLLMBackend to produce a concept vocabulary.
- CLIPAnnotator to score each image against the generated concepts.
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

import argparse

import torch
from torch import nn
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

    llm = LiteLLMBackend(
        model=args.llm_model,
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
    )
    # The generator sees only the training dataset and proposes a shared
    # concept vocabulary for the downstream annotator.
    generator = LLMConceptGenerator(
        llm=llm,
        prompt=(
            "Generate 12 short visual concepts useful for classifying "
            "ColorMNIST images as even or odd. Include digit identity, color, "
            "and simple shape concepts. The classes are: {class_names}. "
            "Return one concept per line and no explanations."
        ),
    )
    # The annotator scores each image against the generated concept vocabulary.
    # In this example we ask it to annotate both train and validation splits.
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
    # The pipeline wires concept generation and annotation together. With
    # routing="merged", all generated concepts are passed to the annotator as a
    # single concept axis.
    pipeline = ConceptSupervisionPipeline(
        generators=generator,
        annotators=annotator,
        routing="merged",
    )

    train_name = "train_CLIPAnnotator"
    val_name = "val_CLIPAnnotator"

    # Build validation first so the training dataset can ask the pipeline to
    # annotate it in the same call that generates concepts from train.
    val_dataset = ColorMNISTDataset(
        root="./data",
        train=False,
        download=True,
        random=False,
        indices=range(10000),
    )
    # Passing concept_pipeline makes ColorMNIST run concept generation inside
    # the dataset constructor. Concepts are generated from this training split;
    # datasets_to_annotate adds the validation split to the annotation pass.
    train_dataset = ColorMNISTDataset(
        root="./data",
        train=True,
        download=True,
        random=False,
        indices=range(10000),
        concept_pipeline=pipeline,
        # Select the generated train annotations as dataset.ground_truth. This
        # keeps the code related to the learning independent of the pipeline output dict.
        use_as_gt=True,
        generated_gt_name=train_name,
        concept_pipeline_kwargs={
            "class_names": ["red digit", "green digit"],
            # Name this dataset "train" in the annotation outputs. This makes
            # the train annotations available as "train_CLIPAnnotator".
            "self_annotation_name": "train",
            "datasets_to_annotate": {
                "val": val_dataset,
            },
        },
    )

    concept_axis = train_dataset.generated_concepts[train_name]
    # Because use_as_gt=True selected train_name above, the training concept
    # matrix is available through the standard dataset.ground_truth field.
    train_concepts = train_dataset.ground_truth.float()
    # Validation was annotated in the same pipeline call, but it is not this
    # dataset's ground truth, so read it from the generated annotation outputs.
    val_concepts = train_dataset.generated_annotations[val_name].float()

    mean = train_concepts.mean(dim=0, keepdim=True)
    std = train_concepts.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_concepts = (train_concepts - mean) / std
    val_concepts = (val_concepts - mean) / std

    # ColorMNIST returns one-hot task labels at index 2; CrossEntropyLoss
    # expects integer class indices, so convert them with argmax.
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
