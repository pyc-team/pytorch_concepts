"""Label-free concept supervision on ColorMNIST with flexible dataset layouts.

This example uses:
- LLMConceptGenerator with LiteLLMBackend to produce a concept vocabulary.
- CLIPAnnotator to produce raw image-concept similarity scores.
- Calibrator and FilterAnnotator stages to turn similarities into
  probabilities and filter uncertain sample-level annotations.
- ConceptGenerationPipeline, whose concept-discovery and annotation targets
  can be chosen independently.
- A PyC concept bottleneck model supervised by the generated concepts.

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
import math
from io import BytesIO

import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, Subset
from PIL import Image

from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.filters import (
    ThresholdAnnotationFilter,
)
from torch_concepts.data.generation.generators import LiteLLMBackend, LLMConceptGenerator
from torch_concepts.data import ColorMNISTDataModule
from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss


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
        "--train-epochs",
        type=int,
        default=5,
        help="Number of CBM training epochs for the train-to-full mode.",
    )
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
    pipeline = ConceptGenerationPipeline(
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

    generated_name = "CLIPAnnotator"
    datamodule.generate_concepts(
        pipeline,
        generation_indices=train_indices,
        use_as_gt=True,
        generated_gt_name=generated_name,
        **pipeline_kwargs,
    )

    generated = dataset.generated_concepts[generated_name]
    generated_annotation = generated.annotation
    if "parity" in generated_annotation.labels:
        raise ValueError(
            "The generated vocabulary contains 'parity', which collides with "
            "the downstream task name. Regenerate concepts without that label."
        )

    parity_annotation = dataset._all_concept_annotation.subset(["parity"])
    model_annotations = generated_annotation.union_with(parity_annotation)

    def cbm_collate(samples):
        batch = dataset.collate(samples)
        generated_target = batch["concepts"]["generated"][generated_name]
        parity_target = batch["concepts"]["native"][["parity"]]
        batch["concepts"]["c"] = generated_target.union_with(parity_target)
        return batch

    train_loader = DataLoader(
        datamodule.trainset,
        batch_size=datamodule.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=datamodule.workers,
        pin_memory=datamodule.pin_memory,
        collate_fn=cbm_collate,
    )
    val_loader = DataLoader(
        datamodule.valset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=datamodule.workers,
        pin_memory=datamodule.pin_memory,
        collate_fn=cbm_collate,
    )

    model = ConceptBottleneckModel(
        input_size=datamodule.n_features,
        annotations=model_annotations,
        task_names=["parity"],
        backbone=nn.Flatten(),
        latent_size=math.prod(datamodule.n_features),
        lightning=True,
        loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01, "weight_decay": 1e-3},
    )
    trainer = Trainer(
        max_epochs=args.train_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"]["x"].to(model.device)
            output = model(query=["parity"], input=inputs)
            prediction = (
                output.logits["parity"].tensor.squeeze(-1) >= 0
            ).long().cpu()
            target = (
                batch["concepts"]["c"]["parity"].tensor.squeeze(-1).long()
            )
            correct += (prediction == target).sum().item()
            total += target.numel()
    val_acc = correct / total

    print("Generated concepts:", generated_annotation.labels)
    print("Model concepts:", model_annotations.labels)
    print(f"Validation accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    main()
