"""
HAM10000: loading and label-free concept generation (Data Interface)
====================================================================

This example demonstrates how to:
1. Load HAM10000 through PyC's datamodule utilities
2. Read the native concepts that ship with the dataset
3. Run the smallest possible concept-generation pipeline over the images

Key Components:
- HAM10000DataModule: PyC datamodule wrapping the dermoscopic image dataset
- FixedConceptGenerator: returns a hand-written concept vocabulary
- CLIPAnnotator: scores every image against every concept with frozen CLIP
- ConceptGenerationPipeline: wires a generator to an annotator

Dataset: HAM10000, 10,015 dermoscopic images, 4 native concepts
(diagnosis, age, sex, localization)

Note: the first run downloads the image archives from Harvard Dataverse (a few
GB) and extracts them under ``<root>/images``.
"""
from torch_concepts import seed_everything
from torch_concepts.data import HAM10000DataModule
from torch_concepts.data.generation import ConceptGenerationPipeline
from torch_concepts.data.generation.annotators import CLIPAnnotator
from torch_concepts.data.generation.calibrators import SigmoidCalibrator
from torch_concepts.data.generation.generators import FixedConceptGenerator

# Dermoscopic criteria, written by hand. The examples in
# examples/utilization/4_label_free/ replace this with an LLM generator.
CONCEPTS = [
    'an asymmetric lesion',
    'an irregular border',
    'more than one colour',
    'a blue-white veil',
    'an image of a dog',
    'an image of a cat'
]


def main():
    seed_everything(42)

    # =========================================================================
    # Load HAM10000
    # =========================================================================
    print("\n1. Loading HAM10000 dataset...")

    # HAM10000 ships no official split, so the datamodule draws a random one.
    dm = HAM10000DataModule(
        root='./data/ham10000',
        batch_size=64,
        max_samples=256,
        seed=42,
    )
    dm.setup()
    print(f"   Dataset size: {dm.n_samples} samples")
    print(f"   Image shape: {dm.n_features}")
    print(f"   Concepts: {dm.concept_names} "
          f"(cardinalities {dm.annotations.cardinalities})")

    batch = next(iter(dm.train_dataloader()))
    print(f"   inputs:   {tuple(batch['inputs']['x'].shape)}")
    print(f"   concepts: {tuple(batch['concepts']['c'].shape)}")

    # =========================================================================
    # Generate concepts
    # =========================================================================
    print("\n2. Annotating images with CLIP...")

    # The smallest useful pipeline: one generator proposing the vocabulary,
    # one annotator scoring it, and a calibrator turning the raw cosine
    # similarities into probabilities.
    pipeline = ConceptGenerationPipeline(
        generators=FixedConceptGenerator(
            labels=CONCEPTS,
            types=['binary'] * len(CONCEPTS),
            cardinalities=[1] * len(CONCEPTS),
        ),
        annotators=CLIPAnnotator(
            model_name='openai/clip-vit-large-patch14',
            prompt_template='dermoscopy of {}',
            batch_size=64,
            show_progress=True,
        ),
        calibrator=SigmoidCalibrator(scale=10.0),
    )
    generated = pipeline(dm.dataset)['CLIPAnnotator']

    print(f"   Generated: {tuple(generated.shape)}")
    for index, label in enumerate(generated.annotation.labels):
        print(f"     P({label}) = {generated.tensor[:, index].mean():.2f}")


if __name__ == "__main__":
    main()
