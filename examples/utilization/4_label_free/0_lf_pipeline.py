"""LF-CBM-style concept supervision with Gemini and CLIP.

Before running, create a Gemini API key in Google AI Studio and export it:

    export GEMINI_API_KEY="your_google_ai_studio_key"

The example downloads official CUB data when needed and creates the small
preprocessed split consumed by ``CUBDataset``. Its ``build()`` runs the pipeline.
"""

import argparse
import hashlib
import json
import os
import pickle
import tarfile
from pathlib import Path
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

from torch_concepts.data.annotators import CLIPAnnotator
from torch_concepts.data.base import ConceptSupervisionPipeline
from torch_concepts.data.concept_generators import LLMConceptGenerator
from torch_concepts.data.datasets import CUBDataset


CUB_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz?download=1"
)
CUB_MD5 = "97eceeb196236b17998738112f37df78"


PROMPT = """
Generate short visual concepts useful for distinguishing these classes:
{class_names}

Return exactly {num_concepts} concepts, one per line.

Rules:
- every concept must be binary: clearly answerable with yes or no
- every concept must represent exactly one visual property
- never combine properties with words such as and, or, or with
- use a short phrase starting with has, for example has red bill
- include only directly visible bird attributes
- do not include class names
- do not use Markdown, bullets, numbering, headings, colons, or descriptions
- output only the concept names
"""


def gemini_llm(prompt: str, **kwargs) -> str:
    """Send a prompt to Gemini using its REST API.
    TODO: generalize this (look for libraries that support multiple LLMs)
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a free key in Google AI Studio "
            "and export it before running this example."
        )

    model = kwargs.pop("model", "gemini-2.5-flash")
    temperature = kwargs.pop("temperature", None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported Gemini request options: {unsupported}")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
    }
    if temperature is not None:
        payload["generationConfig"] = {"temperature": temperature}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8")
        raise RuntimeError(f"Gemini API request failed: {detail}") from error

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected Gemini API response: {data}") from error

    return "\n".join(part.get("text", "") for part in parts).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CUB_DATA_ROOT", "./data"),
        help="Directory used to store/download CUB.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-concepts", type=int, default=10)
    parser.add_argument("--num-preview", type=int, default=4)
    parser.add_argument(
        "--preview-output",
        default="cub_concept_annotations.png",
        help="Path for the image-and-annotation preview.",
    )
    return parser.parse_args()


def prepare_cub(data_root):
    """Download and extract the official CUB archive when needed."""
    data_root = Path(data_root).expanduser()
    cub_root = data_root / "CUB_200_2011"
    if (cub_root / "images.txt").is_file():
        return cub_root

    archive_path = data_root / "CUB_200_2011.tgz"
    data_root.mkdir(parents=True, exist_ok=True)
    if not archive_path.is_file():
        print(f"Downloading CUB-200-2011 to {archive_path}...")
        urllib.request.urlretrieve(CUB_URL, archive_path)

    check_md5(archive_path, CUB_MD5)
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, data_root)

    if not (cub_root / "images.txt").is_file():
        raise RuntimeError(
            f"CUB extraction did not create the expected directory: {cub_root}"
        )
    return cub_root


def check_md5(path, expected_md5):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_md5 = digest.hexdigest()
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}"
        )


def safe_extract(archive, destination):
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise RuntimeError(f"Refusing to extract unsafe path: {member.name}")
    archive.extractall(destination)


def prepare_cub_dataset(data_root, num_samples):
    """Download CUB and create the pickle consumed by CUBDataset."""
    cub_root = prepare_cub(data_root)

    images = _read_indexed_file(cub_root / "images.txt")
    labels = _read_indexed_file(
        cub_root / "image_class_labels.txt", int
    )
    train_flags = _read_indexed_file(
        cub_root / "train_test_split.txt", int
    )
    image_ids = [
        image_id for image_id in images if train_flags[image_id]
    ][:num_samples]

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
    attributes_path = cub_root / "attributes" / "image_attribute_labels.txt"
    with attributes_path.open() as file:
        for line in file:
            image_id, attribute_id, present, certainty = line.split()[:4]
            image_id = int(image_id)
            if image_id not in records:
                continue
            index = int(attribute_id) - 1
            present = int(present)
            records[image_id]["attribute_label"][index] = present
            records[image_id]["uncertain_attribute_label"][index] = present
            records[image_id]["attribute_certainty"][index] = int(certainty)

    processed_dir = cub_root / "class_attr_data_10"
    processed_dir.mkdir(exist_ok=True)
    with (processed_dir / "train.pkl").open("wb") as file:
        pickle.dump([records[image_id] for image_id in image_ids], file)
    return str(cub_root)


def _read_indexed_file(path, parser=str):
    values = {}
    with Path(path).open() as file:
        for line in file:
            index, value = line.strip().split(maxsplit=1)
            values[int(index)] = parser(value)
    return values


def prepare_cub_image(image):
    """Convert CUB normalized tensors to CLIP input tensors."""
    image = image * 2 + 0.5
    image = TF.resize(image, [224, 224], antialias=True)
    return TF.normalize(
        image,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )


def save_annotation_preview(
    dataset,
    values,
    concept_names,
    output_path,
    num_preview,
):
    # Save CUB images beside their generated concept annotations.
    count = min(num_preview, len(dataset))
    figure, axes = plt.subplots(count, 2, figsize=(13, 4 * count))
    if count == 1:
        axes = [axes]

    for index, (image_axis, text_axis) in enumerate(axes):
        image = Image.open(dataset.data[index]["img_path"]).convert("RGB")
        class_index = int(dataset.data[index]["class_label"])
        class_name = dataset.task_names[class_index].replace("_", " ")
        scores = values[index]
        positive = [
            position
            for position in scores.argsort(descending=True).tolist()
            if scores[position] >= 0.5
        ]
        negative = [
            position
            for position in scores.argsort().tolist()
            if scores[position] < 0.5
        ]
        lines = ["PRESENT:"]
        lines.extend(
            f"+ {concept_names[position]}: {scores[position]:.3f}"
            for position in positive
        )
        lines.append("")
        lines.append("ABSENT:")
        lines.extend(
            f"- {concept_names[position]}: {scores[position]:.3f}"
            for position in negative
        )

        image_axis.imshow(image)
        image_axis.set_title(class_name)
        image_axis.axis("off")
        text_axis.text(
            0,
            1,
            "\n".join(lines),
            va="top",
            family="monospace",
            fontsize=10,
        )
        text_axis.set_title("Generated concept probabilities")
        text_axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def clip_binary_prompt(concept_name):
    """Turn a binary concept label into a natural CLIP image prompt."""
    concept = concept_name.strip().rstrip(".")
    if concept.lower().startswith("has "):
        concept = concept[4:]
    return f"a bird with {concept}"


def main():
    args = parse_args()
    cub_root = prepare_cub_dataset(args.data_root, args.num_samples)

    generator = LLMConceptGenerator(
        llm=gemini_llm,
        prompt=PROMPT,
        llm_kwargs={
            "temperature": 0.5,
        },
    )

    annotator = CLIPAnnotator(
        model_name="ViT-B-32",
        pretrained="openai",
        output="binary",
        prompt_template=[
            "a photo of {}",
            "a close-up photo of {}",
        ],
        binary_prompt_formatter=clip_binary_prompt,
        input_transform=prepare_cub_image,
    )

    pipeline = ConceptSupervisionPipeline(
        generators=generator,
        annotators=annotator,
        routing="merged",
    )

    dataset = CUBDataset(
        root=cub_root,
        split="train",
        concept_pipeline=pipeline,
        use_as_gt=False,
        concept_pipeline_kwargs={"num_concepts": args.num_concepts},
    )

    annotator_name = next(iter(dataset.generated_concepts))
    concept_vocabulary = dataset.generated_concepts[annotator_name]
    sample = dataset[0]

    print("Image shape:", tuple(sample["inputs"]["x"].shape))
    print("Native concepts shape:", tuple(sample["concepts"]["native"].shape))
    print(
        "Generated annotations shape:",
        tuple(sample["concepts"]["generated"][annotator_name].shape),
    )
    print(
        "Ground-truth concepts shape:",
        tuple(sample["concepts"]["ground_truth"].shape),
    )
    print("Number of generated concepts:", len(concept_vocabulary.labels))
    print("Generated concept names:", concept_vocabulary.labels)

    save_annotation_preview(
        dataset=dataset,
        values=dataset.generated_annotations[annotator_name],
        concept_names=concept_vocabulary.labels,
        output_path=args.preview_output,
        num_preview=args.num_preview,
    )
    print("Annotation preview saved to:", args.preview_output)


if __name__ == "__main__":
    main()
