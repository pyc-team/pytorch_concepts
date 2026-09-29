from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_concepts import Annotations
from torch_concepts.data.generation.base.annotator import Annotator
from torch_concepts.tensor import AnnotatedTensor


PromptTemplate = str | Sequence[str] | Callable[[str], str | Sequence[str]]
BinaryPromptFormatter = Callable[[str], str]
StatePromptFormatter = Callable[[str, str], str]


def _identity_collate(batch):
    """Return a sample list unchanged for CLIP image preprocessing."""
    return batch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or prefer CUDA, then MPS, then CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_input_getter(sample: Any) -> Any:
    """Extract the image/input from common dataset sample formats."""
    if isinstance(sample, dict):
        if "inputs" in sample and isinstance(sample["inputs"], dict):
            return sample["inputs"]["x"]
        if "x" in sample:
            return sample["x"]
    if isinstance(sample, (tuple, list)):
        return sample[0]
    return sample


def default_binary_prompt_formatter(concept_name: str) -> str:
    return concept_name


def default_state_prompt_formatter(concept_name: str, state_name: str) -> str:
    return f"{concept_name} {state_name}"


class CLIPAnnotator(Annotator):
    """General CLIP-based annotator for label-free concept supervision.

    The annotator maps an image dataset and an :class:`Annotations` to a
    tensor of sample-level similarities. Binary concepts are represented by
    their labels; categorical concepts use one text prompt per state.
    Calibration and filtering are handled by the concept-supervision pipeline.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model identifier. Defaults to
        ``"openai/clip-vit-base-patch32"``.
    batch_size : int, optional
        Batch size used while annotating the dataset. Default is 64.
    device : str or torch.device, optional
        Device on which CLIP inference runs. By default CUDA is preferred,
        followed by MPS and CPU.
    input_getter : callable, optional
        Function used to extract an image from a dataset sample.
    prompt_template : str, sequence of str, or callable, optional
        Template or function applied after concept/state prompt formatting.
    binary_prompt_formatter : callable, optional
        Converts a binary concept name into prompt text.
    state_prompt_formatter : callable, optional
        Converts a categorical concept name and state into prompt text.
    num_workers : int, optional
        Number of data-loading workers. Default is 0.
    show_progress : bool, optional
        Whether to show progress bars while encoding text concepts and image
        batches. Default is False.
    normalize : bool, optional
        Whether to L2-normalize image and text embeddings. The default is
        ``True``, producing cosine similarities. Set to ``False`` to retain
        raw CLIP embeddings and produce dot-product scores.

    Examples
    --------
    Annotate a small image dataset with binary and categorical concepts.
    Constructing the annotator downloads the Hugging Face model on first use.
    This example exercises every constructor option; ``input_getter`` is only
    needed because this dataset stores images under a custom ``"image"`` key.
    For datasets returning ``{"inputs": {"x": image}}``, ``{"x": image}``,
    tuples, or images directly, the default getter is sufficient.

    .. code-block:: python

        import torch
        from torch.utils.data import Dataset

        from torch_concepts import Annotations
        from torch_concepts.data.generation.annotators import CLIPAnnotator

        class ImageDataset(Dataset):
            def __init__(self):
                self.images = torch.rand(4, 3, 224, 224)

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                return {"image": self.images[index]}

        dataset = ImageDataset()
        concepts = Annotations(
            labels=["has feathers", "color"],
            states=[["0"], ["red", "blue"]],
            types=["binary", "categorical"],
        )
        annotator = CLIPAnnotator(
            # Hugging Face model identifier.
            model_name="openai/clip-vit-base-patch32",
            # Encode two images per CLIP batch.
            batch_size=2,
            # Extract images from this dataset's custom sample dictionary.
            input_getter=lambda sample: sample["image"],
            # Add the same text prompt template to every concept/state prompt.
            prompt_template="a photo of {}",
            binary_prompt_formatter=lambda name: name,
            state_prompt_formatter=lambda name, state: f"{state} {name}",
        )
        scores = annotator.annotate(dataset, concepts)
        print(scores.shape)  # torch.Size([4, 3])
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 64,
        device: str | torch.device | None = None,
        input_getter: Callable[[Any], Any] = default_input_getter,
        prompt_template: PromptTemplate = "{}",
        binary_prompt_formatter: BinaryPromptFormatter = (
            default_binary_prompt_formatter
        ),
        state_prompt_formatter: StatePromptFormatter = (
            default_state_prompt_formatter
        ),
        num_workers: int = 0,
        show_progress: bool = False,
        normalize: bool = True,
    ):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise ImportError(
                "CLIPAnnotator requires transformers. Install the "
                "pytorch-concepts data extras or run: pip install transformers"
            ) from error

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.input_getter = input_getter
        self.prompt_template = prompt_template
        self.binary_prompt_formatter = binary_prompt_formatter
        self.state_prompt_formatter = state_prompt_formatter
        self.num_workers = num_workers
        self.show_progress = show_progress
        self.normalize = normalize

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def annotate(
        self,
        dataset: Dataset,
        concepts: Annotations,
        *,
        image_features: Tensor | None = None,
        concept_features: Tensor | None = None,
        **kwargs: Any,
    ) -> AnnotatedTensor:
        """Annotate a dataset, optionally reusing ordered CLIP features."""
        del kwargs
        if not isinstance(concepts, Annotations):
            raise TypeError("concepts must be an Annotations.")

        if concept_features is None:
            concept_features = self.encode_concepts(concepts)
        if image_features is None:
            image_features = self.encode_dataset(dataset)
        if image_features.shape[0] != len(dataset):
            raise ValueError(
                "image_features must contain one row per dataset sample."
            )
        if concept_features.shape[0] != concepts.size:
            raise ValueError(
                "concept_features must contain one row per annotation column."
            )

        concept_features = concept_features.to(self.device)
        score_batches = []
        for batch in image_features.split(self.batch_size):
            with torch.no_grad():
                score_batches.append((batch.to(self.device) @ concept_features.T).cpu())
        concept_data = torch.cat(score_batches) if score_batches else torch.empty(
            (0, concepts.size)
        )
        return AnnotatedTensor(concept_data, concepts, axis=1)

    def encode_concepts(self, concepts: Annotations) -> Tensor:
        """Encode every column represented by an annotation axis."""
        if not isinstance(concepts, Annotations):
            raise TypeError("concepts must be an Annotations.")
        text_concepts = self._flatten_concept_prompts(concepts)
        if not text_concepts:
            raise ValueError("Cannot encode an empty concept axis.")
        return self._encode_text_concepts(text_concepts)

    def encode_dataset(self, dataset: Dataset) -> Tensor:
        """Encode a dataset in order and return CPU image features."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_identity_collate,
        )
        batches = self._progress(loader, desc="CLIP image encoding", total=len(loader))
        features = []
        for batch in batches:
            images = [self.input_getter(sample) for sample in batch]
            with torch.no_grad():
                features.append(self.encode_images(images).cpu())
        if features:
            return torch.cat(features)
        projection_dim = self.model.config.projection_dim
        return torch.empty((0, projection_dim))

    def _flatten_concept_prompts(
        self,
        concepts: Annotations,
    ) -> list[str]:
        prompts: list[str] = []
        for label, states, cardinality in zip(
            concepts.labels,
            concepts.states,
            concepts.cardinalities,
        ):
            if cardinality == 1:
                prompts.append(self.binary_prompt_formatter(label))
            else:
                prompts.extend(
                    self.state_prompt_formatter(label, state)
                    for state in states
                )
        return prompts

    def _encode_text_concepts(self, concepts: Sequence[str]) -> Tensor:
        prompt_groups = [self._make_prompts(concept) for concept in concepts]
        prompts = [prompt for group in prompt_groups for prompt in group]
        starts = []
        position = 0
        for group in prompt_groups:
            starts.append((position, position + len(group)))
            position += len(group)

        batches = range(0, len(prompts), self.batch_size)
        batches = self._progress(
            batches,
            desc="CLIP text encoding",
            total=(len(prompts) + self.batch_size - 1) // self.batch_size,
        )
        with torch.no_grad():
            encoded = torch.cat(
                [
                    self.encode_texts(prompts[start : start + self.batch_size])
                    for start in batches
                ]
            )

        all_features = []
        for start, end in starts:
            text_feature = encoded[start:end].mean(dim=0)
            if self.normalize:
                text_feature = F.normalize(text_feature, dim=0)
            all_features.append(text_feature)
        return torch.stack(all_features, dim=0)

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        """Encode text, normalizing embeddings when configured."""
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1) if self.normalize else features

    def encode_images(self, images: Sequence[Any]) -> Tensor:
        """Preprocess and encode images, normalizing when configured."""
        processor_kwargs = {}
        if images and all(
            isinstance(image, Tensor)
            and image.is_floating_point()
            and image.numel() > 0
            and image.min().item() >= 0.0
            and image.max().item() <= 1.0
            for image in images
        ):
            # Hugging Face image processors otherwise divide by 255 again.
            processor_kwargs["do_rescale"] = False
        inputs = self.processor(
            images=list(images),
            return_tensors="pt",
            **processor_kwargs,
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(features, dim=-1) if self.normalize else features

    def _progress(self, iterable: Any, desc: str, total: int | None = None) -> Any:
        if not self.show_progress:
            return iterable
        try:
            from tqdm import tqdm
        except ImportError:
            return iterable
        return tqdm(iterable, desc=desc, total=total)

    def _make_prompts(self, concept: str) -> list[str]:
        template = self.prompt_template
        if callable(template):
            prompts = template(concept)
            return [prompts] if isinstance(prompts, str) else list(prompts)
        if isinstance(template, str):
            return [template.format(concept)]
        return [item.format(concept) for item in template]
