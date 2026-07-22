from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_concepts import Annotations
from torch_concepts.data.base.annotator import Annotator


PromptTemplate = str | Sequence[str] | Callable[[str], str | Sequence[str]]
BinaryPromptFormatter = Callable[[str], str]
StatePromptFormatter = Callable[[str, str], str]


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
    tensor of sample-level concept values. Binary concepts are represented by
    their labels; categorical concepts use one text prompt per state.

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
    output : str, optional
        Output representation: ``"similarity"``, ``"logit"``,
        ``"probability"``, or ``"binary"``.
    temperature : float, optional
        Multiplicative logit scale. Default is 1.0.
    bias : float, optional
        Additive logit bias. Default is 0.0.
    threshold : float, optional
        Probability threshold used for binary output. Default is 0.5.
    num_workers : int, optional
        Number of data-loading workers. Default is 0.
    show_progress : bool, optional
        Whether to show progress bars while encoding text concepts and image
        batches. Default is False.
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
        output: str = "similarity",
        temperature: float = 1.0,
        bias: float = 0.0,
        threshold: float = 0.5,
        num_workers: int = 0,
        show_progress: bool = False,
    ):
        if output not in {"similarity", "logit", "probability", "binary"}:
            raise ValueError(
                "output must be one of: "
                "'similarity', 'logit', 'probability', 'binary'."
            )

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
        self.output = output
        self.temperature = temperature
        self.bias = bias
        self.threshold = threshold
        self.num_workers = num_workers
        self.show_progress = show_progress

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def annotate(
        self,
        dataset: Dataset,
        concepts: Annotations,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        if not isinstance(concepts, Annotations):
            raise TypeError("concepts must be an Annotations.")

        text_concepts = self._flatten_concept_prompts(concepts)
        if not text_concepts:
            raise ValueError("Cannot annotate an empty concept axis.")
        concept_features = self._encode_text_concepts(text_concepts)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )

        concept_batches = []
        batches = self._progress(
            loader,
            desc="CLIP image annotation",
            total=len(loader),
        )
        for batch in batches:
            images = [self.input_getter(sample) for sample in batch]
            with torch.no_grad():
                image_features = self.encode_images(images)
                scores = image_features @ concept_features.T
                scores = self._postprocess_scores(scores)

            concept_batches.append(scores.detach().cpu())

        concept_data = (
            torch.cat(concept_batches, dim=0)
            if concept_batches
            else torch.empty((0, concepts.size))
        )
        return concept_data

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
        all_features = []
        concept_iterator = self._progress(
            concepts,
            desc="CLIP text encoding",
            total=len(concepts),
        )
        for concept in concept_iterator:
            prompts = self._make_prompts(concept)
            with torch.no_grad():
                text_features = self.encode_texts(prompts)
                text_feature = text_features.mean(dim=0)
                text_feature = F.normalize(text_feature, dim=0)
            all_features.append(text_feature)
        return torch.stack(all_features, dim=0)

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        """Encode and normalize text with the Hugging Face model."""
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

    def encode_images(self, images: Sequence[Any]) -> Tensor:
        """Preprocess, encode, and normalize images with Hugging Face."""
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(features, dim=-1)

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

    def _postprocess_scores(self, similarities: Tensor) -> Tensor:
        if self.output == "similarity":
            return similarities

        logits = similarities * self.temperature + self.bias
        if self.output == "logit":
            return logits

        probabilities = torch.sigmoid(logits)
        if self.output == "probability":
            return probabilities
        return (probabilities >= self.threshold).float()
