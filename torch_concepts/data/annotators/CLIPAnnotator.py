from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_concepts import AxisAnnotation
from torch_concepts.data.base.annotator import Annotator


PromptTemplate = str | Sequence[str] | Callable[[str], str | Sequence[str]]
BinaryPromptFormatter = Callable[[str], str]
StatePromptFormatter = Callable[[str, str], str]


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

    The annotator maps an image dataset and an :class:`AxisAnnotation` to a
    tensor of sample-level concept values. Binary concepts are represented by
    their labels; categorical concepts use one text prompt per state.

    Parameters
    ----------
    model_name : str, optional
        The name of the CLIP model to use. Default is ``"ViT-B-32"``.
    pretrained : str, optional
        The pretrained weights to use. Default is ``"openai"``.
    batch_size : int, optional
        Batch size used while annotating the dataset. Default is 64.
    device : str or torch.device, optional
        Device on which CLIP inference runs. Defaults to CUDA when available.
    input_getter : callable, optional
        Function used to extract an image from a dataset sample.
    input_transform : callable, optional
        Optional transform applied to each extracted image before inference.
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
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 64,
        device: str | torch.device | None = None,
        input_getter: Callable[[Any], Any] = default_input_getter,
        input_transform: Callable[[Any], Tensor] | None = None,
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
    ):
        if output not in {"similarity", "logit", "probability", "binary"}:
            raise ValueError(
                "output must be one of: "
                "'similarity', 'logit', 'probability', 'binary'."
            )

        try:
            import open_clip
        except ImportError as error:
            raise ImportError(
                "open_clip is required for CLIPAnnotator. "
                "Install it with: pip install open-clip-torch"
            ) from error

        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_getter = input_getter
        self.input_transform = input_transform
        self.prompt_template = prompt_template
        self.binary_prompt_formatter = binary_prompt_formatter
        self.state_prompt_formatter = state_prompt_formatter
        self.output = output
        self.temperature = temperature
        self.bias = bias
        self.threshold = threshold
        self.num_workers = num_workers

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def annotate(
        self,
        dataset: Dataset,
        concepts: AxisAnnotation,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        if not isinstance(concepts, AxisAnnotation):
            raise TypeError("concepts must be an AxisAnnotation.")

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
        for batch in loader:
            images = [self.input_getter(sample) for sample in batch]
            clip_images = self._prepare_clip_images(images)

            with torch.no_grad():
                image_features = self.model.encode_image(clip_images)
                image_features = F.normalize(image_features, dim=-1)
                scores = image_features @ concept_features.T
                scores = self._postprocess_scores(scores)

            concept_batches.append(scores.detach().cpu())

        concept_data = (
            torch.cat(concept_batches, dim=0)
            if concept_batches
            else torch.empty((0, concepts.shape))
        )
        return concept_data

    def _flatten_concept_prompts(
        self,
        concepts: AxisAnnotation,
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
        for concept in concepts:
            prompts = self._make_prompts(concept)
            with torch.no_grad():
                tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=-1)
                text_feature = text_features.mean(dim=0)
                text_feature = F.normalize(text_feature, dim=0)
            all_features.append(text_feature)
        return torch.stack(all_features, dim=0)

    def _make_prompts(self, concept: str) -> list[str]:
        template = self.prompt_template
        if callable(template):
            prompts = template(concept)
            return [prompts] if isinstance(prompts, str) else list(prompts)
        if isinstance(template, str):
            return [template.format(concept)]
        return [item.format(concept) for item in template]

    def _prepare_clip_images(self, images: Sequence[Any]) -> Tensor:
        processed = []
        for image in images:
            if self.input_transform is not None:
                image = self.input_transform(image)
            if isinstance(image, Tensor):
                processed.append(image)
            else:
                processed.append(self.preprocess(image))
        return torch.stack(processed).to(self.device)

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
