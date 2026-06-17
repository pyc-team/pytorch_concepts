from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base.annotator import Annotator
from torch_concepts.data.base.dataset import ConceptDataset


Concepts = Sequence[str] | Tensor
PromptTemplate = str | Sequence[str] | Callable[[str], str | Sequence[str]]


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


class CLIPAnnotator(Annotator):
    """General CLIP-based annotator for label-free concept supervision.

    The annotator maps:

        image dataset + textual concepts -> ConceptDataset

    It supports the common CLIP-based annotation pattern used in label-free
    CBMs and related methods:

        C[i, j] = score(CLIP_image(x_i), CLIP_text(concept_j))

    Concepts may also be provided directly as embeddings of shape
    ``(n_concepts, embedding_dim)``.
    
    Parameters
    ----------
    model_name : str, optional
        The name of the CLIP model to use. Default is "ViT-B-32".
    pretrained : str, optional
        The pretrained weights to use. Default is "openai".
    batch_size : int, optional
        The batch size for processing the dataset. Default is 64.
    device : str or torch.device, optional
        The device to run the model on. Default is "cuda" if available, else "cpu".
    input_getter : callable, optional
        A function to extract the input (image) from a dataset sample. Default is ``default_input_getter``.
    input_transform : callable, optional
        A function to transform the input (image) before passing it to the model. Default is None, which uses the CLIP preprocessing.
    prompt_template : str, sequence of str, or callable, optional
        A template or function to generate prompts for the concepts. Default is "a photo of {}".
    output : str, optional
        The type of output to return. One of "similarity", "logit", "probability", or "binary". Default is "similarity".
    temperature : float, optional
        The temperature for scaling logits. Default is 1.0.
    bias : float, optional
        The bias for scaling logits. Default is 0.0.
    threshold : float, optional
        The threshold for converting probabilities to binary outputs. Default is 0.5.
    num_workers : int, optional
        The number of worker processes for data loading. Default is 0 (no additional workers).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 64,
        device: str | torch.device | None = None,
        input_getter: Callable[[Any], Any] = default_input_getter,
        input_transform: Callable[[Any], Tensor] | None = None,
        prompt_template: PromptTemplate = "a photo of {}",
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
        except ImportError as e:
            raise ImportError(
                "open_clip is required for CLIPAnnotator. "
                "Install it with: pip install open-clip-torch"
            ) from e

        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_getter = input_getter
        self.input_transform = input_transform
        self.prompt_template = prompt_template
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
        concepts: Concepts,
        concept_names: Sequence[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> ConceptDataset:
        concept_features, concept_names = self._prepare_concepts(
            concepts=concepts,
            concept_names=concept_names,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )

        xs = []
        cs = []

        for batch in loader:
            images = [self.input_getter(sample) for sample in batch]

            clip_images = self._prepare_clip_images(images)

            with torch.no_grad():
                image_features = self.model.encode_image(clip_images)
                image_features = F.normalize(image_features, dim=-1)

                scores = image_features @ concept_features.T
                scores = self._postprocess_scores(scores)

            xs.extend(self._prepare_dataset_inputs(images))
            cs.append(scores.detach().cpu())

        input_data = torch.stack(xs)
        concept_data = torch.cat(cs, dim=0)

        annotations = self._make_annotations(concept_names)

        return ConceptDataset(
            input_data=input_data,
            concepts=concept_data,
            annotations=annotations,
            name=name or "CLIPConceptDataset",
        )

    def _prepare_concepts(
        self,
        concepts: Concepts,
        concept_names: Sequence[str] | None,
    ) -> tuple[Tensor, list[str]]:
        if isinstance(concepts, Tensor):
            if concepts.ndim != 2:
                raise ValueError(
                    "Concept embeddings must have shape "
                    "(n_concepts, embedding_dim)."
                )

            features = F.normalize(concepts.to(self.device), dim=-1)

            if concept_names is None:
                names = [f"concept_{i}" for i in range(concepts.shape[0])]
            else:
                names = list(concept_names)

            if len(names) != concepts.shape[0]:
                raise ValueError(
                    "concept_names must match the number of concept embeddings."
                )

            return features, names

        if not all(isinstance(c, str) for c in concepts):
            raise TypeError(
                "concepts must be either a sequence of strings or a tensor "
                "of precomputed concept embeddings."
            )

        names = list(concepts) if concept_names is None else list(concept_names)

        if len(names) != len(concepts):
            raise ValueError("concept_names must match the number of concepts.")

        features = self._encode_text_concepts(list(concepts))

        return features, names

    def _encode_text_concepts(self, concepts: Sequence[str]) -> Tensor:
        all_features = []

        for concept in concepts:
            prompts = self._make_prompts(concept)

            with torch.no_grad():
                tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=-1)

                # Prompt ensembling: average all prompt-template embeddings.
                text_feature = text_features.mean(dim=0)
                text_feature = F.normalize(text_feature, dim=0)

            all_features.append(text_feature)

        return torch.stack(all_features, dim=0)

    def _make_prompts(self, concept: str) -> list[str]:
        template = self.prompt_template

        if callable(template):
            prompts = template(concept)

            if isinstance(prompts, str):
                return [prompts]

            return list(prompts)

        if isinstance(template, str):
            return [template.format(concept)]

        return [t.format(concept) for t in template]

    def _prepare_clip_images(self, images: Sequence[Any]) -> Tensor:
        processed = []

        for image in images:
            if isinstance(image, Tensor):
                processed.append(image)
            else:
                processed.append(self.preprocess(image))

        return torch.stack(processed).to(self.device)

    def _prepare_dataset_inputs(self, images: Sequence[Any]) -> list[Tensor]:
        xs = []

        for image in images:
            if self.input_transform is not None:
                x = self.input_transform(image)
            elif isinstance(image, Tensor):
                x = image
            else:
                # Fall back to CLIP preprocessing so that the result can always
                # be represented as a tensor-based ConceptDataset.
                x = self.preprocess(image)

            if not isinstance(x, Tensor):
                raise TypeError(
                    "input_transform must return a torch.Tensor."
                )

            xs.append(x.detach().cpu())

        return xs

    def _postprocess_scores(self, similarities: Tensor) -> Tensor:
        if self.output == "similarity":
            return similarities

        logits = similarities * self.temperature + self.bias

        if self.output == "logit":
            return logits

        probs = torch.sigmoid(logits)

        if self.output == "probability":
            return probs

        return (probs >= self.threshold).float()

    def _make_annotations(self, concept_names: Sequence[str]) -> Annotations:
        return Annotations({
            1: AxisAnnotation(labels=list(concept_names))
        })
