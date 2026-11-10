from __future__ import annotations

import pickle
import re
import types
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_concepts.data.generation.annotators.clip import (
    BinaryPromptFormatter,
    CLIPAnnotator,
    PromptTemplate,
    StatePromptFormatter,
    default_binary_prompt_formatter,
    default_input_getter,
    default_state_prompt_formatter,
    resolve_device,
)

TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
IMAGE_SIZE = 224
# The ResNet-50 checkpoints were trained on ImageNet-normalized images.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


class _Skipped:
    """Placeholder for the checkpoint's pickled OmegaConf training config."""


class _CheckpointUnpickler(pickle.Unpickler):
    """Rebuild tensors and plain containers; refuse every other global."""

    ALLOWED = {
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("__builtin__", "dict"),
        ("__builtin__", "list"),
        ("__builtin__", "long"),
    }

    def find_class(self, module, name):
        if module.split(".")[0] == "omegaconf" or (module, name) == ("typing", "Any"):
            return _Skipped
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unexpected global in checkpoint: {module}.{name}")


def _load_state_dict(checkpoint: str | Path) -> dict[str, Tensor]:
    """Load the model weights of an official CXR-CLIP checkpoint.

    The checkpoint also pickles its OmegaConf training config, which
    ``torch.load(weights_only=True)`` rejects even when allowlisted (it holds a
    ``defaultdict``); the restricted unpickler skips it.
    """
    pickle_module = types.ModuleType("restricted_pickle")
    pickle_module.Unpickler = _CheckpointUnpickler
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        pickle_module=pickle_module,
    )["model"]
    # Weights are nested one level deeper than here (image_encoder.resnet.conv1).
    state = {re.sub(r"^(\w+)\.\w+\.", r"\1.", key): value for key, value in state.items()}
    # Unused: the temperature, the ResNet's ImageNet head and BERT's position ids
    # (a buffer recent transformers no longer saves).
    for key in [
        "logit_scale",
        "image_encoder.fc.weight",
        "image_encoder.fc.bias",
        "text_encoder.embeddings.position_ids",
    ]:
        del state[key]
    return state


class CXRCLIPAnnotator(CLIPAnnotator):
    """CLIP annotator backed by CXR-CLIP, pretrained on chest X-rays and reports.

    Loads a ResNet-50 CXR-CLIP checkpoint (``r50_m.tar``, ``r50_mc.tar`` or
    ``r50_mcc.tar``, CC BY-NC 4.0, linked from
    https://github.com/Soombit-ai/cxr-clip). Its text encoder, Bio_ClinicalBERT,
    is downloaded from Hugging Face. Images may be PIL images or ``(C, H, W)``
    tensors, either float in [0, 1] or uint8, with one or three channels.

    Args:
        checkpoint: Path to the downloaded checkpoint.
        batch_size: Batch size used while annotating the dataset.
        device: Device for inference. Defaults to CUDA, then MPS, then CPU.
        input_getter: Extracts the image from a dataset sample.
        prompt_template: Template applied to every concept or state prompt.
        binary_prompt_formatter: Turns a binary concept name into a prompt.
        state_prompt_formatter: Turns a categorical concept and state into a prompt.
        num_workers: Number of data-loading workers.
        show_progress: Whether to show progress bars.
    """

    def __init__(
        self,
        checkpoint: str | Path,
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
    ):
        from torchvision.models import resnet50
        from transformers import AutoModel, AutoTokenizer

        # CLIPAnnotator.__init__ loads a Hugging Face CLIP, so its options are set here.
        self.model_name = str(checkpoint)
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.input_getter = input_getter
        self.prompt_template = prompt_template
        self.binary_prompt_formatter = binary_prompt_formatter
        self.state_prompt_formatter = state_prompt_formatter
        self.num_workers = num_workers
        self.show_progress = show_progress

        resnet = resnet50()
        resnet.fc = nn.Identity()
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model = nn.ModuleDict(
            {
                "image_encoder": resnet,
                "text_encoder": AutoModel.from_pretrained(TEXT_MODEL),
                "image_projection": nn.Linear(2048, 512),
                "text_projection": nn.Linear(768, 512),
            }
        )
        self.model.load_state_dict(_load_state_dict(checkpoint))
        self.model.to(self.device).eval()

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        """Encode and normalize text with Bio_ClinicalBERT."""
        tokens = self.tokenizer(
            list(texts),
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model["text_encoder"](**tokens).last_hidden_state
        # CXR-CLIP pools the last attended token ([SEP]).
        last = tokens["attention_mask"].sum(dim=-1) - 1
        features = self.model["text_projection"](hidden[torch.arange(len(hidden)), last])
        return F.normalize(features, dim=-1)

    def encode_images(self, images: Sequence[Any]) -> Tensor:
        """Resize, crop, normalize, encode and normalize chest X-rays."""
        import torchvision.transforms.functional as TF

        batch = []
        for image in images:
            if not isinstance(image, Tensor):
                image = TF.pil_to_tensor(image)
            # Other integer depths (16-bit X-rays) fail at normalization, unscaled.
            if image.dtype == torch.uint8:
                image = image.float() / 255
            image = TF.resize(image.expand(3, -1, -1), IMAGE_SIZE)
            batch.append(TF.center_crop(image, IMAGE_SIZE))
        pixels = TF.normalize(torch.stack(batch), IMAGE_MEAN, IMAGE_STD).to(self.device)
        features = self.model["image_projection"](self.model["image_encoder"](pixels))
        return F.normalize(features, dim=-1)
