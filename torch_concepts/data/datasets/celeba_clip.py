import os
import re
import logging
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip is required for CelebACLIPDataset. "
        "Install it with: pip install open-clip-torch"
    ) from e

from torch_concepts import Annotations, AxisAnnotation
from .celeba import CelebADataset

logger = logging.getLogger(__name__)


# Default prompts mirror the 40 CelebA attributes so the class works
# as a drop-in replacement with no configuration required.
DEFAULT_CLIP_CONCEPT_PROMPTS: Dict[str, str] = {
    "5_o_Clock_Shadow":     "a photo of a person with 5 o'clock shadow stubble",
    "Arched_Eyebrows":      "a photo of a person with arched eyebrows",
    "Attractive":           "a photo of an attractive person",
    "Bags_Under_Eyes":      "a photo of a person with bags under their eyes",
    "Bald":                 "a photo of a bald person",
    "Bangs":                "a photo of a person with bangs",
    "Big_Lips":             "a photo of a person with big lips",
    "Big_Nose":             "a photo of a person with a big nose",
    "Black_Hair":           "a photo of a person with black hair",
    "Blond_Hair":           "a photo of a person with blond hair",
    "Blurry":               "a blurry photo of a person",
    "Brown_Hair":           "a photo of a person with brown hair",
    "Bushy_Eyebrows":       "a photo of a person with bushy eyebrows",
    "Chubby":               "a photo of a chubby person",
    "Double_Chin":          "a photo of a person with a double chin",
    "Eyeglasses":           "a photo of a person wearing eyeglasses",
    "Goatee":               "a photo of a person with a goatee",
    "Gray_Hair":            "a photo of a person with gray hair",
    "Heavy_Makeup":         "a photo of a person wearing heavy makeup",
    "High_Cheekbones":      "a photo of a person with high cheekbones",
    "Male":                 "a photo of a man",
    "Mouth_Slightly_Open":  "a photo of a person with their mouth slightly open",
    "Mustache":             "a photo of a person with a mustache",
    "Narrow_Eyes":          "a photo of a person with narrow eyes",
    "No_Beard":             "a photo of a person with no beard",
    "Oval_Face":            "a photo of a person with an oval face",
    "Pale_Skin":            "a photo of a person with pale skin",
    "Pointy_Nose":          "a photo of a person with a pointy nose",
    "Receding_Hairline":    "a photo of a person with a receding hairline",
    "Rosy_Cheeks":          "a photo of a person with rosy cheeks",
    "Sideburns":            "a photo of a person with sideburns",
    "Smiling":              "a photo of a smiling person",
    "Straight_Hair":        "a photo of a person with straight hair",
    "Wavy_Hair":            "a photo of a person with wavy hair",
    "Wearing_Earrings":     "a photo of a person wearing earrings",
    "Wearing_Hat":          "a photo of a person wearing a hat",
    "Wearing_Lipstick":     "a photo of a person wearing lipstick",
    "Wearing_Necklace":     "a photo of a person wearing a necklace",
    "Wearing_Necktie":      "a photo of a person wearing a necktie",
    "Young":                "a photo of a young person",
}


class CelebACLIPDataset(CelebADataset):
    """CelebA dataset with CLIP-generated concept pseudo-labels.

    Replaces the 40 hand-annotated CelebA binary attributes with binary
    pseudo-labels derived by thresholding cosine similarities between image
    embeddings and user-supplied text prompts, computed with an
    ``open_clip`` model.

    The CelebA images are downloaded and cached exactly as in
    :class:`CelebADataset`.  The CLIP pseudo-labels are computed once and
    stored next to the other processed files; subsequent instantiations load
    them from disk without re-running inference.

    Args:
        root: Root directory for the dataset.  Defaults to
            ``<cwd>/data/celeba``.
        concept_prompts: Concept vocabulary as either

            * a ``dict`` mapping concept name → text prompt, or
            * a ``list`` of text prompts (concept names will be the prompt
              strings themselves).

            Defaults to :data:`DEFAULT_CLIP_CONCEPT_PROMPTS`, which mirrors
            all 40 CelebA attributes.
        clip_model: ``open_clip`` model name.
            Default: ``'ViT-SO400M-14-SigLIP2-384'`` (SigLIP2).
        clip_pretrained: ``open_clip`` pretrained weights tag.
            Default: ``'webli'``.
        clip_device: Device used for CLIP inference.  Defaults to CUDA when
            available, otherwise CPU.
        inference_batch_size: Number of images processed per CLIP forward
            pass.  Default: ``64``.
        concept_subset: Optional list of concept names to retain after
            pseudo-label generation.
        label_descriptions: Optional dict mapping concept names to
            human-readable descriptions (metadata only, not used in training).

    Example::

        from torch_concepts.data.datasets import CelebACLIPDataset

        # Drop-in replacement — uses default SigLIP2 prompts for all 40 attrs
        dataset = CelebACLIPDataset(root='./data/celeba')

        # Custom concept vocabulary
        dataset = CelebACLIPDataset(
            root='./data/celeba',
            concept_prompts={
                'smiling':   'a photo of a smiling person',
                'blonde':    'a photo of a person with blond hair',
                'glasses':   'a photo of a person wearing glasses',
            },
        )
    """

    def __init__(
        self,
        root: Optional[str] = None,
        concept_prompts: Optional[Union[Dict[str, str], List[str]]] = None,
        clip_model: str = 'ViT-B-16-SigLIP2',
        clip_pretrained: str = 'webli',
        clip_device: Optional[str] = None,
        threshold: float = 0.0,
        inference_batch_size: int = 64,
        concept_subset: Optional[List[str]] = None,
        label_descriptions: Optional[dict] = None,
    ):
        # Normalise concept_prompts to a dict before super().__init__ is
        # called, because __init__ triggers load() → build() → CLIP inference.
        if concept_prompts is None:
            concept_prompts = DEFAULT_CLIP_CONCEPT_PROMPTS
        elif isinstance(concept_prompts, list):
            concept_prompts = {p: p for p in concept_prompts}

        # Store all CLIP-specific state on self *before* calling super so that
        # overridden processed_filenames / build / load_raw can access them.
        self._concept_prompts: Dict[str, str] = concept_prompts
        self._clip_model_name: str = clip_model
        self._clip_pretrained: str = clip_pretrained
        self._clip_device: str = clip_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._threshold: float = threshold
        self._inference_batch_size: int = inference_batch_size

        super().__init__(
            root=root,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _model_key(self) -> str:
        """Filesystem-safe identifier for the current model + pretrained tag."""
        raw = f"{self._clip_model_name}_{self._clip_pretrained}"
        return re.sub(r'[^a-zA-Z0-9_]', '_', raw)

    # ------------------------------------------------------------------
    # ConceptDataset interface overrides
    # ------------------------------------------------------------------

    @property
    def processed_filenames(self) -> List[str]:
        """Processed files: parent's four files plus two CLIP-specific ones."""
        return [
            "filenames.txt",                            # [0] shared
            "concepts.h5",                              # [1] parent annotated concepts
            "annotations.pt",                           # [2] parent annotations
            "split_mapping.h5",                         # [3] split labels
            f"clip_concepts_{self._model_key}.h5",      # [4] CLIP pseudo-labels
            f"clip_annotations_{self._model_key}.pt",   # [5] CLIP annotations
        ]

    def build(self):
        """Build processed files: parent images/splits then CLIP pseudo-labels."""
        # Ensure raw CelebA files are downloaded, extracted, and the parent's
        # four processed files (filenames, annotated concepts, annotations,
        # splits) are written to disk.
        super().build()

        clip_concepts_path = self.processed_paths[4]
        clip_annotations_path = self.processed_paths[5]

        if os.path.exists(clip_concepts_path) and os.path.exists(clip_annotations_path):
            logger.info("CLIP pseudo-labels already exist, skipping inference.")
            return

        self._compute_clip_pseudo_labels(clip_concepts_path, clip_annotations_path)

    def load_raw(self):
        """Load filenames and CLIP pseudo-labels from processed files."""
        self.maybe_build()

        logger.info(f"Loading CelebACLIPDataset from {self.root_dir}")

        with open(self.processed_paths[0], 'r') as f:
            filenames = f.read().strip().split('\n')

        concepts = pd.read_hdf(self.processed_paths[4], "concepts")
        annotations = torch.load(self.processed_paths[5], weights_only=False)

        return filenames, concepts, annotations, None

    # ------------------------------------------------------------------
    # CLIP pseudo-label computation
    # ------------------------------------------------------------------

    def _compute_clip_pseudo_labels(
        self,
        concepts_out_path: str,
        annotations_out_path: str,
    ) -> None:
        """Run CLIP inference over all images and save binary pseudo-labels.

        Args:
            concepts_out_path: Destination HDF5 file for the concept tensor.
            annotations_out_path: Destination ``.pt`` file for the
                :class:`Annotations` object.
        """

        device = torch.device(self._clip_device)
        logger.info(
            f"Loading CLIP model '{self._clip_model_name}' "
            f"(pretrained='{self._clip_pretrained}') on {device} …"
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name,
            pretrained=self._clip_pretrained,
            device=device,
        )
        tokenizer = open_clip.get_tokenizer(self._clip_model_name)
        model.eval()

        concept_names = list(self._concept_prompts.keys())
        prompts = list(self._concept_prompts.values())

        # Encode text prompts once
        logger.info(f"Encoding {len(prompts)} concept text prompts …")
        with torch.no_grad():
            text_tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Load filenames from parent's processed file
        with open(self.processed_paths[0], 'r') as f:
            filenames = f.read().strip().split('\n')

        n_samples = len(filenames)
        n_concepts = len(concept_names)
        pseudo_labels = torch.zeros(n_samples, n_concepts, dtype=torch.float32)

        img_dir = os.path.join(self.root, "raw", "img_align_celeba")
        batch_size = self._inference_batch_size

        logger.info(
            f"Running CLIP inference on {n_samples} images "
            f"(batch_size={batch_size}, threshold={self._threshold}) …"
        )

        for start in tqdm(range(0, n_samples, batch_size), desc="CLIP pseudo-labels"):
            end = min(start + batch_size, n_samples)
            batch_imgs = []
            for fname in filenames[start:end]:
                img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
                batch_imgs.append(preprocess(img))

            batch_tensor = torch.stack(batch_imgs).to(device)

            with torch.no_grad():
                img_features = model.encode_image(batch_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                # Cosine similarity: (B, n_concepts)
                logits = img_features @ text_features.T * model.logit_scale.exp() + model.logit_bias
                probs = torch.sigmoid(logits)

            pseudo_labels[start:end] = probs.cpu()

        # Save as DataFrame so the parent's set_concepts() path (which expects
        # a DataFrame with named columns) works without modification.
        concepts_df = pd.DataFrame(pseudo_labels.numpy(), columns=concept_names)
        concepts_df.to_hdf(concepts_out_path, key="concepts", mode="w")

        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                cardinalities=tuple([1] * n_concepts),
                metadata={name: {'type': 'discrete'} for name in concept_names},
            )
        })
        torch.save(annotations, annotations_out_path)

        logger.info(f"Saved CLIP pseudo-labels to {concepts_out_path}")
