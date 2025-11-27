import os
import tarfile
import torch
import pandas as pd
import numpy as np
from typing import List, Dict
from PIL import Image, ImageFile
import torchvision.transforms as T
from torch_concepts import Annotations
from torch_concepts.annotations import AxisAnnotation
from torch_concepts.data.base import ConceptDataset
from torch_concepts.data.io import download_url
from torch_concepts.data.backbone import compute_backbone_embs
from torchvision.models import resnet18

# Names of all CUB attributes
CONCEPT_SEMANTICS = [
    "has_bill_shape::curved_(up_or_down)",
    "has_bill_shape::dagger",
    "has_bill_shape::hooked",
    "has_bill_shape::needle",
    "has_bill_shape::hooked_seabird",
    "has_bill_shape::spatulate",
    "has_bill_shape::all-purpose",
    "has_bill_shape::cone",
    "has_bill_shape::specialized",
    "has_wing_color::blue",
    "has_wing_color::brown",
    "has_wing_color::iridescent",
    "has_wing_color::purple",
    "has_wing_color::rufous",
    "has_wing_color::grey",
    "has_wing_color::yellow",
    "has_wing_color::olive",
    "has_wing_color::green",
    "has_wing_color::pink",
    "has_wing_color::orange",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_wing_color::red",
    "has_wing_color::buff",
    "has_upperparts_color::blue",
    "has_upperparts_color::brown",
    "has_upperparts_color::iridescent",
    "has_upperparts_color::purple",
    "has_upperparts_color::rufous",
    "has_upperparts_color::grey",
    "has_upperparts_color::yellow",
    "has_upperparts_color::olive",
    "has_upperparts_color::green",
    "has_upperparts_color::pink",
    "has_upperparts_color::orange",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_upperparts_color::red",
    "has_upperparts_color::buff",
    "has_underparts_color::blue",
    "has_underparts_color::brown",
    "has_underparts_color::iridescent",
    "has_underparts_color::purple",
    "has_underparts_color::rufous",
    "has_underparts_color::grey",
    "has_underparts_color::yellow",
    "has_underparts_color::olive",
    "has_underparts_color::green",
    "has_underparts_color::pink",
    "has_underparts_color::orange",
    "has_underparts_color::black",
    "has_underparts_color::white",
    "has_underparts_color::red",
    "has_underparts_color::buff",
    "has_breast_pattern::solid",
    "has_breast_pattern::spotted",
    "has_breast_pattern::striped",
    "has_breast_pattern::multi-colored",
    "has_back_color::blue",
    "has_back_color::brown",
    "has_back_color::iridescent",
    "has_back_color::purple",
    "has_back_color::rufous",
    "has_back_color::grey",
    "has_back_color::yellow",
    "has_back_color::olive",
    "has_back_color::green",
    "has_back_color::pink",
    "has_back_color::orange",
    "has_back_color::black",
    "has_back_color::white",
    "has_back_color::red",
    "has_back_color::buff",
    "has_tail_shape::forked_tail",
    "has_tail_shape::rounded_tail",
    "has_tail_shape::notched_tail",
    "has_tail_shape::fan-shaped_tail",
    "has_tail_shape::pointed_tail",
    "has_tail_shape::squared_tail",
    "has_upper_tail_color::blue",
    "has_upper_tail_color::brown",
    "has_upper_tail_color::iridescent",
    "has_upper_tail_color::purple",
    "has_upper_tail_color::rufous",
    "has_upper_tail_color::grey",
    "has_upper_tail_color::yellow",
    "has_upper_tail_color::olive",
    "has_upper_tail_color::green",
    "has_upper_tail_color::pink",
    "has_upper_tail_color::orange",
    "has_upper_tail_color::black",
    "has_upper_tail_color::white",
    "has_upper_tail_color::red",
    "has_upper_tail_color::buff",
    "has_head_pattern::spotted",
    "has_head_pattern::malar",
    "has_head_pattern::crested",
    "has_head_pattern::masked",
    "has_head_pattern::unique_pattern",
    "has_head_pattern::eyebrow",
    "has_head_pattern::eyering",
    "has_head_pattern::plain",
    "has_head_pattern::eyeline",
    "has_head_pattern::striped",
    "has_head_pattern::capped",
    "has_breast_color::blue",
    "has_breast_color::brown",
    "has_breast_color::iridescent",
    "has_breast_color::purple",
    "has_breast_color::rufous",
    "has_breast_color::grey",
    "has_breast_color::yellow",
    "has_breast_color::olive",
    "has_breast_color::green",
    "has_breast_color::pink",
    "has_breast_color::orange",
    "has_breast_color::black",
    "has_breast_color::white",
    "has_breast_color::red",
    "has_breast_color::buff",
    "has_throat_color::blue",
    "has_throat_color::brown",
    "has_throat_color::iridescent",
    "has_throat_color::purple",
    "has_throat_color::rufous",
    "has_throat_color::grey",
    "has_throat_color::yellow",
    "has_throat_color::olive",
    "has_throat_color::green",
    "has_throat_color::pink",
    "has_throat_color::orange",
    "has_throat_color::black",
    "has_throat_color::white",
    "has_throat_color::red",
    "has_throat_color::buff",
    "has_eye_color::blue",
    "has_eye_color::brown",
    "has_eye_color::purple",
    "has_eye_color::rufous",
    "has_eye_color::grey",
    "has_eye_color::yellow",
    "has_eye_color::olive",
    "has_eye_color::green",
    "has_eye_color::pink",
    "has_eye_color::orange",
    "has_eye_color::black",
    "has_eye_color::white",
    "has_eye_color::red",
    "has_eye_color::buff",
    "has_bill_length::about_the_same_as_head",
    "has_bill_length::longer_than_head",
    "has_bill_length::shorter_than_head",
    "has_forehead_color::blue",
    "has_forehead_color::brown",
    "has_forehead_color::iridescent",
    "has_forehead_color::purple",
    "has_forehead_color::rufous",
    "has_forehead_color::grey",
    "has_forehead_color::yellow",
    "has_forehead_color::olive",
    "has_forehead_color::green",
    "has_forehead_color::pink",
    "has_forehead_color::orange",
    "has_forehead_color::black",
    "has_forehead_color::white",
    "has_forehead_color::red",
    "has_forehead_color::buff",
    "has_under_tail_color::blue",
    "has_under_tail_color::brown",
    "has_under_tail_color::iridescent",
    "has_under_tail_color::purple",
    "has_under_tail_color::rufous",
    "has_under_tail_color::grey",
    "has_under_tail_color::yellow",
    "has_under_tail_color::olive",
    "has_under_tail_color::green",
    "has_under_tail_color::pink",
    "has_under_tail_color::orange",
    "has_under_tail_color::black",
    "has_under_tail_color::white",
    "has_under_tail_color::red",
    "has_under_tail_color::buff",
    "has_nape_color::blue",
    "has_nape_color::brown",
    "has_nape_color::iridescent",
    "has_nape_color::purple",
    "has_nape_color::rufous",
    "has_nape_color::grey",
    "has_nape_color::yellow",
    "has_nape_color::olive",
    "has_nape_color::green",
    "has_nape_color::pink",
    "has_nape_color::orange",
    "has_nape_color::black",
    "has_nape_color::white",
    "has_nape_color::red",
    "has_nape_color::buff",
    "has_belly_color::blue",
    "has_belly_color::brown",
    "has_belly_color::iridescent",
    "has_belly_color::purple",
    "has_belly_color::rufous",
    "has_belly_color::grey",
    "has_belly_color::yellow",
    "has_belly_color::olive",
    "has_belly_color::green",
    "has_belly_color::pink",
    "has_belly_color::orange",
    "has_belly_color::black",
    "has_belly_color::white",
    "has_belly_color::red",
    "has_belly_color::buff",
    "has_wing_shape::rounded-wings",
    "has_wing_shape::pointed-wings",
    "has_wing_shape::broad-wings",
    "has_wing_shape::tapered-wings",
    "has_wing_shape::long-wings",
    "has_size::large_(16_-_32_in)",
    "has_size::small_(5_-_9_in)",
    "has_size::very_large_(32_-_72_in)",
    "has_size::medium_(9_-_16_in)",
    "has_size::very_small_(3_-_5_in)",
    "has_shape::upright-perching_water-like",
    "has_shape::chicken-like-marsh",
    "has_shape::long-legged-like",
    "has_shape::duck-like",
    "has_shape::owl-like",
    "has_shape::gull-like",
    "has_shape::hummingbird-like",
    "has_shape::pigeon-like",
    "has_shape::tree-clinging-like",
    "has_shape::hawk-like",
    "has_shape::sandpiper-like",
    "has_shape::upland-ground-like",
    "has_shape::swallow-like",
    "has_shape::perching-like",
    "has_back_pattern::solid",
    "has_back_pattern::spotted",
    "has_back_pattern::striped",
    "has_back_pattern::multi-colored",
    "has_tail_pattern::solid",
    "has_tail_pattern::spotted",
    "has_tail_pattern::striped",
    "has_tail_pattern::multi-colored",
    "has_belly_pattern::solid",
    "has_belly_pattern::spotted",
    "has_belly_pattern::striped",
    "has_belly_pattern::multi-colored",
    "has_primary_color::blue",
    "has_primary_color::brown",
    "has_primary_color::iridescent",
    "has_primary_color::purple",
    "has_primary_color::rufous",
    "has_primary_color::grey",
    "has_primary_color::yellow",
    "has_primary_color::olive",
    "has_primary_color::green",
    "has_primary_color::pink",
    "has_primary_color::orange",
    "has_primary_color::black",
    "has_primary_color::white",
    "has_primary_color::red",
    "has_primary_color::buff",
    "has_leg_color::blue",
    "has_leg_color::brown",
    "has_leg_color::iridescent",
    "has_leg_color::purple",
    "has_leg_color::rufous",
    "has_leg_color::grey",
    "has_leg_color::yellow",
    "has_leg_color::olive",
    "has_leg_color::green",
    "has_leg_color::pink",
    "has_leg_color::orange",
    "has_leg_color::black",
    "has_leg_color::white",
    "has_leg_color::red",
    "has_leg_color::buff",
    "has_bill_color::blue",
    "has_bill_color::brown",
    "has_bill_color::iridescent",
    "has_bill_color::purple",
    "has_bill_color::rufous",
    "has_bill_color::grey",
    "has_bill_color::yellow",
    "has_bill_color::olive",
    "has_bill_color::green",
    "has_bill_color::pink",
    "has_bill_color::orange",
    "has_bill_color::black",
    "has_bill_color::white",
    "has_bill_color::red",
    "has_bill_color::buff",
    "has_crown_color::blue",
    "has_crown_color::brown",
    "has_crown_color::iridescent",
    "has_crown_color::purple",
    "has_crown_color::rufous",
    "has_crown_color::grey",
    "has_crown_color::yellow",
    "has_crown_color::olive",
    "has_crown_color::green",
    "has_crown_color::pink",
    "has_crown_color::orange",
    "has_crown_color::black",
    "has_crown_color::white",
    "has_crown_color::red",
    "has_crown_color::buff",
    "has_wing_pattern::solid",
    "has_wing_pattern::spotted",
    "has_wing_pattern::striped",
    "has_wing_pattern::multi-colored",
]

CUB_DIR = os.environ.get("CUB_DIR", './CUB200/')
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CUBDataset(ConceptDataset):
    """
    The CUB dataset is a dataset of bird images with annotated attributes.
    Each image is associated with a set of concept labels (attributes) and
    task labels (bird species).

    Attributes:
        root: The root directory where the dataset is stored.
        split: The dataset split to use ('train' or 'test').
        uncertain_concept_labels: Whether to treat uncertain concept labels as
            positive.
        path_transform: A function to transform the image paths.
    """
    name = "cub"
    n_concepts = 312
    n_tasks = 200

    def __init__(
        self,
        precision : int = 32,
        concepts : np.ndarray | pd.DataFrame | torch.Tensor = None,
        annotations : Annotations | None = None,
        concept_names_subset : List[str] | None = None,
        root : str = CUB_DIR,
        image_transform: object | None = None,
    ) -> None:
        self.root = root
        # ensure images have consistent size for batching
        self.image_transform = image_transform or T.Compose([T.Resize((256, 256)), T.ToTensor()])
        
        embeddings, concepts, annotations, graph = self.load()
        
        super().__init__(
            precision=precision,
            input_data=embeddings,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_names_subset,
        )
    
    @property
    def raw_filenames(self) -> List[str]:
        """List of raw filenames that need to be present in the raw directory
        for the dataset to be considered present."""
        return [
            "CUB_200_2011/images.txt",
            "CUB_200_2011/image_class_labels.txt",
            "CUB_200_2011/train_test_split.txt",
            "CUB_200_2011/bounding_boxes.txt",
            "CUB_200_2011/classes.txt",
            "CUB_200_2011/attributes/image_attribute_labels.txt",
            "CUB_200_2011/attributes/class_attribute_labels_continuous.txt",
            "CUB_200_2011/attributes/certainties.txt",
        ]
        
    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            "cub_concepts.pt",
            "cub_annotations.pt",
            "cub_embeddings.pt",
        ]
    
    def download(self) -> None:
        """Downloads the CUB dataset if it is not already present."""
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
        tgz_path = download_url(url, self.root)
        
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=self.root)
        os.unlink(tgz_path)
                
    def build(self):
        self.maybe_download()
        
        # workaround to get self.n_samples() work in ConceptDataset. We will overwrite later in super().__init__()
        # create a torch tensor with shape (n_samples, whatever) and set self.input_data to it temporarily
        temp_input_data = torch.zeros((11788, 10))  # CUB has 11788 samples
        self.input_data = temp_input_data

        images = pd.read_csv(
            self.raw_paths[0],
            sep=r"\s+",
            header=None,
            names=["image_id", "path"],
        )
        concept_names = CONCEPT_SEMANTICS

        attr_labels = pd.read_csv(
            self.raw_paths[5],
            header=None,
            names=['image_id', 'attr_id', 'is_present', 'certainty', 'time_ms', 'extra'],
            usecols=[0, 1, 2],
            delim_whitespace=True,
            engine="python",
        )
        concepts_df = attr_labels.pivot(index='image_id', columns='attr_id', values='is_present').fillna(0)
        concepts_df = concepts_df.loc[images["image_id"]]
        concepts_tensor = torch.tensor(concepts_df.values, dtype=torch.float32)

        concept_metadata = {name: {'type': 'discrete'} for name in concept_names}
        cardinalities = tuple(1 for _ in concept_names)  # binary concepts
        annotations = Annotations({
            1: AxisAnnotation(labels=concept_names,
                              cardinalities=cardinalities,
                              metadata=concept_metadata)
        })

        torch.save(concepts_tensor, self.processed_paths[0])
        torch.save(annotations, self.processed_paths[1])

        annotations = torch.load(self.processed_paths[1], weights_only=False)
        self._annotations = annotations
        self.maybe_reduce_annotations(annotations, None)
        concepts = torch.load(self.processed_paths[0], weights_only=False)
        # temporary placeholder so set_concepts has a length reference
        self.input_data = torch.zeros((concepts.shape[0], 1))
        self.precision = 32  # set precision before calling set_concepts
        self.set_concepts(concepts)

        # Compute embeddings using a pretrained model (e.g., ResNet) as backbone from torch_concepts.data.backbone
        backbone = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        embeddings = compute_backbone_embs(
            self,
            backbone,
            batch_size=64,
            workers=4,
            verbose=True
        )
        
        torch.save(embeddings, self.processed_paths[2])

    def load_raw(self):
        self.maybe_build()
        concepts = torch.load(self.processed_paths[0], weights_only=False)
        annotations = torch.load(self.processed_paths[1], weights_only=False)
        embeddings = torch.load(self.processed_paths[2], weights_only=False)
        return embeddings, concepts, annotations, None

    def load(self):
        embeddings, concepts, annotations, graph = self.load_raw()
        return embeddings, concepts, annotations, graph

    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.Tensor]]:
        img_rel_path = pd.read_csv( # TODO: optimize by reading this once in __init__
            self.raw_paths[0],
            header=None,
            names=['image_id', 'img_path'],
            delim_whitespace=True,
            engine="python",
        ).set_index('image_id').loc[idx + 1, 'img_path']  # idx +1 because image_id starts from 1
        img_path = os.path.join(self.root, "CUB_200_2011/images", img_rel_path)
        image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        concepts = self.concepts[idx].clone()
        sample = {
            'inputs': {'x': image},
            'concepts': {'c': concepts},
        }
        return sample
