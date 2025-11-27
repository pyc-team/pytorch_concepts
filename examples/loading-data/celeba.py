import torchvision.models as models
from torchvision import transforms

from torch_concepts.data.datasets import CelebADataset


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data = CelebADataset(root='../data', split='test', transform=transform,
                         download=False, class_attributes=['Attractive'])

    # Direct data access
    print(f"Dataset size: {len(data)}")
    print(f"Concept attributes: {data.concept_attr_names}")
    print(f"Task attributes: {data.task_attr_names}")
    return


if __name__ == "__main__":
    main()
