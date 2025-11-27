# TODO: update example when dataset is fixed

import torchvision.models as models
from torchvision import transforms

from torch_concepts.data.datasets import ColorMNISTDataset
# from torch_concepts.data.utils import preprocess_img_data, load_preprocessed_data


def main():
    data = ColorMNISTDataset(root='../data', train=False, download=True, transform=transforms.ToTensor(), random=True)
    # model = models.resnet18(pretrained=True)
    # try:
    #     embeddings, concepts, tasks, concept_names, task_names = load_preprocessed_data('../data/ColorMNISTDataset', 'test')
    # except FileNotFoundError:
    #     preprocess_img_data(data, '../data/ColorMNISTDataset', model, split='test', batch_size=32, n_batches=10)
    #     embeddings, concepts, tasks, concept_names, task_names = load_preprocessed_data('../data/ColorMNISTDataset', 'test')

    # print(embeddings.shape, concepts.shape, tasks.shape, concept_names, task_names)

    # Direct data access
    print(f"Dataset size: {len(data)}")
    print(f"Concept names: {data.concept_attr_names}")
    print(f"Task names: {data.task_attr_names}")
    return


if __name__ == "__main__":
    main()
