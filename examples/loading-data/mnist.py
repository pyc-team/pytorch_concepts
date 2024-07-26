import torchvision.models as models
from torchvision import transforms

from torch_concepts.data import ColorMNISTDataset, load_preprocessed_data, preprocess_img_data


def main():
    data = ColorMNISTDataset(root='../data', train=False, download=True, transform=transforms.ToTensor(), random=True)
    model = models.resnet18(pretrained=True)
    try:
        embeddings, concepts, tasks, concept_names, task_names = load_preprocessed_data('../data/ColorMNISTDataset', 'test')
    except FileNotFoundError:
        preprocess_img_data(data, '../data/ColorMNISTDataset', model, split='test', batch_size=32, n_batches=10)
        embeddings, concepts, tasks, concept_names, task_names = load_preprocessed_data('../data/ColorMNISTDataset', 'test')

    print(embeddings.shape, concepts.shape, tasks.shape, concept_names, task_names)
    return


if __name__ == "__main__":
    main()
