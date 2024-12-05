import torchvision.models as models
from torchvision import transforms

from torch_concepts.data import CelebADataset
from .utils import preprocess_img_data, load_preprocessed_data


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data = CelebADataset(root='../data', split='test', transform=transform,
                         download=False, class_attributes=['Attractive'])
    model = models.resnet18(pretrained=True)
    try:
        embeddings, concepts, tasks, concept_names, task_names = load_preprocessed_data('../data/celeba', 'test')
    except FileNotFoundError:
        preprocess_img_data(data, '../data/celeba', model, split='test', batch_size=32, n_batches=10)
        embeddings, concepts, tasks, concept_names, task_names = load_preprocessed_data('../data/celeba', 'test')

    print(embeddings.shape, concepts.shape, tasks.shape, concept_names, task_names)
    return


if __name__ == "__main__":
    main()
