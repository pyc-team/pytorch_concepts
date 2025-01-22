import os
from typing import Tuple

import torch

from torch.utils.data import DataLoader, Subset


def stratified_train_test_split(
    dataset: torch.utils.data.Dataset,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Split a dataset into stratified training and testing sets

    Args:
        dataset: dataset object.
        test_size: fraction of the dataset to include in the test split.
        random_state: random seed for reproducibility.

    Returns:
        Tuple(Subset, Subset): training and testing datasets.
    """
    n_samples = len(dataset)
    indices = torch.randperm(n_samples)
    test_size = int(n_samples * test_size)
    # stratified sampling
    targets = [batch[-1] for batch in dataset]
    targets = torch.stack(targets).squeeze()

    train_idx, test_idx = [], []
    for target in torch.unique(targets):
        idx = indices[targets == target]
        # shuffle the indices with the random seed for reproducibility
        torch.manual_seed(random_state)
        idx = idx[torch.randperm(len(idx))]
        idx_train, idx_test = idx[:-test_size], idx[-test_size:]
        train_idx.append(idx_train)
        test_idx.append(idx_test)
    train_idx = torch.cat(train_idx)
    test_idx = torch.cat(test_idx)

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    return train_dataset, test_dataset


class InputImgEncoder(torch.nn.Module):
    """
    Initialize the input image encoder.

    Attributes:
        original_model: The original model to extract features from.
    """
    def __init__(self, original_model: torch.nn.Module):
        super(InputImgEncoder, self).__init__()
        self.features = torch.nn.Sequential(
            *list(original_model.children())[:-1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the input image encoder.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The output tensor from the last layer of the model.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def preprocess_img_data(
    dataset: torch.utils.data.Dataset,
    dataset_root: str,
    input_encoder: torch.nn.Module,
    split: str = 'test',
    batch_size: int = 32,
    n_batches: int = None,
) -> None:
    """
    Preprocess an image dataset using a given input encoder.

    Args:
        dataset: dataset object.
        dataset_root: dataset root directory.
        input_encoder: input encoder model.
        split: dataset split to process.
        batch_size: batch size.
        n_batches: number of batches to process.

    Returns:
        None
    """
    model = InputImgEncoder(input_encoder)
    model.eval()

    # Load CelebA dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract embeddings
    embeddings, c, y = [], [], []
    with torch.no_grad():
        for batch_idx, (images, concepts, tasks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")
            emb = model(images)
            embeddings.append(emb)
            c.append(concepts)
            y.append(tasks)
            if n_batches is not None and batch_idx + 1 >= n_batches:
                break

    # Concatenate and save embeddings
    embeddings = torch.cat(embeddings, dim=0)
    c = torch.cat(c, dim=0)
    y = torch.cat(y, dim=0)
    torch.save(embeddings, os.path.join(dataset_root, f'{split}_embeddings.pt'))
    torch.save(c, os.path.join(dataset_root, f'{split}_concepts.pt'))
    torch.save(y, os.path.join(dataset_root, f'{split}_tasks.pt'))
    torch.save(
        dataset.concept_attr_names,
        os.path.join(dataset_root, f'{split}_concept_names.pt'),
    )
    torch.save(
        dataset.task_attr_names,
        os.path.join(dataset_root, f'{split}_task_names.pt'),
    )


def load_preprocessed_data(dataset_root: str, split: str = 'test') -> tuple:
    """
    Load preprocessed embeddings, concepts, tasks, concept names and task names
    from a dataset.

    Args:
        dataset_root: dataset root directory.
        split: dataset split to load.

    Returns:
        embeddings: embeddings tensor.
        concepts: concepts tensor.
        tasks: tasks tensor.
        concept_names: concept names list.
        task_names: task names list.
    """
    embeddings_path = os.path.join(dataset_root, f'{split}_embeddings.pt')
    concepts_path = os.path.join(dataset_root, f'{split}_concepts.pt')
    tasks_path = os.path.join(dataset_root, f'{split}_tasks.pt')
    concept_names_path = os.path.join(dataset_root, f'{split}_concept_names.pt')
    task_names_path = os.path.join(dataset_root, f'{split}_task_names.pt')

    embeddings = torch.load(embeddings_path)
    concepts = torch.load(concepts_path)
    tasks = torch.load(tasks_path)
    concept_names = torch.load(concept_names_path)
    task_names = torch.load(task_names_path)

    concepts = concepts.float()
    if len(tasks.shape) == 1:
        tasks = tasks.unsqueeze(1)
    tasks = tasks.float()
    return embeddings, concepts, tasks, concept_names, task_names
