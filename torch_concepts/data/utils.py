import os
import torch
from torch.utils.data import DataLoader

from torch_concepts.nn import InputImgEncoder


def preprocess_img_data(dataset: torch.utils.data.Dataset,
                        dataset_root: str,
                        input_encoder: torch.nn.Module,
                        split: str = 'test',
                        batch_size: int = 32,
                        n_batches: int = None) -> None:
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
    torch.save(dataset.concept_attr_names, os.path.join(dataset_root, f'{split}_concept_names.pt'))
    torch.save(dataset.task_attr_names, os.path.join(dataset_root, f'{split}_task_names.pt'))


def load_preprocessed_data(dataset_root: str, split: str = 'test') -> tuple:
    """
    Load preprocessed embeddings, concepts, tasks, concept names and task names from a dataset.

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
