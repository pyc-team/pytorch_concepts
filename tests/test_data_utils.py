import unittest
import torch
import os
import shutil
from torch.utils.data import Dataset
from torchvision import models

from torch_concepts.data.utils import preprocess_img_data, load_preprocessed_data


class MockDataset(Dataset):
    def __init__(self, size, num_concepts, num_tasks):
        self.data = torch.randn(size, 3, 224, 224)
        self.concepts = torch.randint(0, 2, (size, num_concepts)).float()
        self.tasks = torch.randint(0, 2, (size, num_tasks)).float()
        self.concept_attr_names = [f'concept_{i}' for i in range(num_concepts)]
        self.task_attr_names = [f'task_{i}' for i in range(num_tasks)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.concepts[index], self.tasks[index]


class TestPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_root = 'test_dataset'
        os.makedirs(cls.dataset_root, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dataset_root)

    def setUp(self):
        self.size = 100
        self.num_concepts = 5
        self.num_tasks = 2
        self.mock_dataset = MockDataset(self.size, self.num_concepts, self.num_tasks)
        self.input_encoder = models.resnet18(pretrained=False)

    def test_preprocess_img_data(self):
        preprocess_img_data(self.mock_dataset, self.dataset_root, self.input_encoder, split='test', batch_size=10,
                            n_batches=2)

        self.assertTrue(os.path.exists(os.path.join(self.dataset_root, 'test_embeddings.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.dataset_root, 'test_concepts.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.dataset_root, 'test_tasks.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.dataset_root, 'test_concept_names.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.dataset_root, 'test_task_names.pt')))

        embeddings = torch.load(os.path.join(self.dataset_root, 'test_embeddings.pt'))
        concepts = torch.load(os.path.join(self.dataset_root, 'test_concepts.pt'))
        tasks = torch.load(os.path.join(self.dataset_root, 'test_tasks.pt'))

        self.assertEqual(embeddings.shape[0], 20)  # 2 batches of 10
        self.assertEqual(concepts.shape[0], 20)
        self.assertEqual(tasks.shape[0], 20)

    def test_load_preprocessed_data(self):
        preprocess_img_data(self.mock_dataset, self.dataset_root, self.input_encoder, split='test', batch_size=10,
                            n_batches=2)

        embeddings, concepts, tasks, concept_names, task_names = load_preprocessed_data(self.dataset_root, split='test')

        self.assertEqual(embeddings.shape[0], 20)
        self.assertEqual(concepts.shape[0], 20)
        self.assertEqual(tasks.shape[0], 20)
        self.assertEqual(len(concept_names), self.num_concepts)
        self.assertEqual(len(task_names), self.num_tasks)
        self.assertEqual(concept_names, self.mock_dataset.concept_attr_names)
        self.assertEqual(task_names, self.mock_dataset.task_attr_names)


if __name__ == '__main__':
    unittest.main()
