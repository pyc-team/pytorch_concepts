import unittest
import torch
from torch import nn
from torch_concepts.nn.models import (
    ConceptExplanationModel,
    AVAILABLE_MODELS
)
from torch_concepts.utils import set_seed

set_seed(42)

# Create dummy data
batch_size = 4
input_dim = 10
latent_dim = 5
embedding_size = 3
n_concepts = 6
n_tasks = 2
class_reg = 0.1
residual_size = 2
memory_size = 2

x = torch.randn(batch_size, input_dim)
c_true = torch.randint(0, 2, (batch_size, n_concepts)).float()

# Initialize encoder and model parameters
encoder = nn.Sequential(
    nn.Linear(input_dim, latent_dim),
    nn.ReLU()
)

concept_names = [f"concept_{i}" for i in range(n_concepts)]
task_names = [f"task_{i}" for i in range(n_tasks)]

models = {
    model_name: model_cls(encoder, latent_dim, concept_names, task_names,
                          class_reg=class_reg, residual_size=residual_size,
                          embedding_size=embedding_size, memory_size=memory_size)
    for model_name, model_cls in AVAILABLE_MODELS.items()
}


class TestModels(unittest.TestCase):

    def test_forward_pass(self):
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                y_pred, c_pred = model(x)
                self.assertEqual(y_pred.shape[0], batch_size)
                self.assertEqual(c_pred.shape[0], batch_size)

                # Check if y_pred are logits (unbounded real numbers)
                self.assertTrue(torch.any(y_pred < 0) or torch.any(y_pred > 1),
                                "y_pred does not contain logits")

                # Check if c_pred are probabilities
                self.assertTrue(torch.all(c_pred >= 0) and torch.all(c_pred <= 1),
                                "c_pred does not contain probabilities")
                print(f"Forward pass successful for {model_name}")


    def test_intervention_functions(self):
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                _, c_pred_initial = model(x)
                c_intervened = torch.randint(0, 2, c_pred_initial.shape).float()
                model.int_prob = 1.0
                _, c_pred_after_intervention = model(x, c_intervened)
                self.assertTrue(torch.allclose(c_pred_after_intervention,
                                               c_intervened),
                                f"Intervention failed for {model_name}")
                print(f"Intervention successful for {model_name}")


    # TODO: not working yet
    # def test_get_local_explanations(self):
    #     for model_name, model in models.items():
    #         with self.subTest(model=model_name):
    #             if isinstance(model, ConceptExplanationModel):
    #                 explanations = model.get_local_explanations(x)
    #                 self.assertIsNotNone(explanations)
    #                 print(f"Local explanations for {model_name}: "
    #                       f"{explanations}")
    #
    # def test_get_global_explanations(self):
    #     for model_name, model in models.items():
    #         with self.subTest(model=model_name):
    #             if isinstance(model, ConceptExplanationModel):
    #                 global_explanations = model.get_global_explanations(x)
    #                 self.assertIsNotNone(global_explanations)
    #                 print(f"Global explanations for {model_name}: "
    #                       f"{global_explanations}")

if __name__ == "__main__":
    unittest.main()