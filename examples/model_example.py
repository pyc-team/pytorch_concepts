# !/usr/local/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import torch
import lightning as L
from torch.utils.data import TensorDataset, random_split

from torch_concepts.data import ToyDataset
from torch_concepts.data.utils import stratified_train_test_split
from torch_concepts.nn.models import (
    ConceptBottleneckModel,
    ConceptResidualModel,
    ConceptEmbeddingModel,
    DeepConceptReasoning,
    LinearConceptEmbeddingModel,
    ConceptMemoryReasoning,
    ConceptEmbeddingReasoning,
    ConceptExplanationModel,
    LinearConceptMemoryReasoning,
    StochasticConceptBottleneckModel,
)
from experiments.utils import set_seed, CustomProgressBar
from torch_concepts.utils import get_most_common_expl


def main():
    latent_dims = 20
    n_epochs = 100
    n_samples = 1000
    class_reg = 0.5
    batch_size = 1024
    residual_size = 20
    embedding_size = 20
    memory_size = 2
    num_monte_carlo = 100
    level = 0.99
    cov_reg = 1.0
    concept_reg = 1.0
    model_kwargs = dict()

    models = [
        ConceptBottleneckModel,
        ConceptResidualModel,
        ConceptEmbeddingModel,
        DeepConceptReasoning,
        LinearConceptEmbeddingModel,
        ConceptMemoryReasoning,
        ConceptEmbeddingReasoning,
        LinearConceptMemoryReasoning,
        StochasticConceptBottleneckModel,
    ]

    set_seed(42)
    data = ToyDataset("xor", size=n_samples, random_state=42)
    x, c, y = data.data, data.concept_labels, data.target_labels

    concept_names, task_names = data.concept_attr_names, data.task_attr_names
    y = y.squeeze()
    task_names = ["xnor", "xor"]

    dataset = TensorDataset(x, c, y)
    # Check: stratified train test split returns twice the amount of test size
    train_set, val_set = random_split(dataset, lengths=[900, 100])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size)

    n_features = x.shape[1]
    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims), torch.nn.LeakyReLU()
    )

    results = {}
    for model_cls in models:
        # Add special kwargs for specific models
        if model_cls.__name__ == "StochasticConceptBottleneckModel":
            model_kwargs.update(
                dict(
                    num_monte_carlo=num_monte_carlo,
                    level=level,
                    n_epochs=n_epochs,
                    cov_reg=cov_reg,
                    concept_reg=concept_reg,
                )
            )
        model = model_cls(
            encoder,
            latent_dims,
            concept_names,
            task_names,
            class_reg=class_reg,
            residual_size=residual_size,
            embedding_size=embedding_size,
            memory_size=memory_size,
            **model_kwargs,
        )
        model.configure_optimizers()

        trainer = L.Trainer(max_epochs=n_epochs, callbacks=[CustomProgressBar()])
        print(
            f"\n\nTraining {model_cls.__name__} "
            f"on device {trainer.strategy.root_device}"
        )
        trainer.fit(model, train_loader, val_loader)

        model_result = trainer.test(model, val_loader)[0]
        results[model_cls.__name__] = model_result

        if isinstance(model, ConceptExplanationModel):
            print("Local Explanations: ")
            local_expl = model.get_local_explanations(x)
            print(local_expl)

            print("Global Explanations: ")
            print(model.get_global_explanations(x))

            print("Explanation Counter: ")
            print(get_most_common_expl(local_expl))

    results = pd.DataFrame(results).T
    print(results[["test_c_acc", "test_c_avg_auc", "test_y_acc", "test_loss"]])
    results.to_csv("model_results.csv")


if __name__ == "__main__":
    main()
