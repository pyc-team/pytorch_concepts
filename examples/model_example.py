# !/usr/local/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import torch
import lightning as L
from torch.utils.data import TensorDataset

from torch_concepts.data import ToyDataset
from torch_concepts.data.utils import stratified_train_test_split
from torch_concepts.nn.models import ConceptBottleneckModel, \
    ConceptResidualModel, ConceptEmbeddingModel


def main():
    latent_dims = 20
    n_epochs = 100
    n_samples = 1000
    class_reg = 0.5
    batch_size = 1024
    residual_size = 4
    embedding_size = 4

    models = [ConceptBottleneckModel,
              ConceptResidualModel,
              ConceptEmbeddingModel]

    data = ToyDataset('xor', size=n_samples, random_state=42)
    x, c, y = data.data, data.concept_labels, data.target_labels
    concept_names, task_names = data.concept_attr_names, data.task_attr_names

    dataset = TensorDataset(x, c, y)
    train_set, val_set = stratified_train_test_split(dataset, test_size=0.2)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size)

    n_features = x.shape[1]
    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, latent_dims),
                                  torch.nn.LeakyReLU())

    results = {}
    for model_cls in models:
        model = model_cls(encoder, latent_dims, concept_names, task_names,
                          class_reg=class_reg, residual_size=residual_size,
                          embedding_size=embedding_size)
        model.configure_optimizers()

        trainer = L.Trainer(max_epochs=n_epochs)
        trainer.fit(model, train_loader)

        model_result = trainer.test(model, val_loader)[0]
        results[model_cls.__name__] = model_result

    print(pd.DataFrame(results))

    return


if __name__ == "__main__":
    main()
