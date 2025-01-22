from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import lightning as L
from sklearn.metrics import accuracy_score, f1_score

from torch_concepts.nn import LinearConceptBottleneck, \
    LinearConceptResidualBottleneck, ConceptEmbeddingBottleneck, \
    LinearConceptLayer


class ConceptModel(ABC, L.LightningModule):
    @abstractmethod
    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 class_reg=0.1, c_loss_fn=nn.BCELoss(),
                 y_loss_fn=nn.BCEWithLogitsLoss(),
                 intervention_prob=0.1, intervention_idxs=None,
                 **kwargs):
        super().__init__()

        assert (len(task_names) > 1 or
                not isinstance(y_loss_fn, nn.CrossEntropyLoss)), \
            "CrossEntropyLoss requires at least two classes"
        assert isinstance(y_loss_fn, nn.BCEWithLogitsLoss) or \
                isinstance(y_loss_fn, nn.CrossEntropyLoss), \
            "y_loss_fn must be either BCEWithLogitsLoss or CrossEntropyLoss"

        self.encoder = encoder
        self.latent_dim = latent_dim
        self.concept_names = concept_names
        self.task_names = task_names
        self.c_loss_fn = c_loss_fn
        self.y_loss_fn = y_loss_fn
        self.multi_class = isinstance(y_loss_fn, nn.CrossEntropyLoss)
        self.class_reg = class_reg
        self.int_prob = intervention_prob
        if intervention_idxs is None:
            intervention_idxs = torch.ones(len(concept_names)).bool()
        self.intervention_idxs = intervention_idxs

    @abstractmethod
    def forward(self, x, c_true=None):
        pass

    def step(self, batch, mode="train") -> torch.Tensor:
        x, c_true, y_true = batch

        # Intervene on concepts only during training step
        if mode == "train":
            c_pred, y_pred = self.forward(x, c_true)
        else:
            c_pred, y_pred = self.forward(x)

        c_loss = 0.
        if c_pred is not None:
            c_loss = self.c_loss_fn(c_pred, c_true)

        y_loss = self.y_loss_fn(y_pred, y_true)

        loss = c_loss + self.class_reg * y_loss

        c_acc, c_f1 = 0., 0.
        if c_pred is not None:
            c_acc = accuracy_score(c_true, c_pred > 0.5)
            c_f1 = f1_score(c_true, c_pred > 0.5, average='macro')
        # Extract most likely class in multi-class classification
        y_pred = y_pred.argmax(dim=1) if self.multi_class else y_pred > 0.
        y_acc = accuracy_score(y_true, y_pred)

        # Log metrics on progress bar only during validation
        prog = mode == "val"
        self.log(f'{mode}_c_acc', c_acc, on_epoch=True, prog_bar=prog)
        self.log(f'{mode}_c_f1', c_f1, on_epoch=True, prog_bar=prog)
        self.log(f'{mode}_y_acc', y_acc, on_epoch=True, prog_bar=prog)
        self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=prog)
        self.log(f'{mode}_c_loss', c_loss, on_epoch=True, prog_bar=prog)
        self.log(f'{mode}_y_loss', y_loss, on_epoch=True, prog_bar=prog)

        return loss

    def training_step(self, batch) -> torch.Tensor:
        return self.step(batch, mode="train")

    def validation_step(self, batch) -> torch.Tensor:
        return self.step(batch, mode="val")

    def test_step(self, batch):
        return self.step(batch, mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        return optimizer


class ConceptBottleneckModel(ConceptModel):
    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 *args, **kwargs):
        super().__init__(encoder, latent_dim, concept_names, task_names,
                         *args, **kwargs)

        self.bottleneck = LinearConceptBottleneck(latent_dim, concept_names)
        self.y_predictor = nn.Sequential(
            nn.Linear(len(concept_names), latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, len(task_names)),
        )

    def forward(self, x, c_true=None):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent, c_true=c_true,
                                    intervention_idxs=self.intervention_idxs,
                                    intervention_rate=self.int_prob)
        c_pred = c_dict['c_pred']
        y_pred = self.y_predictor(c_pred)
        return c_pred, y_pred


class ConceptResidualModel(ConceptModel):
    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 residual_size, **kwargs):
        super().__init__(encoder, latent_dim, concept_names, task_names,
                         **kwargs)

        self.bottleneck = LinearConceptResidualBottleneck(latent_dim,
                                                          concept_names,
                                                          residual_size)
        self.y_predictor = nn.Sequential(
            nn.Linear(len(concept_names) + residual_size, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, len(task_names)),
        )

    def forward(self, x, c_true=None):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_pred']
        y_pred = self.y_predictor(c_emb)
        return c_pred, y_pred


class ConceptEmbeddingModel(ConceptModel):
    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 embedding_size, **kwargs):
        super().__init__(encoder, latent_dim, concept_names, task_names,
                         **kwargs)

        self.bottleneck = ConceptEmbeddingBottleneck(latent_dim, concept_names,
                                                     embedding_size)
        self.y_predictor = nn.Sequential(
            nn.Linear(len(concept_names) * embedding_size, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, len(task_names)),
        )

    def forward(self, x, c_true=None):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_pred']
        y_pred = self.y_predictor(c_emb.flatten(-2))
        return c_pred, y_pred
