from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import lightning as L
from sklearn.metrics import accuracy_score, f1_score

from torch_concepts.nn import LinearConceptBottleneck, \
    LinearConceptResidualBottleneck, ConceptEmbeddingBottleneck, \
    LinearConceptLayer, Annotate
from torch_concepts.semantic import ProductTNorm
from torch_concepts.nn import functional as CF
from torch_concepts.utils import get_global_explanations


class ConceptModel(ABC, L.LightningModule):
    @abstractmethod
    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 class_reg=0.1, c_loss_fn=nn.BCELoss(),
                 y_loss_fn=nn.BCEWithLogitsLoss(),
                 int_prob=0.1, int_idxs=None,
                 **kwargs):
        super().__init__()

        assert (len(task_names) > 1 or
                not isinstance(y_loss_fn, nn.CrossEntropyLoss)), \
            "CrossEntropyLoss requires at least two tasks"
        assert isinstance(y_loss_fn, nn.BCEWithLogitsLoss) or \
                isinstance(y_loss_fn, nn.CrossEntropyLoss), \
            "y_loss_fn must be either BCEWithLogitsLoss or CrossEntropyLoss"

        self.encoder = encoder
        self.latent_dim = latent_dim
        self.concept_names = concept_names
        self.task_names = task_names
        self.n_concepts = len(concept_names)
        self.n_tasks = len(task_names)
        
        self.c_loss_fn = c_loss_fn
        self.y_loss_fn = y_loss_fn
        self.multi_class = isinstance(y_loss_fn, nn.CrossEntropyLoss)
        self.class_reg = class_reg
        self.int_prob = int_prob
        if int_idxs is None:
            int_idxs = torch.ones(len(concept_names)).bool()
        self.int_idxs = int_idxs

    @abstractmethod
    def forward(self, x, c_true=None, **kwargs):
        pass

    def step(self, batch, mode="train") -> torch.Tensor:
        x, c_true, y_true = batch

        # Intervene on concepts and memory reconstruction only on training
        if mode == "train":
            y_pred, c_pred = self.forward(x,
                                          c_true=c_true,
                                          y_true=y_true)
        else:
            y_pred, c_pred = self.forward(x)

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


class ConceptExplanationModel(ConceptModel):
    @abstractmethod
    def get_local_explanations(self, x, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_global_explanations(self, x, **kwargs):
        raise NotImplementedError()


class ConceptBottleneckModel(ConceptModel):
    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 *args, **kwargs):
        super().__init__(encoder, latent_dim, concept_names, task_names,
                         **kwargs)

        self.bottleneck = LinearConceptBottleneck(latent_dim, concept_names)
        self.y_predictor = nn.Sequential(
            nn.Linear(len(concept_names), latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, len(task_names)),
        )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent, c_true=c_true,
                                        intervention_idxs=self.int_idxs,
                                        intervention_rate=self.int_prob)
        c_pred = c_dict['c_pred']
        y_pred = self.y_predictor(c_pred)
        return y_pred, c_pred


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

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent, c_true=c_true,
                                        intervention_idxs=self.int_idxs,
                                        intervention_rate=self.int_prob)
        c_pred = c_dict['c_pred']
        y_pred = self.y_predictor(c_emb)
        return y_pred, c_pred


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

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent, c_true=c_true,
                                        intervention_idxs=self.int_idxs,
                                        intervention_rate=self.int_prob)
        c_pred = c_dict['c_int']
        y_pred = self.y_predictor(c_emb.flatten(-2))
        return y_pred, c_pred


class DeepConceptReasoning(ConceptExplanationModel):
    n_roles = 3
    memory_names = ['Positive', 'Negative', 'Irrelevant']
    temperature = 100

    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 embedding_size, semantic=ProductTNorm(), **kwargs):
        super().__init__(encoder, latent_dim, concept_names, task_names,
                         **kwargs)
        self.semantic = semantic
        self.bottleneck = ConceptEmbeddingBottleneck(latent_dim, concept_names,
                                                     embedding_size)

        # module predicting concept imp. for all concepts tasks and roles
        # its input is batch_size x n_concepts x embedding_size
        # its output is batch_size x n_concepts x n_tasks x n_roles
        self.concept_importance_predictor = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(),
            nn.Linear(embedding_size, self.n_tasks * self.n_roles),
            nn.Unflatten(-1, (self.n_tasks, self.n_roles)),
        )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent, c_true=c_true,
                                        intervention_idxs=self.int_idxs,
                                        intervention_rate=self.int_prob)
        c_pred = c_dict['c_int']
        c_weights = self.concept_importance_predictor(c_emb)
        # adding memory dimension
        c_weights = c_weights.unsqueeze(dim=1)
        # soft selecting important concepts
        relevance = CF.soft_select(c_weights, self.temperature, -3)
        # softmax over roles
        polarity = c_weights.softmax(-1)
        # batch_size x memory_size x n_concepts x n_tasks x n_roles
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)

        y_pred = CF.logic_rule_eval(c_weights, c_pred,
                                    semantic=self.semantic)
        # removing memory dimension
        y_pred = y_pred[:, :, 0]

        # transform probabilities to logits
        y_pred = torch.log(y_pred / (1 - y_pred))

        return y_pred, c_pred

    def get_local_explanations(self, x, return_preds=False, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        c_weights = self.concept_importance_predictor(c_emb)
        relevance = CF.soft_select(c_weights, self.temperature, -3)
        polarity = c_weights.softmax(-1)
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)
        local_explanations = CF.logic_rule_eval(c_weights, c_pred,
                                                semantic=self.semantic)
        if return_preds:
            y_preds = CF.logic_rule_eval(c_weights, c_pred,
                                         semantic=self.semantic)
            return local_explanations, y_preds[:, :, 0]
        return local_explanations

    def get_global_explanations(self, x, **kwargs):
        explanations, y_preds = self.get_local_explanations(x, return_preds=True)
        return get_global_explanations(explanations, y_preds, self.task_names)


class ConceptMemoryReasoning(ConceptExplanationModel):
    n_roles = 3
    memory_names = ['Positive', 'Negative', 'Irrelevant']

    def __init__(self, encoder, latent_dim, concept_names, task_names,
                    memory_size, **kwargs):
        super().__init__(encoder, latent_dim, concept_names, task_names,
                        **kwargs)
        self.memory_size = memory_size
        self.bottleneck = LinearConceptBottleneck(latent_dim, concept_names)

        self.concept_memory = torch.nn.Embedding(memory_size, self.latent_dim)
        self.memory_decoder = LinearConceptLayer(self.latent_dim,
                                                 [self.concept_names,
                                                  self.task_names,
                                                  self.memory_names])
        self.classifier_selector = nn.Sequential(
            LinearConceptLayer(latent_dim, [self.task_names,
                                            memory_size]),
        )

    def forward(self, x, c_true=None, y_true=None, **kwargs):
        # generate concept and task predictions
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent, c_true=c_true,
                                        intervention_idxs=self.int_idxs,
                                        intervention_rate=self.int_prob)
        c_pred = c_dict['c_int']
        classifier_selector_logits = self.classifier_selector(latent)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        # softmax over roles and adding batch dimension to concept memory
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)

        y_per_classifier = CF.logic_rule_eval(concept_weights, c_pred)
        if y_true is not None:
            c_rec_per_classifier = CF.logic_memory_reconstruction(concept_weights,
                                                                c_true, y_true)
            y_pred = CF.selection_eval(prob_per_classifier, y_per_classifier,
                                       c_rec_per_classifier)
        else:
            y_pred = CF.selection_eval(prob_per_classifier, y_per_classifier)

        # transform probabilities to logits
        y_pred = torch.log(y_pred / (1 - y_pred))

        return y_pred, c_pred

    def get_local_explanations(self, x, return_preds=False, **kwargs):
        emb = self.encoder(x)
        c_pred = self.concept_bottleneck(emb).sigmoid()
        classifier_selector_logits = self.classifier_selector(emb)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        # softmax over roles and adding batch dimension to concept memory
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)

        explanations = CF.logic_rule_explanations(concept_weights,
                                               {1: self.concept_names,
                                                2: self.class_names})
        return explanations

    def get_global_explanations(self, x, **kwargs):
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)

        global_explanations = CF.logic_rule_explanations(concept_weights,
                                                         {1: self.concept_names,
                                                          2: self.class_names})

        return global_explanations[0]


class LinearConceptEmbeddingModel(ConceptExplanationModel):
    def __init__(self, encoder, latent_dim, concept_names, task_names,
                 embedding_size, bias=True, **kwargs):
        super().__init__(encoder, latent_dim, concept_names, task_names,
                         **kwargs)
        self.bias = bias

        self.bottleneck = ConceptEmbeddingBottleneck(latent_dim, concept_names,
                                                     embedding_size)
        # module predicting the concept importance for all concepts and tasks
        # input batch_size x concept_number x embedding_size
        # output batch_size x concept_number x task_number
        self.concept_relevance = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, len(task_names)),
            Annotate([concept_names, task_names], [1, 2])
        )
        # module predicting the class bias for each class
        # input batch_size x concept_number x embedding_size
        # output batch_size x task_number
        if self.bias:
            self.bias_predictor = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(self.n_concepts * embedding_size,
                                embedding_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(embedding_size, self.n_tasks),
                Annotate([task_names], 1)
            )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        c_weights = self.concept_relevance(c_emb)

        y_bias = None
        if self.bias:
            y_bias = self.bias_predictor(c_emb)

        c_weights, y_bias = c_weights.unsqueeze(dim=1), y_bias.unsqueeze(dim=1)
        y_pred = CF.linear_equation_eval(c_weights, c_pred, y_bias)
        return y_pred[:, :, 0], c_pred

    def get_local_explanations(self, x, return_preds=False, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_pred']
        c_weights = self.concept_relevance(c_emb)

        y_bias = None
        if self.bias:
            y_bias = self.bias_predictor(c_emb)
        linear_equations = CF.linear_equation_eval(c_weights, c_pred, y_bias)

        if return_preds:
            y_preds = CF.linear_equation_eval(c_weights, c_pred, y_bias)
            return linear_equations, y_preds
        return linear_equations

    def get_global_explanations(self, x, **kwargs):
        explanations, y_preds = self.get_local_explanations(x, return_preds=True)
        return get_global_explanations(explanations, y_preds, self.task_names)

