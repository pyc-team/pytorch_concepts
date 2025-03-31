import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch_concepts.nn as pyc_nn
import torch.nn as nn
import torch.nn.functional as F
import warnings

from abc import abstractmethod, ABC
from sklearn.metrics import accuracy_score, f1_score
from torch_concepts.nn import functional as CF
from torch_concepts.semantic import ProductTNorm
from torch_concepts.utils import get_global_explanations

class ConceptModel(ABC, L.LightningModule):
    @abstractmethod
    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        class_reg=0.1,
        c_loss_fn=nn.BCELoss(),
        y_loss_fn=nn.BCEWithLogitsLoss(),
        int_prob=0.1,
        int_idxs=None,
        l_r=0.01,
        **kwargs,
    ):
        super().__init__()

        assert (len(task_names) > 1 or
                not isinstance(y_loss_fn, nn.CrossEntropyLoss)), \
            "CrossEntropyLoss requires at least two tasks"

        self.encoder = encoder
        self.latent_dim = latent_dim
        self.concept_names = concept_names
        self.task_names = task_names
        self.n_concepts = len(concept_names)
        self.n_tasks = len(task_names)
        self.l_r = l_r

        self.c_loss_fn = c_loss_fn
        self.y_loss_fn = y_loss_fn
        self.multi_class = len(task_names) > 1
        self.class_reg = class_reg
        self.int_prob = int_prob
        if int_idxs is None:
            int_idxs = torch.ones(len(concept_names)).bool()
        self.register_buffer("int_idxs", int_idxs, persistent=True)
        self.test_intervention = False
        self._train_losses = []
        self._val_losses = []

        self._bce_loss = isinstance(y_loss_fn, nn.BCELoss) or \
            isinstance(y_loss_fn, nn.BCEWithLogitsLoss)

    @abstractmethod
    def forward(self, x, c_true=None, **kwargs):
        pass

    def step(self, batch, mode="train") -> torch.Tensor:
        x, c_true, y_true = batch

        # Intervene on concept and memory reconstruction only on training
        # or if explicitly set to True
        if mode == "train":
            y_pred, c_pred = self.forward(x, c_true=c_true, y_true=y_true)
        elif self.test_intervention:
            y_pred, c_pred = self.forward(x, c_true=c_true)
        else:
            y_pred, c_pred = self.forward(x)

        c_loss = 0.
        if c_pred is not None:
            c_loss = self.c_loss_fn(c_pred, c_true)

        # BCELoss requires one-hot encoding
        if self._bce_loss and self.multi_class:
            if y_true.squeeze().dim() == 1:
                y_true = F.one_hot(y_true.long(),
                                   self.n_tasks).squeeze().float()
        elif y_true.dim() == 1 and self._bce_loss:
            y_true = y_true.unsqueeze(-1) # add a dimension

        y_loss = self.y_loss_fn(y_pred, y_true)
        loss = c_loss + self.class_reg * y_loss

        c_acc, c_f1 = 0., 0.
        if c_pred is not None:
            c_acc = accuracy_score(c_true.cpu(), (c_pred.cpu() > 0.5).float())
            c_f1 = f1_score(c_true.cpu(), c_pred.cpu() > 0.5, average='macro')

        # Extract most likely class in multi-class classification
        if self.multi_class and y_true.squeeze().dim() == 1:
            y_pred = y_pred.argmax(dim=1)
        # Extract prediction from sigmoid output
        elif isinstance(self.y_loss_fn, nn.BCELoss):
            y_pred = (y_pred > 0.5).float()
        # Extract prediction from logits
        else:
            y_pred = (y_pred > 0.).float()
        y_acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())
        # manually compute accuracy
        # y_acc = (y_pred == y_true).sum().item() / len(y_true)

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
        loss = self.step(batch, mode="train")
        self._train_losses.append(loss.item())
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        loss = self.step(batch, mode="val")
        self._val_losses.append(loss.item())
        return loss

    def test_step(self, batch):
        return self.step(batch, mode="test")

    def configure_optimizers(self):
        print(f"Employing AdamW optimizer with learning rate {self.l_r}")
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.l_r)
        return optimizer

    def on_train_end(self) -> None:
        # plot losses
        sns.lineplot(
            x=torch.linspace(0, 1, len(self._train_losses)),
            y=self._train_losses,
        )
        sns.lineplot(
            x=torch.linspace(0, 1, len(self._val_losses)),
            y=self._val_losses,
        )
        model_name = INV_AVAILABLE_MODELS[self.__class__]
        plt.title("Train and validation losses -- " + model_name)
        plt.ylabel("Loss")
        plt.xlabel("Step")
        plt.ylim(0.001, 10)
        plt.yscale("log")
        plt.show()


class ConceptExplanationModel(ConceptModel):
    @abstractmethod
    def get_local_explanations(self, x, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_global_explanations(self, x, **kwargs):
        raise NotImplementedError()


class ConceptBottleneckModel(ConceptModel):
    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        *args,
        **kwargs,
    ):
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            latent_dim,
            concept_names,
        )
        self.y_predictor = nn.Sequential(
            nn.Linear(len(concept_names), latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, len(task_names)),
        )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_pred, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=self.int_idxs,
            intervention_rate=self.int_prob,
        )
        y_pred = self.y_predictor(c_pred)
        return y_pred, c_pred


class ConceptResidualModel(ConceptModel):
    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        residual_size,
        **kwargs,
    ):
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )

        self.bottleneck = pyc_nn.LinearConceptResidualBottleneck(
            latent_dim,
            concept_names,
            residual_size,
        )
        self.y_predictor = nn.Sequential(
            nn.Linear(len(concept_names) + residual_size, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, len(task_names)),
        )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=self.int_idxs,
            intervention_rate=self.int_prob,
        )
        c_pred = c_dict['c_int']
        y_pred = self.y_predictor(c_emb)
        return y_pred, c_pred


class ConceptEmbeddingModel(ConceptModel):
    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        embedding_size,
        **kwargs,
    ):
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_dim,
            concept_names,
            embedding_size,
        )
        self.y_predictor = nn.Sequential(
            nn.Linear(len(concept_names) * embedding_size, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, len(task_names)),
        )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=self.int_idxs,
            intervention_rate=self.int_prob,
        )
        c_pred = c_dict['c_int']
        y_pred = self.y_predictor(c_emb.flatten(-2))
        return y_pred, c_pred


class DeepConceptReasoning(ConceptExplanationModel):
    n_roles = 3
    memory_names = ['Positive', 'Negative', 'Irrelevant']
    temperature = 100

    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        embedding_size,
        semantic=ProductTNorm(),
        **kwargs,
    ):
        if 'y_loss_fn' in kwargs:
            if not isinstance(kwargs['y_loss_fn'], nn.BCELoss):
                warnings.warn(
                    "DCR y_loss_fn must be a BCELoss. Changing to BCELoss."
                )

        kwargs['y_loss_fn'] = nn.BCELoss()
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )
        self.semantic = semantic
        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_dim,
            concept_names,
            embedding_size,
        )

        # module predicting concept imp. for all concepts tasks and roles
        # its input is batch_size x n_concepts x embedding_size
        # its output is batch_size x n_concepts x n_tasks x n_roles
        self.concept_importance_predictor = nn.Sequential(
            nn.Linear(embedding_size, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, self.n_tasks * self.n_roles),
            nn.Unflatten(-1, (self.n_tasks, self.n_roles)),
        )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=self.int_idxs,
            intervention_rate=self.int_prob,
        )
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

        # transform probabilities to logits # does not work, numerical problems
        # y_pred = torch.log(y_pred) - torch.log(1 - y_pred + eps)

        return y_pred, c_pred

    def get_local_explanations(self, x, return_preds=False, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        c_weights = self.concept_importance_predictor(c_emb)
        relevance = CF.soft_select(c_weights, self.temperature, -3)
        polarity = c_weights.softmax(-1)
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)
        local_explanations = CF.logic_rule_eval(
            c_weights,
            c_pred,
            semantic=self.semantic,
        )
        if return_preds:
            y_preds = CF.logic_rule_eval(
                c_weights,
                c_pred,
                semantic=self.semantic,
            )
            return local_explanations, y_preds[:, :, 0]
        return local_explanations

    def get_global_explanations(self, x, **kwargs):
        explanations, y_preds = self.get_local_explanations(
            x,
            return_preds=True,
        )
        return get_global_explanations(explanations, y_preds, self.task_names)


class ConceptMemoryReasoning(ConceptExplanationModel):
    n_roles = 3
    memory_names = ['Positive', 'Negative', 'Irrelevant']

    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        memory_size,
        **kwargs,
    ):
        if 'y_loss_fn' in kwargs:
            if not isinstance(kwargs['y_loss_fn'], nn.BCELoss):
                warnings.warn(
                    "CMR y_loss_fn must be a BCELoss. Changing to BCELoss."
                )
        kwargs['y_loss_fn'] = nn.BCELoss()
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )

        self.memory_size = memory_size
        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            latent_dim,
            concept_names,
        )

        self.concept_memory = torch.nn.Embedding(
            memory_size,
            self.latent_dim,
        )
        self.memory_decoder = pyc_nn.LinearConceptLayer(
            self.latent_dim,
            [
                self.concept_names,
                self.task_names,
                self.memory_names,
            ],
        )
        self.classifier_selector = nn.Sequential(
            pyc_nn.LinearConceptLayer(
                latent_dim,
                [self.task_names, memory_size],
            ),
        )

    def forward(self, x, c_true=None, y_true=None, **kwargs):
        # generate concept and task predictions
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=self.int_idxs,
            intervention_rate=self.int_prob,
        )
        c_pred = c_dict['c_int']
        classifier_selector_logits = self.classifier_selector(latent)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        # softmax over roles and adding batch dimension to concept memory
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)

        y_per_classifier = CF.logic_rule_eval(concept_weights, c_pred)
        if y_true is not None:
            c_rec_per_classifier = self._conc_recon(concept_weights, 
                                                    c_true, 
                                                    y_true)
            y_pred = CF.selection_eval(
                prob_per_classifier,
                y_per_classifier,
                c_rec_per_classifier,
            )
        else:
            y_pred = CF.selection_eval(prob_per_classifier,
                                       y_per_classifier)

        return y_pred, c_pred

    def _conc_recon(self, concept_weights, c_true, y_true):
        # check if y_true is an array (label encoding) or a matrix
        # (one-hot encoding) in case it is an array convert it to a matrix
        # if it is a multi-class task
        if len(y_true.squeeze().shape) == 1 and self.multi_class:
            y_true = torch.nn.functional.one_hot(
                y_true.squeeze().long(),
                len(self.task_names),
            )

        elif len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(-1)
        c_rec_per_classifier = CF.logic_memory_reconstruction(
            concept_weights,
            c_true,
            y_true,
        )
        # weighting the reconstruction loss - lower reconstruction weights
        # brings values closer to 1 thus influencing less the prediction
        c_rec_per_classifier = c_rec_per_classifier * self.rec_weight + \
                               (1 - self.rec_weight)
        return c_rec_per_classifier
    

        return y_pred, c_pred

    def get_local_explanations(self, x, return_preds=False, **kwargs):
        emb = self.encoder(x)
        classifier_selector_logits = self.classifier_selector(emb)
        # softmax over roles and adding batch dimension to concept memory
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)

        explanations = CF.logic_rule_explanations(
            concept_weights,
            {
                1: self.concept_names,
                2: self.task_names,
            },
        )
        return explanations

    def get_global_explanations(self, x, **kwargs):
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)

        global_explanations = CF.logic_rule_explanations(
            concept_weights,
            {
                1: self.concept_names,
                2: self.task_names,
            },
        )

        return global_explanations[0]


class LinearConceptEmbeddingModel(ConceptExplanationModel):
    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        embedding_size,
        bias=True,
        **kwargs,
    ):
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )
        self.bias = bias

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_dim,
            concept_names,
            embedding_size,
        )
        # module predicting the concept importance for all concepts and tasks
        # input batch_size x concept_number x embedding_size
        # output batch_size x concept_number x task_number
        self.concept_relevance = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, latent_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(latent_dim, len(task_names)),
            pyc_nn.Annotate([concept_names, task_names], [1, 2])
        )
        # module predicting the class bias for each class
        # input batch_size x concept_number x embedding_size
        # output batch_size x task_number
        if self.bias:
            self.bias_predictor = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(
                    self.n_concepts * embedding_size,
                    embedding_size,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(embedding_size, self.n_tasks),
                pyc_nn.Annotate([task_names], 1)
            )

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=self.int_idxs,
            intervention_rate=self.int_prob,
        )
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
        c_pred = c_dict['c_int']
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
        explanations, y_preds = self.get_local_explanations(
            x,
            return_preds=True,
        )
        return get_global_explanations(explanations, y_preds, self.task_names)


AVAILABLE_MODELS = {
    "ConceptBottleneckModel": ConceptBottleneckModel,
    "ConceptResidualModel": ConceptResidualModel,
    "ConceptEmbeddingModel": ConceptEmbeddingModel,
    "DeepConceptReasoning": DeepConceptReasoning,
    "LinearConceptEmbeddingModel": LinearConceptEmbeddingModel,
    "ConceptMemoryReasoning": ConceptMemoryReasoning,
}

MODELS_ACRONYMS = {
    "ConceptBottleneckModel": "CBM",
    "ConceptResidualModel": "CRM",
    "ConceptEmbeddingModel": "CEM",
    "DeepConceptReasoning": "DCR",
    "LinearConceptEmbeddingModel": "LICEM",
    "ConceptMemoryReasoning": "CMR",
}