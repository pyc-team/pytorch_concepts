import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch_concepts.nn as pyc_nn
import torch.nn as nn
import torch.nn.functional as F
import warnings

from abc import abstractmethod, ABC
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Optional, List, Dict

from packaging import version
if version.parse(torch.__version__) < version.parse("2.0.0"):
    # Then we will use pytorch lightning's version compatible with PyTorch < 2.0
    import pytorch_lightning as L
else:
    import lightning as L

from torch_concepts.nn import functional as CF
from torch_concepts.semantic import ProductTNorm
from torch.distributions import RelaxedBernoulli
from torch_concepts.utils import compute_temperature

class ConceptModel(ABC, L.LightningModule):
    """
    Abstract class for concept-based models. It defines the basic structure
    of a concept-based model and the methods that should be implemented by
    the subclasses. The concept-based models are models that predict the
    output of a task based on the concepts extracted from the input data.

    Attributes:
        encoder (torch.nn.Module): The encoder module that extracts the
            features from the input data.
        latent_dim (int): The dimension of the latent space.
        concept_names (list[str]): The names of the concepts extracted from
            the input data.
        task_names (list[str]): The names of the tasks to predict.
        class_reg (float): The regularization factor for the task
            classification loss.
        c_loss_fn (torch.nn.Module): The loss function for learning the
            concepts.
        y_loss_fn (torch.nn.Module): The loss function for learning the
            tasks.
        int_prob (float): The probability of intervening on the concepts at
            training time.
        int_idxs (torch.Tensor): The indices of the concepts to intervene
            on.
        l_r (float): The learning rate for the optimizer.

    """

    @abstractmethod
    def __init__(
        self,
        encoder: torch.nn.Module,
        latent_dim: int,
        concept_names : list[str],
        task_names: list[str],
        class_reg: float=0.1,
        concept_reg: float=1,
        c_loss_fn=nn.BCELoss(),
        y_loss_fn=nn.BCEWithLogitsLoss(),
        int_prob=0.1,
        int_idxs=None,
        l_r=0.01,
        optimizer_config=None,
        concept_weights=None,
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
        self.optimizer_config = optimizer_config \
            if optimizer_config is not None else {}
        self.optimizer_config['learning_rate'] = self.l_r

        self.c_loss_fn = c_loss_fn
        self.y_loss_fn = y_loss_fn
        self.class_reg = class_reg
        self.concept_reg = concept_reg
        self.int_prob = int_prob
        if int_idxs is None:
            int_idxs = torch.ones(len(concept_names)).bool()
        self.register_buffer("int_idxs", int_idxs, persistent=True)
        self.test_intervention = False
        self._train_losses = []
        self._val_losses = []

        self._bce_loss = isinstance(y_loss_fn, nn.BCELoss) or \
            isinstance(y_loss_fn, nn.BCEWithLogitsLoss)
        self._multi_class = len(task_names) > 1

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
        if (self._bce_loss and self._multi_class and
                y_true.squeeze().dim() == 1):
            y_true_loss = F.one_hot(
                y_true.long(),
                self.n_tasks,
            ).squeeze().float()
        elif self._bce_loss and y_true.squeeze().dim() == 1 :
            y_true_loss = y_true.unsqueeze(-1) # add a dimension
        else:
            y_true_loss = y_true

        y_loss = self.y_loss_fn(y_pred, y_true_loss)
        loss = self.concept_reg * c_loss + self.class_reg * y_loss

        c_acc, c_avg_auc = 0., 0.
        if c_pred is not None:
            c_acc = accuracy_score(c_true.cpu(), (c_pred.cpu() > 0.5).float())
            c_avg_auc = roc_auc_score(
                c_true.cpu().view(-1),
                (c_pred.cpu().view(-1) > 0.5).float()
            )

        # Extract most likely class in multi-class classification
        if self._multi_class and y_true.squeeze().dim() == 1:
            y_pred = y_pred.argmax(dim=1)
        # Extract prediction from sigmoid output
        elif isinstance(self.y_loss_fn, nn.BCELoss):
            y_pred = (y_pred > 0.5).float()
        # Extract prediction from logits
        else:
            y_pred = (y_pred > 0.).float()
        y_acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())

        # Log metrics on progress bar only during validation
        if mode == "train":
            self.log(f'c_avg_auc', c_avg_auc, on_step=True, on_epoch=False, prog_bar=True)
            self.log(f'y_acc', y_acc, on_step=True, on_epoch=False, prog_bar=True)
            self.log(f'loss', loss, on_step=True, on_epoch=False, prog_bar=False)
        else:
            prog = mode == "val"
            self.log(f'{mode}_c_acc', c_acc, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_c_avg_auc', c_avg_auc, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_y_acc', y_acc, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_c_loss', c_loss, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_y_loss', y_loss, on_epoch=True, prog_bar=prog)

        return loss

    def training_step(self, batch, batch_no=None) -> torch.Tensor:
        loss = self.step(batch, mode="train")
        self._train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_no=None) -> torch.Tensor:
        loss = self.step(batch, mode="val")
        self._val_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_no=None):
        return self.step(batch, mode="test")

    def configure_optimizers(self):
        optimizer_name = self.optimizer_config.get('name', 'adamw')
        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer_config.get('learning_rate', 1e-3),
                weight_decay=self.optimizer_config.get('weight_decay', 0),
            )
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_config.get('learning_rate', 1e-3),
                weight_decay=self.optimizer_config.get('weight_decay', 0),
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.optimizer_config.get('learning_rate', 1e-3),
                weight_decay=self.optimizer_config.get('weight_decay', 0),
                momentum=self.optimizer_config.get('momentum', 0),
            )
        else:
            raise ValueError(
                f'Unsupported optimizer {optimizer_name}'
            )

        if self.optimizer_config.get('lr_scheduler_patience', 0) != 0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                patience=self.optimizer_config.get('lr_scheduler_patience', 0),
                factor=self.optimizer_config.get('lr_scheduler_factor', 0.1),
                min_lr=self.optimizer_config.get('lr_scheduler_min_lr', 1e-5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "loss",
            }
        return {
            "optimizer": optimizer,
            "monitor": "loss",
        }

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
    """
        Abstract class for concept-based models that provide local and global
        explanations. It extends the ConceptModel class and adds the methods
        get_local_explanations and get_global_explanations. The local
        explanations are the explanations for each input in the batch, while
        the global explanations are the explanations for the whole model.
    """

    @abstractmethod
    def get_local_explanations(
        self,
        x: torch.Tensor,
        multi_label=False,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Get local explanations for the model given a batch of inputs.
        It returns a list of dictionaries where each entry correspond
        to the local explanation for each input. This is a dictionary with
        the task name as key and the explanation as value. Only the predicted
        task is included in the explanation. In case of multi-label tasks,
        all tasks with a probability higher than 0.5 are included.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features).
            multi_label: boolean indicating if the task is multi-label.

        Returns:
            local_explanations (list[dict]): List of dictionaries with the
                local explanations for each input.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_global_explanations(
        self,
        x: Optional[torch.Tensor]=None,
        **kwargs,
    ) -> Dict[str, Dict[str, str]]:
        """
        Get the global explanations for the model. This is a dictionary of
        explanations for each task. Each task has a dictionary with all
        the explanations reported. Some models might require the input
        tensor x to compute the global explanations.

        Args:
            x (Optional[torch.Tensor]): Input tensor of shape (batch_size,
                n_features). Required for some models to compute the global
                explanations.

        Returns:
            global_explanations (dict[str, dict]): Dictionary with the global
                explanations for each task.
        """
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
    """
        DCR is a concept-based model that makes task prediction by means of
        a locally constructed logic rule made of concepts. The model uses a
        concept embedding bottleneck to extract the concepts from the input
        data. The concept roles are computed from the concept embeddings
        and are used to construct the define how concept enter the logic rule.
        The model uses a fuzzy system based on some semantic to compute
        the final prediction according to the predicted rules.

        Paper: https://arxiv.org/abs/2304.14068
    """
    n_roles = 3
    memory_names = ['Positive', 'Negative', 'Irrelevant']

    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        embedding_size,
        semantic=ProductTNorm(),
        temperature=100,
        use_bce=True,
        **kwargs,
    ):
        self.temperature = temperature
        if 'y_loss_fn' in kwargs:
            if isinstance(kwargs['y_loss_fn'], nn.CrossEntropyLoss):
                if use_bce:
                    warnings.warn(
                        "DCR y_loss_fn must operate with probabilities, not "
                        "logits. Changing CrossEntropyLoss to BCE."
                    )
                    kwargs['y_loss_fn'] = nn.BCELoss()
                else:
                    warnings.warn(
                        "DCR y_loss_fn must operate with probabilities, not "
                        "logits. Changing CrossEntropyLoss to NLLLoss with "
                        "a log."
                    )
                    kwargs['y_loss_fn'] = lambda input, target, **kwargs: \
                        torch.nn.functional.nll_loss(
                            torch.log(input / (input.sum(dim=-1, keepdim=True) + 1e-8) + 1e-8),
                            target,
                            **kwargs,
                        )
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
        self.temperature = temperature
        self._y_pred = None
        print(f"Setting concept temperature to {self.temperature}")

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
        # soft selecting concept relevance (last role) among concepts
        relevance = CF.soft_select(
            c_weights[:, :, :, :, -2:-1],
            self.temperature,
            -3,
        )
        # softmax over positive/negative roles
        polarity = c_weights[:, :, :, :, :-1].softmax(-1)
        # batch_size x memory_size x n_concepts x n_tasks x n_roles
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)

        y_pred = CF.logic_rule_eval(
            c_weights,
            c_pred,
            semantic=self.semantic,
        )
        # removing memory dimension
        y_pred = y_pred[:, :, 0]

        # converting probabilities to logits # REMOVED! it makes rules
        # difficult to learn. They might be false but they still get predicted
        # y_pred = torch.log(y_pred / (1 - y_pred + 1e-8) + 1e-8)

        return y_pred, c_pred

    def get_local_explanations(self, x, multi_label=False, **kwargs):
        assert not multi_label or self._multi_class, \
            "Multi-label explanations are supported only for multi-class tasks"
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        c_weights = self.concept_importance_predictor(c_emb)
        c_weights = c_weights.unsqueeze(dim=1) # add memory dimension
        relevance = CF.soft_select(
            c_weights[:, :, :, :, -2:-1],
            self.temperature,
            -3,
        )
        polarity = c_weights[:, :, :, :, :-1].softmax(-1)
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)
        explanations = CF.logic_rule_explanations(
            c_weights,
            {
                1: self.concept_names,
                2: self.task_names,
            },
        )
        y_pred = CF.logic_rule_eval(c_weights, c_pred,
                                     semantic=self.semantic)[:, :, 0]

        local_explanations = []
        for i in range(x.shape[0]):
            sample_expl = {}
            for j in range(self.n_tasks):
                # a task is predicted if it is the most likely task or is
                # a multi-label task with probability higher than 0.5 or is
                # a binary task with probability higher than 0.5
                if self._multi_class and not multi_label:
                    predicted_task = j == y_pred[i].argmax()
                else: # multi-label or binary
                    predicted_task = y_pred[i,j] > 0.5

                if predicted_task:
                    task_rules = explanations[i][self.task_names[j]]
                    predicted_rule = task_rules[f'Rule {0}']
                    sample_expl.update({self.task_names[j]: predicted_rule})
            local_explanations.append(sample_expl)
        return local_explanations

    def get_global_explanations(self, x=None, multi_label=False, **kwargs):
        assert x is not None, \
            "DCR requires input x to compute global explanations"

        local_explanations = self.get_local_explanations(x, multi_label)

        global_explanations = {}
        for i in range(self.n_tasks):
            task_explanations = {
                exp[self.task_names[i]]
                for exp in local_explanations if self.task_names[i] in exp
            }
            global_explanations[self.task_names[i]] = {
                f"Rule {j}": exp for j, exp in enumerate(task_explanations)
            }
        return global_explanations


class ConceptMemoryReasoning(ConceptExplanationModel):
    """
    This model represent an advancement of DCR as it stores the rules in a
    memory and selects the right one for the given input. The memory is
    a tensor of shape memory_size x n_concepts x n_tasks x n_roles. Each entry
    in the memory represents a rule for a task. The model predicts the current
    task according to which task is most likely given the predicted concepts.

    Paper: https://arxiv.org/abs/2407.15527
    """

    n_roles = 3
    memory_names = ['Positive', 'Negative', 'Irrelevant']

    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        memory_size,
        rec_weight=1,
        use_bce=True,
        **kwargs,
    ):
        if 'y_loss_fn' in kwargs:
            if isinstance(kwargs['y_loss_fn'], nn.CrossEntropyLoss):
                if use_bce:
                    warnings.warn(
                        "DCR y_loss_fn must operate with probabilities, not "
                        "logits. Changing CrossEntropyLoss to BCE."
                    )
                    kwargs['y_loss_fn'] = nn.BCELoss()
                else:
                    warnings.warn(
                        "DCR y_loss_fn must operate with probabilities, not "
                        "logits. Changing CrossEntropyLoss to NLLLoss with "
                        "a log."
                    )
                    kwargs['y_loss_fn'] = lambda input, target, **kwargs: \
                        torch.nn.functional.nll_loss(
                            torch.log(input / (input.sum(dim=-1, keepdim=True) + 1e-8) + 1e-8),
                            target,
                            **kwargs,
                        )
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )

        self.memory_size = memory_size
        self.rec_weight = rec_weight

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

        # converting probabilities to logits # REMOVED! it makes rules
        # difficult to learn. They might be false but they still get predicted
        # y_pred = torch.log(y_pred / (1 - y_pred + 1e-8) + 1e-8)

        return y_pred, c_pred

    def _conc_recon(self, concept_weights, c_true, y_true):
        # check if y_true is an array (label encoding) or a matrix
        # (one-hot encoding) in case it is an array convert it to a matrix
        # if it is a multi-class task
        if len(y_true.squeeze().shape) == 1 and self._multi_class:
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
        c_rec_per_classifier = torch.pow(c_rec_per_classifier, self.rec_weight)

        return c_rec_per_classifier


    def get_local_explanations(self, x, multi_label=False, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        classifier_selector_logits = self.classifier_selector(latent)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)
        y_per_classifier = CF.logic_rule_eval(concept_weights, c_pred)
        rule_probs = prob_per_classifier * y_per_classifier
        rule_preds = rule_probs.argmax(dim=-1) # = CF.most_likely_expl(rule_probs, multi_label)
        global_explanations = CF.logic_rule_explanations(
            concept_weights,
            {
                1: self.concept_names,
                2: self.task_names,
            },
        )
        local_expl = []
        y_pred = rule_probs.sum(dim=-1)
        for i in range(x.shape[0]):
            sample_expl = {}
            for j in range(self.n_tasks):
                # a task is predicted if it is the most likely task or is
                # a multi-label task with probability higher than 0.5 or is
                # a binary task with probability higher than 0.5
                predicted_task = (j == y_pred[i].argmax()) or \
                                 (multi_label and y_pred[i,j] > 0.5) or \
                                 (not self._multi_class and y_pred[i,j] > 0.5)
                if predicted_task:
                    task_rules = global_explanations[0][self.task_names[j]]
                    predicted_rule = task_rules[f'Rule {rule_preds[i, j]}']
                    sample_expl.update({self.task_names[j]: predicted_rule})
            local_expl.append(sample_expl)
        return local_expl

    def get_global_explanations(self, x=None, **kwargs):
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
        use_bias=True,
        weight_reg=1e-4,
        bias_reg=1e-4,
        **kwargs,
    ):
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )
        self.use_bias = use_bias

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
        if self.use_bias:
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

        self.weight_reg = weight_reg
        self.bias_reg = bias_reg
        self.__predicted_weights = None
        if self.use_bias:
            self.__predicted_bias = None

    def forward(self, x, c_true=None, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=self.int_idxs,
            intervention_rate=self.int_prob,
        )
        c_pred = c_dict['c_int']
        # adding memory dimension to concept weights
        c_weights = self.concept_relevance(c_emb).unsqueeze(dim=1)
        self.__predicted_weights = c_weights

        y_bias = None
        if self.use_bias:
            # adding memory dimension to bias
            y_bias = self.bias_predictor(c_emb).unsqueeze(dim=1)
            self.__predicted_bias = y_bias

        y_pred = CF.linear_equation_eval(c_weights, c_pred, y_bias)
        return y_pred[:, :, 0], c_pred

    def step(self, batch, mode="train") -> torch.Tensor:
        loss = super().step(batch, mode)

        # adding l2 regularization to the weights
        w_loss = self.weight_reg * self.__predicted_weights.norm(p=2)
        loss += w_loss

        prog = mode == "val"
        self.log(f'{mode}_weight_loss', w_loss, on_epoch=True,
                 prog_bar=prog)

        if self.use_bias:
            b_loss = self.bias_reg * self.__predicted_bias.norm(p=1)
            loss += b_loss
            self.log(f'{mode}_bias_loss', b_loss,
                     on_epoch=True, prog_bar=prog)

        return loss

    def get_local_explanations(self, x, multi_label=False, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        c_weights = self.concept_relevance(c_emb)

        y_bias = None
        if self.use_bias:
            y_bias = self.bias_predictor(c_emb)

        # adding memory dimension to concept weights and bias
        c_weights, y_bias = c_weights.unsqueeze(dim=1), y_bias.unsqueeze(dim=1)
        linear_equations = CF.linear_equation_expl(c_weights, y_bias, {
            1: self.concept_names,
            2: self.task_names,
        })
        y_pred = CF.linear_equation_eval(c_weights, c_pred, y_bias)

        local_expl = []
        for i in range(x.shape[0]):
            sample_expl = {}
            for j in range(self.n_tasks):
                # a task is predicted if it is the most likely task or if it is
                # a multi-label task and the probability is higher than 0.5
                # or is a binary task with probability higher than 0.5
                predicted_task = (j == y_pred[i].argmax()) or \
                                 (multi_label and y_pred[i, j] > 0.5)
                if predicted_task:
                    task_eqs = linear_equations[i]
                    predicted_eq = task_eqs[self.task_names[j]]['Equation 0']
                    sample_expl.update(
                        {self.task_names[j]: predicted_eq})
            local_expl.append(sample_expl)

        return local_expl

    def get_global_explanations(self, x=None, **kwargs):
        assert x is not None, "LinearConceptEmbeddingModel requires input x "\
                              "to compute global explanations"

        local_explanations = self.get_local_explanations(x, **kwargs)

        global_explanations = {}
        for i in range(self.n_tasks):
            task_explanations = {
                exp[self.task_names[i]]
                for exp in local_explanations if self.task_names[i] in exp
            }
            global_explanations[self.task_names[i]] = {
                f"Equation {j}": exp for j, exp in enumerate(task_explanations)
            }

        return global_explanations


class ConceptEmbeddingReasoning(ConceptMemoryReasoning):
    """
    This model is a combination of the ConceptEmbeddingModel and the
    ConceptMemoryReasoning model. It uses the concept embedding bottleneck
    to both to predict the concept and to select the rule from the concept
    memory. The concept memory is used to store the rules for each task.
    """
    n_roles = 3
    memory_names = ['Positive', 'Negative', 'Irrelevant']


    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        embedding_size,
        memory_size,
        use_bce=True,
        **kwargs,
    ):
        if 'y_loss_fn' in kwargs:
            if isinstance(kwargs['y_loss_fn'], nn.CrossEntropyLoss):
                if use_bce:
                    warnings.warn(
                        "DCR y_loss_fn must operate with probabilities, not "
                        "logits. Changing CrossEntropyLoss to BCE."
                    )
                    kwargs['y_loss_fn'] = nn.BCELoss()
                else:
                    warnings.warn(
                        "DCR y_loss_fn must operate with probabilities, not "
                        "logits. Changing CrossEntropyLoss to NLLLoss with "
                        "a log."
                    )
                    kwargs['y_loss_fn'] = lambda input, target, **kwargs: \
                        torch.nn.functional.nll_loss(
                            torch.log(input / (input.sum(dim=-1, keepdim=True) + 1e-8) + 1e-8),
                            target,
                            **kwargs,
                        )
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            memory_size,
            **kwargs,
        )

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_dim,
            concept_names,
            embedding_size,
        )

        self.classifier_selector = nn.Sequential(
            torch.nn.Linear(embedding_size*len(concept_names), latent_dim),
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
        classifier_selector_logits = self.classifier_selector(c_emb.flatten(-2))
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

        # converting probabilities to logits # REMOVED! it makes rules
        # difficult to learn. They might be false but they still get predicted
        # y_pred = torch.log(y_pred / (1 - y_pred + 1e-8) + 1e-8)

        return y_pred, c_pred

    def get_local_explanations(self, x, multi_label=False, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        classifier_selector_logits = self.classifier_selector(c_emb.flatten(-2))
        prob_per_classifier = torch.softmax(
            classifier_selector_logits,
            dim=-1,
        )
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)
        y_per_classifier = CF.logic_rule_eval(concept_weights, c_pred)
        rule_probs = prob_per_classifier * y_per_classifier
        rule_preds = rule_probs.argmax(
            dim=-1)  # = CF.most_likely_expl(rule_probs, multi_label)
        global_explanations = CF.logic_rule_explanations(
            concept_weights,
            {
                1: self.concept_names,
                2: self.task_names,
            },
        )
        local_expl = []
        y_pred = rule_probs.sum(dim=-1)
        for i in range(x.shape[0]):
            sample_expl = {}
            for j in range(self.n_tasks):
                # a task is predicted if it is the most likely task or is
                # a multi-label task with probability higher than 0.5 or is
                # a binary task with probability higher than 0.5
                if self._multi_class and not multi_label:
                    predicted_task = j == y_pred[i].argmax()
                else: # multi-label or binary
                    predicted_task = y_pred[i,j] > 0.5

                if predicted_task:
                    task_rules = global_explanations[0][self.task_names[j]]
                    predicted_rule = task_rules[f'Rule {rule_preds[i, j]}']
                    sample_expl.update(
                        {self.task_names[j]: predicted_rule})
            local_expl.append(sample_expl)
        return local_expl


class LinearConceptMemoryReasoning(ConceptExplanationModel):
    """
    This model is a combination of the LinearConceptEmbeddingModel and the
    ConceptMemoryReasoning model. It uses the concept embedding bottleneck
    to both to predict the concept and to select the equations from the
    memory. The memory is used to store the equations that can be used for each task.
    The model uses a linear equation to compute the final prediction according
    to the predicted equation. Differently from LICEM it does not use the bias.
    """

    def __init__(
        self,
        encoder,
        latent_dim,
        concept_names,
        task_names,
        embedding_size,
        memory_size,
        weight_reg=1e-4,
        negative_concepts=True,
        **kwargs,
    ):
        super().__init__(
            encoder,
            latent_dim,
            concept_names,
            task_names,
            **kwargs,
        )
        self.memory_size = memory_size
        self.weight_reg = weight_reg
        self.negative_concepts = negative_concepts

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_dim,
            concept_names,
            embedding_size,
        )

        self.classifier_selector = nn.Sequential(
            torch.nn.Linear(embedding_size * len(concept_names),
                            latent_dim),
            pyc_nn.LinearConceptLayer(
                latent_dim,
                [self.task_names, memory_size],
            ),
        )
        self.equation_memory = torch.nn.Embedding(
            memory_size,
            latent_dim
        )

        self.equation_decoder = pyc_nn.LinearConceptLayer(
            latent_dim,
            [
                self.concept_names,
                self.task_names,
            ],
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
        classifier_selector_logits = self.classifier_selector(c_emb.flatten(-2))
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        # adding batch dimension to concept memory
        equation_weights = self.equation_decoder(
            self.equation_memory.weight).unsqueeze(dim=0)

        if self.negative_concepts:
            c_mapped = 2*c_pred - 1
        else:
            c_mapped = c_pred

        y_per_classifier = CF.linear_equation_eval(equation_weights, c_mapped)
        y_pred = CF.selection_eval(prob_per_classifier,
                                   y_per_classifier)

        return y_pred, c_pred

    def step(self, batch, mode="train") -> torch.Tensor:
        loss = super().step(batch, mode)

        # adding l2 regularization to the weights
        w_loss = self.weight_reg * self.equation_memory.weight.norm(p=2)
        loss += w_loss

        prog = mode == "val"
        self.log(f'{mode}_weight_loss', w_loss, on_epoch=True,
                 prog_bar=prog)

        return loss

    def get_local_explanations(
        self,
        x: torch.Tensor,
        multi_label=False,
        **kwargs,
    ) -> List[Dict[str, str]]:
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict['c_int']
        classifier_selector_logits = self.classifier_selector(c_emb.flatten(-2))
        prob_per_classifier = torch.softmax(classifier_selector_logits,
                                            dim=-1)
        equation_weights = self.equation_decoder(
            self.equation_memory.weight).unsqueeze(dim=0)
        c_mapped = 2*c_pred - 1 if self.negative_concepts else c_pred
        y_per_classifier = CF.linear_equation_eval(equation_weights, c_mapped)
        equation_probs = prob_per_classifier * y_per_classifier
        y_pred = equation_probs.sum(dim=-1)

        global_explanations = CF.linear_equation_expl(
            equation_weights, None, {
            1: self.concept_names,
            2: self.task_names,
        })

        local_expl = []
        for i in range(x.shape[0]):
            sample_expl = {}
            for j in range(self.n_tasks):
                # a task is predicted if it is the most likely task or is
                # a multi-label task with probability higher than 0.5 or is
                # a binary task with probability higher than 0.5
                if self._multi_class and not multi_label:
                    predicted_task = j == y_pred[i].argmax()
                else: # multi-label or binary
                    predicted_task = y_pred[i,j] > 0.5

                if predicted_task:
                    task_eqs = global_explanations[0][self.task_names[j]]
                    predicted_eq = task_eqs[f'Equation 0']
                    sample_expl.update(
                        {self.task_names[j]: predicted_eq})
            local_expl.append(sample_expl)
        return local_expl

    def get_global_explanations(self, x=None, **kwargs):
        concept_weights = self.equation_decoder(
            self.equation_memory.weight).unsqueeze(dim=0)

        global_explanations = CF.linear_equation_expl(
            concept_weights, None,
            {
                1: self.concept_names,
                2: self.task_names,
            },
        )

        return global_explanations[0]

class StochasticConceptBottleneckModel(ConceptModel):
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
        self.num_monte_carlo = kwargs['num_monte_carlo']
        self.num_epochs = kwargs['n_epochs']

        self.cov_reg = kwargs['cov_reg']
        self.concept_reg = kwargs['concept_reg']
        self.y_loss_fn = nn.BCELoss()
        
        self.bottleneck = pyc_nn.StochasticConceptBottleneck(latent_dim, concept_names, num_monte_carlo= self.num_monte_carlo, level=kwargs['level'])
        self.y_predictor = nn.Sequential(torch.nn.Linear(len(concept_names), latent_dim),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(latent_dim, len(task_names)),
                                      torch.nn.Sigmoid())
    
    def step(self, batch, mode="train") -> torch.Tensor:

        x, c_true, y_true = batch
        y_pred, c_pred, c_pred_av, emb = self.forward(x, c_true=c_true, current_epoch=self.trainer.current_epoch)

        # Monte Carlo concept loss
        c_true_exp = c_true.unsqueeze(-1).expand_as(c_pred).float()
        bce_loss = F.binary_cross_entropy(c_pred, c_true_exp, reduction="none")
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B, MCMC]
        mcmc_loss = -torch.logsumexp(intermediate_concepts_loss, dim=1)
        concept_loss = torch.mean(mcmc_loss)

        # Task loss
        # BCELoss requires one-hot encoding
        if (self._bce_loss and self._multi_class and
                y_true.squeeze().dim() == 1):
            y_true_loss = F.one_hot(
                y_true.long(),
                self.n_tasks,
            ).squeeze().float()
        elif self._bce_loss and y_true.squeeze().dim() == 1 :
            y_true_loss = y_true.unsqueeze(-1) # add a dimension
        else:
            y_true_loss = y_true
        task_loss = self.y_loss_fn(y_pred, y_true_loss)

        # Precision matrix regularization
        c_triang_cov = self.bottleneck.predict_sigma(emb)
        c_triang_inv = torch.inverse(c_triang_cov)
        prec_matrix = torch.matmul(c_triang_inv.transpose(1, 2), c_triang_inv)
        prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
            dim1=1, dim2=2).abs().sum(dim=1)

        if prec_matrix.size(1) > 1:
            prec_loss = prec_loss / (prec_matrix.size(1) * (prec_matrix.size(1) - 1))
        else:  # Univariate case, can happen when intervening
            prec_loss = prec_loss
        cov_loss = prec_loss.mean()

        # Final loss
        total_loss = self.concept_reg * concept_loss + task_loss + self.cov_reg * cov_loss

        # Metrics
        c_acc, c_avg_auc = 0., 0.
        if c_pred_av is not None:
            c_acc = accuracy_score(c_true.cpu(), (c_pred_av.cpu() > 0.5).float())
            c_avg_auc = roc_auc_score(
                c_true.cpu().view(-1),
                (c_pred_av.cpu().view(-1) > 0.5).float()
            )

        # Extract most likely class in multi-class classification
        if self._multi_class and y_true.squeeze().dim() == 1:
            y_pred = y_pred.argmax(dim=1)
        # Extract prediction from sigmoid output
        elif isinstance(self.y_loss_fn, nn.BCELoss):
            y_pred = (y_pred > 0.5).float()
        y_acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())

        if mode == "train":
            self.log(f'c_avg_auc', c_avg_auc, on_step=True, on_epoch=False, prog_bar=True)
            self.log(f'y_acc', y_acc, on_step=True, on_epoch=False, prog_bar=True)
            self.log(f'loss', total_loss, on_step=True, on_epoch=False, prog_bar=False)
        else:
            prog = mode == "val"
            self.log(f'{mode}_loss', total_loss, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_c_loss', concept_loss, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_y_loss', task_loss, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_c_acc', c_acc, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_c_avg_auc', c_avg_auc, on_epoch=True, prog_bar=prog)
            self.log(f'{mode}_y_acc', y_acc, on_epoch=True, prog_bar=prog)
        return total_loss

    def forward(self, x, c_true=None, **kwargs):
        # generate concept and task predictions
        emb = self.encoder(x)
        c_pred, _ = self.bottleneck(emb)
        c_pred_av = c_pred.mean(-1) 
        # Hard MC concepts
        temp = compute_temperature(kwargs["current_epoch"], self.num_epochs).to(c_pred.device)
        c_pred_relaxed = RelaxedBernoulli(temp, probs=c_pred).rsample()
        c_pred_hard = (c_pred_relaxed > 0.5) * 1
        c_pred_hard = c_pred_hard - c_pred_relaxed.detach() + c_pred_relaxed
        y_pred = 0
        for i in range(self.num_monte_carlo):
            c_i = c_pred_hard[:, :, i]
            y_pred += self.y_predictor(c_i)
        y_pred /= self.num_monte_carlo
        return y_pred, c_pred, c_pred_av, emb

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
        plt.show()

AVAILABLE_MODELS = {
    "ConceptBottleneckModel": ConceptBottleneckModel,
    "ConceptResidualModel": ConceptResidualModel,
    "ConceptEmbeddingModel": ConceptEmbeddingModel,
    "DeepConceptReasoning": DeepConceptReasoning,
    "LinearConceptEmbeddingModel": LinearConceptEmbeddingModel,
    "ConceptMemoryReasoning": ConceptMemoryReasoning,
    "ConceptMemoryReasoning (embedding)": ConceptEmbeddingReasoning,
    "LinearConceptMemoryReasoning": LinearConceptMemoryReasoning,
    "StochasticConceptBottleneckModel": StochasticConceptBottleneckModel,
}

INV_AVAILABLE_MODELS = {v: k for k, v in AVAILABLE_MODELS.items()}

MODELS_ACRONYMS = {
    "ConceptBottleneckModel": "CBM",
    "ConceptResidualModel": "CRM",
    "ConceptEmbeddingModel": "CEM",
    "DeepConceptReasoning": "DCR",
    "LinearConceptEmbeddingModel": "LICEM",
    "ConceptMemoryReasoning": "CMR",
    "ConceptMemoryReasoning (embedding)": "CMR (emb)",
    "LinearConceptMemoryReasoning": "LCMR",
    "StochasticConceptBottleneckModel": "SCBM",
}
