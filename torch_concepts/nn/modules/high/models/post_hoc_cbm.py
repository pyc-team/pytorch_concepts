"""Post-hoc Concept Bottleneck Model (PCBM) (Yuksekgonul et al., ICLR 2023).

A recipe for turning any already trained model into a CBM. The pretrained
backbone is frozen, each concept becomes a concept-activation vector (CAV) in
that backbone's embedding space, the concept scores are the normalised signed
distances to those CAVs, and ONLY a sparse linear head from the
scores to the tasks is ever trained.

The hybrid variant (PCBM-h) additionally fits, sequentially, a residual
linear head from the raw embedding, which recovers the accuracy that an
incomplete concept bank costs the bottleneck while keeping the interpretable
pathway intact (with the obvious caveat that the residual is an uninterpretable
side channel around the bottleneck).

Yuksekgonul et al. (2023) proposed both variants, building the concept bank on
the CAVs of Kim et al. (2018).

References
----------
Yuksekgonul et al. "Post-hoc Concept Bottleneck Models", ICLR 2023.
https://arxiv.org/abs/2205.15480

Kim et al. "Interpretability Beyond Feature Attribution: Quantitative Testing
with Concept Activation Vectors (TCAV)", ICML 2018.
https://proceedings.mlr.press/v80/kim18d
"""
import warnings

from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from torch.distributions import Bernoulli, OneHotCategorical

from .....annotations import Annotations
from .....distributions import Delta
from .....utils import ensure_list

from ...low.encoders.cav import CAVEmbeddingToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...low.predictors.residual import ResidualConceptEmbeddingToConcept
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.variable import ConceptVariable

from .cbm import ConceptBottleneckModel


class PostHocCBM(ConceptBottleneckModel):
    """
    A Post-hoc Concept Bottleneck Model as described in Yuksekgonul et al.
    (2023), covering both the interpretable PCBM and the hybrid PCBM-h.

    This is a model that follows the following structure:
    ``input -> latent (frozen pretrained backbone) -> concept scores -> tasks``,
    where the bottleneck here holds real-valued CAV scores read off a frozen
    backbone. Only the (sparse) task head is trained.
    With ``residual=True`` we get the hybrid PCBM-h, whose task logits
    additionally receive a linear residual from the raw embedding.

    Pre-fitted CAVs are passed in via ``concept_vectors`` /
    ``concept_intercepts`` into the
    :class:`~torch_concepts.nn.CAVEmbeddingToConcept` bank, frozen by default.

    Notice that if they are not frozeb, the CAVs can be supervised with
    BCE-with-logits against concept labels, turning each CAV into a
    logistic-regression probe. This, however, is not the default behaviour as
    it is not what the paper suggests doing.

    To fit a PCBM-h, train the interpretable head first with the residual
    disabled (:meth:`set_residual_use`), then call :meth:`freeze_non_residual_components`
    and train the re-enabled residual.

    This model works as a pure PyTorch module by default, or as a Lightning
    module when ``lightning=True``.

    With ``lightning=True`` the class implements the paper's suggested training
    pipeline (which can be then easily kickstarted with a ``Trainer.fit``).
    Note that fitting the CAVs is NOT part of it (that happens beforehand, on
    the frozen backbone, and is what makes the model post-hoc).
    See example 17.5 for an example of how to train a PCBM using lightning.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types). Every non-task
        concept must be **binary**, as each one is represented by a single
        concept-activation vector. Tasks may be binary or categorical.
    task_names : Union[List[str], str]
        Names of the task variables (a subset of the annotation labels).
    concept_vectors : torch.Tensor, optional
        Pre-fitted CAVs of shape ``(n_concepts, latent_size)``, e.g. from
        per-concept SVM or logistic probes on the backbone embeddings. When
        omitted the CAV table is randomly initialised (and typically left
        trainable).
    concept_intercepts : torch.Tensor, optional
        Pre-fitted CAV intercepts of shape ``(n_concepts,)``. Defaults to zeros.
    residual : bool, default False
        If True, builds the hybrid PCBM-h: task logits are the sum of the
        interpretable head over the scores and a linear residual over the
        backbone embedding.
    freeze_backbone : bool, default True
        Freeze the pretrained backbone's parameters, i.e. the post-hoc setting.
        The backbone is frozen *in place*, so a backbone that arrives still
        requiring gradients is mutated and a warning is issued.
    freeze_concept_vectors : bool, default True
        Freeze the CAV table. Set to False to learn the CAVs in place as
        logistic probes.
    reg_strength : float, default 1e-5
        Strength of the elastic-net regulariser on the interpretable head (see
        :meth:`elastic_net`).
    l1_ratio : float, default 0.99
        Mixing weight of the L1 term in the elastic net.
    inference : BaseInference, optional
        Evaluation inference engine class. Defaults to ``DeterministicInference``.
    inference_kwargs : dict, optional
        Keyword arguments forwarded to the evaluation inference engine.
    train_inference : BaseInference, optional
        Training inference engine class (defaults to ``inference``).
    train_inference_kwargs : dict, optional
        Keyword arguments forwarded to the training inference engine.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default None
        Per-level plate preference (forwarded to :class:`BaseModel`). ``None``
        or ``True`` groups the concept scores into a single plate; ``False``
        uses one score variable per concept (less efficient).
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``, and
        the Lightning training arguments).

    Lightning Training Parameters (only used when lightning=True)
    -------------------------------------------------------------
    interpretable_epochs : int, default 20
        Length of the interpretable stage, which fits the sparse head over the
        concept scores under the elastic net.
    residual_epochs : int, default 20
        Length of the residual stage, which fits PCBM-h's residual with
        everything else frozen. Ignored when ``residual=False``.
    residual_l2_penalty : float, default 1e-3
        Penalty on the residual head during its own stage, as a mean over its
        weights. This is the paper's ``l2_penalty``.
    concept_loss_weight : float, default 0.0
        Weight of a concept supervision term on the CAV margins. Zero, the
        default, is the paper: the concept bank is fitted beforehand and left
        alone. Raise it together with ``freeze_concept_vectors=False`` to learn
        the bank in place as logistic probes.
    lr : float, default 0.01
        Learning rate.
    weight_decay : float, default 0.0
        Weight decay of the optimizer, kept separate from the elastic net and
        the residual penalty above, which are part of the objective itself.
    """

    supported_concept_types = frozenset({"binary", "categorical"})
    param_for_discrete_var = "logits"

    # Per-type distribution policy for the *tasks* (the concept scores are
    # modelled as deterministic Delta variables regardless of type).
    variable_distributions = {
        'binary': Bernoulli,
        'categorical': OneHotCategorical,
    }
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        concept_vectors: Optional[torch.Tensor] = None,
        concept_intercepts: Optional[torch.Tensor] = None,
        residual: bool = False,
        freeze_backbone: bool = True,
        freeze_concept_vectors: bool = True,
        reg_strength: float = 1e-5,
        l1_ratio: float = 0.99,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        # -- Lightning-related args ---------------------------------------
        interpretable_epochs: int = 20,
        residual_epochs: int = 20,
        residual_l2_penalty: float = 1e-3,
        concept_loss_weight: float = 0.0,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        # -----------------------------------------------------------------
        **kwargs,
    ):
        # There is one CAV per concept, so let's first make sure every non-task
        # concept is binary.
        task_list = ensure_list(task_names)
        non_binary = [
            name
            for name, concept_type in zip(annotations.labels, annotations.types)
            if name not in task_list and concept_type != 'binary'
        ]
        if non_binary:
            raise ValueError(
                f"PostHocCBM only supports binary (non-task) concepts as each "
                f"concept is represented by a single concept-activation "
                f"vector. Non-binary concepts found: {non_binary}."
            )

        # These attributes have to be set before super().__init__ as that is
        # what dispatches to the (overridden) ``_build_model`` below.
        self.residual = residual
        self.freeze_concept_vectors = freeze_concept_vectors
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio
        self._given_concept_vectors = concept_vectors
        self._given_concept_intercepts = concept_intercepts

        # Now the lightning-specific args
        self.interpretable_epochs = interpretable_epochs
        self.residual_epochs = residual_epochs
        self.residual_l2_penalty = residual_l2_penalty
        self.concept_loss_weight = concept_loss_weight
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            inference=inference,
            inference_kwargs=inference_kwargs,
            train_inference=train_inference,
            train_inference_kwargs=train_inference_kwargs,
            lightning=lightning,
            **kwargs,
        )

        # And we can't forget about freezing the backbone.
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            # Note for self: we freeze the backbone we were handed, not a
            # copy of it, so whoever else holds a reference to it -- another
            # model sharing the same trunk, say -- sees it frozen too. Say
            # so when we are the ones flipping it, as that is easy to miss.
            if any(p.requires_grad for p in self.backbone.parameters()):
                warnings.warn(
                    "PostHocCBM was given a backbone whose parameters still "
                    "require gradients, and is freezing them in place. Any "
                    "other model sharing this backbone will see it frozen "
                    "too; pass freeze_backbone=False to leave it untouched, "
                    "or hand over a copy.",
                    stacklevel=2,
                )
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def elastic_net(self) -> torch.Tensor:
        """
        The elastic-net regulariser on the interpretable head's weights, i.e.
        the sparsity penalty of Eq. 1 of Yuksekgonul et al.

        As in the original PCBM paper, we normalize by the number of classes and
        concepts so that the penalty is scale independent of the bottleneck's
        size.
        """
        weights = [head.weight for head in self._interpretable_heads]

        l1_norm = sum(w.norm(p=1) for w in weights)
        squared_l2_norm = sum(w.pow(2).sum() for w in weights)
        elastic = (
            self.l1_ratio * l1_norm +
            (1.0 - self.l1_ratio) * squared_l2_norm
        )
        n_concepts = len(self.intermediate_concept_names)
        n_outputs = sum(
            self.concept_annotations.concept(t).cardinality
            for t in self.task_names
        )
        return elastic * self.reg_strength / (n_concepts * n_outputs)

    def set_residual_use(self, enabled: bool) -> None:
        """
        Toggle the residual term of every (hybrid) task head. Disabling it
        recovers the purely interpretable predictions of the concept
        bottleneck, and it is a no-op when ``residual=False``.
        """
        for head in self._task_heads:
            if isinstance(head, ResidualConceptEmbeddingToConcept):
                head.residual_use = enabled

    def freeze_non_residual_components(self) -> None:
        """
        Freeze everything but the residual heads, which is stage 2 of PCBM-h.
        """
        for param in self.parameters():
            param.requires_grad_(False)
        for head in self._task_heads:
            if isinstance(head, ResidualConceptEmbeddingToConcept):
                for param in head.residual.parameters():
                    param.requires_grad_(True)

    def _make_task_head(self, n_concepts: int, out_size: int):
        """
        Build the interpretable (or hybrid) head for one task CPD, returning the
        CPD layer and registering its interpretable ``nn.Linear`` so that the
        elastic-net regulariser can reach it.
        """
        if self.residual:
            head = ResidualConceptEmbeddingToConcept(
                in_concepts=n_concepts,
                in_embeddings=self.latent_size,
                out_concepts=out_size,
            )
            self._interpretable_heads.append(head.c2y)
        else:
            head = LinearConceptToConcept(
                in_concepts=n_concepts,
                out_concepts=out_size,
            )
            self._interpretable_heads.append(head.predictor)
        return head

    # ------------------------------------------------------------------
    # Lightning training (i.e., when lightning=True)
    # ------------------------------------------------------------------
    @property
    def total_epochs(self) -> int:
        """
        How many epochs a full run takes in total.
        """
        if not self.residual:
            return self.interpretable_epochs
        return self.interpretable_epochs + self.residual_epochs

    def training_stage(self, epoch: int) -> str:
        """
        Which of the paper's two stages ``epoch`` belongs to.
        """
        if (not self.residual) or (epoch < self.interpretable_epochs):
            return 'interpretable'
        return 'residual'

    def set_training_stage(self, stage: str) -> None:
        """
        Freeze whatever the given stage should not be training, and switch the
        residual pathway on only once it is being fitted. Public so that a
        manual training loop can follow the same schedule as the Lightning one.
        """
        if stage == 'residual':
            self.freeze_non_residual_components()
            self.set_residual_use(True)
            return

        # Interpretable stage: the head (and the bank, if it was left
        # trainable) learn; the residual is off, and so is anything the
        # post-hoc setting fixed.
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        for head in self._task_heads:
            if isinstance(head, ResidualConceptEmbeddingToConcept):
                for parameter in head.residual.parameters():
                    parameter.requires_grad_(False)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.set_residual_use(False)

    def on_train_epoch_start(self) -> None:
        """
        Move to the stage this epoch belongs to.
        """
        self.set_training_stage(self.training_stage(self.current_epoch))

    def _residual_penalty(self) -> torch.Tensor:
        """
        The paper's ``l2_penalty`` on the residual heads, as a mean over their
        weights so its scale does not track the embedding size.
        """
        weights = [
            head.residual.weight for head in self._task_heads
            if isinstance(head, ResidualConceptEmbeddingToConcept)
        ]
        if not weights:
            return torch.zeros(
                (), device=self.concept_encoders[0].cavs.device,
            )
        return sum(w.pow(2).mean() for w in weights) / len(weights)

    def _concept_penalty(self, out, target) -> torch.Tensor:
        """
        Optional supervision of the CAV scores, treating each margin as a
        logit. Off by default, since the paper fits the bank beforehand.
        """
        names = self.intermediate_concept_names
        return F.binary_cross_entropy_with_logits(
            out.value[names],
            target[names].float(),
        )

    def shared_step(self, batch, step):
        """
        One train/val/test step under the paper's objective, rather than the
        generic composed loss.
        """
        inputs, concepts, transforms = self.unpack_batch(batch)
        batch_size = batch['inputs']['x'].size(0)
        inputs = self.maybe_scale_inputs(inputs, transforms)
        c_loss = self.maybe_scale_concepts(concepts, transforms).get('c', None)

        out = self.forward(
            query=self.default_query(c_loss),
            evidence=self.default_evidence(inputs),
        )
        target = self.prepare_target(c_loss)

        task_loss = self.supervised_loss(out, target, self.task_names)
        stage = self.training_stage(self.current_epoch)
        if stage == 'residual':
            loss = (
                task_loss +
                self.residual_l2_penalty * self._residual_penalty()
            )
        else:
            loss = task_loss + self.elastic_net()
            if self.concept_loss_weight:
                loss = loss + self.concept_loss_weight * self._concept_penalty(
                    out,
                    target,
                )

        for name, value in (('task', task_loss), ('', loss)):
            self.log_loss(
                f"{step}_{name}".rstrip('_'),
                value,
                batch_size=batch_size,
            )

        out = self.unscale_output(out, transforms)
        self.update_and_log_metrics(
            out,
            self.prepare_target(concepts.get('c', None)),
            step,
            batch_size,
        )
        return loss

    def configure_optimizers(self):
        """
        Adam over whatever the current stage left trainable, as in the paper.

        No schedule: the reference fits the interpretable head with a sparse
        solver and the residual with a plain constant-rate Adam, so there is
        nothing to anneal.
        """
        return dict(
            optimizer=torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            ),
        )

    # ------------------------------------------------------------------
    # Model construction (this is very similar to a traditional CBM)
    # ------------------------------------------------------------------
    def _build_model(self) -> BayesianNetwork:
        """
        The key method for building the graph for the Post-hoc CBM. This method
        is called by the parent constructor and returns a
        :class:`BayesianNetwork` object that describes the structure of the
        model. The graph is built to satisfy the following structure:
        ``input -> latent -> concept scores -> tasks``.

        The concepts are modelled as deterministic (``Delta``) real-valued
        scores, the normalised signed distances of the latent to each CAV; every
        task consumes all of those scores through a sparse interpretable head,
        plus, for PCBM-h, a linear residual over the raw latent.
        """
        # First build the input and latent variables and CPDs
        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        concept_names = self.intermediate_concept_names
        n_concepts = len(concept_names)

        # Note for self: this maps each concept to its row in the CAV tables,
        # so a group's encoder can slice out the vectors it owns.
        idx_of = {name: i for i, name in enumerate(concept_names)}

        # With no CAVs supplied the bank starts from random directions: only
        # meaningful once trained (or overwritten with pre-fitted ones), but it
        # keeps the model buildable either way.
        vectors = self._given_concept_vectors
        if vectors is None:
            vectors = torch.randn(n_concepts, self.latent_size)
            vectors /= self.latent_size ** 0.5

        # Note for self: these are plain lists rather than ``nn.ModuleList``s
        # because they hold *references* to layers the PGM already owns
        self.concept_encoders = []
        self._interpretable_heads = []
        self._task_heads = []

        # Let's get the concept-score variables. We build them by hand rather
        # than with ``build_concept_variables`` because a score is a ``Delta``
        # whatever the concept's annotated type says.
        concept_vars = []
        encoders = []
        for kind, cname, members in self._plate_layout(
            concept_names,
            "concepts",
        ):
            concept_vars.append(ConceptVariable(
                names=cname,
                members=(members if kind == "plate" else None),
                distribution=Delta,
                size=1,
            ))

            # latent -> concept scores (the signed distances to the CAVs)
            idx = [idx_of[m] for m in members]
            self.concept_encoders.append(CAVEmbeddingToConcept(
                in_embeddings=self.latent_size,
                out_concepts=len(members),
                cavs=vectors[idx],
                bias=(
                    None if self._given_concept_intercepts is None
                    else self._given_concept_intercepts[idx]
                ),
                trainable=not self.freeze_concept_vectors,
            ))
            encoders.append(ParametricCPD(
                variable=concept_vars[-1],
                parents=[latent_var],
                parametrization=dict(value=self.concept_encoders[-1]),
            ))

        # And the downstream end tasks
        tasks = self.build_concept_variables(
            self.task_names,
            plate_name="tasks",
        )

        # Now time to build the label predictor: concept scores -> tasks, plus,
        # for PCBM-h, the raw latent so the residual head can reach it.
        task_parents = (
            [*concept_vars, latent_var] if self.residual
            else [*concept_vars]
        )
        task_parametrizations = []
        for t in tasks:
            head = self._make_task_head(n_concepts, t.size)
            self._task_heads.append(head)
            task_parametrizations.append({"logits": head})
        predictors = ParametricCPD(
            variable=tasks,
            parents=task_parents,
            parametrization=task_parametrizations,
        )

        # Aaaaand that's it
        return BayesianNetwork(
            variables=[
                input_var,
                latent_var,
                *concept_vars,
                *tasks,
            ],
            factors=[
                input_cpd,
                latent_cpd,
                *encoders,
                *predictors,
            ],
        )
