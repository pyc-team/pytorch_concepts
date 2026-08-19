"""Probabilistic Concept Bottleneck Model (ProbCBM) (Kim et al., ICML 2023).

A Concept Bottleneck Model (CBM) in which every concept is a probabilistic
embedding rather than a scalar logit. Specifically, every concept is represented
as a Gaussian embedding whose mean and diagonal covariance are both predicted
from the input (akin to traditional SAEs). Concepts are then decoded based on
the (Euclidean) distance between that embedding and a learnable pair of
positive/negative anchors representing the concept when is present/absent,
respectively.

Contrary to the standard simple linear concept-to-task probe used in CBMs, in
ProbCBMs tasks are decoded based on the distance between a projection of the
concept embeddings and learnable class anchors embeddings. This allows one to
easily compute the task uncertainty based on the concept embedding uncertainty.

ProbCBMs allow better estimates of concept uncertainty by leveraging the
convariance matrices they produce during inference. Specifically, the embedding
variance  is used as a sampling-free measure of how ambiguous a concept is for
a given input (e.g., think an occluded concept), and it propagates naturally
into the class prediction to yield a measure of task uncertainty.

References
----------
Kim et al. "Probabilistic Concept Bottleneck Models", ICML 2023.
https://arxiv.org/abs/2306.01574

"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....utils import ensure_list

from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.anchor import AnchorPredictor, EmbeddingAnchors
from ...low.sequential import Sequential
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.variable import EmbeddingVariable
from ...outputs import ModelOutput

from .cbm import ConceptBottleneckModel


class _NormalizePerConcept(nn.Module):
    """
    This module normalises each concept's chunk of a flat embedding tensor (based
    on the L2 norm). This is needed because ProbCBM uses unit-length concept
    embedding when determining distances to anchors.
    Because embeddings are proeduced as a flat tensor with dimension
    ``(batch, n_concepts * embedding_size)``, this module is a simple way of
    performing reshape -> normalise -> (re)flatten.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, self.embedding_size)
        return F.normalize(x, p=2, dim=-1).flatten(start_dim=-2)


class ProbCBM(ConceptBottleneckModel):
    """
    A Probabilistic Concept Bottleneck Model as described in Kim et al. (2023).

    This is a model that follows the following structure:
    ``input -> latent -> concept embeddings -> concepts -> tasks``, where
    each concept here is represented via a Gaussian embedding
    ``z_i ~ Normal(loc_i(latent), scale_i(latent))``.

    The concept probability is computed using a contrastive function, whereas
    we predict a concept's state based on whether ``z_i`` is closer to a
    learnable anchor representing concept c_i when it is "present" vs another
    anchor representing the concept when it is "absent". Tasks are then decoded
    from the distances between a projection of the concept embeddings and
    learnable set of class anchors.

    We note that, when using this model, the inference engine selects how
    concept embeddings are produced. This means that the default
    ``DeterministicInference`` propagates the means as they are (the paper's
    sampling-free inference), while
    :class:`~torch_concepts.nn.AncestralSamplingInference` draws
    reparameterised samples, so repeated forward passes give its Monte-Carlo
    estimate.

    By default the task head reads the predicted concept embeddings directly,
    exactly as in the paper. Interventions are then performed the way the paper
    performs them, by replacing a predicted concept embedding with that
    concept's ground-truth anchor.

    With ``lightning=True`` the model also carries the paper's training
    recipe, so a plain ``Trainer.fit`` reproduces the training pipeline
    used for the experiments of the original paper.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types). Every non-task
        concept must be **binary**, as the anchor construction gives each
        concept one positive/negative pair. Tasks may be binary or categorical.
    task_names : Union[List[str], str]
        Names of the task variables (a subset of the annotation labels).
    embedding_size : int, default 16
        Dimensionality of each probabilistic concept embedding.
    class_embedding_size : int, default 128
        Dimensionality of the class-embedding space used by the task heads.
    init_negative_scale : float, default 5.0
        Initial value of the learnable concept distance scale.
    init_class_scale : float, default 5.0
        Initial value of the learnable class distance scale.
    per_item_scale : bool, default False
        Whether every concept (and every task) calibrates its own distance
        scale rather than sharing one per level. The paper shares one, which is
        the default. Nevertheless, we support a more general version.
    use_anchor_interpolation : bool, default False
        If True, the task head consumes the anchor interpolation of the concept
        *activations* rather than the predicted concept embeddings, which keeps
        the task prediction behind the concept bottleneck at the cost of
        departing from the paper. Defaults to False, i.e. the paper's own head.
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
        or ``True`` groups the concepts into a single plate, giving one batched
        Gaussian embedding and one anchor-decoded concept variable; ``False``
        uses one of each per concept (less efficient).
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``, and
        the Lightning training arguments).

    Lightning Training Parameters (only relevant when lightning=True)
    -------------------------------------------------------------
    vib_beta : float, default 5e-5
        Weight of the VIB regulariser in the concept loss.
    intervention_prob : float, default 0.5
        The probability that,while the class predictor trains, a sample's
        concept embeddings are replaced by their ground-truth anchors.
    train_class_mode : str, default 'sequential'
        ``'sequential'`` reproduces the paper, fitting the concept predictor
        for ``concept_epochs`` and then the class predictor for
        ``class_epochs`` with everything else frozen; ``'joint'`` fits both at
        once, in which case ``intervention_prob`` is not applied.
        Must be one of ``'sequential'`` or ``'joint'``.
    concept_epochs : int, default 50
        Length of the concept stage, and of a ``'joint'`` run.
    class_epochs : int, default 20
        Length of the class stage. Ignored when ``train_class_mode='joint'``.
    warm_epochs : int, default 5
        Number of leading epochs during which the backbone stays frozen, so the
        randomly initialised heads settle before it is fine-tuned.
    lr : float, default 1e-3
        Base learning rate, applied to the (pretrained) backbone.
    lr_ratio : float, default 10.0
        Multiplier on ``lr`` for everything the model adds on top of the
        backbone, which starts from scratch and so wants a larger step.
    weight_decay : float, default 0.0
        Weight decay, never applied to the anchors or the distance scales.
    """

    supported_concept_types = frozenset({"binary", "categorical"})
    param_for_discrete_var = "logits"

    # Per-type distribution policy: how this model models each concept type.
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
        embedding_size: int = 16,
        class_embedding_size: int = 128,
        init_negative_scale: float = 5.0,
        init_class_scale: float = 5.0,
        per_item_scale: bool = False,
        use_anchor_interpolation: bool = False,
        # -- Training recipe: only read when lightning=True ---------------
        vib_beta: float = 5e-5,
        intervention_prob: float = 0.5,
        train_class_mode: str = 'sequential',
        concept_epochs: int = 50,
        class_epochs: int = 20,
        warm_epochs: int = 5,
        lr: float = 1e-3,
        lr_ratio: float = 10.0,
        weight_decay: float = 0.0,
        # -----------------------------------------------------------------
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs,
    ):
        # The anchor construction owns one positive/negative embedding pair per
        # concept, so let's first make sure every non-task concept is binary.
        task_list = ensure_list(task_names)
        non_binary = [
            name
            for name, concept_type in zip(annotations.labels, annotations.types)
            if name not in task_list and concept_type != 'binary'
        ]
        if non_binary:
            raise ValueError(
                f"ProbCBM only supports binary (non-task) concepts as each "
                f"concept is represented by a positive/negative anchor pair. "
                f"Non-binary concepts found: {non_binary}."
            )

        # These hyperparameters have to be set before super().__init__ as that
        # is what dispatches to the (overridden) ``_build_model`` below.
        self.embedding_size = embedding_size
        self.class_embedding_size = class_embedding_size
        self.init_negative_scale = init_negative_scale
        self.init_class_scale = init_class_scale
        self.per_item_scale = per_item_scale
        self.use_anchor_interpolation = use_anchor_interpolation

        # Now let's handle all the lightning specific hyperparameters/arguments
        if train_class_mode not in ['sequential', 'joint']:
            raise ValueError(
                f"train_class_mode must be 'sequential' or 'joint', got "
                f"{train_class_mode!r}."
            )
        self.vib_beta = vib_beta
        self.intervention_prob = intervention_prob
        self.train_class_mode = train_class_mode
        self.concept_epochs = concept_epochs
        self.class_epochs = class_epochs
        self.warm_epochs = warm_epochs
        self.lr = lr
        self.lr_ratio = lr_ratio
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

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def embedding_query_names(self) -> List[str]:
        """
        Query names of the probabilistic concept-embedding variables, in
        :attr:`intermediate_concept_names` order. Add these to a forward
        ``query`` to get the embeddings' ``loc``/``scale`` parameters in the
        output, as required by :meth:`vib_kl` and :meth:`concept_uncertainty`.

        Note for self: read this off the groups :meth:`_build_model` recorded
        rather than by scanning the graph for ``Normal`` embeddings. The scan
        is not safe, because it would also pick up any *other* Normal embedding
        a subclass introduces (a variational ``latent``, say), and the helpers
        above would then silently reinterpret that variable's width as more
        concepts instead of failing.
        """
        return [name for name, _ in self._embedding_groups]

    def _gaussian_embedding_parametrization(self, n_concepts: int) -> dict:
        """
        The ``{loc, scale}`` heads of a Gaussian concept-embedding CPD.

        The ``loc`` head is a linear map onto the unit hypersphere, which is
        where the anchor distances live, while ``scale`` is a linear map through
        a ``Softplus`` that keeps the standard deviation strictly positive.
        Note that the reference implementation exponentiates a clamped
        log-variance instead; ``Softplus`` is the standard, better-behaved
        equivalent and needs no clamping.
        """
        out_size = n_concepts * self.embedding_size
        return dict(
            loc=Sequential(
                LinearEmbeddingToConcept(
                    in_embeddings=self.latent_size,
                    out_concepts=out_size,
                ),
                _NormalizePerConcept(self.embedding_size),
            ),
            scale=Sequential(
                LinearEmbeddingToConcept(
                    in_embeddings=self.latent_size,
                    out_concepts=out_size,
                ),
                nn.Softplus(),
            ),
        )

    def _embedding_moments(self, out: ModelOutput):
        """
        The queried embeddings' ``(loc, scale)``, each of shape
        ``(batch, n_concepts, embedding_size)``.

        Embeddings the caller left out of the query are skipped, as indexing an
        unqueried variable raises. If none of them were queried then there is
        nothing to compute at all, which is a caller error we report.
        """
        locs, scales = [], []
        for name in self.embedding_query_names:
            try:
                params = out.params[name]
                loc, scale = params['loc'], params['scale']
            except KeyError:
                continue
            # The CPD emits one flat row per sample; split it back per concept.
            locs.append(loc.reshape(loc.shape[0], -1, self.embedding_size))
            scales.append(
                scale.reshape(scale.shape[0], -1, self.embedding_size)
            )
        if not locs:
            raise ValueError(
                "No probabilistic embedding parameters in the output: add "
                "`model.embedding_query_names` to the forward query."
            )
        return torch.cat(locs, dim=1), torch.cat(scales, dim=1)

    def anchor_embeddings(
        self,
        concepts: Union[torch.Tensor, Dict[str, torch.Tensor]],
        out: Optional[ModelOutput] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build the concept embeddings implied by a set of concept values, keyed
        by embedding variable so the result can be passed straight back to the
        model.

        This is how ProbCBM intervenes: the paper replaces a predicted concept
        embedding ``z_i`` by that concept's ground-truth anchor, ``z_i^+`` when
        the concept is present and ``z_i^-`` when it is absent. Since the task
        head reads the embeddings, an intervention has to be expressed on them
        rather than on the concept variables, and the returned dict does
        exactly that. Pass it as ``evidence`` to intervene at test time, or
        merge it into a training ``query`` to have the engine teacher-force it
        at its ``p_int`` rate, which is the paper's ``p_replace``.

        Parameters
        ----------
        concepts : torch.Tensor or Dict[str, torch.Tensor]
            Either a ``(batch, n_concepts)`` tensor of values whose columns are
            ordered as :attr:`intermediate_concept_names`, or a dict mapping a
            subset of concept names to their ``(batch, 1)`` values. Values are
            usually hard 0/1 labels, but any value in ``[0, 1]`` interpolates
            between the two anchors.
        out : ModelOutput, optional
            A forward output whose query included
            :attr:`embedding_query_names`. Required when ``concepts`` names
            only a subset of the concepts: the ones left out keep the
            embedding the model predicted for them (their ``loc``).

        Returns
        -------
        Dict[str, torch.Tensor]
            Map from embedding variable name to its ``(batch, m * D)`` value.
        """
        concept_names = self.intermediate_concept_names
        if not isinstance(concepts, dict):
            concepts = {
                name: concepts[:, i:i + 1]
                for i, name in enumerate(concept_names)
            }
        missing = [name for name in concept_names if name not in concepts]
        if missing and out is None:
            raise ValueError(
                f"anchor_embeddings was given no value for {missing}, so it "
                f"needs `out` (a forward output including "
                f"`model.embedding_query_names`) to keep those concepts' "
                f"predicted embeddings."
            )

        evidence = {}
        for (name, members), predictor in zip(
            self._embedding_groups,
            self.concept_predictors,
        ):
            present = [m for m in members if m in concepts]
            if not present:
                continue
            values = torch.cat([concepts[m].reshape(-1, 1) for m in present], 1)
            # ``items`` are positions in this group's own table, which is the
            # one its concepts were decoded against.
            anchored = predictor.anchors.interpolate(
                values,
                items=[members.index(m) for m in present],
            )
            if missing:
                predicted = out.params[name]['loc']
                block = predicted.reshape(
                    predicted.shape[0],
                    len(members),
                    self.embedding_size,
                ).clone()
            else:
                block = anchored.new_empty(
                    anchored.shape[0],
                    len(members),
                    self.embedding_size,
                )
            for position, member in enumerate(members):
                if member not in concepts:
                    continue
                block[:, position] = anchored[:, present.index(member)]
            # Reshape to the variable's own event shape, which is what teacher
            # forcing compares against (evidence would accept either).
            evidence[name] = block.reshape(
                block.shape[0],
                *self.pgm.variables[name].shape,
            )
        return evidence

    def vib_kl(self, out: ModelOutput) -> torch.Tensor:
        """
        The variational information bottleneck regulariser of Kim et al.
        that is ``KL(Normal(loc, scale) || Normal(0, I))`` summed over the
        embedding dimensions and averaged over the batch and the concepts.

        This is what stops variance collapse.

        Parameters
        ----------
        out : ModelOutput
            A forward output whose query included :attr:`embedding_query_names`.

        Returns
        -------
        torch.Tensor
            Scalar KL regulariser.
        """
        loc, scale = self._embedding_moments(out)
        kl = 0.5 * (loc.pow(2) + scale.pow(2) - 1.0 - 2.0 * scale.log())
        return kl.sum(-1).mean()

    def concept_uncertainty(self, out: ModelOutput) -> torch.Tensor:
        """
        The per-concept embedding uncertainty of Kim et al. (Sec. 4.4), that is
        the determinant of the diagonal covariance matrix, or equivalently the
        geometric mean of the embedding variances.

        Parameters
        ----------
        out : ModelOutput
            A forward output whose query included :attr:`embedding_query_names`.

        Returns
        -------
        torch.Tensor
            Uncertainties of shape ``(batch, n_concepts)``, with columns ordered
            as in :attr:`intermediate_concept_names`.
        """
        _, scale = self._embedding_moments(out)
        return (2.0 * scale.log()).mean(-1).exp()

    # ------------------------------------------------------------------
    # Lightning training (i.e., when lightning=True)
    # ------------------------------------------------------------------
    @property
    def total_epochs(self) -> int:
        """
        How many epochs a full run takes. Only relevant when lightning = True.
        """
        if self.train_class_mode == 'joint':
            return self.concept_epochs
        return self.concept_epochs + self.class_epochs

    def training_stage(self, epoch: int) -> str:
        """
        Which of the paper's stages ``epoch`` belongs to. Only relevant when
        lightning = True.
        """
        if self.train_class_mode == 'joint':
            return 'joint'
        return 'concept' if epoch < self.concept_epochs else 'class'

    def class_parameters(self):
        """
        The parameters of the class predictor. Only relevant when
        lightning = True.
        """
        modules = [self.class_projection]
        modules += [
            factor.parametrization['logits']
            for name, factor in self.pgm.factors.items()
            if name in self._task_variable_names
        ]
        # The heads hold a reference to the shared projection, so the same
        # tensor shows up twice; keep one of each, in a stable order.
        unique = {}
        for module in modules:
            for parameter in module.parameters():
                unique.setdefault(id(parameter), parameter)
        return list(unique.values())

    def set_training_stage(self, stage: str, warm: bool = False) -> None:
        """
        Freeze whatever the current stage should not be training. Only relevant
        when lightning = True.
        """
        class_ids = {id(p) for p in self.class_parameters()}
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        for parameter in self.parameters():
            is_class = id(parameter) in class_ids
            trainable = is_class if stage == 'class' else (
                True if stage == 'joint' else not is_class
            )
            if warm and (id(parameter) in backbone_ids):
                trainable = False
            parameter.requires_grad_(trainable)

    def on_train_epoch_start(self) -> None:
        """
        Move to the right training stage based on the current epoch. Only
        relevant when lightning = True.
        """
        stage = self.training_stage(self.current_epoch)
        self.set_training_stage(
            stage,
            warm=(self.current_epoch < self.warm_epochs),
        )
        self.train_inference.p_int = (
            self.intervention_prob if (stage == 'class')
            else 0.0
        )

    def default_query(self, c):
        """
        Query every concept and task, plus the embeddings.
        """
        query = super().default_query(c)
        labels = self.prepare_target(c)[self.intermediate_concept_names]
        return {**query, **self.anchor_embeddings(labels)}

    def shared_step(self, batch, step):
        """
        One step under the training objective. Only relevant when
        lightning = True.
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

        concept_loss = (
            self.supervised_loss(out, target, self.intermediate_concept_names) +
            self.vib_beta * self.vib_kl(out)
        )
        class_loss = self.supervised_loss(out, target, self.task_names)

        stage = self.training_stage(self.current_epoch)
        loss = 0.0
        if stage != 'class':
            loss = loss + concept_loss
        if stage != 'concept':
            loss = loss + class_loss

        for name, value in (
            ('concept', concept_loss),
            ('class', class_loss),
            ('', loss),
        ):
            self.log_loss(
                f"{step}_{name}".rstrip('_'),
                value,
                batch_size=batch_size
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
        Adam with the original paper's two learning rates and a cosine schedule.
        Only relevant when lightning = True.

        Everything the model adds on top of the backbone is trained from
        scratch and so gets ``lr * lr_ratio``, while the pretrained backbone
        gets the base ``lr``. The anchors and the distance scales are excluded
        from weight decay, matching the reference's ``no_weight_decay``.
        """
        decay_free = {id(p) for p in self._no_decay_parameters()}
        backbone = {id(p) for p in self.backbone.parameters()}

        groups = {}
        for parameter in self.parameters():
            is_backbone = id(parameter) in backbone
            key = (is_backbone, id(parameter) in decay_free)
            groups.setdefault(key, []).append(parameter)

        optimizer = torch.optim.Adam([
            dict(
                params=params,
                lr=(self.lr if is_backbone else self.lr * self.lr_ratio),
                weight_decay=(0.0 if no_decay else self.weight_decay),
            )
            for (is_backbone, no_decay), params in groups.items()
        ])
        return dict(
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.total_epochs,
            ),
        )

    def _no_decay_parameters(self):
        """
        The anchors and distance scales, which the reference keeps out of
        weight decay so that decay cannot quietly shrink them.
        Only relevant when lightning = True.
        """
        return [
            p
            for module in self.modules()
            if isinstance(module, EmbeddingAnchors)
            for p in module.parameters()
        ]

    # ------------------------------------------------------------------
    # Model assembly (written once for both layouts)
    # ------------------------------------------------------------------
    def _build_model(self) -> BayesianNetwork:
        """
        The key method for building the graph for the ProbCBM.
        The graph is built to satisfy the following structure:
        ``input -> latent -> embeddings -> concepts -> tasks``.

        Each concept group owns one ``Normal`` embedding variable and is decoded
        from its anchor distances; every task consumes all of the concepts'
        activations and is decoded from class-anchor distances through the
        shared projection trunk.
        """
        # First build the input and latent variables and CPDs
        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        concept_names = self.intermediate_concept_names

        # Let's get the concept variables and, for each of them, the Gaussian
        # embedding it is decoded from.
        concepts = self.build_concept_variables(
            concept_names,
            plate_name="concepts",
        )
        embeddings = []
        emb_cpds = []
        encoders = []
        concept_predictors = []
        # Which concepts sit in which embedding variable, and in what order:
        # ``anchor_embeddings`` needs it to address one concept's block.
        self._embedding_groups = []
        for concept in concepts:
            n_members = len(concept.members)
            self._embedding_groups.append(
                (f"{concept.name}__emb", list(concept.members))
            )

            embeddings.append(EmbeddingVariable(
                names=f"{concept.name}__emb",
                distribution=Normal,
                # A plate stacks its members' embeddings into one event.
                shape=(
                    (n_members, self.embedding_size) if concept.is_plate
                    else (self.embedding_size,)
                ),
            ))

            # latent -> Gaussian embedding
            emb_cpds.append(ParametricCPD(
                variable=embeddings[-1],
                parents=[latent_var],
                parametrization=self._gaussian_embedding_parametrization(
                    n_members
                ),
            ))

            # embedding -> concept (via the anchor distances). Each group's
            # predictor owns the anchors of its own concepts, which is also
            # what an intervention on them substitutes.
            concept_predictors.append(AnchorPredictor(
                n_items=n_members,
                embedding_size=self.embedding_size,
                init_scale=self.init_negative_scale,
                per_item_scale=self.per_item_scale,
                distance_reduction="sum",
                normalize=True,
                eps=1e-6,
            ))
            encoders.append(ParametricCPD(
                variable=concept,
                parents=[embeddings[-1]],
                parametrization={"logits": concept_predictors[-1]},
            ))

        # And the downstream end tasks
        tasks = self.build_concept_variables(
            self.task_names,
            plate_name="tasks",
        )

        # Now time to build the label predictor, via the distances to a set of
        # learnable class anchors. When use_anchor_interpolation is True,
        # the head is handed the concept predictors' own anchor tables, so that
        # it interpolates exactly the anchors the concepts were decoded against.
        self.class_projection = nn.Linear(
            len(concept_names) * self.embedding_size,
            self.class_embedding_size,
        )
        predictors = ParametricCPD(
            variable=tasks,
            parents=concepts if self.use_anchor_interpolation else embeddings,
            parametrization=[
                {
                    "logits": AnchorPredictor(
                        n_items=len(t.members),
                        embedding_size=self.class_embedding_size,
                        cardinality=t.member_size,
                        projection=self.class_projection,
                        source_anchors=[
                            predictor.anchors
                            for predictor in concept_predictors
                        ],
                        init_distribution=Normal(0.0, 1.0),
                        init_scale=self.init_class_scale,
                        per_item_scale=self.per_item_scale,
                        distance_reduction="mean",
                        normalize=False,
                        eps=1e-10,
                    )
                }
                for t in tasks
            ],
        )

        self.concept_predictors = nn.ModuleList(concept_predictors)
        # And the task variables, so the Lightning stages can tell the class
        # predictor's parameters apart from the rest.
        self._task_variable_names = [t.name for t in tasks]

        # Aaaaand that's it
        return BayesianNetwork(
            variables=[
                input_var,
                latent_var,
                *embeddings,
                *concepts,
                *tasks,
            ],
            factors=[
                input_cpd,
                latent_cpd,
                *emb_cpds,
                *encoders,
                *predictors,
            ],
        )
