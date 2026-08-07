"""Post-hoc Concept Bottleneck Model (PCBM).

This is an implementation of the Post-hoc Concept Bottleneck Model (PCBM) as
described in the paper: Yuksekgonul, Mert, et al. "Post-hoc Concept Bottleneck
Models." ICLR (2023).

Link: https://arxiv.org/abs/2205.15480

A PCBM turns any *pretrained* model into a concept bottleneck model without
retraining it: the pretrained backbone is frozen, each concept is a
concept-activation vector (CAV) in the backbone's embedding space (typically
fitted post-hoc with per-concept SVM or logistic-regression probes), the
concept scores are the normalised signed distances to the concept hyperplanes,
and only a sparse (elastic-net regularised) linear head from the scores to the
tasks is trained. The hybrid variant (PCBM-h) additionally fits — sequentially
— a residual linear head from the raw embedding to recover the accuracy the
bottleneck loses, while keeping the interpretable pathway intact.
"""
from typing import List, Optional, Union

import torch

from torch.distributions import Bernoulli, OneHotCategorical

from .....annotations import Annotations
from .....distributions import Delta
from .....utils import ensure_list
from ...low.encoders.cav import CAVEmbeddingToConcept, ConceptActivationVectors
from ...low.predictors.linear import LinearConceptToConcept
from ...low.predictors.residual import ResidualConceptEmbeddingToConcept
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.models.bayesian_network import BayesianNetwork
from ...mid.models.cpd import ParametricCPD
from ...mid.models.variable import ConceptVariable, _DEFAULT_DIST_KWARGS
from .cbm import ConceptBottleneckModel


class PostHocCBM(ConceptBottleneckModel):
    """Post-hoc Concept Bottleneck Model (PCBM).

    ``latent (frozen pretrained backbone) → concept scores → tasks``
    bottleneck (Yuksekgonul et al., ICLR 2023). Concepts are *deterministic
    real-valued scores*: the normalised signed distance of the backbone
    embedding to each concept's activation vector (CAV),
    ``s_i = (<f(x), v_i> + b_i) / ||v_i||``. The task head is a linear layer
    over the scores, regularised with an elastic net (:meth:`elastic_net`) to
    stay sparse and interpretable. With ``residual=True`` (PCBM-h) the task
    logits additionally receive a linear residual from the raw embedding.

    PyC mapping and semantics
    -------------------------
    * The pretrained model is passed as the ``backbone`` (frozen by default);
      the concept-score variables are ``Delta`` :class:`ConceptVariable` nodes
      whose value is the raw score — never squashed through a sigmoid, exactly
      as in the reference implementation.
    * Pre-fitted CAVs are passed via ``concept_vectors`` /
      ``concept_intercepts`` and frozen by default. When they are left
      trainable (``freeze_concept_vectors=False``) and the scores are trained
      with a BCE-with-logits loss against concept labels, each CAV is exactly
      a logistic-regression probe, so the concept bank can also be learned
      in-place.
    * Interventions are graph-native: supply concept evidence (e.g. ``+1`` /
      ``-1`` scores, or the score of your choice) and the task head consumes
      the clamped values. Teacher forcing via the engine's ``p_int`` works the
      same way.
    * For PCBM-h, train the interpretable head first with the residual
      disabled (:meth:`set_residual_use`), then call
      :meth:`freeze_non_residual_components` and train the (re-enabled)
      residual — the paper's sequential fitting recipe.

    Works as a pure PyTorch module by default, or as a Lightning module when
    ``lightning=True``.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types). Every non-task
        concept must be **binary** (one CAV per concept); tasks may be binary
        or categorical.
    task_names : Union[List[str], str]
        Names of the task variables (a subset of the annotation labels).
    concept_vectors : torch.Tensor, optional
        Pre-fitted CAVs of shape ``(n_concepts, latent_size)`` (e.g. from
        per-concept logistic probes on the backbone embeddings). When omitted,
        the CAV table is randomly initialised (and typically left trainable).
    concept_intercepts : torch.Tensor, optional
        Pre-fitted CAV intercepts of shape ``(n_concepts,)``. Defaults to
        zeros.
    residual : bool, default False
        If True, builds the hybrid PCBM-h: task logits are the sum of the
        interpretable head over the scores and a linear residual over the
        backbone embedding.
    freeze_backbone : bool, default True
        Freeze the pretrained backbone's parameters (the post-hoc setting).
    freeze_concept_vectors : bool, default True
        Freeze the CAV table. Set to False to learn the CAVs in-place as
        logistic probes.
    reg_strength : float, default 1e-5
        Strength of the elastic-net regulariser on the interpretable head
        (see :meth:`elastic_net`).
    l1_ratio : float, default 0.99
        Mixing weight of the L1 term in the elastic net.
    inference : BaseInference, optional
        Evaluation inference engine class. Defaults to
        ``DeterministicInference``.
    inference_kwargs : dict, optional
        Keyword arguments forwarded to the evaluation inference engine.
    train_inference : BaseInference, optional
        Training inference engine class (defaults to ``inference``).
    train_inference_kwargs : dict, optional
        Keyword arguments forwarded to the training inference engine.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default None
        Controls which building path is used.  ``None`` (default) auto-detects:
        uses plates only when **all** graph levels are plate-compatible (see
        :meth:`~torch_concepts.nn.modules.high.base.graph.DirectedGraphModel.plate_compatible_levels`),
        otherwise falls back to individual variables.  Pass ``True`` to force
        plates or ``False`` to force individual variables.
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``,
        and the Lightning training arguments).

    References
    ----------
    Yuksekgonul et al. "Post-hoc Concept Bottleneck Models", ICLR 2023.
    https://arxiv.org/abs/2205.15480
    """

    supported_concept_types = frozenset({"binary", "categorical"})
    param_for_discrete_var = "logits"

    # Per-type distribution policy for the *tasks* (the concept scores are
    # modelled as deterministic Delta variables regardless of type).
    variable_distributions = {
        'binary': Bernoulli,
        'categorical': OneHotCategorical,
    }
    variable_dist_kwargs = dict(_DEFAULT_DIST_KWARGS)

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
        **kwargs,
    ):
        # One CAV per concept, so every non-task concept must be binary.
        task_list = ensure_list(task_names)
        non_binary = [
            name
            for name, concept_type in zip(annotations.labels, annotations.types)
            if name not in task_list and concept_type != 'binary'
        ]
        if non_binary:
            raise ValueError(
                f"PostHocCBM only supports binary (non-task) concepts — each "
                f"concept is represented by a single concept-activation "
                f"vector. Non-binary concepts found: {non_binary}."
            )

        # Attributes needed by the (overridden) building paths, which run
        # inside super().__init__ below.
        self.residual = residual
        self.freeze_concept_vectors = freeze_concept_vectors
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio
        self._given_concept_vectors = concept_vectors
        self._given_concept_intercepts = concept_intercepts

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

        # The post-hoc setting: the pretrained backbone is not retrained.
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _setup_cav_components(self) -> None:
        """Create the shared CAV table used by every concept-score CPD."""
        self.cavs = ConceptActivationVectors(
            n_concepts=len(self.intermediate_concept_names),
            embedding_size=self.latent_size,
            vectors=self._given_concept_vectors,
            intercepts=self._given_concept_intercepts,
            trainable=not self.freeze_concept_vectors,
        )

    def _make_task_head(self, n_concepts: int, out_size: int):
        """Interpretable (or hybrid) head for one task CPD.

        Returns the CPD layer and registers its interpretable ``nn.Linear``
        for the elastic-net regulariser.
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

    def elastic_net(self) -> torch.Tensor:
        """Elastic-net regulariser on the interpretable head's weights.

        ``reg_strength * (l1_ratio * ||W||_1 + (1 - l1_ratio) * ||W||_2) /
        (n_concepts * n_task_outputs)`` — the sparsity penalty of the
        reference implementation, which keeps the concept-to-task mapping
        interpretable. Add it to the task loss during training.
        """
        weights = [head.weight for head in self._interpretable_heads]
        l1_norm = sum(w.norm(p=1) for w in weights)
        l2_norm = torch.sqrt(sum(w.pow(2).sum() for w in weights))
        elastic = self.l1_ratio * l1_norm + (1.0 - self.l1_ratio) * l2_norm
        n_concepts = len(self.intermediate_concept_names)
        n_outputs = sum(
            self.concept_annotations.concept(t).cardinality
            for t in self.task_names
        )
        return elastic * self.reg_strength / (n_concepts * n_outputs)

    def set_residual_use(self, enabled: bool) -> None:
        """Toggle the residual term of every (hybrid) task head.

        Disabling it recovers the purely interpretable predictions of the
        concept bottleneck; a no-op when ``residual=False``.
        """
        for head in self._task_heads:
            if isinstance(head, ResidualConceptEmbeddingToConcept):
                head.residual_use = enabled

    def freeze_non_residual_components(self) -> None:
        """Freeze everything but the residual heads (PCBM-h stage 2).

        The paper fits the hybrid model sequentially: first the interpretable
        head (with the residual disabled), then — with the backbone, the CAVs
        and the interpretable head frozen — the residual. Call this before
        the residual stage; only the residual heads keep ``requires_grad``.
        """
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.cavs.vectors.requires_grad_(False)
        self.cavs.intercepts.requires_grad_(False)
        for head in self._interpretable_heads:
            head.weight.requires_grad_(False)
            if head.bias is not None:
                head.bias.requires_grad_(False)
        for head in self._task_heads:
            if isinstance(head, ResidualConceptEmbeddingToConcept):
                for param in head.residual.parameters():
                    param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Building paths
    # ------------------------------------------------------------------
    def _build_plate_model(self) -> BayesianNetwork:
        """Build using one plate variable per bipartite
        level (concept scores, tasks).
        """
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        concept_names = self.intermediate_concept_names
        n_concepts = len(concept_names)
        task0 = axis.concept(self.task_names[0])

        self._setup_cav_components()
        self._interpretable_heads = []
        self._task_heads = []

        # Concept scores are deterministic (Delta) real values: the
        # normalised signed distances to the concept hyperplanes.
        concepts = ConceptVariable(
            names="concepts",
            members=concept_names,
            distribution=Delta,
            size=1,
        )
        tasks = ConceptVariable(
            names="tasks",
            members=self.task_names,
            distribution=self.distribution_of(task0.name),
            dist_kwargs=self.dist_kwargs_of(task0.name),
            size=task0.cardinality,
        )

        encoders = ParametricCPD(
            variable=concepts,
            parents=[latent_var],
            parametrization={
                "value": CAVEmbeddingToConcept(self.cavs),
            },
        )
        task_head = self._make_task_head(n_concepts, tasks.size)
        self._task_heads.append(task_head)
        predictors = ParametricCPD(
            variable=tasks,
            parents=(
                [concepts, latent_var] if self.residual else [concepts]
            ),
            parametrization={"logits": task_head},
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, concepts, tasks],
            factors=[input_cpd, latent_cpd, encoders, predictors],
        )

    def _build_individual_model(self) -> BayesianNetwork:
        """Build with one score variable per concept and one variable per
        task."""
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        concept_names = self.intermediate_concept_names
        n_concepts = len(concept_names)
        task_concepts = [axis.concept(name) for name in self.task_names]

        self._setup_cav_components()
        self._interpretable_heads = []
        self._task_heads = []

        concepts = ConceptVariable(
            names=concept_names,
            distribution=Delta,
            size=1,
        )
        tasks = ConceptVariable(
            names=self.task_names,
            distribution=[self.distribution_of(t.name) for t in task_concepts],
            dist_kwargs=[self.dist_kwargs_of(t.name) for t in task_concepts],
            size=[t.cardinality for t in task_concepts],
        )

        encoders = ParametricCPD(
            variable=concepts,
            parents=[latent_var],
            parametrization=[{
                "value": CAVEmbeddingToConcept(self.cavs, concept_idx=[i]),
            } for i in range(n_concepts)],
        )
        task_parents = (
            [*concepts, latent_var] if self.residual else [*concepts]
        )
        task_parametrizations = []
        for t in task_concepts:
            head = self._make_task_head(n_concepts, t.cardinality)
            self._task_heads.append(head)
            task_parametrizations.append({"logits": head})
        predictors = ParametricCPD(
            variable=tasks,
            parents=task_parents,
            parametrization=task_parametrizations,
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *concepts, *tasks],
            factors=[input_cpd, latent_cpd, *encoders, *predictors],
        )
