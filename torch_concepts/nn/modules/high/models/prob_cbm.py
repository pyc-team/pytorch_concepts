"""Probabilistic Concept Bottleneck Model (ProbCBM).

This is an implementation of the Probabilistic Concept Bottleneck Model
(ProbCBM) as described in the paper: Kim, Eunji, et al. "Probabilistic concept
bottleneck models." ICML (2023).

Link: https://arxiv.org/abs/2306.01574

A ProbCBM replaces the CBM's deterministic concept predictions with
*probabilistic concept embeddings*: each concept is represented by a Gaussian
distribution in an embedding space whose mean and (diagonal) standard
deviation are predicted from the input. Concepts are decoded from the
distances between the (sampled or mean) embedding and a learnable pair of
positive/negative concept anchors, and tasks are predicted from distances to
learnable class anchors in a projected class-embedding space. The variance of
the embeddings quantifies concept uncertainty, which propagates to class
uncertainty.
"""
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Bernoulli, OneHotCategorical, Normal

from .....annotations import Annotations
from .....utils import ensure_list
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.anchor import (
    AnchorConceptToConcept,
    AnchorEmbeddingToConcept,
    ConceptAnchorProjection,
    ConceptAnchors,
)
from ...low.sequential import Sequential
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.models.bayesian_network import BayesianNetwork
from ...mid.models.cpd import ParametricCPD
from ...mid.models.variable import ConceptVariable, EmbeddingVariable, \
    _DEFAULT_DIST_KWARGS
from ...outputs import ModelOutput
from .cbm import ConceptBottleneckModel


class _NormalizePerConcept(nn.Module):
    """L2-normalise each concept's chunk of a flat embedding tensor.

    ProbCBM places the concept embedding *means* on the unit hypersphere (the
    anchor distances are only meaningful there). The location head emits a
    flat ``(batch, n_concepts * embedding_size)`` tensor — the layout a
    ``Normal`` CPD's ``loc`` must have — so this reshapes it per concept,
    normalises on the embedding axis, and flattens it back.

    Private to this model: it is the only ProbCBM-specific piece of the
    location head, so it is not part of the public low-level layer API.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, self.embedding_size)
        return F.normalize(x, p=2, dim=-1).flatten(start_dim=-2)


class ProbCBM(ConceptBottleneckModel):
    """Probabilistic Concept Bottleneck Model (ProbCBM).

    ``latent → probabilistic concept embeddings → concepts → tasks``
    bottleneck (Kim et al., ICML 2023). Each (binary) concept ``i`` owns a
    Gaussian embedding ``z_i ~ Normal(loc_i(latent), scale_i(latent))`` whose
    mean is L2-normalised; the concept probability is a two-way softmax over
    the scaled distances between ``z_i`` and the concept's learnable
    positive/negative anchors; and tasks are predicted from the distances
    between a projection of the (anchor-interpolated) concept embeddings and
    learnable class anchors.

    PyC mapping and semantics
    -------------------------
    * The probabilistic embeddings enter the PGM as ``Normal``
      :class:`EmbeddingVariable` nodes. With the default
      ``DeterministicInference`` the *mean* embedding is propagated (the
      paper's sampling-free evaluation mode); with
      :class:`~torch_concepts.nn.AncestralSamplingInference` the embeddings
      are reparameterised samples, so repeated forward passes give the paper's
      Monte-Carlo estimates and class uncertainty derives from concept
      (embedding) uncertainty.
    * The task head consumes the concept *activations* through an anchor
      interpolation ``v_i * anchor_i^+ + (1 - v_i) * anchor_i^-``. Under
      interventions or teacher forcing (hard ``v_i``) this is exactly the
      paper's replacement of predicted embeddings with ground-truth anchor
      embeddings; during free prediction the activation is the model's own
      concept probability (the one deviation from the reference, which feeds
      the raw predicted embedding to the class head — routing through the
      concept activations keeps the bottleneck leak-free and makes
      interventions graph-native).
    * The paper's VIB regulariser on the embeddings is available as
      :meth:`vib_kl`, and per-concept embedding uncertainty as
      :meth:`concept_uncertainty`; both consume a forward output whose query
      includes :attr:`embedding_query_names`.

    Works as a pure PyTorch module by default, or as a Lightning module when
    ``lightning=True``.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types). Every non-task
        concept must be **binary** (the anchor construction is a
        positive/negative pair per concept); tasks may be binary or
        categorical.
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
    Kim et al. "Probabilistic Concept Bottleneck Models", ICML 2023.
    https://arxiv.org/abs/2306.01574
    """

    supported_concept_types = frozenset({"binary", "categorical"})
    param_for_discrete_var = "logits"

    # Per-type distribution policy: how this model models each concept type.
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
        embedding_size: int = 16,
        class_embedding_size: int = 128,
        init_negative_scale: float = 5.0,
        init_class_scale: float = 5.0,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs,
    ):
        # The anchor construction owns one positive/negative embedding pair
        # per concept, so every non-task concept must be binary.
        task_list = ensure_list(task_names)
        non_binary = [
            name
            for name, concept_type in zip(annotations.labels, annotations.types)
            if name not in task_list and concept_type != 'binary'
        ]
        if non_binary:
            raise ValueError(
                f"ProbCBM only supports binary (non-task) concepts — each "
                f"concept is represented by a positive/negative anchor pair. "
                f"Non-binary concepts found: {non_binary}."
            )

        # Hyperparameters must be set before super().__init__, which
        # dispatches to the (overridden) building paths below.
        self.embedding_size = embedding_size
        self.class_embedding_size = class_embedding_size
        self.init_negative_scale = init_negative_scale
        self.init_class_scale = init_class_scale

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
    def _setup_anchor_components(self) -> None:
        """Create the anchor table and the class-head trunk (both shared).

        The concept predictors and the task heads reference the *same*
        :class:`ConceptAnchors` instance, so the embeddings the concepts are
        decoded against are exactly the ones interventions substitute.
        """
        n_concepts = len(self.intermediate_concept_names)
        self.concept_anchors = ConceptAnchors(
            n_concepts=n_concepts,
            embedding_size=self.embedding_size,
            init_negative_scale=self.init_negative_scale,
        )
        self.class_projection = ConceptAnchorProjection(
            anchors=self.concept_anchors,
            class_embedding_size=self.class_embedding_size,
            init_scale=self.init_class_scale,
        )

    def _gaussian_embedding_parametrization(self, n_concepts: int) -> dict:
        """``{loc, scale}`` heads for a Gaussian concept-embedding CPD.

        Both are composed from the standard
        :class:`~torch_concepts.nn.LinearEmbeddingToConcept` encoder:

        * ``loc`` — a linear map followed by the per-concept L2 normalisation
          onto the unit hypersphere (:class:`_NormalizePerConcept`);
        * ``scale`` — a linear map followed by ``Softplus``, which keeps the
          standard deviation in the strictly positive domain a ``Normal``
          requires. (The reference implementation instead exponentiates a
          clamped log-variance; ``Softplus`` is the standard, better-behaved
          equivalent and needs no clamp.)
        """
        out_size = n_concepts * self.embedding_size
        return {
            "loc": Sequential(
                LinearEmbeddingToConcept(
                    in_embeddings=self.latent_size,
                    out_concepts=out_size,
                ),
                _NormalizePerConcept(self.embedding_size),
            ),
            "scale": Sequential(
                LinearEmbeddingToConcept(
                    in_embeddings=self.latent_size,
                    out_concepts=out_size,
                ),
                nn.Softplus(),
            ),
        }

    @property
    def embedding_query_names(self) -> List[str]:
        """Query names of the probabilistic (Normal) embedding variables.

        Add these to a forward ``query`` to obtain the embeddings' ``loc`` /
        ``scale`` parameters in the output — required by :meth:`vib_kl` and
        :meth:`concept_uncertainty`.
        """
        return [
            variable.name
            for variable in self.pgm.variables.values()
            if variable.variable_type == 'embedding'
            and variable.distribution is Normal
        ]

    def vib_kl(self, out: ModelOutput) -> torch.Tensor:
        """Variational information bottleneck regulariser of ProbCBM.

        Computes ``KL(Normal(loc, scale) || Normal(0, 1))`` for every
        probabilistic concept embedding, summed over embedding dimensions and
        averaged over the batch and the concepts (the reference
        implementation's ``mean`` reduction). Add ``vib_beta * vib_kl(out)``
        to the concept loss as in the paper.

        Parameters
        ----------
        out : ModelOutput
            A forward output whose query included
            :attr:`embedding_query_names`.

        Returns
        -------
        torch.Tensor
            Scalar KL regulariser.
        """
        per_concept = []
        for name in self.embedding_query_names:
            params = out.params.get(name)
            if params is None or 'loc' not in params:
                continue
            loc = params['loc'].reshape(
                params['loc'].shape[0], -1, self.embedding_size
            )
            scale = params['scale'].reshape(
                params['scale'].shape[0], -1, self.embedding_size
            )
            kl = 0.5 * (loc.pow(2) + scale.pow(2) - 1.0 - 2.0 * scale.log())
            per_concept.append(kl.sum(-1))  # (batch, n_concepts_in_var)
        if not per_concept:
            raise ValueError(
                "vib_kl needs the embedding parameters in the output: add "
                "`model.embedding_query_names` to the forward query."
            )
        return torch.cat(per_concept, dim=1).mean()

    def concept_uncertainty(self, out: ModelOutput) -> torch.Tensor:
        """Per-concept embedding uncertainty (Kim et al., Eq. for u_i).

        The geometric mean of the embedding variances across the embedding
        dimensions, per concept — the concept uncertainty measure of the
        reference implementation.

        Parameters
        ----------
        out : ModelOutput
            A forward output whose query included
            :attr:`embedding_query_names`.

        Returns
        -------
        torch.Tensor
            Uncertainties of shape ``(batch, n_concepts)``, columns ordered as
            :attr:`intermediate_concept_names`.
        """
        parts = []
        for name in self.embedding_query_names:
            params = out.params.get(name)
            if params is None or 'scale' not in params:
                continue
            scale = params['scale'].reshape(
                params['scale'].shape[0], -1, self.embedding_size
            )
            parts.append((2.0 * scale.log()).mean(-1).exp())
        if not parts:
            raise ValueError(
                "concept_uncertainty needs the embedding parameters in the "
                "output: add `model.embedding_query_names` to the forward "
                "query."
            )
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Building paths
    # ------------------------------------------------------------------
    def _build_plate_model(self) -> BayesianNetwork:
        """Build using one plate variable per bipartite level.

        A single ``Normal`` embedding variable holds every concept's
        probabilistic embedding (event shape ``(n_concepts, embedding_size)``),
        one plate concept variable decodes all concepts from anchor distances
        in one shot, and one plate task variable holds all tasks.
        """
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        concept_names = self.intermediate_concept_names
        n_concepts = len(concept_names)
        concept0 = axis.concept(concept_names[0])
        task0 = axis.concept(self.task_names[0])

        self._setup_anchor_components()

        # All concepts' probabilistic embeddings in one Normal variable.
        embeddings = EmbeddingVariable(
            "embeddings",
            distribution=Normal,
            shape=(n_concepts, self.embedding_size),
        )
        concepts = ConceptVariable(
            names="concepts",
            members=concept_names,
            distribution=self.distribution_of(concept0.name),
            dist_kwargs=self.dist_kwargs_of(concept0.name),
            size=concept0.cardinality,
        )
        tasks = ConceptVariable(
            names="tasks",
            members=self.task_names,
            distribution=self.distribution_of(task0.name),
            dist_kwargs=self.dist_kwargs_of(task0.name),
            size=task0.cardinality,
        )

        emb_cpd = ParametricCPD(
            variable=embeddings,
            parents=[latent_var],
            parametrization=self._gaussian_embedding_parametrization(
                n_concepts
            ),
        )
        encoders = ParametricCPD(
            variable=concepts,
            parents=[embeddings],
            parametrization={
                "logits": AnchorEmbeddingToConcept(self.concept_anchors),
            },
        )
        predictors = ParametricCPD(
            variable=tasks,
            parents=[concepts],
            parametrization={
                "logits": AnchorConceptToConcept(
                    projection=self.class_projection,
                    cardinality=task0.cardinality,
                    n_heads=len(self.task_names),
                ),
            },
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, embeddings, concepts, tasks],
            factors=[input_cpd, latent_cpd, emb_cpd, encoders, predictors],
        )

    def _build_individual_model(self) -> BayesianNetwork:
        """Build with one embedding/concept variable per concept and one
        variable per task."""
        axis = self.concept_annotations

        input_var, latent_var, input_cpd, latent_cpd = \
            self._input_latent_block()

        concept_names = self.intermediate_concept_names
        intermediate = [axis.concept(name) for name in concept_names]
        task_concepts = [axis.concept(name) for name in self.task_names]

        self._setup_anchor_components()

        # One Normal embedding variable per concept.
        embeddings = EmbeddingVariable(
            names=[f"emb_{c.name}" for c in intermediate],
            distribution=Normal,
            size=self.embedding_size,
        )
        concepts = ConceptVariable(
            names=concept_names,
            distribution=[self.distribution_of(c.name) for c in intermediate],
            dist_kwargs=[self.dist_kwargs_of(c.name) for c in intermediate],
            size=[c.cardinality for c in intermediate],
        )
        tasks = ConceptVariable(
            names=self.task_names,
            distribution=[self.distribution_of(t.name) for t in task_concepts],
            dist_kwargs=[self.dist_kwargs_of(t.name) for t in task_concepts],
            size=[t.cardinality for t in task_concepts],
        )

        emb_cpds = ParametricCPD(
            variable=embeddings,
            parents=[latent_var],
            # One independently-parametrized CPD per concept.
            parametrization=[
                self._gaussian_embedding_parametrization(1)
                for _ in intermediate
            ],
        )
        # One CPD per concept: each concept is decoded from its *own*
        # embedding against its own anchor pair (shared anchor table).
        encoders = [
            ParametricCPD(
                variable=concept,
                parents=[embedding],
                parametrization={
                    "logits": AnchorEmbeddingToConcept(
                        self.concept_anchors,
                        concept_idx=[i],
                    ),
                },
            )
            for i, (concept, embedding) in enumerate(zip(concepts, embeddings))
        ]
        predictors = ParametricCPD(
            variable=tasks,
            parents=[*concepts],
            parametrization=[{
                "logits": AnchorConceptToConcept(
                    projection=self.class_projection,
                    cardinality=t.cardinality,
                    n_heads=1,
                ),
            } for t in task_concepts],
        )

        return BayesianNetwork(
            variables=[input_var, latent_var, *embeddings, *concepts, *tasks],
            factors=[
                input_cpd, latent_cpd, *emb_cpds, *encoders, *predictors,
            ],
        )
