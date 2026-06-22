"""Causally-Reliable Concept Bottleneck Model (C2BM).

A directed concept model over an explicit (causal) DAG: every concept owns a
per-state embedding produced from the latent; root concepts are decoded from
that embedding, and each internal concept is predicted from its parent concepts
mixed with its embedding through a hypernetwork. The graph assembly and inference
wiring live in :class:`~torch_concepts.nn.modules.high.base.homogen.HomogenGraphModel`;
this model only chooses the encoder and predictor layers.

References
----------
Dominici et al. "Causal Concept Graph Models: Beyond Causal Opacity in Deep
Learning", ICLR 2025. https://arxiv.org/abs/2405.16507
De Felice et al. "Causally Reliable Concept Bottleneck Models", NeurIPS.
https://arxiv.org/abs/2503.04363
"""
from typing import Optional

from torch.distributions import Bernoulli, OneHotCategorical

from .....annotations import Annotations, AxisAnnotation
from .....concept_graph import ConceptGraph
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.hypernet import HyperlinearConceptEmbeddingToConcept
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.models.variable import _DEFAULT_DIST_KWARGS
from ..base.homogen import HomogenGraphModel


class CausallyReliableConceptBottleneckModel(HomogenGraphModel):
    """Causally-reliable concept bottleneck model over an explicit DAG.

    Every concept is decoded from its own per-state embedding; root concepts read
    that embedding directly, internal concepts mix their parent concepts'
    activations with the embedding through a hypernetwork predictor.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, distributions).
    graph : ConceptGraph
        Directed acyclic graph over the concepts (node names must match labels).
    embedding_size : int, default 16
        Width of each per-concept embedding.
    hypernet_hidden_size : int, default 16
        Hidden width of the hypernetwork that generates the predictor weights.
    hypernet_use_bias : bool, default False
        Whether the hypernetwork predictor adds a stochastic bias.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration (see :class:`ConceptBottleneckModel`).
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``).
    """

    supported_concept_types = frozenset({"binary", "categorical"})
    param_for_discrete_var = "logits"
    source_embeddings = True
    internal_embeddings = True

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
        graph: ConceptGraph,
        embedding_size: int = 8,
        hypernet_hidden_size: int = 8,
        hypernet_use_bias: bool = False,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            graph=graph,
            lightning=lightning,
            **kwargs,
        )
        self.embedding_size = embedding_size
        self.hypernet_hidden_size = hypernet_hidden_size
        self.hypernet_use_bias = hypernet_use_bias
        self.pgm = self._build_individual_model()

        # once self.pgm is built, we can set up the inference engines (train and eval)
        self.setup_inference(
            inference,
            inference_kwargs,
            train_inference,
            train_inference_kwargs,
        )

    # ------------------------------------------------------------------
    # Layer hooks (the only model-specific pieces)
    # ------------------------------------------------------------------
    def build_encoder(self, in_embeddings, out_concepts):
        return LinearEmbeddingToConcept(in_embeddings=in_embeddings, out_concepts=out_concepts)

    def build_predictor(self, in_concepts: AxisAnnotation, in_embeddings, out_concepts):
        return HyperlinearConceptEmbeddingToConcept(
            in_concepts=int(sum(in_concepts.cardinalities)),
            in_embeddings=in_embeddings,
            hidden_size=self.hypernet_hidden_size,
            use_bias=self.hypernet_use_bias,
        )
