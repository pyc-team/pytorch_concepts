"""Graph Concept Bottleneck Model — a CBM over an explicit DAG.

A worked example that the :class:`~torch_concepts.nn.modules.high.base.homogen.HomogenGraphModel`
assembler is genuinely extendible: a plain linear concept bottleneck defined over
an arbitrary DAG (concepts may have concept parents) is obtained by supplying
*only* the encoder and predictor layers — no embeddings, no custom assembly.
"""
from typing import Optional

from .....annotations import Annotations, AxisAnnotation
from .....concept_graph import ConceptGraph
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.predictors.linear import LinearConceptToConcept
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ..base.homogen import HomogenGraphModel


class GraphConceptBottleneckModel(HomogenGraphModel):
    """Linear concept bottleneck over a DAG: root concepts encoded from the latent,
    internal concepts predicted from their parent concepts.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types).
    graph : ConceptGraph
        Directed acyclic graph over the concepts (node names must match labels).
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration (see :class:`ConceptBottleneckModel`).
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``).
    """

    supported_concept_types = frozenset({"binary", "categorical"})
    param_for_discrete_var = "logits"
    source_embeddings = False
    internal_embeddings = False

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        graph: ConceptGraph,
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
        return LinearEmbeddingToConcept(
            in_embeddings=in_embeddings, 
            out_concepts=out_concepts
        )

    def build_predictor(self, in_concepts: AxisAnnotation, in_embeddings, out_concepts):
        return LinearConceptToConcept(
            in_concepts=int(sum(in_concepts.cardinalities)), 
            out_concepts=out_concepts
        )
