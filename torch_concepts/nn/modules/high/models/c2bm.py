
from typing import Dict, List, Optional
import torch

from torch_concepts.nn.modules.low.encoders.exogenous import LinearLatentToExogenous

from .....annotations import Annotations

from ...low.base.inference import BaseInference
from ...low.encoders.linear import LinearExogenousToConcept
from ...low.predictors.hypernet import HyperlinearConceptExogenousToConcept
from ...low.lazy import LazyConstructor

from ...mid.inference.deterministic import DeterministicInference
from ...mid.constructors.concept_graph import ConceptGraph
from ...mid.constructors.graph import GraphModel

from ..base.model import BaseModel


class CausallyReliableConceptBottleneckModel(BaseModel):
    
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        graph: ConceptGraph,
        exogenous_size: int = 16,
        hypernet_hidden_size: int = 16,
        hypernet_use_bias: bool = False,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = DeterministicInference,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False, # wrap the Torch model with Lightning capabilities
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            **kwargs
        )
        
        # TODO: implement utils to provide a causal discovery 
        # pipeline to generate the graph structure from data

        self.model = GraphModel(
            model_graph=graph,
            input_size=self.latent_size,
            annotations=annotations,
            source_exogenous=LazyConstructor(
                LinearLatentToExogenous, 
                out_exogenous=exogenous_size
            ),
            internal_exogenous=LazyConstructor(
                LinearLatentToExogenous, 
                out_exogenous=exogenous_size
            ),
            encoder=LazyConstructor(LinearExogenousToConcept),
            predictor=LazyConstructor(
                HyperlinearConceptExogenousToConcept, 
                hidden_size=hypernet_hidden_size,
                use_bias=hypernet_use_bias
            ),
            use_source_exogenous=True
        )

        self.eval_inference = inference(
            self.model.probabilistic_model, 
            **(inference_kwargs or {})
        )
        self.train_inference = train_inference(
            self.model.probabilistic_model, 
            **(train_inference_kwargs or {})
        )
