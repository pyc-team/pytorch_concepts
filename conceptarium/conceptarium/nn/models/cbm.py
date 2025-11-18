"""Concept Bottleneck Model (CBM) implementations.

This module provides two implementations of CBM:
1. CBM: High-level implementation using BipartiteModel
2. CBM_factors: Mid-level implementation using Variables, Factors, and PGM

CBM enforces a strict information bottleneck where task predictions must go 
through interpretable concept representations.

Reference:
    Koh et al. "Concept Bottleneck Models" (ICML 2020)
"""

from typing import Any, Dict, List, Optional, Union, Mapping
from torch import nn
import torch

from torch_concepts import Annotations, Variable
from torch_concepts.distributions import Delta
from torch_concepts.nn import BipartiteModel, ProbEncoderFromEmb, ProbPredictor, ProbabilisticGraphicalModel, \
                              Factor, Propagator, BaseInference

from ..base.model import BaseModel

class CBM(BaseModel):
    """High-level Concept Bottleneck Model using BipartiteModel.
    
    Implements a two-stage architecture:
    1. Backbone + Encoder → Concept predictions
    2. Concept predictions → Task predictions
    
    The concept bottleneck enforces interpretability by forcing all task-relevant
    information to flow through a set of predefined concepts.
    
    Args:
        task_names (Union[List[str], str, List[int]]): Names or indices of task 
            variables to predict.
        inference (BaseInference): Inference strategy class (uninstantiated, 
            e.g., MaximumLikelihood).
        input_size (int): Dimensionality of input features (after backbone).
        annotations (Annotations): Concept and task annotations.
        variable_distributions (Mapping): Distribution types for each variable.
        embs_precomputed (bool, optional): Skip backbone if True. Defaults to False.
        backbone (Optional[callable], optional): Feature extraction module. 
            Defaults to None.
        encoder_kwargs (Dict, optional): Arguments for MLP encoder. Defaults to None.
        **kwargs: Additional arguments (reserved for future use).
        
    Attributes:
        pgm (ProbabilisticGraphicalModel): The underlying PGM structure.
        inference (BaseInference): Instantiated inference object.
        
    Example:
        >>> from torch_concepts import Annotations
        >>> from torch_concepts.nn import DeterministicInference
        >>> 
        >>> annotations = Annotations(...)  # Define all concept annotations
        >>> model = CBM(
        ...     task_names=['diagnosis'],
        ...     inference=DeterministicInference,
        ...     input_size=512,
        ...     annotations=annotations,
        ...     variable_distributions={'symptom1': 'binary', 'diagnosis': 'categorical'},
        ...     encoder_kwargs={'hidden_size': 64, 'n_layers': 1}
        ... )
        >>> 
        >>> # Forward pass
        >>> x = torch.randn(32, 512)  # batch_size=32
        >>> concepts_and_tasks = model(x, query=['symptom1', 'symptom2', 'diagnosis'])
    """
    def __init__(
        self,
        task_names: Union[List[str], str, List[int]],
        inference: BaseInference,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Mapping,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        # loss_type: str = 'standard',
        **kwargs
    ) -> None:
        super().__init__(
            annotations=annotations,
            variable_distributions=variable_distributions,
            # encoder params
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder_kwargs=encoder_kwargs,
        )
        # self.loss_type = loss_type
        # if loss_type == 'weighted':
        #     self.task_names = task_names
        #     self.task_idxes = [annotations.get_axis_annotation(1).get_index(tn) for tn in task_names]
        #     self.concept_idxes = [i for i in range(len(annotations.get_axis_annotation(1).labels)) if i not in self.task_idxes]

        model = BipartiteModel(task_names=task_names,
                               input_size=self.encoder_out_features,
                               annotations=annotations,
                               encoder=Propagator(ProbEncoderFromEmb),
                               predictor=Propagator(ProbPredictor))
        self.pgm = model.pgm

        self.inference = inference(self.pgm)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
                *args,
                backbone_kwargs: Optional[Mapping[str, Any]] = None,
                **kwargs
            ) -> torch.Tensor:
        """Forward pass through CBM.
        
        Args:
            x (torch.Tensor): Input data (raw or pre-computed embeddings).
            query (List[str], optional): Variables to query from PGM. 
                Typically all concepts and tasks. Defaults to None.
            backbone_kwargs (Optional[Mapping[str, Any]], optional): Arguments 
                for backbone. Defaults to None.
            *args, **kwargs: Additional arguments for future extensions.
            
        Returns:
            torch.Tensor: Concatenated logits for queried variables.
                Shape: (batch_size, sum of variable cardinalities).
        """
        
        # (b, input_size) -> (b, backbone_out_features)
        features = self.maybe_apply_backbone(x, backbone_kwargs)

        # (b, backbone_out_features) -> (b, encoder_out_features)
        features = self.encoder(features)

        # inference
        # get logits for the query concepts
        # (b, encoder_out_features) -> (b, sum(concept_cardinalities))
        out = self.inference.query(query, evidence={'embedding': features})
        return out

    def filter_output_for_loss(self, forward_out):
        """No filtering needed - return raw logits for standard loss computation.
        
        Args:
            forward_out: Model output logits.
            
        Returns:
            Unmodified forward output.
        """
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        """No filtering needed - return raw logits for metric computation.
        
        Args:
            forward_out: Model output logits.
            
        Returns:
            Unmodified forward output.
        """
        # forward_out: logits
        # return: logits
        return forward_out




class CBM_factors(BaseModel):
    """Mid-level Concept Bottleneck Model using Variables, Factors, and PGM.
    
    Provides more explicit control over the PGM structure compared to the 
    high-level CBM implementation. Useful for:
    - Custom factor definitions
    - Advanced PGM modifications
    - Research on probabilistic concept models
    
    The structure mirrors CBM but constructs the PGM manually:
    embedding → concepts → tasks
    
    Args:
        task_names (Union[List[str], str, List[int]]): Task variable names/indices.
        inference (BaseInference): Inference strategy class (uninstantiated).
        input_size (int): Input feature dimensionality.
        annotations (Annotations): Variable annotations.
        variable_distributions (Mapping): Distribution types.
        embs_precomputed (bool, optional): Skip backbone. Defaults to False.
        backbone (Optional[callable], optional): Feature extractor. Defaults to None.
        encoder_kwargs (Dict, optional): MLP encoder config. Defaults to None.
        **kwargs: Reserved for future use.
        
    Example:
        >>> # More control over PGM structure
        >>> model = CBM_factors(
        ...     task_names=['disease'],
        ...     inference=DeterministicInference,
        ...     input_size=512,
        ...     annotations=annotations,
        ...     variable_distributions={'fever': 'binary', 'disease': 'categorical'},
        ...     encoder_kwargs={'hidden_size': 64, 'n_layers': 1}
        ... )
        >>> 
        >>> # Access PGM components directly
        >>> print(model.pgm.variables)  # [embedding, fever, cough, disease]
        >>> print(model.pgm.factors)    # [embedding_factor, encoders, predictors]
    """
    def __init__(
        self,
        task_names: Union[List[str], str, List[int]],
        inference: BaseInference,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Mapping,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        **kwargs
    ) -> None:
        # Initialize the BaseModel
        # this will setup the encoder (torch) layers and the annotations metadata
        super().__init__(
            annotations=annotations,
            variable_distributions=variable_distributions,
            # encoder params
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder_kwargs=encoder_kwargs,
        )
        # init variable for the latent embedding from the encoder
        embedding = Variable("embedding", parents=[], distribution=Delta, size=self.encoder_out_features)
        embedding_factor = Factor("embedding", module_class=nn.Identity())

        # variables initialization
        concept_names = [c for c in annotations.get_axis_labels(1) if c not in task_names]
        concepts = Variable(concept_names,
                            parents=['embedding'], # all concepts have the same parent='embedding'
                            distribution=[annotations[1].metadata[c]['distribution'] for c in concept_names],
                            size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in concept_names])
        
        tasks = Variable(task_names,
                         parents=concept_names, # all tasks have the same parents='concepts'
                         distribution=[annotations[1].metadata[c]['distribution'] for c in task_names],
                         size=[annotations[1].cardinalities[annotations[1].get_index(c)] for c in task_names])

        # layers initialization
        concept_encoders = Factor(concept_names, 
                                  module_class=[ProbEncoderFromEmb(in_features_embedding=embedding.size, 
                                                                   out_features=c.size) for c in concepts])
        
        task_predictors = Factor(task_names, 
                                 module_class=[ProbPredictor(in_features_logits=sum([c.size for c in concepts]), 
                                                             out_features=t.size) for t in tasks])

        # PGM Initialization
        self.pgm = ProbabilisticGraphicalModel(
            variables=[embedding, *concepts, *tasks],
            factors=[embedding_factor, *concept_encoders, *task_predictors]
        )

        self.inference = inference(self.pgm)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
                *args,
                backbone_kwargs: Optional[Mapping[str, Any]] = None,
                **kwargs
            ) -> torch.Tensor:
        """Forward pass through CBM_factors.
        
        Identical behavior to CBM.forward() but uses manually constructed PGM.
        
        Args:
            x (torch.Tensor): Input data.
            query (List[str], optional): Variables to query. Defaults to None.
            backbone_kwargs (Optional[Mapping[str, Any]], optional): Backbone args. 
                Defaults to None.
                
        Returns:
            torch.Tensor: Logits for queried variables.
        """
        
        # (b, input_size) -> (b, backbone_out_features)
        features = self.maybe_apply_backbone(x, backbone_kwargs)

        # (b, backbone_out_features) -> (b, encoder_out_features)
        features = self.encoder(features)

        # inference
        # get logits for the query concepts
        # (b, encoder_out_features) -> (b, sum(concept_cardinalities))
        out = self.inference.query(query, evidence={'embedding': features})
        return out

    def filter_output_for_loss(self, forward_out):
        """Return logits unchanged for loss computation."""
        # forward_out: logits
        # return: logits
        return forward_out

    def filter_output_for_metric(self, forward_out):
        """Return logits unchanged for metric computation."""
        # forward_out: logits
        # return: logits
        return forward_out