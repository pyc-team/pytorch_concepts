from typing import List, Optional, Union, Mapping
from torch import nn
import torch

from .....annotations import Annotations
from .....data.utils import ensure_list

from ....modules.mid.constructors.bipartite import BipartiteModel
from ....modules.low.encoders.exogenous import LinearZU
from ....modules.low.encoders.linear import LinearUC
from ....modules.low.predictors.exogenous import MixCUC
from ....modules.low.lazy import LazyConstructor
from ....modules.low.base.inference import BaseInference
from ....modules.mid.inference.forward import DeterministicInference

from ..base.model import BaseModel
from ..learners import JointLearner


class ConceptEmbeddingModel_Joint(BaseModel, JointLearner):
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        exogenous_size: int = 16,
        variable_distributions: Optional[Mapping] = None,
        inference: Optional[BaseInference] = DeterministicInference,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Mapping] = None,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            variable_distributions=variable_distributions,
            loss=loss,
            metrics=metrics,
            **kwargs
        )

        # Ensure task_names is a list
        task_names = ensure_list(task_names)
        
        # Extract concept cardinalities (excluding tasks)
        concept_idxs = [self.concept_names.index(name) for name in self.concept_names
                        if name not in task_names]
        cardinalities = [self.concept_annotations.cardinalities[i] for i in concept_idxs]

        self.model = BipartiteModel(
            task_names=task_names,
            input_size=self.latent_size,
            annotations=annotations,
            source_exogenous=LazyConstructor(LinearZU, exogenous_size=exogenous_size),
            encoder=LazyConstructor(LinearUC, n_exogenous_per_concept=1),
            predictor=LazyConstructor(MixCUC, cardinalities=cardinalities),
            use_source_exogenous=True
        )

        # self.inference = AncestralSamplingInference(self.model.probabilistic_model, temperature=1.0)
        self.inference = inference(self.model.probabilistic_model)

    def forward(
        self, 
        x: torch.Tensor, 
        query: List[str] = None
    ) -> torch.Tensor:
        """Forward pass through CBM.

        Args:
            x (torch.Tensor): Input data (raw or pre-computed inputs).
            query (List[str], optional): Variables to query from PGM.
                Typically all concepts and tasks. Defaults to None.
            backbone_kwargs (Optional[Mapping[str, Any]], optional): Arguments
                for backbone. Defaults to None.
            *args, **kwargs: Additional arguments for future extensions.

        Returns:
            torch.Tensor: Concatenated endogenous for queried variables.
                Shape: (batch_size, sum of variable cardinalities).
        """

        # (b, input_size) -> (b, backbone_out_features)
        features = self.maybe_apply_backbone(x)

        # (b, backbone_out_features) -> (b, latent_size)
        latent = self.latent_encoder(features)

        # inference
        # get endogenous for the query concepts
        # (b, latent_size) -> (b, sum(concept_cardinalities))
        endogenous = self.inference.query(query, evidence={'input': latent})
        return endogenous

    def filter_output_for_loss(self, forward_out, target):
        """No filtering needed - return raw endogenous for standard loss computation.

        Args:
            forward_out: Model output endogenous.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for loss computation.
        """
        # forward_out: endogenous
        # return: endogenous
        return {'input': forward_out,
                'target': target}

    def filter_output_for_metrics(self, forward_out, target):
        """No filtering needed - return raw endogenous for metric computation.

        Args:
            forward_out: Model output endogenous.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for metric computation.
        """
        # forward_out: endogenous
        # return: endogenous
        return {'preds': forward_out,
                'target': target}


class ConceptEmbeddingModel(ConceptEmbeddingModel_Joint):
    """Alias for ConceptEmbeddingModel_Joint."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)