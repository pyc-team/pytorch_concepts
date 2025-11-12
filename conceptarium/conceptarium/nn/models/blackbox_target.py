import torch.nn as nn
from typing import List, Optional, Union
from torch import Tensor

from torch_concepts import ConceptTensor, AnnotatedTensor, Annotations

from experiments.conceptarium.nn.layers import MLP
from conceptarium.typing import BackboneType, BaseModel


class BB_Target(BaseModel):
    def __init__(self,
                 task_names: Union[List[str], List[int]],
                 input_size: int,
                 concept_annotations: Annotations,
                 hidden_size: Optional[int] = 64,
                 n_layers: Optional[int] = 1,
                 activation: Optional[str] = 'leaky_relu',
                 dropout: Optional[float] = 0.0,

                 embs_precomputed: Optional[bool] = False,
                 backbone: Optional[BackboneType] = None,
                 ):
        super(BB_Target, self).__init__(concept_annotations=concept_annotations,
                                       embs_precomputed=embs_precomputed,
                                       backbone=backbone)

        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=self.total_concept_dim,
                           n_layers=n_layers,
                           activation=activation,
                           dropout=dropout)

        self.reasoner = nn.Identity()  # Placeholder for compatibility

    def forward(self, 
                x: Tensor, 
                c: AnnotatedTensor = None, 
                interv_idx: Optional[AnnotatedTensor] = None):
        h = self.maybe_apply_backbone(x)
        y_hat = self.encoder(h)
        y_hat = self.reasoner(y_hat)
        return y_hat, None