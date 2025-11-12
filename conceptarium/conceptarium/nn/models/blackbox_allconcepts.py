import torch
import torch.nn as nn
from typing import Optional

from torch_concepts import AnnotatedTensor, Annotations, ConceptTensor

from experiments.conceptarium.nn.layers import MLP
from conceptarium.nn.base.model import BaseModel
from conceptarium.typing import BackboneType


class BB_AllConcepts(BaseModel):
    def __init__(self,
                 input_size: int,
                 concept_annotations: Annotations,
                 hidden_size: Optional[int] = 64,
                 n_layers: Optional[int] = 1,
                 activation: Optional[str] = 'leaky_relu',
                 dropout: Optional[float] = 0.0,

                 embs_precomputed: Optional[bool] = False,
                 backbone: Optional[BackboneType] = None,
                 ):
        super(BB_AllConcepts, self).__init__(concept_annotations=concept_annotations,
                                            embs_precomputed=embs_precomputed,
                                            backbone=backbone)

        # TODO: redo this with root layer and internal layer

        total_concept_dim = concept_annotations[1].get_total_cardinality()
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=total_concept_dim,
                           n_layers=n_layers,
                           activation=activation,
                           dropout=dropout)

        self.reasoner = nn.Identity()  # Placeholder for compatibility

        self.activation = nn.Sigmoid()

    def forward(self, 
                x: torch.Tensor, 
                c: AnnotatedTensor = None, 
                interv_idx: Optional[AnnotatedTensor] = None):
        h = self.maybe_apply_backbone(x)
        y_hat = self.encoder(h)
        y_hat = self.reasoner(y_hat)
        # activate from logits to probs
        y_hat = self.activation(y_hat).unsqueeze(-1)
        # reshape to nested tensor for concept probabilities
        # concatenate 1 - y_hat for concept neg probs
        y_hat = torch.cat([1 - y_hat, y_hat], dim=-1)
        out = ConceptTensor(annotations=self.concept_annotations,
                            concept_probs=y_hat, 
                            concept_embs=None, 
                            residual=None)
        return out, None