"""Elastic-net linear prediction using the optional GLM-SAGA solver."""
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from torch_concepts import Annotations
from .linear import LinearConceptToConcept


class SparseLinearConceptToConcept(LinearConceptToConcept):
    """Multinomial linear predictor fitted with proximal SAGA.

    Inputs are already standardized sample-by-concept tensors and targets are
    integer class indices. ``fit`` freezes the fitted layer and exposes solver
    settings and losses in ``diagnostics_``. Fitting requires the ``lfcbm``
    extra; forward inference uses only PyTorch.
    """

    def __init__(self, in_concepts: Union[int, Annotations], out_concepts: Union[int, Annotations]):
        super().__init__(in_concepts, out_concepts)

    @torch.no_grad()
    def fit(self, concepts: torch.Tensor, targets: torch.Tensor,
            val_concepts: Optional[torch.Tensor] = None, val_targets: Optional[torch.Tensor] = None, *,
            epochs=1000, batch_size=256, lr=.1, alpha=.99, lam=.0007,
            tolerance=1e-4, seed=0) -> "SparseLinearConceptToConcept":
        """Fit one penalty, ``lam / max(.001, alpha)``, with an unpenalized bias.

        Validation data is optional and used only to report validation loss.
        No regularization search or validation-based classifier selection runs.
        Every fit resets weights and bias to zero. ``epochs`` bounds passes
        through the training data, and ``tolerance`` controls parameter-change
        stopping. Inputs need not retain an autograd graph.
        """
        try:
            from glm_saga.elasticnet import IndexedTensorDataset, elastic_loss, train_saga
        except ImportError as error:
            raise ImportError("Sparse fitting requires glm-saga; install 'pytorch_concepts[lfcbm]'.") from error
        if not 0 <= alpha <= 1 or lam < 0:
            raise ValueError("Classifier alpha must be in [0, 1] and lam nonnegative.")
        if epochs < 1 or batch_size < 1 or lr <= 0 or tolerance < 0:
            raise ValueError("Classifier steps, batch size, and learning rate must be positive; tolerance nonnegative.")
        if (val_concepts is None) != (val_targets is None):
            raise ValueError("Provide both validation concepts and targets.")
        datasets = [("train", concepts, targets)]
        if val_concepts is not None:
            datasets.append(("val", val_concepts, val_targets))
        for _, x, y in datasets:
            if (x.ndim != 2 or x.shape[1] != self.predictor.in_features or len(x) == 0
                    or y.shape != (len(x),) or y.dtype != torch.long
                    or not torch.isfinite(x).all() or (y < 0).any()
                    or (y >= self.predictor.out_features).any()):
                raise ValueError("Expected finite sample-by-concept inputs and valid integer class targets.")
        self.predictor.weight.zero_()
        self.predictor.bias.zero_()
        self.requires_grad_(True)
        loader = DataLoader(
            IndexedTensorDataset(concepts.detach().to(device="cpu", dtype=self.predictor.weight.dtype),
                                 targets.cpu()), batch_size=batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(seed),
        )
        penalty = lam / max(.001, alpha)
        train_saga(
            self.predictor, loader, lr, epochs, penalty, alpha, group=False,
            n_ex=len(concepts), n_classes=self.predictor.out_features, tol=tolerance,
        )
        diagnostics = {"solver": "glm_saga", "regularization": penalty, "alpha": alpha}
        weight = self.predictor.weight
        for name, x, y in datasets:
            diagnostics[f"{name}_loss"] = float(elastic_loss(
                self.predictor, x.to(weight), y.to(weight.device), penalty, alpha,
            ))
        self.diagnostics_ = diagnostics
        return self.eval().requires_grad_(False)
