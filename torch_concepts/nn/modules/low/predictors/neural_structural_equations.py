"""Reusable neural structural equations."""

import torch
import torch.nn as nn

from ..base.layer import BaseConceptLayer
from ..dense_layers import MLP, get_layer_activation
from ...utils import state_embedding_counts


class NeuralStructuralEquations(BaseConceptLayer):
    """Map graph-aggregated target contexts to distribution parameters.

    Each concept has its own final structural equation head in
    ``concept_structural_equations``. The preceding shared block is grouped by
    the number of state embeddings required by the target concept: binary and
    continuous concepts usually share the one-state block, while categorical
    concepts with the same cardinality share a wider block. Concepts with
    different state counts do not share that block.

    In CGM this module sits after the mixer and graph aggregator. For non-root
    evaluation factors it receives the graph-aggregated context and then
    selects the target concept's column before the concept-specific head. Root
    evaluation factors bypass this module and reuse the corresponding copy CPD
    parametrization.
    """

    def __init__(
        self,
        in_embeddings,
        out_concept_types,
        cardinalities_out_concepts,
        shared_n_layers=0,
        shared_hidden_size=None,
        concept_n_layers=1,
        concept_hidden_size=None,
        shared_activation="leaky_relu",
        concept_activation="leaky_relu",
    ):
        if shared_n_layers < 0:
            raise ValueError("shared_n_layers must be non-negative.")

        if concept_n_layers < 1:
            raise ValueError("concept_n_layers must be positive.")

        if len(out_concept_types) != len(cardinalities_out_concepts):
            raise ValueError(
                "out_concept_types and cardinalities_out_concepts "
                "must have the same length."
            )

        out_concepts = sum(
            cardinality if concept_type == "categorical" else 1
            for concept_type, cardinality in zip(
                out_concept_types,
                cardinalities_out_concepts,
            )
        )

        super().__init__(
            in_concepts=len(out_concept_types),
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )

        self.out_concept_types = list(out_concept_types)
        self.cardinalities_out_concepts = list(
            cardinalities_out_concepts
        )
        self.n_state_embeddings = state_embedding_counts(
            out_concept_types,
            cardinalities_out_concepts,
        )

        shared_hidden_size = shared_hidden_size or in_embeddings
        concept_hidden_size = concept_hidden_size or in_embeddings

        self.shared_activation = get_layer_activation(
            shared_activation
        )()

        # Concepts with the same number of state embeddings share this part
        # of their structural equation.
        self.shared_structural_equations = nn.ModuleDict({
            str(n_states): self._make_shared_equation(
                n_states=n_states,
                hidden_size=shared_hidden_size,
                n_layers=shared_n_layers,
                activation=shared_activation,
            )
            for n_states in sorted(set(self.n_state_embeddings))
        })

        # The final part of the equation is specific to each concept.
        self.concept_structural_equations = nn.ModuleList([
            MLP(
                n_states * in_embeddings,
                concept_hidden_size,
                output_size=(
                    cardinality
                    if concept_type == "categorical"
                    else 1
                ),
                n_layers=concept_n_layers,
                activation=concept_activation,
            )
            for concept_type, cardinality, n_states in zip(
                self.out_concept_types,
                self.cardinalities_out_concepts,
                self.n_state_embeddings,
            )
        ])

    def _make_shared_equation(
        self,
        n_states,
        hidden_size,
        n_layers,
        activation,
    ):
        output_size = n_states * self.in_embeddings

        if n_layers == 0:
            return nn.Linear(
                self.in_embeddings,
                output_size,
                bias=False,
            )

        return MLP(
            self.in_embeddings,
            hidden_size,
            output_size=output_size,
            n_layers=n_layers,
            activation=activation,
        )

    def _validate_inputs(self, contexts, target_concept=None):
        expected = (self.in_concepts, self.in_embeddings)
        if target_concept is not None and contexts.shape[-1:] == (self.in_embeddings,):
            return
        if contexts.shape[-2:] != expected:
            raise ValueError(
                f"contexts must have trailing shape {expected}, got "
                f"{tuple(contexts.shape)}."
            )

    def forward(
        self, contexts, *, target_concept=None, target_concepts=None
    ):
        """Parametrize all targets, one target, or a selected target group.

        Usually ``contexts`` has shape ``(..., n_concepts, embedding_dim)``.
        The concept axis is the second-to-last one, so ``contexts[..., j, :]``
        is the graph-aggregated parent context used to predict concept ``j``.

        During evaluation a caller may already pass one target context with
        shape ``(..., embedding_dim)`` together with ``target_concept=j``.

        When all concepts have the same number of state embeddings, the shared
        block is applied once to the full context tensor and target columns are
        selected afterwards. With mixed variable types/cardinalities, the shared
        block is chosen by each target's state count and evaluated for that
        target group; no result cache is kept inside this module.
        """
        if target_concept is not None and target_concepts is not None:
            raise ValueError(
                "Pass either target_concept or target_concepts, not both."
            )
        self._validate_inputs(contexts, target_concept)

        single_target_context = False
        if target_concept is not None:
            if not 0 <= target_concept < self.in_concepts:
                raise IndexError(
                    f"target_concept must be in [0, {self.in_concepts}), "
                    f"got {target_concept}."
                )
            targets = [target_concept]
            single_target_context = contexts.shape[-1:] == (self.in_embeddings,)
        elif target_concepts is not None:
            targets = list(target_concepts)
        else:
            targets = list(range(self.in_concepts))

        if single_target_context:
            contexts = contexts.unsqueeze(-2)

        if len(self.shared_structural_equations) != 1:
            return self._mixed_forward(
                contexts, targets, target_concept, single_target_context,
            )

        # Original CGM flow after GraphAggregator has already applied
        # torch.matmul(x, fc1_weight) and restored the node axis.
        x = self.shared_activation(contexts)
        x = next(iter(self.shared_structural_equations.values()))(x)
        x = self.shared_activation(x)

        outputs = [
            self.concept_structural_equations[target](
                x[
                    ...,
                    0 if single_target_context else target,
                    :,
                ]
            )
            for target in targets
        ]
        if target_concept is not None:
            return outputs[0]
        return torch.cat(outputs, dim=-1)

    def _mixed_forward(
        self, contexts, targets, target_concept, single_target_context,
    ):
        """Evaluate targets whose state counts may require different blocks."""
        x = self.shared_activation(contexts)
        outputs = [
            self.concept_structural_equations[target](
                self.shared_activation(
                    self.shared_structural_equations[
                        str(self.n_state_embeddings[target])
                    ](x)
                )[
                    ...,
                    0 if single_target_context else target,
                    :,
                ]
            )
            for target in targets
        ]
        if target_concept is not None:
            return outputs[0]
        return torch.cat(outputs, dim=-1)

    def for_targets(self, targets):
        """Return a lightweight view evaluating selected equations."""
        return _SelectedStructuralEquations(self, targets)


class _SelectedStructuralEquations(nn.Module):
    """Lightweight view that restricts an equation module to selected targets."""

    def __init__(self, equations, targets):
        super().__init__()
        self.equations = equations
        self.targets = list(targets)

    def forward(self, contexts):
        return self.equations(contexts, target_concepts=self.targets)


__all__ = [
    "NeuralStructuralEquations",
]
