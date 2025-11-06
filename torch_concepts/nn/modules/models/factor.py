import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from typing import List, Optional, Tuple
from itertools import product

from ....concepts.variable import Variable
from torch_concepts.distributions import Delta


class Factor:
    def __init__(self, concepts: List[str], module_class: nn.Module):
        self.concepts = concepts
        self.module_class = module_class
        self.variable: Optional[Variable] = None
        self.parents: List[Variable] = []

    def forward(self, **kwargs) -> torch.Tensor:
        return self.module_class(**kwargs)

    def _get_parent_combinations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates:
        1. all_full_inputs: Full feature vectors used as input to the module.
        2. all_discrete_state_vectors: State vectors for discrete parents (for potential table rows).
        """
        if not self.parents:
            in_features = self.module_class.in_features
            placeholder_input = torch.zeros((1, in_features))
            return placeholder_input, torch.empty((1, 0))

        discrete_combinations_list = []
        discrete_state_vectors_list = []
        continuous_tensors = []

        for parent in self.parents:
            parent_var = parent

            if parent_var.distribution in [Bernoulli, Categorical]:
                out_dim = parent_var.out_features

                input_combinations = []
                state_combinations = []

                if parent_var.distribution is Bernoulli:
                    input_combinations = list(product([0.0, 1.0], repeat=out_dim))
                    state_combinations = input_combinations

                elif parent_var.distribution is Categorical:
                    for i in range(out_dim):
                        one_hot = torch.zeros(out_dim)
                        one_hot[i] = 1.0
                        input_combinations.append(one_hot.tolist())
                        state_combinations.append([float(i)])  # State is the category index

                discrete_combinations_list.append(
                    [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in input_combinations])
                discrete_state_vectors_list.append(
                    [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in state_combinations])

            elif parent_var.distribution is Delta or parent_var.distribution is torch.distributions.Normal:
                fixed_value = torch.zeros(parent_var.out_features).unsqueeze(0)
                continuous_tensors.append(fixed_value)

            else:
                raise TypeError(f"Unsupported distribution type {parent_var.distribution.__name__} for CPT generation.")

        # Product across discrete parents
        all_discrete_product = list(product(*discrete_combinations_list))
        all_discrete_states_product = list(product(*discrete_state_vectors_list))

        all_full_inputs = []
        all_discrete_state_vectors = []

        fixed_continuous_input = torch.cat(continuous_tensors, dim=-1) if continuous_tensors else torch.empty((1, 0))

        # Build combined input tensors for the module
        for discrete_inputs in all_discrete_product:
            discrete_part = torch.cat(list(discrete_inputs), dim=-1)
            full_input_tensor = torch.cat([discrete_part, fixed_continuous_input], dim=-1)
            all_full_inputs.append(full_input_tensor)

        # Build combined state vectors for the potential table rows
        for discrete_states in all_discrete_states_product:
            discrete_state_vector = torch.cat(list(discrete_states), dim=-1)
            all_discrete_state_vectors.append(discrete_state_vector)

        if not all_full_inputs and continuous_tensors:
            all_full_inputs = [fixed_continuous_input]

        return torch.cat(all_full_inputs, dim=0), torch.cat(all_discrete_state_vectors, dim=0)

    def build_cpt(self) -> torch.Tensor:
        if not self.variable:
            raise RuntimeError("Factor not linked to a Variable in PGM.")

        all_full_inputs, discrete_state_vectors = self._get_parent_combinations()

        input_batch = all_full_inputs

        if input_batch.shape[-1] != self.module_class.in_features:
            raise RuntimeError(
                f"Input tensor dimension mismatch for CPT building. "
                f"Factor module expects {self.module_class.in_features} features, "
                f"but parent combinations resulted in {input_batch.shape[-1]} features. "
                f"Check Variable definition and PGM resolution."
            )

        logits = self.module_class(input=input_batch)
        probabilities = None

        if self.variable.distribution is Bernoulli:
            # Traditional P(X=1) output
            p_c1 = torch.sigmoid(logits)

            # ACHIEVE THE REQUESTED 4x3 STRUCTURE: [Parent States | P(X=1)]
            probabilities = torch.cat([discrete_state_vectors, p_c1], dim=-1)

        elif self.variable.distribution is Categorical:
            probabilities = torch.softmax(logits, dim=-1)

        elif self.variable.distribution is Delta:
            probabilities = logits

        else:
            raise NotImplementedError(f"CPT for {self.variable.distribution.__name__} not supported.")

        return probabilities

    def build_potential(self) -> torch.Tensor:
        if not self.variable:
            raise RuntimeError("Factor not linked to a Variable in PGM.")

        # We need the core probability part for potential calculation
        all_full_inputs, discrete_state_vectors = self._get_parent_combinations()
        logits = self.module_class(input=all_full_inputs)

        if self.variable.distribution is Bernoulli:
            cpt_core = torch.sigmoid(logits)
        elif self.variable.distribution is Categorical:
            cpt_core = torch.softmax(logits, dim=-1)
        elif self.variable.distribution is Delta:
            cpt_core = logits
        else:
            raise NotImplementedError("Potential table construction not supported for this distribution.")

        # --- Potential Table Construction ---

        if self.variable.distribution is Bernoulli:
            p_c1 = cpt_core
            p_c0 = 1.0 - cpt_core

            child_states_c0 = torch.zeros_like(p_c0)
            child_states_c1 = torch.ones_like(p_c1)

            # Rows for X=1: [Parent States | Child State (1) | P(X=1)]
            rows_c1 = torch.cat([discrete_state_vectors, child_states_c1, p_c1], dim=-1)
            # Rows for X=0: [Parent States | Child State (0) | P(X=0)]
            rows_c0 = torch.cat([discrete_state_vectors, child_states_c0, p_c0], dim=-1)

            potential_table = torch.cat([rows_c1, rows_c0], dim=0)

        elif self.variable.distribution is Categorical:
            n_classes = self.variable.size
            all_rows = []
            for i in range(n_classes):
                child_state_col = torch.full((cpt_core.shape[0], 1), float(i), dtype=torch.float32)
                prob_col = cpt_core[:, i].unsqueeze(-1)

                # [Parent States | Child State (i) | P(X=i)]
                rows_ci = torch.cat([discrete_state_vectors, child_state_col, prob_col], dim=-1)
                all_rows.append(rows_ci)

            potential_table = torch.cat(all_rows, dim=0)

        elif self.variable.distribution is Delta:
            # [Parent States | Child Value]
            child_value = cpt_core
            potential_table = torch.cat([discrete_state_vectors, child_value], dim=-1)

        else:
            raise NotImplementedError("Potential table construction not supported for this distribution.")

        return potential_table

    def __repr__(self):
        return f"Factor(concepts={self.concepts}, module={self.module_class.__class__.__name__})"
