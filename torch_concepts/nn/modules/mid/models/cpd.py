import copy

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical
from typing import List, Optional, Tuple, Union
from itertools import product

from .variable import Variable
from .....distributions import Delta


class ParametricCPD(nn.Module):
    """
    A ParametricCPD represents a conditional probability distribution (CPD) in a probabilistic graphical model.

    A ParametricCPD links concepts to neural network modules that compute probability distributions.
    It can automatically split multiple concepts into separate CPD and supports building
    conditional probability tables (CPTs) and potential tables for inference.

    Parameters
    ----------
    concepts : Union[str, List[str]]
        A single concept name or a list of concept names. If a list of N concepts is provided,
        the ParametricCPD automatically splits into N separate ParametricCPD instances.
    module : Union[nn.Module, List[nn.Module]]
        A neural network module or list of modules that compute the probability distribution.
        If concepts is a list of length N, module can be:
        - A single module (will be replicated for all concepts)
        - A list of N modules (one per concept)

    Attributes
    ----------
    concepts : List[str]
        List of concept names associated with this CPD.
    module : nn.Module
        The neural network module used to compute probabilities.
    variable : Optional[Variable]
        The Variable instance this CPD is linked to (set by ProbabilisticModel).
    parents : List[Variable]
        List of parent Variables in the graphical model.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch_concepts.nn import ParametricCPD
    >>>
    >>> # Create different modules for different concepts
    >>> module_a = nn.Linear(in_features=10, out_features=1)
    >>> module_b = nn.Sequential(
    ...     nn.Linear(in_features=10, out_features=5),
    ...     nn.ReLU(),
    ...     nn.Linear(in_features=5, out_features=1)
    ... )
    >>>
    >>> # Create CPD with different modules
    >>> cpd = ParametricCPD(
    ...     concepts=["binary_concept", "complex_concept"],
    ...     parametrization=[module_a, module_b]
    ... )
    >>>
    >>> print(cpd[0].parametrization)
    Linear(in_features=10, out_features=1, bias=True)
    >>> print(cpd[1].parametrization)
    Sequential(...)

    Notes
    -----
    - The ParametricCPD class uses a custom `__new__` method to automatically split multiple concepts
      into separate ParametricCPD instances when a list is provided.
    - ParametricCPDs are typically created and managed by a ProbabilisticModel rather than directly.
    - The module should accept an 'input' keyword argument in its forward pass.
    - Supported distributions for CPT/potential building: Bernoulli, Categorical, Delta, Normal.

    See Also
    --------
    Variable : Represents a random variable in the probabilistic model.
    ProbabilisticModel : Container that manages CPD and variables.
    """
    def __new__(cls, concepts: Union[str, List[str]],
                parametrization: Union[nn.Module, List[nn.Module]]):

        if isinstance(concepts, str):
            assert not isinstance(parametrization, list)
            return object.__new__(cls)

        n_concepts = len(concepts)

        # If single concept in list, treat as single ParametricCPD
        if n_concepts == 1:
            assert not isinstance(parametrization, list), "For single concept, modules must be a single nn.Module."
            return object.__new__(cls)

        # Standardize module: single value -> list of N values
        if not isinstance(parametrization, list):
            module_list = [parametrization] * n_concepts
        else:
            module_list = parametrization

        if len(module_list) != n_concepts:
            raise ValueError("If concepts list has length N > 1, module must either be a single value or a list of length N.")

        new_cpd = []
        for i in range(n_concepts):
            instance = object.__new__(cls)
            instance.__init__(
                concepts=[concepts[i]],
                parametrization=copy.deepcopy(module_list[i])
            )
            new_cpd.append(instance)
        return new_cpd

    def __init__(self, concepts: Union[str, List[str]],
                 parametrization: Union[nn.Module, List[nn.Module]]):
        super().__init__()

        if isinstance(concepts, str):
            concepts = [concepts]

        self.concepts = concepts
        self.parametrization = parametrization
        self.variable: Optional[Variable] = None
        self.parents: List[Variable] = []

    def forward(self, **kwargs) -> torch.Tensor:
        return self.parametrization(**kwargs)

    def _get_parent_combinations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates:
        1. all_full_inputs: Full feature vectors used as input to the module.
        2. all_discrete_state_vectors: State vectors for discrete parents (for potential table rows).
        """
        if not self.parents:
            in_features = self.parametrization.in_features
            placeholder_input = torch.zeros((1, in_features))
            return placeholder_input, torch.empty((1, 0))

        discrete_combinations_list = []
        discrete_state_vectors_list = []
        continuous_tensors = []

        for parent in self.parents:
            parent_var = parent

            if parent_var.distribution in [Bernoulli, RelaxedBernoulli, Categorical, RelaxedOneHotCategorical]:
                out_dim = parent_var.out_features

                input_combinations = []
                state_combinations = []

                if parent_var.distribution in [Bernoulli, RelaxedBernoulli]:
                    input_combinations = list(product([0.0, 1.0], repeat=out_dim))
                    state_combinations = input_combinations

                elif parent_var.distribution in [Categorical, RelaxedOneHotCategorical]:
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

        # Handle case with only continuous parents (no discrete parents)
        if not discrete_combinations_list:
            fixed_continuous_input = torch.cat(continuous_tensors, dim=-1) if continuous_tensors else torch.empty((1, 0))
            return fixed_continuous_input, torch.empty((1, 0))

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


        return torch.cat(all_full_inputs, dim=0), torch.cat(all_discrete_state_vectors, dim=0)

    def build_cpt(self) -> torch.Tensor:
        if not self.variable:
            raise RuntimeError("ParametricCPD not linked to a Variable in ProbabilisticModel.")

        all_full_inputs, discrete_state_vectors = self._get_parent_combinations()

        input_batch = all_full_inputs

        if input_batch.shape[-1] != self.parametrization.in_features:
            raise RuntimeError(
                f"Input tensor dimension mismatch for CPT building. "
                f"ParametricCPD module expects {self.parametrization.in_features} features, "
                f"but parent combinations resulted in {input_batch.shape[-1]} features. "
                f"Check Variable definition and ProbabilisticModel resolution."
            )

        endogenous = self.parametrization(input=input_batch)
        probabilities = None

        if self.variable.distribution is Bernoulli:
            # Traditional P(X=1) output
            p_c1 = torch.sigmoid(endogenous)

            # ACHIEVE THE REQUESTED 4x3 STRUCTURE: [Parent States | P(X=1)]
            probabilities = torch.cat([discrete_state_vectors, p_c1], dim=-1)

        elif self.variable.distribution is Categorical:
            probabilities = torch.softmax(endogenous, dim=-1)

        elif self.variable.distribution is Delta:
            probabilities = endogenous

        else:
            raise NotImplementedError(f"CPT for {self.variable.distribution.__name__} not supported.")

        return probabilities

    def build_potential(self) -> torch.Tensor:
        if not self.variable:
            raise RuntimeError("ParametricCPD not linked to a Variable in ProbabilisticModel.")

        # We need the core probability part for potential calculation
        all_full_inputs, discrete_state_vectors = self._get_parent_combinations()
        endogenous = self.parametrization(input=all_full_inputs)

        if self.variable.distribution is Bernoulli:
            cpt_core = torch.sigmoid(endogenous)
        elif self.variable.distribution is Categorical:
            cpt_core = torch.softmax(endogenous, dim=-1)
        elif self.variable.distribution is Delta:
            cpt_core = endogenous
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
        return f"ParametricCPD(concepts={self.concepts}, parametrization={self.parametrization.__class__.__name__})"
