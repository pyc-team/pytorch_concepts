"""
Conditional Probability Distribution (CPD) for directed probabilistic graphical models.

This module defines the ParametricCPD class, a ParametricFactor subclass that represents
conditional probability distributions in Bayesian Networks.  Unlike the base
ParametricFactor (which only knows about a scope of concepts), a CPD introduces the
notion of **parents** — giving the factor directed semantics.
"""
import copy
import torch
from typing import List, Optional, Tuple, Union
from itertools import product

import torch.nn as nn

from .factor import Factor
from .parametric_factor import ParametricFactor
from .variable import (
    Variable,
    _BINARY_DISTRIBUTIONS,
    _CATEGORICAL_DISTRIBUTIONS,
    _CONTINUOUS_DISTRIBUTIONS,
)


class ParametricCPD(ParametricFactor):
    """
    Conditional probability distribution parameterised by a neural network.

    Extends :class:`ParametricFactor` with directed-edge semantics: each CPD
    has a list of **parent** concept-variables and computes
    ``P(child | parents)`` via its ``parametrization`` module.

    Parameters
    ----------
    concepts : Union[str, List[str]]
        Concept name(s).  When a list is provided, ``__new__`` returns a list
        of independent ``ParametricCPD`` instances (one per concept).
    parametrization : Union[nn.Module, List[nn.Module]]
        Neural network(s) that compute the conditional distribution.
    parents : List[Union[Variable, str]], optional
        Parent concept-variables (or their names as strings, resolved later
        by :class:`ProbabilisticModel`).  Defaults to ``[]``.

    Attributes
    ----------
    parents : List[Variable]
        Parent concept-variables in the directed graphical model.

    See Also
    --------
    ParametricFactor : Base (undirected) factor class.
    ProbabilisticModel : PGM container that resolves parent references.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __new__(cls, 
                concepts: Union[str, List[str]],
                parametrization: Union[nn.Module, List[nn.Module]],
                shared: bool = False,
                shared_name: Optional[str] = None,
                **kwargs):
        """
        Create new ParametricCPD instance(s).

        If ``concepts`` is a string, returns a single instance.
        If ``concepts`` is a list and ``shared=False`` (default), returns a
        list of instances (one per concept), each with a deep-copied
        parametrization.
        If ``concepts`` is a list and ``shared=True``, returns a **single**
        instance whose parametrization is shared across all concepts.  The
        parametrization must output concatenated logits for all concepts
        (i.e. ``(batch, n_concepts * size)``).
        """
        if isinstance(concepts, str):
            if isinstance(parametrization, list):
                raise ValueError(
                    "When 'concepts' is a string, 'parametrization' must be a single module, not a list.")
            return object.__new__(cls)

        # --- shared=True: single instance, no deepcopy ---
        if shared:
            if isinstance(parametrization, list):
                raise ValueError(
                    "When shared=True, 'parametrization' must be a single module, not a list.")
            return object.__new__(cls)

        # --- shared=False (default): one deepcopied instance per concept ---
        n_concepts = len(concepts)
        if not isinstance(parametrization, list):
            module_list = [parametrization] * n_concepts
        else:
            module_list = parametrization

        if len(module_list) != n_concepts:
            raise ValueError(
                f"If concepts is a list of length {n_concepts}, parametrization must either be "
                f"a single module or a list of length {n_concepts}.")

        instances = []
        for i in range(n_concepts):
            instance = object.__new__(cls)
            instance.__init__(
                concepts=concepts[i],
                parametrization=copy.deepcopy(module_list[i]),
                **kwargs,
            )
            instances.append(instance)
        return instances

    def __init__(self, 
                 concepts: Union[str, List[str]],
                 parametrization: Union[nn.Module, List[nn.Module]],
                 parents: List[Union[Variable, str]] = None,
                 shared: bool = False,
                 shared_name: Optional[str] = None,
                 **kwargs):
        super().__init__(concepts=concepts, parametrization=parametrization, **kwargs)
        self.parents: List[Variable] = list(parents) if parents is not None else []
        self.shared: bool = shared
        self.shared_name: Optional[str] = shared_name

    # ------------------------------------------------------------------
    # Directed-model helpers (moved from ParametricFactor)
    # ------------------------------------------------------------------
    @property
    def in_features(self) -> int:
        """Sum of parent variable sizes."""
        if not self.parents:
            return 0
        return sum(p.size for p in self.parents)

    _MAX_DISCRETE_BITS = 20  # cap on total discrete parent bits for table construction

    def _get_parent_combinations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enumerate all discrete parent-state combinations for table construction.

        Continuous (Delta / Normal) parents are held at zero; discrete parents
        (Bernoulli, Categorical and their relaxed variants) are exhaustively
        enumerated.

        Returns
        -------
        all_full_inputs : torch.Tensor
            Input tensors for the parametrization, one row per combination.
        all_discrete_state_vectors : torch.Tensor
            Corresponding state vectors for the table rows.

        Raises
        ------
        RuntimeError
            If the total number of discrete parent bits exceeds
            ``_MAX_DISCRETE_BITS`` (default 20), which would require
            enumerating more than ~1 million combinations.
        """
        if not self.parents:
            in_features = self.parametrization.in_features
            placeholder_input = torch.zeros((1, in_features))
            return placeholder_input, torch.empty((1, 0))

        # --- guard against combinatorial explosion ---
        total_bits = 0
        for p in self.parents:
            if p.distribution in _BINARY_DISTRIBUTIONS:
                total_bits += p.size
            elif p.distribution in _CATEGORICAL_DISTRIBUTIONS:
                total_bits += p.size  # one-hot dims
        if total_bits > self._MAX_DISCRETE_BITS:
            raise RuntimeError(
                f"Total discrete parent bits ({total_bits}) exceeds the "
                f"maximum of {self._MAX_DISCRETE_BITS}. Table construction "
                f"would require 2^{total_bits} rows."
            )

        discrete_combinations_list = []
        discrete_state_vectors_list = []
        continuous_tensors = []

        for parent_var in self.parents:
            if parent_var.distribution in _BINARY_DISTRIBUTIONS | _CATEGORICAL_DISTRIBUTIONS:
                out_dim = parent_var.size
                input_combinations = []
                state_combinations = []

                if parent_var.distribution in _BINARY_DISTRIBUTIONS:
                    input_combinations = list(product([0.0, 1.0], repeat=out_dim))
                    state_combinations = input_combinations
                elif parent_var.distribution in _CATEGORICAL_DISTRIBUTIONS:
                    for i in range(out_dim):
                        one_hot = torch.zeros(out_dim)
                        one_hot[i] = 1.0
                        input_combinations.append(one_hot.tolist())
                        state_combinations.append([float(i)])

                discrete_combinations_list.append(
                    [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in input_combinations])
                discrete_state_vectors_list.append(
                    [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in state_combinations])

            elif parent_var.distribution in _CONTINUOUS_DISTRIBUTIONS:
                fixed_value = torch.zeros(parent_var.size).unsqueeze(0)
                continuous_tensors.append(fixed_value)
            else:
                raise TypeError(
                    f"Unsupported distribution type {parent_var.distribution.__name__} for table generation.")

        if not discrete_combinations_list:
            fixed_continuous_input = (torch.cat(continuous_tensors, dim=-1)
                                      if continuous_tensors else torch.empty((1, 0)))
            return fixed_continuous_input, torch.empty((1, 0))

        all_discrete_product = list(product(*discrete_combinations_list))
        all_discrete_states_product = list(product(*discrete_state_vectors_list))

        fixed_continuous_input = (torch.cat(continuous_tensors, dim=-1)
                                  if continuous_tensors else torch.empty((1, 0)))

        all_full_inputs = []
        for discrete_inputs in all_discrete_product:
            discrete_part = torch.cat(list(discrete_inputs), dim=-1)
            all_full_inputs.append(torch.cat([discrete_part, fixed_continuous_input], dim=-1))

        all_discrete_state_vectors = []
        for discrete_states in all_discrete_states_product:
            all_discrete_state_vectors.append(torch.cat(list(discrete_states), dim=-1))

        return torch.cat(all_full_inputs, dim=0), torch.cat(all_discrete_state_vectors, dim=0)

    # ------------------------------------------------------------------
    # Factor construction (for inference algorithms)
    # ------------------------------------------------------------------

    @staticmethod
    def _variable_cardinality(var: Variable) -> int:
        """Return the number of discrete states for a variable."""
        if var.distribution in _BINARY_DISTRIBUTIONS:
            return 2
        elif var.distribution in _CATEGORICAL_DISTRIBUTIONS:
            return var.size
        elif var.distribution in _CONTINUOUS_DISTRIBUTIONS:
            raise ValueError(
                f"Continuous variable '{var.concept}' "
                f"(distribution={var.distribution.__name__}, size={var.size}) "
                f"cannot be discretized into a factor."
            )
        else:
            raise NotImplementedError(
                f"Cannot determine cardinality for distribution "
                f"{var.distribution.__name__}."
            )

    def build_factor(self, cardinalities: dict = None,
                     input: torch.Tensor = None) -> "Factor":
        """
        Build a :class:`Factor` representing this CPD as a multi-dimensional
        tensor ``P(child | parents)``.

        The tensor has one axis per variable in ``{parents} ∪ {child}``,
        and is filled by evaluating the parametrization over every
        parent-state combination followed by sigmoid (Bernoulli) or
        softmax (Categorical).

        Parameters
        ----------
        cardinalities : dict, optional
            Pre-computed ``{variable_name: num_states}`` mapping.  If
            ``None`` the cardinalities are inferred from the
            :class:`Variable` objects.
        input : torch.Tensor, optional
            Input embedding of shape ``(batch, emb_dim)``.  Only used for
            root nodes (CPDs with no parents).  When provided, the
            embedding is fed to the parametrization to produce per-sample
            factor values (returned :class:`Factor` has ``batched=True``).
            Child nodes ignore this parameter — their factors are
            determined entirely by parent-state combinations.

        Returns
        -------
        Factor
            A factor over ``[*parent_names, child_name]``.
        """
        if cardinalities is None:
            cardinalities = {}

        # --- determine variable names and cardinalities -----------------
        child_name = self.concept
        child_var = self.variable
        child_card = cardinalities.get(
            child_name, self._variable_cardinality(child_var)
        )
        cardinalities[child_name] = child_card

        parent_names = []
        parent_cards = []
        for p in self.parents:
            # Continuous parents (Delta, Normal, …) are held at fixed values
            # during table construction and do not contribute discrete axes.
            if p.distribution in _CONTINUOUS_DISTRIBUTIONS:
                continue
            pname = p.concept
            pcard = cardinalities.get(
                pname, self._variable_cardinality(p)
            )
            cardinalities[pname] = pcard
            parent_names.append(pname)
            parent_cards.append(pcard)

        # --- evaluate the neural network over all parent combinations ---
        if input is not None and not parent_names:
            # Input-conditioned mode for root nodes (no discrete parents):
            # produce per-sample factors.
            B = input.size(0)
            logits = self.parametrization(input).unsqueeze(1)  # (B, 1, out)

            if child_var.distribution in _BINARY_DISTRIBUTIONS | _CONTINUOUS_DISTRIBUTIONS:
                p1 = torch.sigmoid(logits)
                probs = torch.cat([1.0 - p1, p1], dim=-1)  # (B, 1, 2)
            elif child_var.distribution in _CATEGORICAL_DISTRIBUTIONS:
                probs = torch.softmax(logits, dim=-1)
            else:
                raise NotImplementedError(
                    f"build_factor() not supported for "
                    f"{child_var.distribution.__name__}."
                )

            values = probs.reshape([B, child_card])
            variables = parent_names + [child_name]
            return Factor(values, variables, cardinalities, batched=True)

        # --- non-batched path (for child nodes, or when input is None) --
        all_inputs, _ = self._get_parent_combinations()
        logits = self.parametrization(all_inputs)  # (n_combos, out)

        if child_var.distribution in _BINARY_DISTRIBUTIONS | _CONTINUOUS_DISTRIBUTIONS:
            p1 = torch.sigmoid(logits)  # P(child=1 | parents)
            probs = torch.cat([1.0 - p1, p1], dim=-1)  # (n_combos, 2)
        elif child_var.distribution in _CATEGORICAL_DISTRIBUTIONS:
            probs = torch.softmax(logits, dim=-1)  # (n_combos, K)
        else:
            raise NotImplementedError(
                f"build_factor() not supported for "
                f"{child_var.distribution.__name__}."
            )

        # --- reshape into multi-dimensional tensor ----------------------
        # Shape: (*parent_cards, child_card)
        if parent_cards:
            shape = parent_cards + [child_card]
        else:
            shape = [child_card]
        values = probs.reshape(shape)

        variables = parent_names + [child_name]
        return Factor(values, variables, cardinalities)
