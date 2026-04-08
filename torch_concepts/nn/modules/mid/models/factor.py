"""
Factor operations for probabilistic graphical models.

This module defines the :class:`Factor` class — a lightweight wrapper around a
multi-dimensional :class:`torch.Tensor` that represents a factor (potential)
in a factor graph.  Each tensor axis corresponds to a random variable in the
factor's scope, and all operations (product, marginalisation, evidence
conditioning) are implemented as standard differentiable PyTorch operations
so that gradients flow back through the neural-network parameters that
produced the factor values.

The class is intentionally agnostic to whether the underlying model is
directed (Bayesian Network) or undirected (Markov Random Field): both
produce :class:`Factor` instances that the inference algorithms manipulate
identically.
"""

import torch
from typing import Dict, List, Tuple


class Factor:
    """
    A factor (potential function) in a probabilistic graphical model.

    Wraps a multi-dimensional :class:`torch.Tensor` whose axes are named
    random variables.  Provides differentiable operations commonly used by
    exact and approximate inference algorithms:

    * :meth:`product` — factor product via broadcasting.
    * :meth:`marginalize` — sum out one or more variables.
    * :meth:`set_evidence` — condition on observed variable values.
    * :meth:`normalize` — compute the partition function and return a
      normalised copy.

    Parameters
    ----------
    values : torch.Tensor
        The factor tensor.  Its shape must match the cardinalities of the
        variables listed in *variables* (in the same order).
    variables : List[str]
        Ordered list of variable names — ``variables[i]`` labels axis *i* of
        *values*.
    cardinalities : Dict[str, int]
        Mapping from every variable name that may appear in the model to its
        number of states.  The dict may contain entries for variables not
        currently in this factor's scope (they are simply ignored).

    Raises
    ------
    ValueError
        If the length of *variables* does not match the number of dimensions
        of *values*, or if any axis size disagrees with *cardinalities*.
    """

    def __init__(
        self,
        values: torch.Tensor,
        variables: List[str],
        cardinalities: Dict[str, int],
    ):
        if values.ndim != len(variables):
            raise ValueError(
                f"Tensor has {values.ndim} dimensions but {len(variables)} "
                f"variable names were given."
            )
        for i, var in enumerate(variables):
            if var in cardinalities and values.shape[i] != cardinalities[var]:
                raise ValueError(
                    f"Axis {i} ('{var}') has size {values.shape[i]} but "
                    f"cardinality is {cardinalities[var]}."
                )

        self.values = values
        self.variables: List[str] = list(variables)
        self.cardinalities: Dict[str, int] = dict(cardinalities)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def product(self, other: "Factor") -> "Factor":
        """
        Compute the factor product of ``self`` and *other*.

        The resulting factor has scope
        ``self.variables U other.variables``.  Shared variables are
        aligned and element-wise multiplied; non-shared variables are
        broadcast.

        Parameters
        ----------
        other : Factor
            The factor to multiply with.

        Returns
        -------
        Factor
            A new factor whose values are the element-wise product over
            the union of scopes.
        """
        # Build the union scope: self's variables first, then new ones
        # from other.
        new_vars = list(self.variables)
        for v in other.variables:
            if v not in new_vars:
                new_vars.append(v)

        # Reshape self.values so that it has a dimension for every
        # variable in new_vars (size-1 for variables not in self).
        a = self._align(new_vars)
        b = other._align(new_vars)

        new_cardinalities = {**self.cardinalities, **other.cardinalities}
        return Factor(a * b, new_vars, new_cardinalities)

    def marginalize(self, variable: str) -> "Factor":
        """
        Sum out *variable* from this factor.

        Parameters
        ----------
        variable : str
            Name of the variable to eliminate.

        Returns
        -------
        Factor
            A new factor whose scope no longer contains *variable*.

        Raises
        ------
        ValueError
            If *variable* is not in this factor's scope.
        """
        if variable not in self.variables:
            raise ValueError(
                f"Variable '{variable}' is not in the factor scope "
                f"{self.variables}."
            )
        axis = self.variables.index(variable)
        new_values = self.values.sum(dim=axis)
        new_vars = [v for v in self.variables if v != variable]
        return Factor(new_values, new_vars, self.cardinalities)

    def set_evidence(self, variable: str, state: int) -> "Factor":
        """
        Condition on ``variable = state`` by slicing the tensor.

        The returned factor no longer contains *variable* in its scope
        (the axis is removed via indexing).

        Parameters
        ----------
        variable : str
            Name of the observed variable.
        state : int
            Observed state index (0-based).

        Returns
        -------
        Factor
            A new factor with *variable* fixed to *state*.

        Raises
        ------
        ValueError
            If *variable* is not in the scope or *state* is out of range.
        """
        if variable not in self.variables:
            raise ValueError(
                f"Variable '{variable}' is not in the factor scope "
                f"{self.variables}."
            )
        axis = self.variables.index(variable)
        if state < 0 or state >= self.values.shape[axis]:
            raise ValueError(
                f"State {state} is out of range for variable '{variable}' "
                f"with {self.values.shape[axis]} states."
            )
        # torch.select removes the dimension (like numpy basic indexing).
        new_values = self.values.select(axis, state)
        new_vars = [v for v in self.variables if v != variable]
        return Factor(new_values, new_vars, self.cardinalities)

    def normalize(self) -> Tuple[torch.Tensor, "Factor"]:
        """
        Normalise the factor so that its values sum to one.

        Returns
        -------
        Z : torch.Tensor
            The partition function (sum of all values), scalar tensor.
        normalized : Factor
            A new factor with the same scope whose values sum to 1.
        """
        Z = self.values.sum()
        normalized_values = self.values / Z
        return Z, Factor(normalized_values, list(self.variables), self.cardinalities)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _align(self, target_vars: List[str]) -> torch.Tensor:
        """
        Reshape ``self.values`` so that it is broadcastable to a tensor
        whose axes correspond to *target_vars*.

        For every variable in *target_vars* that is **not** in
        ``self.variables`` a size-1 dimension is inserted.  For variables
        that *are* in ``self.variables`` the original size is kept, and
        axes are permuted to match the order in *target_vars*.

        Parameters
        ----------
        target_vars : List[str]
            The desired axis ordering (superset of ``self.variables``).

        Returns
        -------
        torch.Tensor
            View / expansion of ``self.values`` with ``len(target_vars)``
            dimensions.
        """
        # Step 1: build a permutation + insertion plan.
        # For each position in target_vars, record either the source axis
        # index (if the variable exists in self) or None (needs unsqueeze).
        source_index = {v: i for i, v in enumerate(self.variables)}

        # Permute existing axes to the right relative order, then
        # unsqueeze for missing variables.
        #
        # Strategy: first permute self.values so that the variables that
        # *do* appear in target_vars are in the correct relative order,
        # then unsqueeze at positions where variables are missing.

        # Which of target_vars are in self?
        present = [v for v in target_vars if v in source_index]
        perm = [source_index[v] for v in present]
        t = self.values.permute(*perm) if perm != list(range(len(perm))) else self.values

        # Now insert size-1 dims for missing variables.
        result = t
        for i, v in enumerate(target_vars):
            if v not in source_index:
                result = result.unsqueeze(i)

        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Factor(variables={self.variables}, "
            f"shape={list(self.values.shape)})"
        )

    def __mul__(self, other: "Factor") -> "Factor":
        """Allow ``f1 * f2`` as shorthand for ``f1.product(f2)``."""
        return self.product(other)
