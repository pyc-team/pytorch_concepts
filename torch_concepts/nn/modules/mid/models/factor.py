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

# Subscript pool for einsum: a-z + A-Y = 51 chars (Z reserved for batch dim).
_EINSUM_SUBSCRIPTS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY'


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
        variables listed in *variables* (in the same order).  When
        *batched* is ``True`` the first dimension is a batch dimension
        and the variable axes start at dimension 1.
    variables : List[str]
        Ordered list of variable names — ``variables[i]`` labels axis *i* of
        *values* (or axis *i + 1* when *batched*).
    cardinalities : Dict[str, int]
        Mapping from every variable name that may appear in the model to its
        number of states.  The dict may contain entries for variables not
        currently in this factor's scope (they are simply ignored).
    batched : bool, optional
        If ``True`` the leading dimension of *values* is treated as a
        batch dimension and all factor operations preserve it.

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
        batched: bool = False,
    ):
        expected_ndim = len(variables) + (1 if batched else 0)
        if values.ndim != expected_ndim:
            raise ValueError(
                f"Tensor has {values.ndim} dimensions but expected "
                f"{expected_ndim} ({len(variables)} variables"
                f"{' + 1 batch dim' if batched else ''})."
            )
        offset = 1 if batched else 0
        for i, var in enumerate(variables):
            if var in cardinalities and values.shape[i + offset] != cardinalities[var]:
                raise ValueError(
                    f"Axis {i + offset} ('{var}') has size "
                    f"{values.shape[i + offset]} but "
                    f"cardinality is {cardinalities[var]}."
                )

        self.values = values
        self.variables: List[str] = list(variables)
        self.cardinalities: Dict[str, int] = cardinalities
        self.batched: bool = batched

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

        new_batched = self.batched or other.batched

        # --- einsum-based product ---
        # Assign each variable a unique subscript letter.
        if len(new_vars) > len(_EINSUM_SUBSCRIPTS):
            raise ValueError(
                f"Factor product scope has {len(new_vars)} variables, "
                f"exceeding the einsum limit of {len(_EINSUM_SUBSCRIPTS)}."
            )
        var_to_sub = {v: _EINSUM_SUBSCRIPTS[i] for i, v in enumerate(new_vars)}
        batch_sub = 'Z'

        lhs = ''.join(var_to_sub[v] for v in self.variables)
        rhs = ''.join(var_to_sub[v] for v in other.variables)
        out = ''.join(var_to_sub[v] for v in new_vars)

        a, b = self.values, other.values

        if new_batched:
            if self.batched:
                lhs = batch_sub + lhs
            if other.batched:
                rhs = batch_sub + rhs
            out = batch_sub + out
            # Promote unbatched operand so einsum sees a matching batch dim
            if self.batched and not other.batched:
                b = b.unsqueeze(0).expand(a.shape[0], *b.shape)
                rhs = batch_sub + rhs
            elif other.batched and not self.batched:
                a = a.unsqueeze(0).expand(b.shape[0], *a.shape)
                lhs = batch_sub + lhs

        result = torch.einsum(f'{lhs},{rhs}->{out}', a, b)

        new_cardinalities = self.cardinalities if self.cardinalities is other.cardinalities else {**self.cardinalities, **other.cardinalities}
        return Factor(result, new_vars, new_cardinalities, batched=new_batched)

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
        offset = 1 if self.batched else 0
        axis = self.variables.index(variable) + offset
        new_values = self.values.sum(dim=axis)
        new_vars = [v for v in self.variables if v != variable]
        return Factor(new_values, new_vars, self.cardinalities,
                      batched=self.batched)

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
        offset = 1 if self.batched else 0
        axis = self.variables.index(variable) + offset
        if state < 0 or state >= self.values.shape[axis]:
            raise ValueError(
                f"State {state} is out of range for variable '{variable}' "
                f"with {self.values.shape[axis]} states."
            )
        # torch.select removes the dimension (like numpy basic indexing).
        new_values = self.values.select(axis, state)
        new_vars = [v for v in self.variables if v != variable]
        return Factor(new_values, new_vars, self.cardinalities,
                      batched=self.batched)

    def normalize(self) -> Tuple[torch.Tensor, "Factor"]:
        """
        Normalise the factor so that its values sum to one.

        When *batched* is ``True`` normalisation is performed
        independently for each sample in the batch.

        Returns
        -------
        Z : torch.Tensor
            The partition function.  Scalar when unbatched, shape
            ``(batch,)`` when batched.
        normalized : Factor
            A new factor with the same scope whose values sum to 1.
        """
        if self.batched:
            var_dims = list(range(1, self.values.ndim))
            Z = self.values.sum(dim=var_dims, keepdim=True)
            normalized_values = self.values / Z
            Z_out = Z.reshape(self.values.size(0))
        else:
            Z_out = self.values.sum()
            normalized_values = self.values / Z_out
        return Z_out, Factor(normalized_values, self.variables,
                             self.cardinalities, batched=self.batched)

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

        if self.batched:
            # Batch dim stays at position 0; variable dims are offset by 1.
            perm_full = [0] + [p + 1 for p in perm]
            t = (self.values.permute(*perm_full)
                 if perm_full != list(range(len(perm_full)))
                 else self.values)
            result = t
            for i, v in enumerate(target_vars):
                if v not in source_index:
                    result = result.unsqueeze(i + 1)
        else:
            t = (self.values.permute(*perm)
                 if perm != list(range(len(perm)))
                 else self.values)
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
