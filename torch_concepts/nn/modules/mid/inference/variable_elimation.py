"""
Variable Elimination inference for probabilistic graphical models.

This module implements the Sum-Product Variable Elimination algorithm for
computing exact conditional probabilities in both Bayesian Networks and
Markov Random Fields.  All operations are differentiable so that gradients
flow back through the neural-network parameters that produced the factor
potentials — enabling end-to-end training through inference.
"""

from typing import Dict, List, Optional

import torch

from ..models.factor import Factor
from ..models.probabilistic_model import ProbabilisticModel
from ...low.base.inference import BaseInference


# ──────────────────────────────────────────────────────────────────────
# Elimination-ordering heuristics
# ──────────────────────────────────────────────────────────────────────

def _min_degree_order(
    factors: List[Factor],
    variables_to_eliminate: List[str],
) -> List[str]:
    """
    Compute a greedy min-degree elimination ordering.

    At each step the variable whose current factor-neighbourhood is
    smallest (fewest other variables sharing a factor) is chosen.

    Parameters
    ----------
    factors : List[Factor]
        The current set of factors.
    variables_to_eliminate : List[str]
        Variables that must be eliminated.

    Returns
    -------
    List[str]
        The variables in elimination order.
    """
    remaining = set(variables_to_eliminate)
    order: List[str] = []

    # Build an adjacency-like structure: for each variable, which other
    # variables share a factor with it?
    def _neighbours(var: str) -> set:
        nbrs: set = set()
        for f in factors:
            if var in f.variables:
                nbrs.update(f.variables)
        nbrs.discard(var)
        return nbrs & remaining

    for _ in range(len(variables_to_eliminate)):
        # pick the remaining variable with the fewest neighbours
        best = min(remaining, key=lambda v: len(_neighbours(v)))
        order.append(best)
        remaining.remove(best)

    return order


# ──────────────────────────────────────────────────────────────────────
# Core VE routines
# ──────────────────────────────────────────────────────────────────────

def _eliminate_var(factors: List[Factor], variable: str) -> List[Factor]:
    """
    Eliminate a single variable from the factor set.

    1. Collect all factors whose scope contains *variable* (Φ').
    2. Multiply them together (ψ).
    3. Marginalise *variable* out of ψ (τ).
    4. Return the factors that did *not* mention *variable* plus τ.

    Parameters
    ----------
    factors : List[Factor]
        Current set of factors.
    variable : str
        Variable to eliminate.

    Returns
    -------
    List[Factor]
        Updated factor set with *variable* removed.
    """
    phi_prime: List[Factor] = []
    phi_rest: List[Factor] = []
    for f in factors:
        if variable in f.variables:
            phi_prime.append(f)
        else:
            phi_rest.append(f)

    if not phi_prime:
        return phi_rest

    # multiply all factors that contain the variable
    psi = phi_prime[0]
    for f in phi_prime[1:]:
        psi = psi.product(f)

    # marginalise out the variable
    tau = psi.marginalize(variable)

    phi_rest.append(tau)
    return phi_rest


def _sum_product_ve(
    factors: List[Factor],
    elimination_order: List[str],
) -> Factor:
    """
    Run Sum-Product Variable Elimination.

    Parameters
    ----------
    factors : List[Factor]
        Initial factor set Φ.
    elimination_order : List[str]
        Ordered list of hidden variables to eliminate.

    Returns
    -------
    Factor
        The product of all remaining factors after elimination (φ*).
    """
    for var in elimination_order:
        factors = _eliminate_var(factors, var)

    # multiply remaining factors
    if not factors:
        raise RuntimeError("No factors remain after variable elimination.")

    result = factors[0]
    for f in factors[1:]:
        result = result.product(f)

    return result


# ──────────────────────────────────────────────────────────────────────
# Inference class
# ──────────────────────────────────────────────────────────────────────

class VariableEliminationInference(BaseInference):
    """
    Exact inference via Sum-Product Variable Elimination.

    Supports both Bayesian Networks (all factors are :class:`ParametricCPD`)
    and Markov Random Fields (all factors are :class:`ParametricFactor`).

    The ``query`` method:

    1. Builds discrete :class:`Factor` instances from the neural-network
       parametrised CPDs / potentials (differentiable).
    2. Conditions on the evidence by slicing factors.
    3. Eliminates hidden variables in a chosen order.
    4. Normalises the remaining factor to obtain ``P(query | evidence)``.

    All tensor operations (product, marginalisation, slicing,
    normalisation) are standard PyTorch ops and therefore
    **differentiable** — gradients propagate back through the network
    weights that produced the factor values.

    Parameters
    ----------
    probabilistic_model : ProbabilisticModel
        The graphical model whose factors are queried.
    elimination_order : List[str], optional
        A fixed elimination ordering for the hidden variables.  If
        ``None`` a greedy min-degree heuristic is used.

    Example
    -------
    >>> import torch
    >>> from torch.distributions import Bernoulli
    >>> from torch_concepts import ConceptVariable
    >>> from torch_concepts.nn import ParametricCPD, ProbabilisticModel
    >>> from torch_concepts.nn.modules.mid.inference.variable_elimination import (
    ...     VariableEliminationInference,
    ... )
    >>>
    >>> A = ConceptVariable('A', distribution=Bernoulli)
    >>> B = ConceptVariable('B', distribution=Bernoulli)
    >>> cpd_A = ParametricCPD('A', parametrization=torch.nn.Linear(1, 1))
    >>> cpd_B = ParametricCPD('B', parametrization=torch.nn.Linear(1, 1),
    ...                       parents=['A'])
    >>> model = ProbabilisticModel(variables=[A, B],
    ...                            factors=[cpd_A, cpd_B])
    >>> ve = VariableEliminationInference(model)
    >>> result = ve.query(query=['B'], evidence={'A': 1})
    >>> result.values  # normalised P(B | A=1)
    """

    def __init__(
        self,
        probabilistic_model: ProbabilisticModel,
        elimination_order: Optional[List[str]] = None,
    ):
        super().__init__()
        self.probabilistic_model = probabilistic_model
        self.elimination_order = elimination_order

    # ------------------------------------------------------------------
    # BaseInference interface
    # ------------------------------------------------------------------

    def query(
        self,
        query: List[str],
        evidence: Optional[Dict[str, int]] = None,
        return_logits: bool = False,
    ) -> Factor:
        """
        Compute the conditional distribution ``P(query | evidence)``.

        Parameters
        ----------
        query : List[str]
            Names of the query variables (𝒴).
        evidence : Dict[str, int], optional
            Mapping from observed variable names to their observed state
            index (0-based).  For example ``{'A': 1}`` means A is observed
            in state 1.
        return_logits : bool, optional
            If ``True``, return log-probabilities (unnormalised) instead
            of normalised probabilities.  Useful during training when the
            loss expects log-scale values.  Default: ``False``.

        Returns
        -------
        Factor
            A factor over the query variables.  When ``return_logits`` is
            ``False`` (default) the values are normalised probabilities
            that sum to 1.  When ``True`` the values are
            log-potentials (unnormalised).
        """
        if evidence is None:
            evidence = {}

        # 1. Build factors from the parametric model
        factors: List[Factor] = self.probabilistic_model.build_factors()

        # 2. Set evidence: replace each factor by its slice
        for var_name, state in evidence.items():
            factors = [
                f.set_evidence(var_name, state) if var_name in f.variables else f
                for f in factors
            ]

        # 3. Determine hidden variables: X - Y - E
        all_vars: set = set()
        for f in factors:
            all_vars.update(f.variables)
        query_set = set(query)
        evidence_set = set(evidence.keys())
        hidden = all_vars - query_set - evidence_set

        # 4. Elimination ordering
        if self.elimination_order is not None:
            # Filter user-provided order to only include actual hidden vars
            elim_order = [v for v in self.elimination_order if v in hidden]
        else:
            elim_order = _min_degree_order(factors, list(hidden))

        # 5. Sum-Product VE
        phi_star = _sum_product_ve(factors, elim_order)

        # 6. Return logits or normalised probabilities
        if return_logits:
            log_values = torch.log(phi_star.values.clamp(min=1e-10))
            return Factor(log_values, phi_star.variables,
                          phi_star.cardinalities)

        _Z, normalised = phi_star.normalize()
        return normalised

    def ground_truth_to_evidence(
        self,
        value: torch.Tensor,
        cardinality: int,
    ) -> torch.Tensor:
        """
        Convert ground-truth labels to state indices.

        For Variable Elimination, evidence is simply integer state indices.
        This method returns the input unchanged (already in the correct
        format).

        Parameters
        ----------
        value : torch.Tensor
            Ground truth tensor with integer state indices.
        cardinality : int
            Number of states for the variable (unused, kept for API
            compatibility).

        Returns
        -------
        torch.Tensor
            The same integer indices.
        """
        return value
