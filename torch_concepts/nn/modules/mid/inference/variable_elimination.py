"""
Variable Elimination inference for probabilistic graphical models.

This module implements the Sum-Product Variable Elimination algorithm for
computing exact conditional probabilities in both Bayesian Networks and
Markov Random Fields.  All operations are differentiable so that gradients
flow back through the neural-network parameters that produced the factor
potentials — enabling end-to-end training through inference.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..models.factor import Factor
from ..models.probabilistic_model import ProbabilisticModel
from ..models.variable import _BINARY_DISTRIBUTIONS, _CATEGORICAL_DISTRIBUTIONS
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

    # Pre-build adjacency: for each variable, the set of other
    # variables that share at least one factor with it.
    adj: dict = {v: set() for v in remaining}
    for f in factors:
        scope = [v for v in f.variables if v in remaining]
        for v in scope:
            for u in scope:
                if u != v:
                    adj[v].add(u)

    for _ in range(len(variables_to_eliminate)):
        # Pick the remaining variable with the fewest active neighbours
        best = min(remaining,
                   key=lambda v: sum(1 for u in adj[v] if u in remaining))
        order.append(best)
        remaining.remove(best)

    return order


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
    >>>
    >>> # With input embedding:
    >>> result = ve.query(query=['B'], evidence={'input': x, 'A': 1})
    """

    def __init__(
        self,
        probabilistic_model: ProbabilisticModel,
        elimination_order: Optional[List[str]] = None,
    ):
        super().__init__()
        self.probabilistic_model = probabilistic_model
        self.elimination_order = elimination_order
        self._order_cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], List[str]] = {}

    # ------------------------------------------------------------------
    # BaseInference interface
    # ------------------------------------------------------------------

    def query(
        self,
        query: List[str],
        evidence: Optional[Dict[str, int]] = None,
        return_logits: bool = False,
        return_log_joint: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the conditional distribution ``P(query | evidence)``.

        Parameters
        ----------
        query : List[str]
            Names of the query variables.
        evidence : dict, optional
            Mapping from observed variable names to their observed state
            index (0-based).  For example ``{'A': 1}`` means A is observed
            in state 1.  The special key ``'input'`` may hold the input
            embedding tensor of shape ``(batch, emb_dim)``.  When present,
            each factor is conditioned on the input, yielding per-sample
            distributions.
        return_logits : bool, optional
            If ``True``, return unnormalised log-potentials instead
            of normalised probabilities.  Useful during training when the
            loss expects log-scale values.  Default: ``False``.
        return_log_joint : bool, optional
            If ``True``, return a dict with keys ``'log_joint'`` (the
            log of the normalised joint distribution, shape
            ``(batch, *cardinalities)``) and ``'logits'`` (per-concept
            marginal logits, shape ``(batch, n_features)``).  Intended
            for use with :class:`JointNLLLoss`.  Takes precedence over
            ``return_logits``.

        Returns
        -------
        torch.Tensor or dict
            When ``return_log_joint=True``: a dict with ``'log_joint'``
            and ``'logits'``.  Otherwise a ``(batch, n_features)`` tensor
            (or ``(n_features,)`` when unbatched).
        """
        if evidence is None:
            evidence = {}

        # Extract input embedding from evidence (if present)
        input_tensor = evidence.pop('input', None) if isinstance(
            evidence, dict) else None

        # 1. Build factors from the parametric model
        # If input_tensor is not None, each factor will be conditioned on it.
        factors: List[Factor] = self.probabilistic_model.build_factors(
            input=input_tensor)

        # 2. Set evidence: replace each factor by its slice
        for var_name, state in evidence.items():
            factors = [
                f.set_evidence(var_name, state) if var_name in f.variables else f
                for f in factors
            ]

        # 3. Determine hidden variables
        all_vars: set = set()
        for f in factors:
            all_vars.update(f.variables)
        query_set = set(query)
        evidence_set = set(evidence.keys())
        hidden = all_vars - query_set - evidence_set

        # 4. Elimination ordering (cached by query/evidence pattern)
        cache_key = (tuple(sorted(query)), tuple(sorted(evidence.keys())))
        if cache_key in self._order_cache:
            elim_order = self._order_cache[cache_key]
        elif self.elimination_order is not None:
            # Filter user-provided order to only include actual hidden vars
            elim_order = [v for v in self.elimination_order if v in hidden]
            self._order_cache[cache_key] = elim_order
        else:
            elim_order = _min_degree_order(factors, list(hidden))
            self._order_cache[cache_key] = elim_order

        # 5. Sum-Product VE
        phi_star = self._sum_product_ve(factors, elim_order)

        # 6. Normalise
        _Z, normalised = phi_star.normalize()

        # 7. Return as log-joint dict or flat Tensor
        if return_log_joint:
            log_joint = torch.log(normalised.values.clamp(min=1e-10))
            logits = self._factor_to_tensor(normalised, query,
                                            return_logits=True)
            return {'log_joint': log_joint, 'logits': logits}

        return self._factor_to_tensor(normalised, query, return_logits)

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

    # ------------------------------------------------------------------
    # Factor → Tensor conversion
    # ------------------------------------------------------------------

    def _factor_to_tensor(
        self,
        joint: Factor,
        query: List[str],
        return_logits: bool,
    ) -> torch.Tensor:
        """
        Convert a normalised joint factor into a flat tensor.

        For each query variable the joint is marginalised to a univariate
        distribution and then converted to either a logit or a
        probability column:

        * **Binary** (cardinality 2): one column - the logit
          ``log P(v=1) - log P(v=0)`` when *return_logits* is ``True``,
          or ``P(v=1)`` otherwise.
        * **Categorical** (cardinality K): *K* columns - the log-probs
          when *return_logits* is ``True``, or the probabilities otherwise.

        All marginals are computed directly on the raw tensor with
        :func:`torch.sum` - no intermediate :class:`Factor` objects are
        created.

        Parameters
        ----------
        joint : Factor
            Normalised factor over the query variables (possibly batched).
        query : List[str]
            Ordered concept names.
        return_logits : bool
            Whether to return logits or probabilities.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_features)`` when batched, else
            ``(n_features,)``.
        """
        concept_to_var = self.probabilistic_model.concept_to_variable
        values = joint.values                        # (batch?, *var_cards)
        offset = 1 if joint.batched else 0
        eps = 1e-10
        columns: List[torch.Tensor] = []

        for var_name in query:
            # Compute marginal by summing over all dims except batch + this var
            var_dim = joint.variables.index(var_name) + offset
            sum_dims = [d for d in range(offset, values.ndim) if d != var_dim]
            probs = values.sum(dim=sum_dims) if sum_dims else values
            # Re-normalise (floating-point drift)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            var = concept_to_var[var_name]
            if var.distribution in _BINARY_DISTRIBUTIONS:
                if return_logits:
                    col = (torch.log(probs[..., 1].clamp(min=eps))
                           - torch.log(probs[..., 0].clamp(min=eps)))
                else:
                    col = probs[..., 1]
                columns.append(col.unsqueeze(-1))
            elif var.distribution in _CATEGORICAL_DISTRIBUTIONS:
                if return_logits:
                    columns.append(torch.log(probs.clamp(min=eps)))
                else:
                    columns.append(probs)
            else:
                raise NotImplementedError(
                    f"_factor_to_tensor: unsupported distribution "
                    f"{var.distribution.__name__} for variable '{var_name}'."
                )

        return torch.cat(columns, dim=-1)


    # ──────────────────────────────────────────────────────────────────────
    # Core VE routines
    # ──────────────────────────────────────────────────────────────────────

    def _sum_product_ve(
        self,
        factors: List[Factor],
        elimination_order: List[str],
    ) -> Factor:
        """
        Run Sum-Product Variable Elimination.

        Uses a variable-to-factor index for O(degree) factor lookup
        per elimination step instead of scanning all factors.

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
        # Build variable → factor-id index
        var_to_fids: Dict[str, set] = defaultdict(set)
        fid_to_factor: Dict[int, Factor] = {}
        for i, f in enumerate(factors):
            fid_to_factor[i] = f
            for v in f.variables:
                var_to_fids[v].add(i)
        next_fid = len(factors)

        for var in elimination_order:
            # Collect factors mentioning var — O(degree) via index
            fids = var_to_fids.pop(var, set())
            phi_prime: List[Factor] = []
            for fid in fids:
                f = fid_to_factor.pop(fid, None)
                if f is not None:
                    phi_prime.append(f)
                    # Remove this factor from index entries of other variables
                    for v in f.variables:
                        if v != var and v in var_to_fids:
                            var_to_fids[v].discard(fid)

            if not phi_prime:
                continue

            # Multiply all factors that contain the variable
            psi = phi_prime[0]
            for f in phi_prime[1:]:
                psi = psi.product(f)

            # Marginalise out the variable
            tau = psi.marginalize(var)

            # Register the new factor in the index
            fid_to_factor[next_fid] = tau
            for v in tau.variables:
                var_to_fids[v].add(next_fid)
            next_fid += 1

        # Multiply remaining factors
        remaining = list(fid_to_factor.values())
        if not remaining:
            raise RuntimeError("No factors remain after variable elimination.")

        result = remaining[0]
        for f in remaining[1:]:
            result = result.product(f)

        return result
