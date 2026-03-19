"""
Probabilistic Model implementation for concept-based architectures.

This module provides :class:`ProbabilisticModel` — a unified factor-graph
container that holds variables and parametric factors.  When initialised with
:class:`ParametricCPD` factors (via the ``parametric_cpds`` keyword), the model
automatically resolves parent references and lazy constructors, acting as a
Bayesian Network.  When initialised with generic :class:`ParametricFactor`
instances (via the ``factors`` keyword), the model stores them without
directed-graph semantics.
"""

from torch import nn
from torch.distributions import Distribution
from typing import List, Dict, Optional, Type

from .variable import Variable, ExogenousVariable, ConceptVariable
from .factor import ParametricFactor
from .cpd import ParametricCPD


# ---------------------------------------------------------------------------
# Unified PGM container
# ---------------------------------------------------------------------------

class ProbabilisticModel(nn.Module):
    """
    Unified factor-graph container for concept-based probabilistic models.

    Stores a set of :class:`Variable` nodes and :class:`ParametricFactor`
    (or :class:`ParametricCPD`) factors.  When ``parametric_cpds`` is used,
    the model behaves as a Bayesian Network — resolving lazy constructors,
    converting string parent references to :class:`Variable` objects, and
    providing helpers for CPT / potential-table construction and parent
    queries.

    Parameters
    ----------
    variables : List[Variable]
        All concept-variables in the model.
    factors : List[ParametricFactor], optional
        Generic (undirected) factors.  Mutually exclusive with
        ``parametric_cpds``.
    parametric_cpds : List[ParametricCPD], optional
        Directed factors (CPDs).  Triggers directed-model initialisation
        (parent resolution, lazy-constructor building).  Mutually exclusive
        with ``factors``.

    Attributes
    ----------
    variables : List[Variable]
        Stored variable list.
    factors : nn.ModuleDict
        Concept-name → factor mapping (registers parameters).
    concept_to_variable : Dict[str, Variable]
        Concept-name → :class:`Variable` lookup.

    Raises
    ------
    TypeError
        If neither ``factors`` nor ``parametric_cpds`` is provided, or if
        both are provided simultaneously.
    """

    def __init__(self, variables: List[Variable],
                 factors: List[ParametricFactor] = None,
                 parametric_cpds: List[ParametricCPD] = None):
        super().__init__()
        if parametric_cpds is not None and factors is not None:
            raise TypeError("Provide either 'factors' or 'parametric_cpds', not both.")
        if parametric_cpds is not None:
            self._is_directed = True
            input_factors = parametric_cpds
        elif factors is not None:
            self._is_directed = False
            input_factors = factors
        else:
            raise TypeError("ProbabilisticModel requires either 'factors' or 'parametric_cpds'.")
        self.variables = variables
        self.factors = nn.ModuleDict()
        self.concept_to_variable: Dict[str, Variable] = {}
        self._initialize_model(input_factors)

    # ---- properties --------------------------------------------------------

    @property
    def parametric_cpds(self) -> nn.ModuleDict:
        """Alias for ``self.factors`` (useful when the model is directed)."""
        return self.factors

    # ---- initialisation ----------------------------------------------------

    def _initialize_model(self, input_factors: List[ParametricFactor]):
        """Build concept→variable mapping and register factors."""
        self.concept_to_variable = {var.concept: var for var in self.variables}

        if self._is_directed:
            self._initialize_directed(input_factors)
        else:
            for factor in input_factors:
                concept = factor.concept
                if concept in self.concept_to_variable:
                    factor.variable = self.concept_to_variable[concept]
                self.factors[concept] = factor

    def _initialize_directed(self, input_factors: List[ParametricFactor]):
        """Directed-model initialisation: lazy constructors + parent resolution."""
        from ...low.lazy import LazyConstructor

        for cpd in input_factors:
            concept = cpd.concept
            if concept in self.concept_to_variable:
                cpd.variable = self.concept_to_variable[concept]

            if isinstance(cpd.parametrization, LazyConstructor):
                parent_vars = self._resolve_parent_refs(cpd.parents)
                in_concepts = in_exogenous = in_latent = 0
                for pv in parent_vars:
                    if isinstance(pv, ExogenousVariable):
                        in_exogenous = pv.size
                    elif isinstance(pv, ConceptVariable):
                        in_concepts += pv.size
                    else:
                        in_latent += pv.size

                out_concepts = (1 if isinstance(cpd.variable, ExogenousVariable)
                                else self.concept_to_variable[concept].size)

                initialized_layer = cpd.parametrization.build(
                    in_latent=in_latent,
                    in_concepts=in_concepts,
                    in_exogenous=in_exogenous,
                    out_concepts=out_concepts,
                )
                new_cpd = ParametricCPD(
                    concepts=concept,
                    parametrization=initialized_layer,
                    parents=cpd.parents,
                )
                new_cpd.variable = cpd.variable
                cpd = new_cpd

            self.factors[concept] = cpd

        # resolve string parent references to Variable objects
        for concept, cpd in self.factors.items():
            cpd.parents = self._resolve_parent_refs(cpd.parents)

    def _resolve_parent_refs(self, parents: list) -> List[Variable]:
        """Resolve a mixed list of Variable / str references to Variables."""
        resolved = []
        for ref in parents:
            if isinstance(ref, str):
                if ref not in self.concept_to_variable:
                    raise ValueError(f"Parent concept '{ref}' not found in any variable.")
                resolved.append(self.concept_to_variable[ref])
            elif isinstance(ref, Variable):
                resolved.append(ref)
            elif hasattr(ref, 'concept'):
                resolved.append(self.concept_to_variable[ref.concept])
            else:
                raise TypeError(f"Invalid parent reference type: {type(ref)}")
        # deduplicate while preserving order
        return list({id(p): p for p in resolved}.values())

    # ---- queries -----------------------------------------------------------

    def get_by_distribution(self, distribution_class: Type[Distribution]) -> List[Variable]:
        """Return all variables with a given distribution type."""
        return [var for var in self.variables if var.distribution is distribution_class]

    def get_module_of_concept(self, concept_name: str) -> Optional[ParametricFactor]:
        """Return the factor for *concept_name*, or ``None``."""
        return self.factors[concept_name] if concept_name in self.factors else None

    def get_variable_parents(self, concept_name: str) -> List[Variable]:
        """Return the parent variables of a concept (empty if none / undirected)."""
        cpd = self.factors[concept_name] if concept_name in self.factors else None
        return cpd.parents if cpd is not None and hasattr(cpd, 'parents') else []

    # ---- CPT / potential-table helpers (directed models) -------------------

    def _make_temp_parametric_cpd(self, concept: str, module: nn.Module) -> ParametricCPD:
        """Create a temporary ParametricCPD for table-building helpers."""
        if isinstance(module, ParametricCPD):
            parametrization = module.parametrization
        else:
            parametrization = module
        f = ParametricCPD(concepts=concept, parametrization=parametrization)
        f.variable = self.concept_to_variable[concept]
        stored = self.factors[concept] if concept in self.factors else None
        f.parents = stored.parents if stored is not None else []
        return f

    def build_potentials(self):
        """Build potential tables for all concepts."""
        return {
            concept: self._make_temp_parametric_cpd(concept, module).build_potential()
            for concept, module in self.factors.items()
        }

    def build_cpts(self):
        """Build Conditional Probability Tables for all concepts."""
        return {
            concept: self._make_temp_parametric_cpd(concept, module).build_cpt()
            for concept, module in self.factors.items()
        }
