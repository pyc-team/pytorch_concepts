"""
Probabilistic Model implementation for concept-based architectures.

This module provides :class:`ProbabilisticModel` — a unified factor-graph
container that holds variables and parametric factors.  When the factors list
contains :class:`ParametricCPD` instances, the model automatically resolves
parent references and lazy constructors, acting as a Bayesian Network.  When
the factors are plain :class:`ParametricFactor` instances, the model stores
them without directed-graph semantics.
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
    (or :class:`ParametricCPD`) factors.  The model inspects the factor types
    at construction time:

    * If all factors are :class:`ParametricCPD` — the model behaves as a
      Bayesian Network (resolving lazy constructors, string parent references,
      and providing CPT / potential-table helpers).
    * If all factors are plain :class:`ParametricFactor` — the model stores
      them without directed-graph semantics.
    * Mixing both types raises a :class:`TypeError`.

    Parameters
    ----------
    variables : List[Variable]
        All concept-variables in the model.
    factors : List[ParametricFactor]
        The factors (either all :class:`ParametricCPD` or all plain
        :class:`ParametricFactor`).

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
        If the factors list mixes :class:`ParametricCPD` and plain
        :class:`ParametricFactor` instances.
    """

    def __init__(self, variables: List[Variable],
                 factors: List[ParametricFactor]):
        super().__init__()
        has_cpds = any(isinstance(f, ParametricCPD) for f in factors)
        has_plain = any(not isinstance(f, ParametricCPD) for f in factors)
        if has_cpds and has_plain:
            raise TypeError(
                "All factors must be the same type: either all ParametricCPD "
                "or all ParametricFactor, not a mix of both."
            )
        self._is_directed = has_cpds
        self.variables = variables
        self.factors = nn.ModuleDict()
        self.concept_to_variable: Dict[str, Variable] = {}
        self._initialize_model(factors)

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
