"""
Probabilistic Model implementation for concept-based architectures.

This module provides a framework for building and managing probabilistic models over concepts.
"""
import inspect

from torch import nn
from torch.distributions import Distribution
from typing import List, Dict, Optional, Type

from torch_concepts.nn import LazyConstructor
from .variable import Variable, ExogenousVariable, ConceptVariable, LatentVariable
from .cpd import ParametricCPD


class ProbabilisticModel(nn.Module):
    """
    Probabilistic Model for concept-based reasoning.

    This class represents a directed acyclic graph (DAG) where nodes are concept
    variables and edges represent probabilistic dependencies. Each variable has
    an associated CPD (neural network module) that computes its conditional
    probability given its parents.

    Attributes:
        variables (List[Variable]): List of concept variables in the model.
        parametric_cpds (nn.ModuleDict): Dictionary mapping concept names to their CPDs.
        concept_to_variable (Dict[str, Variable]): Mapping from concept names to variables.

    Args:
        variables: List of Variable objects defining the concepts.
        parametric_cpds: List of ParametricCPD objects defining the conditional distributions.

    Example:
        >>> import torch
        >>> from torch_concepts import LatentVariable, ConceptVariable
        >>> from torch_concepts.nn import ProbabilisticModel
        >>> from torch_concepts.nn import ParametricCPD
        >>> from torch_concepts.nn import LinearLatentToConcept
        >>> from torch_concepts.nn import LinearConceptToConcept
        >>> from torch_concepts.distributions import Delta
        >>>
        >>> # Define variables
        >>> emb_var = LatentVariable(concepts='input', parents=[], distribution=Delta, size=32)
        >>> c1_var = ConceptVariable(concepts='c1', parents=[emb_var], distribution=Delta, size=1)
        >>> c2_var = ConceptVariable(concepts='c2', parents=[c1_var], distribution=Delta, size=1)
        >>>
        >>> # Define CPDs (neural network modules)
        >>> backbone = torch.nn.Linear(in_features=128, out_features=32)
        >>> encoder = LinearLatentToConcept(in_latent=32, out_features=1)
        >>> predictor = LinearConceptToConcept(in_concepts=1, out_features=1)
        >>>
        >>> parametric_cpds = [
        ...     ParametricCPD(concepts='input', parametrization=backbone),
        ...     ParametricCPD(concepts='c1', parametrization=encoder),
        ...     ParametricCPD(concepts='c2', parametrization=predictor)
        ... ]
        >>>
        >>> # Create ProbabilisticModel
        >>> probabilistic_model = ProbabilisticModel(
        ...     variables=[emb_var, c1_var, c2_var],
        ...     parametric_cpds=parametric_cpds
        ... )
        >>>
        >>> print(f"Number of variables: {len(probabilistic_model.variables)}")
        Number of variables: 3
    """
    def __init__(self, variables: List[Variable], parametric_cpds: List[ParametricCPD]):
        super().__init__()
        self.variables = variables

        # single source of truth: concept -> module
        self.parametric_cpds = nn.ModuleDict()

        self.concept_to_variable: Dict[str, Variable] = {}

        # initialize using the input CPDs list; we don't store that list
        self._initialize_model(parametric_cpds)

    def _initialize_model(self, input_parametric_cpds: List[ParametricCPD]):
        """
        Initialize the ProbabilisticModel by splitting multi-concept variables and resolving parents.

        This internal method processes the input variables and CPDs to create
        an atomic representation where each variable represents a single concept.

        Args:
            input_parametric_cpds: List of ParametricCPD objects to initialize.
        """
        # Build concept_to_variable mapping (each Variable has exactly one concept)
        self.concept_to_variable: Dict[str, Variable] = {
            var.concept: var for var in self.variables
        }

        # ---- ParametricCPD modules: fill only self.parametric_cpds (ModuleDict) ----
        for parametric_cpd in input_parametric_cpds:
            concept = parametric_cpd.concept
            # Link the parametric_cpd to its variable
            if concept in self.concept_to_variable:
                parametric_cpd.variable = self.concept_to_variable[concept]
                parametric_cpd.parents = self.concept_to_variable[concept].parents

            if isinstance(parametric_cpd.parametrization, LazyConstructor):
                # parametric_cpd.parents is a list of Variable objects (or strings)
                # We need to get the actual Variable objects to compute in_features
                parent_vars = []
                for parent_ref in parametric_cpd.parents:
                    if isinstance(parent_ref, str):
                        # Parent is a concept name string
                        parent_vars.append(self.concept_to_variable[parent_ref])
                    elif hasattr(parent_ref, 'concept'):
                        # Parent is a Variable object, use its concept name to look up
                        parent_concept = parent_ref.concept
                        parent_vars.append(self.concept_to_variable[parent_concept])
                    else:
                        raise ValueError(f"Unknown parent type: {type(parent_ref)}")
                in_concepts = in_exogenous = in_latent = 0
                for pv in parent_vars:
                    if isinstance(pv, ExogenousVariable):
                        in_exogenous = pv.size
                    elif isinstance(pv, ConceptVariable):
                        in_concepts += pv.size
                    else:
                        in_latent += pv.size

                if isinstance(parametric_cpd.variable, ExogenousVariable):
                    out_concepts = 1
                else:
                    out_concepts = self.concept_to_variable[concept].size

                initialized_layer = parametric_cpd.parametrization.build(
                    in_latent=in_latent,
                    in_concepts=in_concepts,
                    in_exogenous=in_exogenous,
                    out_concepts=out_concepts,
                )
                new_parametrization = ParametricCPD(concepts=concept, parametrization=initialized_layer)
                # Copy parents and variable from old CPD to new CPD
                new_parametrization.parents = parametric_cpd.parents
                new_parametrization.variable = parametric_cpd.variable
            else:
                new_parametrization = parametric_cpd

            self.parametric_cpds[concept] = new_parametrization

        # ---- Parent resolution (unchanged) ----
        for var in self.variables:
            resolved_parents = []
            for parent_ref in var.parents:
                if isinstance(parent_ref, str):
                    if parent_ref not in self.concept_to_variable:
                        raise ValueError(f"Parent concept '{parent_ref}' not found in any variable.")
                    resolved_parents.append(self.concept_to_variable[parent_ref])
                elif isinstance(parent_ref, Variable):
                    resolved_parents.append(parent_ref)
                else:
                    raise TypeError(f"Invalid parent reference type: {type(parent_ref)}")

            var.parents = list({id(p): p for p in resolved_parents}.values())

    def get_by_distribution(self, distribution_class: Type[Distribution]) -> List[Variable]:
        """
        Get all variables with a specific distribution type.

        Args:
            distribution_class: The distribution class to filter by.

        Returns:
            List[Variable]: Variables using the specified distribution.
        """
        return [var for var in self.variables if var.distribution is distribution_class]

    # concept_to_parametric_cpd removed; if you need the module, use the method below
    def get_variable_parents(self, concept_name: str) -> List[Variable]:
        """
        Get the parent variables of a concept.

        Args:
            concept_name: Name of the concept to query.

        Returns:
            List[Variable]: List of parent variables, or empty list if none.
        """
        var = self.concept_to_variable.get(concept_name)
        return var.parents if var else []

    def get_module_of_concept(self, concept_name: str) -> Optional[nn.Module]:
        """
        Return the neural network module for a given concept.

        Args:
            concept_name: Name of the concept.

        Returns:
            Optional[nn.Module]: The parametric_cpd module for the concept, or None if not found.
        """
        return self.parametric_cpds[concept_name] if concept_name in self.parametric_cpds else None

    def _make_temp_parametric_cpd(self, concept: str, module: nn.Module) -> ParametricCPD:
        """
        Create a temporary ParametricCPD object for internal use.

        Small helper to reuse existing ParametricCPD.build_* logic without keeping a ParametricCPD list.

        Args:
            concept: Concept name.
            module: Neural network module or ParametricCPD instance.

        Returns:
            ParametricCPD: Temporary parametric_cpd object.
        """
        # module may be either an nn.Module (the parametrization) or a ParametricCPD
        if isinstance(module, ParametricCPD):
            parametrization = module.parametrization
        else:
            parametrization = module

        f = ParametricCPD(concepts=concept, parametrization=parametrization)
        target_var = self.concept_to_variable[concept]
        f.variable = target_var
        f.parents = target_var.parents
        return f

    def build_potentials(self):
        """
        Build potential functions for all concepts in the ProbabilisticModel.

        Returns:
            Dict[str, callable]: Dictionary mapping concept names to their potential functions.
        """
        potentials = {}
        for concept, module in self.parametric_cpds.items():
            temp_parametric_cpd = self._make_temp_parametric_cpd(concept, module)
            potentials[concept] = temp_parametric_cpd.build_potential()
        return potentials

    def build_cpts(self):
        """
        Build Conditional Probability Tables (CPTs) for all concepts.

        Returns:
            Dict[str, callable]: Dictionary mapping concept names to their CPT functions.
        """
        cpts = {}
        for concept, module in self.parametric_cpds.items():
            temp_parametric_cpd = self._make_temp_parametric_cpd(concept, module)
            cpts[concept] = temp_parametric_cpd.build_cpt()
        return cpts
