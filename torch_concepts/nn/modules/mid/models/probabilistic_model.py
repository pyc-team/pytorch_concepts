"""
Probabilistic Model implementation for concept-based architectures.

This module provides a framework for building and managing probabilistic models over concepts.
"""
import inspect

from torch import nn
from torch.distributions import Distribution
from typing import List, Dict, Optional, Type

from torch_concepts.nn import LazyConstructor
from .variable import Variable, ExogenousVariable, EndogenousVariable, InputVariable
from .cpd import ParametricCPD


def _reinitialize_with_new_param(instance, key, new_value):
    """
    Create a new instance with one parameter changed.

    Creates a new instance of the same class, retaining all current initialization
    parameters except the one specified by 'key', which gets 'new_value'.

    Args:
        instance: The instance to recreate with modified parameters.
        key: The parameter name to change.
        new_value: The new value for the specified parameter.

    Returns:
        A new instance with the modified parameter.
    """
    cls = instance.__class__

    # 1. Get current state (attributes) and create a dictionary of arguments
    # 2. Update the specific parameter
    # 3. Create a new instance

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    allowed = {
        name for name, p in params.items()
        if name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    new_dict = {}
    for k in allowed:
        if k == key:
            new_dict[k] = new_value
        else:
            if k == 'bias':
                new_dict[k] = False if instance.bias is None else True
            else:
                new_dict[k] = getattr(instance, k, None)

    new_instance = cls(**new_dict)

    return new_instance


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
        >>> from torch_concepts import InputVariable, EndogenousVariable
        >>> from torch_concepts.nn import ProbabilisticModel
        >>> from torch_concepts.nn import ParametricCPD
        >>> from torch_concepts.nn import LinearZC
        >>> from torch_concepts.nn import LinearCC
        >>> from torch_concepts.distributions import Delta
        >>>
        >>> # Define variables
        >>> emb_var = InputVariable(concepts='input', parents=[], distribution=Delta, size=32)
        >>> c1_var = EndogenousVariable(concepts='c1', parents=[emb_var], distribution=Delta, size=1)
        >>> c2_var = EndogenousVariable(concepts='c2', parents=[c1_var], distribution=Delta, size=1)
        >>>
        >>> # Define CPDs (neural network modules)
        >>> backbone = torch.nn.Linear(in_features=128, out_features=32)
        >>> encoder = LinearZC(in_features=32, out_features=1)
        >>> predictor = LinearCC(in_features_endogenous=1, out_features=1)
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
        new_variables = []
        temp_concept_to_variable: Dict[str, Variable] = {}

        # ---- Variable splitting (unchanged) ----
        for var in self.variables:
            if len(var.concepts) > 1:
                for concept in var.concepts:
                    atomic_var = var[[concept]]
                    atomic_var.parents = var.parents
                    atomic_var.metadata = var.metadata.copy()
                    new_variables.append(atomic_var)
                    temp_concept_to_variable[concept] = atomic_var
            else:
                new_variables.append(var)
                temp_concept_to_variable[var.concepts[0]] = var

        self.variables = new_variables
        self.concept_to_variable = temp_concept_to_variable

        # ---- ParametricCPD modules: fill only self.parametric_cpds (ModuleDict) ----
        for parametric_cpd in input_parametric_cpds:
            for concept in parametric_cpd.concepts:
                # Link the parametric_cpd to its variable
                if concept in self.concept_to_variable:
                    parametric_cpd.variable = self.concept_to_variable[concept]
                    parametric_cpd.parents = self.concept_to_variable[concept].parents

                if isinstance(parametric_cpd.parametrization, LazyConstructor):
                    parent_vars = [self.concept_to_variable[parent_ref] for parent_ref in parametric_cpd.variable.parents]
                    in_features_endogenous = in_features_exogenous = in_features = 0
                    for pv in parent_vars:
                        if isinstance(pv, ExogenousVariable):
                            in_features_exogenous = pv.size
                        elif isinstance(pv, EndogenousVariable):
                            in_features_endogenous += pv.size
                        else:
                            in_features += pv.size

                    if isinstance(parametric_cpd.variable, ExogenousVariable):
                        out_features = 1
                    else:
                        out_features = self.concept_to_variable[concept].size

                    initialized_layer = parametric_cpd.parametrization.build(
                        in_features=in_features,
                        in_features_endogenous=in_features_endogenous,
                        in_features_exogenous=in_features_exogenous,
                        out_features=out_features,
                    )
                    new_parametrization = ParametricCPD(concepts=[concept], parametrization=initialized_layer)
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

        f = ParametricCPD(concepts=[concept], parametrization=parametrization)
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
