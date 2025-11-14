import copy
import inspect

from torch import nn
from torch.distributions import Distribution
from typing import List, Dict, Optional, Type

from ....concepts.variable import Variable
from .factor import Factor


def _reinitialize_with_new_param(instance, key, new_value):
    """
    Creates a new instance of the same class, retaining all current init
    parameters except the one specified by 'key', which gets 'new_value'.
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
            new_dict[k] = getattr(instance, k, None)

    new_instance = cls(**new_dict)

    return new_instance


class ProbabilisticGraphicalModel(nn.Module):
    def __init__(self, variables: List[Variable], factors: List[Factor]):
        super().__init__()
        self.variables = variables

        # single source of truth: concept -> module
        self.factors = nn.ModuleDict()

        self.concept_to_variable: Dict[str, Variable] = {}

        # initialize using the input factors list; we don't store that list
        self._initialize_model(factors)

    def _initialize_model(self, input_factors: List[Factor]):
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

        # ---- Factor modules: fill only self.factors (ModuleDict) ----
        for factor in input_factors:
            original_module = factor.module_class
            if len(factor.concepts) > 1:
                for concept in factor.concepts:
                    self.factors[concept] = copy.deepcopy(original_module)
            else:
                concept = factor.concepts[0]
                self.factors[concept] = factor

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
        return [var for var in self.variables if var.distribution is distribution_class]

    # concept_to_factor removed; if you need the module, use the method below
    def get_variable_parents(self, concept_name: str) -> List[Variable]:
        var = self.concept_to_variable.get(concept_name)
        return var.parents if var else []

    def get_module_of_concept(self, concept_name: str) -> Optional[nn.Module]:
        """Return the nn.Module for a given concept name."""
        return self.factors[concept_name] if concept_name in self.factors else None

    def _make_temp_factor(self, concept: str, module: nn.Module) -> Factor:
        """
        Small helper to reuse existing Factor.build_* logic without keeping a Factor list.
        """
        f = Factor(concepts=[concept], module_class=module)
        target_var = self.concept_to_variable[concept]
        f.variable = target_var
        f.parents = target_var.parents
        return f

    def build_potentials(self):
        potentials = {}
        for concept, module in self.factors.items():
            temp_factor = self._make_temp_factor(concept, module)
            potentials[concept] = temp_factor.build_potential()
        return potentials

    def build_cpts(self):
        cpts = {}
        for concept, module in self.factors.items():
            temp_factor = self._make_temp_factor(concept, module)
            cpts[concept] = temp_factor.build_cpt()
        return cpts
