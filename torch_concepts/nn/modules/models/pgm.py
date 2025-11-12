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


class ProbabilisticGraphicalModel(nn.Module):  # 1. Inherit from nn.Module
    def __init__(self, variables: List[Variable], factors: List[Factor]):
        super().__init__()  # Initialize nn.Module base class
        self.variables = variables
        self.factors = factors
        self.concept_to_variable: Dict[str, Variable] = {}
        self.concept_to_factor: Dict[str, Factor] = {}

        # 2. Add a ModuleDict to store the actual PyTorch modules from the factors
        self.factor_modules = nn.ModuleDict()

        self._initialize_model()

    def _initialize_model(self):
        new_variables = []
        new_factors = []

        temp_concept_to_variable: Dict[str, Variable] = {}

        # ... (Variable initialization logic remains the same) ...
        # (Assuming the original Variable initialization is correct and omitted here for brevity)

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

        # New list to temporarily hold new Factor objects
        temp_new_factors = []

        for factor in self.factors:
            original_module = factor.module_class
            # original_module_class = original_module.__class__ # This line isn't used

            if len(factor.concepts) > 1:
                for concept in factor.concepts:
                    # atomic_module = copy.deepcopy(original_module) # Original code

                    # 3. Store the module in ModuleDict using the concept as the key
                    atomic_module = copy.deepcopy(original_module)
                    self.factor_modules[concept] = atomic_module

                    # Create the Factor object with the module
                    atomic_factor = Factor(concepts=[concept], module_class=atomic_module)
                    temp_new_factors.append(atomic_factor)
            else:
                concept = factor.concepts[0]  # Get the single concept
                # 3. Store the module in ModuleDict using the concept as the key
                self.factor_modules[concept] = original_module
                temp_new_factors.append(factor)  # The original factor object is kept

        self.factors = temp_new_factors  # Update the instance's factors list

        # ... (Parent resolution logic remains the same) ...
        # (Assuming the original Parent resolution is correct and omitted here for brevity)

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

        for factor in self.factors:
            if not factor.concepts:
                raise ValueError("Factor must model at least one concept.")

            target_concept = factor.concepts[0]
            target_var = self.concept_to_variable[target_concept]

            factor.variable = target_var
            factor.parents = target_var.parents
            self.concept_to_factor[target_concept] = factor

    def get_by_distribution(self, distribution_class: Type[Distribution]) -> List[Variable]:
        return [var for var in self.variables if var.distribution is distribution_class]

    def get_factor_of_variable(self, concept_name: str) -> Optional[Factor]:
        return self.concept_to_factor.get(concept_name)

    def get_variable_parents(self, concept_name: str) -> List[Variable]:
        var = self.concept_to_variable.get(concept_name)
        if var:
            return var.parents
        return []

    def get_module_of_concept(self, concept_name: str) -> Optional[nn.Module]:
        """Easily get the model (module_class) for a given concept name."""
        return self.concept_to_module.get(concept_name)

    def build_potentials(self):
        potentials = {}
        for factor in self.factors:
            concept = factor.concepts[0]
            potentials[concept] = factor.build_potential()
        return potentials

    def build_cpts(self):
        cpts = {}
        for factor in self.factors:
            concept = factor.concepts[0]
            cpts[concept] = factor.build_cpt()
        return cpts
