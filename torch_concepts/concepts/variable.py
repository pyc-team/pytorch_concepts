import torch
from torch.distributions import Distribution, Bernoulli, Categorical
from typing import List, Dict, Any, Union, Optional, Type

from torch_concepts.distributions import Delta


class Variable:
    def __new__(cls, concepts: List[str], parents: List[Union['Variable', str]],
                distribution: Optional[Union[Type[Distribution], List[Type[Distribution]]]] = None,
                size: Union[int, List[int]] = 1, metadata: Optional[Dict[str, Any]] = None):

        # 1. Handle the case for creating multiple Variable objects (e.g., c1_var, c2_var = Variable([...]))
        if isinstance(concepts, list) and len(concepts) > 1:
            n_concepts = len(concepts)

            # Standardize distribution: single value -> list of N values
            if distribution is None:
                distribution_list = [Delta] * n_concepts
            elif not isinstance(distribution, list):
                distribution_list = [distribution] * n_concepts
            else:
                distribution_list = distribution

            # Standardize size: single value -> list of N values
            if not isinstance(size, list):
                size_list = [size] * n_concepts
            else:
                size_list = size

            # Validation checks for list lengths
            if len(distribution_list) != n_concepts or len(size_list) != n_concepts:
                raise ValueError(
                    "If concepts list has length N > 1, distribution and size must either be single values or lists of length N.")

            # Create and return a list of individual Variable instances
            new_vars = []
            for i in range(n_concepts):
                # Use object.__new__(cls) to bypass this __new__ logic for the sub-creation
                instance = object.__new__(cls)
                instance.__init__(
                    concepts=[concepts[i]],  # Pass as single-element list
                    parents=parents,
                    distribution=distribution_list[i],
                    size=size_list[i],
                    metadata=metadata.copy() if metadata else None
                )
                new_vars.append(instance)
            return new_vars

        # 2. Default: Single instance creation (either from a direct call or a recursive call from step 1)
        return object.__new__(cls)

    def __init__(self, concepts: List[str], parents: List[Union['Variable', str]], distribution: Optional[Type[Distribution]] = None,
                 size: int = 1, metadata: Optional[Dict[str, Any]] = None):

        # Ensure concepts is a list (important if called internally after __new__ splitting)
        if isinstance(concepts, str):
            concepts = [concepts]

        # Original validation logic
        if distribution is None:
            distribution = Delta

        if distribution is Categorical:
            if len(concepts) != 1:
                # This validation is slightly tricky now, but generally still relevant
                # if a single Variable is constructed with multiple concepts and is Categorical.
                pass
            if size <= 1:
                raise ValueError("Categorical Variable must have a size > 1 (number of classes).")

        if distribution is Bernoulli and size != 1:
            raise ValueError("Bernoulli Variable must have size=1 as it represents a binary outcome per concept.")

        self.concepts = concepts
        self.concept_to_var = {c: self for c in concepts}
        self.parents = parents
        self.distribution = distribution
        self.size = size
        self.metadata = metadata if metadata is not None else {}
        self._out_features = None

    @property
    def out_features(self) -> int:
        if self._out_features is not None:
            return self._out_features

        n_concepts = len(self.concepts)
        if self.distribution in [Delta, torch.distributions.Normal]:
            self._out_features = self.size * n_concepts
        elif self.distribution is Bernoulli:
            self._out_features = n_concepts
        elif self.distribution is Categorical:
            self._out_features = self.size
        else:
            self._out_features = self.size * n_concepts

        return self._out_features

    @property
    def in_features(self) -> int:
        total_in = 0
        for parent in self.parents:
            if isinstance(parent, Variable):
                total_in += parent.out_features
            else:
                raise TypeError(f"Parent '{parent}' is not a Variable object. PGM initialization error.")
        return total_in

    def __getitem__(self, key: Union[str, List[str]]) -> 'Variable':
        if isinstance(key, str):
            concepts = [key]
        else:
            concepts = key

        if not all(c in self.concepts for c in concepts):
            raise ValueError(f"Concepts {concepts} not found in variable {self.concepts}")

        if self.distribution is Categorical and len(concepts) != 1:
            raise ValueError(
                "Slicing a Categorical Variable into a new Variable is not supported as it must represent a single, multi-class concept.")

        # This call will hit __new__, but since len(concepts) is <= 1, it proceeds to single instance creation
        new_var = Variable(
            concepts=concepts,
            parents=self.parents,
            distribution=self.distribution,
            size=self.size,
            metadata=self.metadata.copy()
        )
        n_concepts = len(concepts)

        if self.distribution in [Delta, torch.distributions.Normal]:
            new_var._out_features = self.size * n_concepts
        elif self.distribution is Bernoulli:
            new_var._out_features = n_concepts
        elif self.distribution is Categorical:
            new_var._out_features = self.size
        else:
            new_var._out_features = self.size * n_concepts

        return new_var

    def __repr__(self):
        meta_str = f", metadata={self.metadata}" if self.metadata else ""
        return f"Variable(concepts={self.concepts}, dist={self.distribution.__name__}, size={self.size}, out_features={self.out_features}{meta_str})"
