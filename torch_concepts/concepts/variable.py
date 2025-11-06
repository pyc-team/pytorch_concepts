import torch
from torch.distributions import Distribution, Bernoulli, Categorical
from typing import List, Dict, Any, Union, Optional, Type

from torch_concepts.distributions import Delta


class Variable:
    def __init__(self, concepts: List[str], parents: List[Union['Variable', str]], distribution: Type[Distribution],
                 size: int = 1, metadata: Optional[Dict[str, Any]] = None):

        if distribution is Categorical:
            if len(concepts) != 1:
                raise ValueError("Categorical Variable must have exactly 1 concept string in the concepts list.")
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
