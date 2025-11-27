"""
Variable representation for concept-based Probabilistic Models.

This module defines the Variable class, which represents random variables in
concept-based models. Variables can have different probability distributions
and support hierarchical concept structures.
"""
import torch
from torch.distributions import Distribution, Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical
from typing import List, Dict, Any, Union, Optional, Type

from .....distributions import Delta


class Variable:
    """
    Represents a random variable in a concept-based Probabilistic Model.

    A Variable encapsulates one or more concepts along with their associated
    probability distribution, parent variables, and metadata. It supports
    multiple distribution types including Delta (deterministic), Bernoulli,
    Categorical, and Normal distributions.

    The Variable class implements a special __new__ method that allows creating
    multiple Variable instances when initialized with multiple concepts, or a
    single instance for a single concept.

    Attributes:
        concepts (List[str]): List of concept names represented by this variable.
        parents (List[Variable]): List of parent variables in the graphical model.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Size/cardinality of the variable (e.g., number of classes for Categorical).
        metadata (Dict[str, Any]): Additional metadata associated with the variable.

    Properties:
        out_features (int): Number of output features this variable produces.
        in_features (int): Total input features from all parent variables.

    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli, Categorical, Normal
        >>> from torch_concepts import Variable
        >>> from torch_concepts.distributions import Delta
        >>>
        >>> # Create a binary concept variable
        >>> var_binary = Variable(
        ...     concepts='has_wheels',
        ...     parents=[],
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> print(var_binary.concepts)  # ['has_wheels']
        >>> print(var_binary.out_features)  # 1
        >>>
        >>> # Create a categorical variable with 3 color classes
        >>> var_color = Variable(
        ...     concepts=['color'],
        ...     parents=[],
        ...     distribution=Categorical,
        ...     size=3  # red, green, blue
        ... )
        >>> print(var_color.out_features)  # 3
        >>>
        >>> # Create a deterministic (Delta) variable
        >>> var_delta = Variable(
        ...     concepts=['continuous_feature'],
        ...     parents=[],
        ...     distribution=Delta,
        ...     size=1
        ... )
        >>>
        >>> # Create multiple variables at once
        >>> vars_list = Variable(
        ...     concepts=['A', 'B', 'C'],
        ...     parents=[],
        ...     distribution=Delta,
        ...     size=1
        ... )
        >>> print(len(vars_list))  # 3
        >>> print(vars_list[0].concepts)  # ['A']
        >>> print(vars_list[1].concepts)  # ['B']
        >>>
        >>> # Create variables with parent dependencies
        >>> parent_var = Variable(
        ...     concepts=['parent_concept'],
        ...     parents=[],
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> child_var = Variable(
        ...     concepts=['child_concept'],
        ...     parents=[parent_var],
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> print(child_var.in_features)  # 1 (from parent)
        >>> print(child_var.out_features)  # 1
    """

    def __new__(cls, concepts: Union[List[str]], parents: List[Union['Variable', str]],
                distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                size: Union[int, List[int]] = 1, metadata: Optional[Dict[str, Any]] = None):
        """
        Create new Variable instance(s).

        If concepts is a list with multiple elements, returns a list of Variable
        instances (one per concept). Otherwise, returns a single Variable instance.

        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances.
            distribution: Distribution type or list of distribution types.
            size: Size parameter(s) for the distribution.
            metadata: Optional metadata dictionary.

        Returns:
            Variable instance or list of Variable instances.
        """
        if isinstance(concepts, str):
            assert not isinstance(distribution, list)
            assert isinstance(size, int)
            return object.__new__(cls)

        n_concepts = len(concepts)

        # If single concept in list, normalize parameters and return single instance
        if n_concepts == 1:
            # This will return a new instance and Python will automatically call __init__
            # We don't call __init__ manually - just return the instance
            return object.__new__(cls)

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

    def __init__(self, concepts: Union[str, List[str]],
                 parents: List[Union['Variable', str]],
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int]] = 1,
                 metadata: Dict[str, Any] = None):
        """
        Initialize a Variable instance.

        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances.
            distribution: Distribution type (Delta, Bernoulli, Categorical, or Normal).
            size: Size parameter for the distribution.
            metadata: Optional metadata dictionary.

        Raises:
            ValueError: If Categorical variable doesn't have size > 1.
            ValueError: If Bernoulli variable doesn't have size=1.
        """
        # Ensure concepts is a list (important if called internally after __new__ splitting)
        if isinstance(concepts, str):
            concepts = [concepts]

        # Handle case where distribution/size are lists with single element (for single concept)
        if len(concepts) == 1:
            if isinstance(distribution, list) and len(distribution) == 1:
                distribution = distribution[0]
            if isinstance(size, list) and len(size) == 1:
                size = size[0]

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
        """
        Calculate the number of output features for this variable.

        The calculation depends on the distribution type:
        - Delta/Normal: size * n_concepts
        - Bernoulli: n_concepts (binary per concept)
        - Categorical: size (single multi-class variable)

        Returns:
            int: Number of output features.
        """
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
        """
        Calculate total input features from all parent variables.

        Returns:
            int: Sum of out_features from all parent variables.

        Raises:
            TypeError: If any parent is not a Variable instance.
        """
        total_in = 0
        for parent in self.parents:
            if isinstance(parent, Variable):
                total_in += parent.out_features
            else:
                raise TypeError(f"Parent '{parent}' is not a Variable object. ProbabilisticModel initialization error.")
        return total_in

    def __getitem__(self, key: Union[str, List[str]]) -> 'Variable':
        """
        Slice the variable to create a new variable with subset of concepts.

        Args:
            key: Single concept name or list of concept names.

        Returns:
            Variable: New variable instance with specified concepts.

        Raises:
            ValueError: If concepts not found in this variable.
            ValueError: If slicing a Categorical variable with multiple concepts.
        """
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
        elif self.distribution in [Bernoulli, RelaxedBernoulli]:
            new_var._out_features = n_concepts
        elif self.distribution is [Categorical, RelaxedOneHotCategorical]:
            new_var._out_features = self.size
        else:
            new_var._out_features = self.size * n_concepts

        return new_var

    def __repr__(self):
        """
        Return string representation of the Variable.

        Returns:
            str: String representation including concepts, distribution, size, and metadata.
        """
        meta_str = f", metadata={self.metadata}" if self.metadata else ""
        return f"Variable(concepts={self.concepts}, dist={self.distribution.__name__}, size={self.size}, out_features={self.out_features}{meta_str})"


class EndogenousVariable(Variable):
    """
    Represents an endogenous variable in a concept-based model.
    
    Endogenous variables are observable and supervisable concepts that can be
    directly measured or annotated in the data. These are typically the concepts
    that we want to learn and predict, such as object attributes, semantic features,
    or intermediate representations that have ground truth labels.
    
    Attributes:
        concepts (List[str]): List of concept names represented by this variable.
        parents (List[Variable]): List of parent variables in the graphical model.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Size/cardinality of the variable.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'endogenous'.
        
    Example:
        >>> from torch.distributions import Bernoulli, Categorical
        >>> from torch_concepts import EndogenousVariable
        >>> # Observable binary concept
        >>> has_wings = EndogenousVariable(
        ...     concepts='has_wings',
        ...     parents=[],
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> 
        >>> # Observable categorical concept (e.g., color)
        >>> color = EndogenousVariable(
        ...     concepts=['color'],
        ...     parents=[],
        ...     distribution=Categorical,
        ...     size=3  # red, green, blue
        ... )
    """
    
    def __init__(self, concepts: Union[str, List[str]],
                 parents: List[Union['Variable', str]],
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int]] = 1,
                 metadata: Dict[str, Any] = None):
        """
        Initialize an EndogenousVariable instance.
        
        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances.
            distribution: Distribution type (Delta, Bernoulli, Categorical, or Normal).
            size: Size parameter for the distribution.
            metadata: Optional metadata dictionary.
        """
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'endogenous'
        super().__init__(concepts, parents, distribution, size, metadata)


class ExogenousVariable(Variable):
    """
    Represents an exogenous variable in a concept-based model.
    
    Exogenous variables are high-dimensional representations related to a single
    endogenous variable. They capture rich, detailed information about a specific
    concept (e.g., image patches, embeddings, or feature vectors) that can be used
    to predict or explain the corresponding endogenous concept.
    
    Attributes:
        concepts (List[str]): List of concept names represented by this variable.
        parents (List[Variable]): List of parent variables in the graphical model.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Dimensionality of the high-dimensional representation.
        endogenous_var (Optional[EndogenousVariable]): The endogenous variable this exogenous variable is related to.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'exogenous'.
        
    Example:
        >>> from torch.distributions import Normal, Bernoulli
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts import EndogenousVariable, ExogenousVariable
        >>> # Endogenous concept
        >>> has_wings = EndogenousVariable(
        ...     concepts='has_wings',
        ...     parents=[],
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> 
        >>> # Exogenous high-dim representation for has_wings
        >>> wings_features = ExogenousVariable(
        ...     concepts='wings_exogenous',
        ...     parents=[],
        ...     distribution=Delta,
        ...     size=128,  # 128-dimensional exogenous
        ... )
    """
    
    def __init__(self, concepts: Union[str, List[str]],
                 parents: List[Union['Variable', str]],
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int]] = 1,
                 endogenous_var: Optional['EndogenousVariable'] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize an ExogenousVariable instance.
        
        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances.
            distribution: Distribution type (typically Delta or Normal for continuous representations).
            size: Dimensionality of the high-dimensional representation.
            endogenous_var: Optional reference to the related endogenous variable.
            metadata: Optional metadata dictionary.
        """
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'exogenous'
        if endogenous_var is not None:
            metadata['endogenous_var'] = endogenous_var
        super().__init__(concepts, parents, distribution, size, metadata)
        self.endogenous_var = endogenous_var
    
    def __repr__(self):
        """Return string representation including endogenous variable reference."""
        meta_str = f", metadata={self.metadata}" if self.metadata else ""
        endo_str = f", endogenous={self.endogenous_var.concepts if self.endogenous_var else None}"
        return f"ExogenousVariable(concepts={self.concepts}, dist={self.distribution.__name__}, size={self.size}, out_features={self.out_features}{endo_str}{meta_str})"


class InputVariable(Variable):
    """
    Represents a latent variable in a concept-based model.
    
    Latent variables are high-dimensional global representations of the whole input
    object (e.g., raw input images, text, or sensor data). They capture the complete
    information about the input before it is decomposed into specific concepts.
    These are typically unobserved, learned representations that encode all relevant
    information from the raw input.
    
    Attributes:
        concepts (List[str]): List of concept names represented by this variable.
        parents (List[Variable]): List of parent variables in the graphical model (typically empty).
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Dimensionality of the latent representation.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'input'.
        
    Example:
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts import InputVariable
        >>> # Global latent representation from input image
        >>> image_latent = InputVariable(
        ...     concepts='global_image_features',
        ...     parents=[],
        ...     distribution=Delta,
        ...     size=512  # 512-dimensional global latent
        ... )
        >>> 
        >>> # Multiple latent variables for hierarchical representation
        >>> low_level_features = InputVariable(
        ...     concepts='low_level_features',
        ...     parents=[],
        ...     distribution=Delta,
        ...     size=256
        ... )
        >>> high_level_features = InputVariable(
        ...     concepts='high_level_features',
        ...     parents=[low_level_features],
        ...     distribution=Delta,
        ...     size=512
        ... )
    """
    
    def __init__(self, concepts: Union[str, List[str]],
                 parents: List[Union['Variable', str]],
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int]] = 1,
                 metadata: Dict[str, Any] = None):
        """
        Initialize a InputVariable instance.
        
        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances (often empty for root latent variables).
            distribution: Distribution type (typically Delta or Normal for continuous representations).
            size: Dimensionality of the latent representation.
            metadata: Optional metadata dictionary.
        """
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'input'
        super().__init__(concepts, parents, distribution, size, metadata)
