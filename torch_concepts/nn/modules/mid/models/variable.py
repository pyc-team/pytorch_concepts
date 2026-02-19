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
        concept (str): The concept name represented by this variable.
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
        >>> print(var_binary.concept)  # 'has_wheels'
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
        >>> print(vars_list[0].concept)  # 'A'
        >>> print(vars_list[1].concept)  # 'B'
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

    def __new__(cls, concepts: Union[str, List[str]], parents: List[Union['Variable', str]],
                distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                size: Union[int, List[int]] = 1, metadata: Optional[Dict[str, Any]] = None):
        """
        Create new Variable instance(s).

        If concepts is a string, returns a single Variable instance.
        If concepts is a list, returns a list of Variable instances (one per concept).

        Args:
            concepts: Single concept name (str) or list of concept names.
            parents: List of parent Variable instances.
            distribution: Distribution type or list of distribution types.
            size: Size parameter(s) for the distribution.
            metadata: Optional metadata dictionary.

        Returns:
            Variable: Single instance if concepts is str.
            List[Variable]: List of instances if concepts is list.

        Raises:
            ValueError: If concepts is str but distribution or size is a list.
            ValueError: If list lengths don't match when concepts is a list.
        """
        if isinstance(concepts, str):
            # Single concept: other fields must NOT be lists
            if isinstance(distribution, list):
                raise ValueError(
                    "When 'concepts' is a string, 'distribution' must be a single value, not a list.")
            if isinstance(size, list):
                raise ValueError(
                    "When 'concepts' is a string, 'size' must be a single value, not a list.")
            return object.__new__(cls)

        # concepts is a list -> return list of Variables
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
                f"If concepts is a list of length {n_concepts}, distribution and size must either be "
                f"single values or lists of length {n_concepts}.")

        # Create and return a list of individual Variable instances
        new_vars = []
        for i in range(n_concepts):
            # Use object.__new__(cls) to bypass this __new__ logic for the sub-creation
            instance = object.__new__(cls)
            instance.__init__(
                concepts=concepts[i],  # Pass as string to create single Variable
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
            concepts: Single concept name (stored as string).
            parents: List of parent Variable instances.
            distribution: Distribution type (Delta, Bernoulli, Categorical, or Normal).
            size: Size parameter for the distribution.
            metadata: Optional metadata dictionary.

        Raises:
            ValueError: If Categorical variable doesn't have size > 1.
            ValueError: If Bernoulli variable doesn't have size=1.
        """
        # Original validation logic
        if distribution is None:
            distribution = Delta

        if distribution is Categorical:
            if size <= 1:
                raise ValueError("Categorical Variable must have a size > 1 (number of classes).")

        if distribution is Bernoulli and size != 1:
            raise ValueError("Bernoulli Variable must have size=1 as it represents a binary outcome per concept.")

        self.concept = concepts
        self.parents = parents
        self.distribution = distribution
        self.size = size
        self.metadata = metadata if metadata is not None else {}

    @property
    def out_features(self) -> int:
        """
        Number of output features for this variable.

        This is an alias for `size`, provided for consistency with neural network
        module interfaces where `out_features` is the conventional name.

        Returns:
            int: Number of output features (equals `size`).
        """
        return self.size

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

    def __repr__(self):
        """
        Return string representation of the Variable.

        Returns:
            str: String representation including concepts, distribution, size, and metadata.
        """
        meta_str = f", metadata={self.metadata}" if self.metadata else ""
        return f"Variable(concept='{self.concept}', dist={self.distribution.__name__}, size={self.size}, out_features={self.out_features}{meta_str})"


class ConceptVariable(Variable):
    """
    Represents a concept variable in a concept-based model.
    
    Concept variables are observable and supervisable variables that can be
    directly measured or annotated in the data. These are typically the concepts
    that we want to learn and predict, such as object attributes, semantic features,
    or intermediate representations that have ground truth labels.
    
    Attributes:
        concept (str): The concept name represented by this variable.
        parents (List[Variable]): List of parent variables in the graphical model.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Size/cardinality of the variable.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'concept'.
        
    Example:
        >>> from torch.distributions import Bernoulli, Categorical
        >>> from torch_concepts import ConceptVariable
        >>> # Observable binary concept
        >>> has_wings = ConceptVariable(
        ...     concepts='has_wings',
        ...     parents=[],
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> 
        >>> # Observable categorical concept (e.g., color)
        >>> color = ConceptVariable(
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
        Initialize a ConceptVariable instance.
        
        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances.
            distribution: Distribution type (Delta, Bernoulli, Categorical, or Normal).
            size: Size parameter for the distribution.
            metadata: Optional metadata dictionary.
        """
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'concept'
        super().__init__(concepts, parents, distribution, size, metadata)


# Backward compatibility alias
EndogenousVariable = ConceptVariable


class ExogenousVariable(Variable):
    """
    Represents an exogenous variable in a concept-based model.
    
    Exogenous variables are high-dimensional representations related to a single
    concept variable. They capture rich, detailed information about a specific
    concept (e.g., image patches, embeddings, or feature vectors) that can be used
    to predict or explain the corresponding concept.
    
    Attributes:
        concept (str): The concept name represented by this variable.
        parents (List[Variable]): List of parent variables in the graphical model.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Dimensionality of the high-dimensional representation.
        concept_var (Optional[ConceptVariable]): The concept variable this exogenous variable is related to.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'exogenous'.
        
    Example:
        >>> from torch.distributions import Normal, Bernoulli
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts import ConceptVariable, ExogenousVariable
        >>> # Concept variable
        >>> has_wings = ConceptVariable(
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
                 concept_var: Optional['ConceptVariable'] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize an ExogenousVariable instance.
        
        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances.
            distribution: Distribution type (typically Delta or Normal for continuous representations).
            size: Dimensionality of the high-dimensional representation.
            concept_var: Optional reference to the related concept variable.
            metadata: Optional metadata dictionary.
        """
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'exogenous'
        if concept_var is not None:
            metadata['concept_var'] = concept_var
        super().__init__(concepts, parents, distribution, size, metadata)
        self.concept_var = concept_var


class LatentVariable(Variable):
    """
    Represents a latent variable in a concept-based model.
    
    Latent variables are high-dimensional global representations of the whole input
    object (e.g., raw input images, text, or sensor data). They capture the complete
    information about the input before it is decomposed into specific concepts.
    These are typically unobserved, learned representations that encode all relevant
    information from the raw input.
    
    Attributes:
        concept (str): The concept name represented by this variable.
        parents (List[Variable]): List of parent variables in the graphical model (typically empty).
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Dimensionality of the latent representation.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'latent'.
        
    Example:
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts import LatentVariable
        >>> # Global latent representation from input image
        >>> image_latent = LatentVariable(
        ...     concepts='global_image_features',
        ...     parents=[],
        ...     distribution=Delta,
        ...     size=512  # 512-dimensional global latent
        ... )
        >>> 
        >>> # Multiple latent variables for hierarchical representation
        >>> low_level_features = LatentVariable(
        ...     concepts='low_level_features',
        ...     parents=[],
        ...     distribution=Delta,
        ...     size=256
        ... )
        >>> high_level_features = LatentVariable(
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
        Initialize a LatentVariable instance.
        
        Args:
            concepts: Single concept name or list of concept names.
            parents: List of parent Variable instances (often empty for root latent variables).
            distribution: Distribution type (typically Delta or Normal for continuous representations).
            size: Dimensionality of the latent representation.
            metadata: Optional metadata dictionary.
        """
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'latent'
        super().__init__(concepts, parents, distribution, size, metadata)


# Backward compatibility alias
InputVariable = LatentVariable
