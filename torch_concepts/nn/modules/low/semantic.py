"""
Semantic operations for fuzzy logic and t-norms.

This module provides various semantic implementations for logical operations
in fuzzy logic, including different t-norms (triangular norms) and their
corresponding operations.
"""
import abc
import torch

from typing import Iterable



class Semantic:
    """
    Abstract base class for semantic operations in fuzzy logic.

    This class defines the interface for implementing logical operations
    such as conjunction, disjunction, negation, and biconditional in
    fuzzy logic systems.
    """

    @abc.abstractmethod
    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute the conjunction (AND operation) of multiple tensors.

        Args:
            *tensors: Variable number of tensors to combine with conjunction.

        Returns:
            torch.Tensor: The result of the conjunction operation.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute the disjunction (OR operation) of multiple tensors.

        Args:
            *tensors: Variable number of tensors to combine with disjunction.

        Returns:
            torch.Tensor: The result of the disjunction operation.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    def iff(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute the biconditional (IFF/equivalence) operation of multiple tensors.

        The biconditional is computed using the equivalence:
        A ⟺ B ≡ (¬A ∨ B) ∧ (A ∨ ¬B)

        Args:
            *tensors: Variable number of tensors to combine with biconditional.

        Returns:
            torch.Tensor: The result of the biconditional operation.
        """
        result = tensors[0]
        for tensor in tensors[1:]:
            result = self.conj(self.disj(self.neg(result), tensor),
                               self.disj(result, self.neg(tensor)))
        return result

    @abc.abstractmethod
    def neg(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the negation (NOT operation) of a tensor.

        Args:
            tensor: The tensor to negate.

        Returns:
            torch.Tensor: The negated tensor.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError


class CMRSemantic(Semantic):
    """
    CMR (Concept Masking and Reasoning) Semantic implementation.

    This semantic uses simple arithmetic operations for fuzzy logic:
    - Conjunction: multiplication
    - Disjunction: addition
    - Negation: 1 - x
    """

    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute conjunction using multiplication.

        Args:
            *tensors: Variable number of tensors to combine.

        Returns:
            torch.Tensor: Product of all input tensors.
        """
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result * tensor
        return result

    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute disjunction using addition.

        Args:
            *tensors: Variable number of tensors to combine.

        Returns:
            torch.Tensor: Sum of all input tensors.
        """
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result + tensor
        return result

    def neg(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute negation using 1 - x.

        Args:
            tensor: The tensor to negate.

        Returns:
            torch.Tensor: 1 - tensor.
        """
        return 1 - tensor


class ProductTNorm(Semantic):
    """
    Product t-norm semantic implementation.

    This is a standard fuzzy logic t-norm where:
    - Conjunction: product (a * b)
    - Disjunction: probabilistic sum (a + b - a*b)
    - Negation: 1 - x
    """

    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute disjunction using probabilistic sum: a + b - a*b.

        Args:
            *tensors: Variable number of tensors to combine.

        Returns:
            torch.Tensor: Probabilistic sum of all input tensors.
        """
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result + tensor - result * tensor
        return result

    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute conjunction using product.

        Args:
            *tensors: Variable number of tensors to combine.

        Returns:
            torch.Tensor: Product of all input tensors.
        """
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result * tensor
        return result

    def neg(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute negation using 1 - a.

        Args:
            a: The tensor to negate.

        Returns:
            torch.Tensor: 1 - a.
        """
        return 1 - a


class GodelTNorm(Semantic):
    """
    Gödel t-norm semantic implementation.

    This is a standard fuzzy logic t-norm where:
    - Conjunction: minimum (min(a, b))
    - Disjunction: maximum (max(a, b))
    - Negation: 1 - x
    """

    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute conjunction using minimum operation.

        Args:
            *tensors: Variable number of tensors to combine.

        Returns:
            torch.Tensor: Element-wise minimum of all input tensors.
        """
        result = tensors[0]
        for tensor in tensors[1:]:
            result = torch.min(result, tensor)
        return result

    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute disjunction using maximum operation.

        Args:
            *tensors: Variable number of tensors to combine.

        Returns:
            torch.Tensor: Element-wise maximum of all input tensors.
        """
        result = tensors[0]
        for tensor in tensors[1:]:
            result = torch.max(result, tensor)
        return result

    def neg(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute negation using 1 - a.

        Args:
            a: The tensor to negate.

        Returns:
            torch.Tensor: 1 - a.
        """
        return 1 - a
