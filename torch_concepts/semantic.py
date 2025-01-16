import abc
from typing import Iterable

import torch


class Semantic:
    @abc.abstractmethod
    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def iff(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        result = tensors[0]
        for tensor in tensors[1:]:
            result = self.conj(self.disj(self.neg(result), tensor),
                               self.disj(result, self.neg(tensor)))
        return result

    @abc.abstractmethod
    def neg(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CMRSemantic(Semantic):
    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result * tensor
        return result

    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result + tensor
        return result

    def neg(self, tensor: torch.Tensor) -> torch.Tensor:
        return 1 - tensor


class ProductTNorm(Semantic):

    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result + tensor - result * tensor
        return result

    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result * tensor
        return result

    def neg(self, a: torch.Tensor) -> torch.Tensor:
        return 1 - a


class GodelTNorm(Semantic):
    def conj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        result = tensors[0]
        for tensor in tensors[1:]:
            result = torch.min(result, tensor)
        return result

    def disj(self, *tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        result = tensors[0]
        for tensor in tensors[1:]:
            result = torch.max(result, tensor)
        return result

    def neg(self, a: torch.Tensor) -> torch.Tensor:
        return 1 - a
