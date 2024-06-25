import abc
import torch


class Logic:
    """
    Abstract base class representing a logical system.

    This class defines the interface for logical operations such as conjunction, disjunction,
    and negation. It also provides methods for pairwise logical operations and updating the state.

    Methods:
        update(): Update the state of the logical system.
        conj(a, dim=1): Perform conjunction (logical AND) on the input tensor along the specified dimension.
        disj(a, dim=1): Perform disjunction (logical OR) on the input tensor along the specified dimension.
        conj_pair(a, b): Perform pairwise conjunction (logical AND) between two tensors.
        disj_pair(a, b): Perform pairwise disjunction (logical OR) between two tensors.
        iff_pair(a, b): Perform pairwise biconditional (logical IFF) between two tensors.
        neg(a): Perform negation (logical NOT) on the input tensor.
    """
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError

    @abc.abstractmethod
    def conj(self, a, dim=1):
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, a, dim=1):
        raise NotImplementedError

    def conj_pair(self, a, b):
        raise NotImplementedError

    def disj_pair(self, a, b):
        raise NotImplementedError

    def iff_pair(self, a, b):
        raise NotImplementedError

    @abc.abstractmethod
    def neg(self, a):
        raise NotImplementedError


class ProductTNorm(Logic):
    """
    ProductTNorm is a logical system implementing the Product T-norm.

    The Product T-norm defines conjunction as the product of truth values and disjunction as
    one minus the product of one minus the truth values. This class also provides methods for
    pairwise logical operations and prediction.

    Attributes:
        current_truth (torch.Tensor): Tensor representing the current truth value.
        current_false (torch.Tensor): Tensor representing the current false value.
    """
    def __init__(self):
        super(ProductTNorm, self).__init__()
        self.current_truth = torch.tensor(1)
        self.current_false = torch.tensor(0)

    def update(self):
        pass

    def conj(self, a, dim=1):
        return torch.prod(a, dim=dim, keepdim=True)

    def conj_pair(self, a, b):
        return a * b

    def disj(self, a, dim=1):
        return 1 - torch.prod(1 - a, dim=dim, keepdim=True)

    def disj_pair(self, a, b):
        return a + b - a * b

    def iff_pair(self, a, b):
        return self.conj_pair(self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)


class GodelTNorm(Logic):
    """
    GodelTNorm is a logical system implementing the Godel T-norm.

    The Godel T-norm defines conjunction as the minimum of truth values and disjunction as
    the maximum of truth values. This class also provides methods for pairwise logical operations
    and prediction.

    Attributes:
        current_truth (int): Integer representing the current truth value.
        current_false (int): Integer representing the current false value.
    """
    def __init__(self):
        super(GodelTNorm, self).__init__()
        self.current_truth = 1
        self.current_false = 0

    def update(self):
        pass

    def conj(self, a, dim=1):
        return torch.min(a, dim=dim, keepdim=True)[0]

    def disj(self, a, dim=1):
        return torch.max(a, dim=dim, keepdim=True)[0]

    def conj_pair(self, a, b):
        return torch.minimum(a, b)

    def disj_pair(self, a, b):
        return torch.maximum(a, b)

    def iff_pair(self, a, b):
        return self.conj_pair(self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)
