"""
ParametricFactor base class for probabilistic graphical models.

This module defines the ParametricFactor base class, which represents a factor
in a factor graph. Factors associate concept-variables with neural network
parametrizations and form the building blocks for both directed (Bayesian
Networks) and undirected (Markov Random Fields) graphical models.
"""
import torch.nn as nn
from typing import List, Optional, Union

from .variable import Variable


class ParametricFactor(nn.Module):
    """
    Base class for factors in a probabilistic graphical model.

    A ParametricFactor associates a set of named concepts (its *scope*) with a
    single neural-network parametrization.  The factor produces one potential
    over all the concepts in its scope.

    This base class is agnostic to the directionality of the graphical model.
    Subclasses specialise the semantics:

    * :class:`ParametricCPD` — directed factor with explicit ``parents``,
      used in Bayesian Networks.
    * (future) undirected factors for Markov Random Fields.

    Parameters
    ----------
    concepts : Union[str, List[str]]
        A single concept name or a list of concept names defining the scope
        of this factor.  A single ``ParametricFactor`` instance is always
        created (never a list).
    parametrization : nn.Module
        A single neural-network module that computes the factor potential.

    Attributes
    ----------
    concept : str
        Primary concept name (first element of the scope).
    concepts : List[str]
        Full scope of concept names.
    parametrization : nn.Module
        The neural network module used to compute factor values.
    variable : Optional[Variable]
        The :class:`Variable` instance this factor is linked to
        (set by :class:`ProbabilisticModel` during initialisation).

    See Also
    --------
    ParametricCPD : Directed factor for conditional probability distributions.
    Variable : Represents a random variable (concept) in the model.
    ProbabilisticModel : Generic container that manages factors and variables.
    """

    def __init__(self, concepts: Union[str, List[str]],
                 parametrization: nn.Module,
                 **kwargs):
        """
        Initialize a ParametricFactor instance.

        Parameters
        ----------
        concepts : Union[str, List[str]]
            Single concept name (stored as ``self.concept``).
        parametrization : Union[nn.Module, List[nn.Module]]
            Neural network module for computing factor values.
        **kwargs
            Ignored at this level; accepted so that subclass keyword
            arguments (e.g. ``parents``) pass through ``__new__`` without error.
        """
        super().__init__()
        if isinstance(concepts, str):
            self.concepts: List[str] = [concepts]
        else:
            self.concepts: List[str] = list(concepts)
        self.concept: str = self.concepts[0]
        self.parametrization = parametrization
        self.variable: Optional[Variable] = None

    def forward(self, **kwargs):
        """
        Compute the factor output by running the parametrization module.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the parametrization module.

        Returns
        -------
        torch.Tensor
            Output of the parametrization module.
        """
        return self.parametrization(**kwargs)

    def __repr__(self):
        scope = self.concepts if len(self.concepts) > 1 else self.concept
        return f"{self.__class__.__name__}(concepts={scope!r}, parametrization={self.parametrization.__class__.__name__})"
