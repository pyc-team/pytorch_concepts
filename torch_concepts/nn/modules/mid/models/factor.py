"""
ParametricFactor base class for probabilistic graphical models.

This module defines the ParametricFactor base class, which represents a factor
in a factor graph. Factors associate concept-variables with neural network
parametrizations and form the building blocks for both directed (Bayesian
Networks) and undirected (Markov Random Fields) graphical models.
"""
import torch.nn as nn
from typing import List



class ParametricFactor(nn.Module):
    """
    Base class for factors in a probabilistic graphical model.

    A ParametricFactor associates a set of named concepts (its *scope*) with a
    single neural-network parametrization.  The factor produces one potential
    over all the concepts in its scope.

    Parameters
    ----------
    concepts : List[str]
        A list of concept names defining the scope of this factor.
    parametrization : nn.Module
        A single neural-network module that computes the factor potential.

    Attributes
    ----------
    concepts : List[str]
        Full scope of concept names.
    parametrization : nn.Module
        The neural network module used to compute factor values.

    See Also
    --------
    ParametricCPD : Directed factor for conditional probability distributions.
    Variable : Represents a random variable (concept) in the model.
    ProbabilisticModel : Generic container that manages factors and variables.
    """

    def __init__(self, 
                 concepts: List[str],
                 parametrization: nn.Module):
        """
        Initialize a ParametricFactor instance.

        Parameters
        ----------
        concepts : List[str]
            A list of concept names defining the scope of this factor.
        parametrization : nn.Module
            Neural network module for computing factor values.
        """
        super().__init__()
        raise NotImplementedError("" \
        "ParametricFactor is an abstract base class that is yet to be implemented. " \
        "Please use ParametricCPD")
        self.concepts = concepts
        self.parametrization = parametrization
        self.scope = concepts
