Mid-Level API
=============

Random variables, factors, probabilistic models and inference engines. See the
:doc:`Mid-Level user guide </guides/using_mid_level>` for explanations and examples; the docstrings
of each class below document their parameters and behaviour.

.. warning::

   This API is still under development and interfaces might change in future releases.

Variables
---------

.. currentmodule:: torch_concepts

.. autosummary::
   :toctree: generated
   :nosignatures:

   Variable
   ConceptVariable
   EmbeddingVariable

Factors
-------

.. currentmodule:: torch_concepts.nn

.. autosummary::
   :toctree: generated
   :nosignatures:

   ParametricFactor
   ParametricCPD

Activations
-----------

A CPD applies no activation, so a parametrization module must already emit a
value in its parameter's domain. ``DefaultActivation`` supplies the family's
standard mapping for a given parameter.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DefaultActivation

Probabilistic Models
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ProbabilisticModel
   BayesianNetwork

Inference
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ForwardInference
   DeterministicInference
   IndependentInference
   AncestralSamplingInference
   MAPForwardInference
   BeliefPropagation
   RejectionSampling
   ImportanceSampling
   VariationalInference
   PyroImportanceSampling
   BaseProposal
   MutilatedNetworkProposal

Base Classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseInference
   TorchBaseInference
   PyroBaseInference
