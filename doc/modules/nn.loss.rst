Loss Functions
===============

Concept-aware loss functions with automatic routing and weighting. The docstrings of each
class below document their parameters and behaviour; see :doc:`Losses </guides/using_loss>`
for how they fit together.

.. currentmodule:: torch_concepts.nn

.. autosummary::
   :toctree: generated
   :nosignatures:

   PyCLoss
   ConceptLoss
   CompositeLoss
   ConceptSubset
   WeightedConceptLoss
   DepthWeightedConceptLoss
   ReconstructionLoss
   KLDivergenceLoss
   OrthogonalityLoss
   NLLProbLoss
   L1LogitRegularizer
   LossWeightWarmup
