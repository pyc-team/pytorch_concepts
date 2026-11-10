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
   MSEReconstructionLoss
   KLDivergenceLoss
   OrthogonalityLoss
   NLLProbLoss
   CosineCubedLoss
   L1LogitRegularizer
   LossWeightWarmup

``CosineCubedLoss`` returns the negative mean of
:func:`~torch_concepts.nn.functional.cosine_cubed_similarity` across concepts:

.. code-block:: python

   from torch_concepts.nn import CosineCubedLoss

   loss = CosineCubedLoss()(projection(embeddings), reference_scores)
   loss.backward()

Inputs are matching sample-by-concept floating-point tensors. This objective
aligns centered activation patterns and can also serve as the continuous loss
in ``ConceptLoss``.
