High-Level API
==============

Ready-to-use concept-based models and the structures that configure them. See the
:doc:`High-Level user guide </guides/using_high_level>` for explanations and examples; the
docstrings of each class below document their parameters and behaviour.

Annotations & Configuration
---------------------------

.. currentmodule:: torch_concepts

.. autosummary::
   :toctree: generated
   :nosignatures:

   Annotations
   GroupConfig
   AnnotatedTensor
   ConceptGraph

.. currentmodule:: torch_concepts.annotations

.. autosummary::
   :toctree: generated
   :nosignatures:

   Concept

Models
------

.. currentmodule:: torch_concepts.nn

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptBottleneckModel
   ConceptEmbeddingModel
   GraphConceptBottleneckModel
   CausallyReliableConceptBottleneckModel
   BlackBox
   BlackBoxTaskOnly

Outputs
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ModelOutput
   InferenceOutput

Base Classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseModel
