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
   LabelFreeCBM
   ConceptEmbeddingModel
   ConceptBottleneckVAE
   ConditionalVAE
   GraphConceptBottleneckModel
   CausallyReliableConceptBottleneckModel
   BlackBox
   BlackBoxTaskOnly

``ConceptBottleneckModel.set_linear_heads`` installs externally fitted linear
heads for scalar deterministic continuous concepts and one categorical task.
``LabelFreeCBM`` subclasses ``ConceptBottleneckModel`` and accepts a generation
pipeline to discover and construct its bottleneck internally during ``fit``.
See :doc:`/guides/lfcbm` for lifecycle, ownership, and numerical equivalence tests.

Concept Generation Helpers
--------------------------

See :doc:`/guides/lfcbm` for LF-CBM training and reproduction settings.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torch_concepts.data.generation.filters.SemanticConceptFilter

Outputs
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   InferenceOutput

Base Classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseModel
