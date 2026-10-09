Concept Generation
==================

Generate candidate concepts, annotate samples with them, and transform the
resulting scores into supervision. Start with the
:doc:`concept generation guide </guides/using_generation>` for a complete example.
The docstrings below describe the supported providers and pipeline stages.

Pipeline
--------

.. currentmodule:: torch_concepts.data.generation

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   ConceptGenerationPipeline

Generators
----------

.. currentmodule:: torch_concepts.data.generation.generators

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   LLMConceptGenerator
   LiteLLMBackend

Annotators
----------

.. currentmodule:: torch_concepts.data.generation.annotators

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   CLIPAnnotator

Calibrators
-----------

.. currentmodule:: torch_concepts.data.generation.calibrators

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   SigmoidCalibrator

Filters
-------

.. currentmodule:: torch_concepts.data.generation.filters

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   DeduplicateConcepts
   ThresholdAnnotationFilter

Base Classes
------------

.. currentmodule:: torch_concepts.data.generation

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   Generator
   Annotator
   FilterGenerator
   FilterAnnotator
   Calibrator
