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

   FixedConceptGenerator
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
   CXRCLIPAnnotator
   RandomAnnotator

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
   SemanticConceptFilter
   ThresholdAnnotationFilter

Semantic vocabulary filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`SemanticConceptFilter` removes overly long labels, class aliases,
class-similar concepts, and semantic near-duplicates before sample annotation.
Supply two text-encoding callables; the filter downloads no models itself.

.. code-block:: python

   from torch_concepts.data.generation.filters import SemanticConceptFilter

   concept_filter = SemanticConceptFilter(
       class_names=["cat", "dog"],
       clip_text_encoder=clip_encoder,
       sentence_encoder=sentence_encoder,
       class_cutoff=0.85,
       similarity_cutoff=0.9,
   )
   filtered = concept_filter.filter(candidate_annotations)
   # Or pass concept_filter as a pipeline's generator_filter.

The similarity is ``(sentence_inner_product + 3 * clip_cosine) / 4``.
CLIP embeddings are normalized internally; provide normalized sentence
embeddings when cosine similarity is intended for both encoders. Labels longer
than ``max_length`` (default 30) and exact class-name aliases are removed first.
Class similarity at or above ``class_cutoff`` removes a concept. For pairs at
or above ``similarity_cutoff``, pruning keeps the concept with greater summed
similarity to the vocabulary after class filtering, with alphabetical order
breaking ties. Output labels are alphabetical; selected states, types, and
applicable named groups are retained by annotation subsetting.

All class/concept pairs are compared by default. Explicitly set
``reference_class_filter=True`` for the LF-CBM reference index exception,
which skips comparisons when class and concept list indices coincide.

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
