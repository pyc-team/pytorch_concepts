Base classes (low level)
==========================

This module provides abstract base classes for building concept-based neural networks at the low level.
These classes define the fundamental interfaces for encoders, predictors, graph learners, and inference modules.

.. currentmodule:: torch_concepts.nn

Summary
-------

**Base Layer Classes**

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseConceptLayer
   BaseEncoder
   BasePredictor

**Graph Learning Classes**

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseGraphLearner

**Inference Classes**

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseInference
   BaseIntervention


Class Documentation
-------------------

Layer Classes
~~~~~~~~~~~~~

.. autoclass:: BaseConceptLayer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BaseEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BasePredictor
   :members:
   :undoc-members:
   :show-inheritance:

Graph Learning Classes
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseGraphLearner
   :members:
   :undoc-members:
   :show-inheritance:

Inference Classes
~~~~~~~~~~~~~~~~~

.. autoclass:: BaseInference
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BaseIntervention
   :members:
   :undoc-members:
   :show-inheritance:
