Functional API
===============

This module provides functional operations for concept-based computations.

.. currentmodule:: torch_concepts.nn.functional

Summary
-------

**Concept Operations**

.. autosummary::
   :toctree: generated
   :nosignatures:

   grouped_concept_exogenous_mixture
   selection_eval
   confidence_selection
   soft_select

**Linear and Logic Operations**

.. autosummary::
   :toctree: generated
   :nosignatures:

   linear_equation_eval
   linear_equation_expl
   logic_rule_eval
   logic_memory_reconstruction
   logic_rule_explanations

**Evaluation Metrics**

.. autosummary::
   :toctree: generated
   :nosignatures:

   completeness_score
   intervention_score
   cace_score
   residual_concept_causal_effect

**Calibration and Selection**

.. autosummary::
   :toctree: generated
   :nosignatures:

   selective_calibration

**Graph Utilities**

.. autosummary::
   :toctree: generated
   :nosignatures:

   edge_type
   hamming_distance

**Model Utilities**

.. autosummary::
   :toctree: generated
   :nosignatures:

   prune_linear_layer


Function Documentation
----------------------

Concept Operations
~~~~~~~~~~~~~~~~~~

.. autofunction:: grouped_concept_exogenous_mixture

.. autofunction:: selection_eval

.. autofunction:: confidence_selection

.. autofunction:: soft_select


Linear and Logic Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: linear_equation_eval

.. autofunction:: linear_equation_expl

.. autofunction:: logic_rule_eval

.. autofunction:: logic_memory_reconstruction

.. autofunction:: logic_rule_explanations


Evaluation Metrics
~~~~~~~~~~~~~~~~~~

.. autofunction:: completeness_score

.. autofunction:: intervention_score

.. autofunction:: cace_score

.. autofunction:: residual_concept_causal_effect


Calibration and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: selective_calibration


Graph Utilities
~~~~~~~~~~~~~~~

.. autofunction:: edge_type

.. autofunction:: hamming_distance


Model Utilities
~~~~~~~~~~~~~~~

.. autofunction:: prune_linear_layer
