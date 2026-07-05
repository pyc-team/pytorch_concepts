Functional API
===============

Functional operations for concept-based computations. The docstrings of each function below
document their parameters and behaviour.

.. currentmodule:: torch_concepts.nn.functional

Concept Operations
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   grouped_concept_exogenous_mixture
   selection_eval
   confidence_selection
   soft_select

Linear and Logic Operations
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   linear_equation_eval
   linear_equation_expl
   logic_rule_eval
   logic_memory_reconstruction
   logic_rule_explanations

Evaluation Metrics
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   completeness_score
   intervention_score
   cace_score
   residual_concept_causal_effect

Calibration and Selection
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   selective_calibration

Graph Utilities
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   edge_type

Model Utilities
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   prune_linear_layer
