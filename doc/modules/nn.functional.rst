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
   concept_orthogonality
   cosine_cubed_similarity
   selection_eval
   confidence_selection
   soft_select

Cosine-cubed alignment
~~~~~~~~~~~~~~~~~~~~~~

``cosine_cubed_similarity`` centers each concept column across samples, cubes
its entries, then measures cosine similarity between corresponding columns.
It returns one score per concept; constant columns return zero. This measures
agreement between activation patterns, including those used by LF-CBM
(Oikarinen et al., ICLR 2023), without imposing concept-selection thresholds.

.. code-block:: python

   from torch_concepts.nn.functional import cosine_cubed_similarity

   scores = cosine_cubed_similarity(projected_activations, reference_scores)
   loss = -scores.mean()
   loss.backward()

Both inputs have shape ``(samples, concepts)`` and share column order. Gradients
are preserved, including for reference scores when required. Computation uses
at least float32 precision.

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

``logic_rule_eval`` evaluates every rule when ``memory_idxs=None``, returning
shape ``(batch_size, n_tasks, memory_size)``. To select one rule per sample and
task, pass a long tensor of indices with shape ``(batch_size, n_tasks)`` on the
same device as the rule weights. The selected output has shape
``(batch_size, n_tasks, 1)``.

Evaluation Metrics
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   completeness_score
   intervention_score
   tcav_score
   number_of_effective_concepts
   number_of_contributing_concepts

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
