Loss Functions
===============

Concept-aware loss functions with automatic routing and weighting.

.. currentmodule:: torch_concepts.nn.modules.loss

Summary
-------

**High-Level Losses**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptLoss
   WeightedConceptLoss

**Low-Level Losses**

.. autosummary::
   :toctree: generated
   :nosignatures:

   WeightedBCEWithLogitsLoss
   WeightedCrossEntropyLoss
   WeightedMSELoss


Overview
--------

High-level losses automatically route to appropriate loss functions based on concept types (binary, categorical, continuous) using annotation metadata.

Quick Start
-----------

.. code-block:: python

   import torch
   from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
   from torch_concepts import Annotations, AxisAnnotation, GroupConfig
   from torch_concepts.nn import ConceptLoss, ConceptBottleneckModel
   from torch.distributions import Bernoulli, Categorical
   
   # Define annotations with mixed types
   ann = Annotations({
       1: AxisAnnotation(
           labels=['is_round', 'is_smooth', 'color', 'class_A'],
           cardinalities=[1, 1, 3, 1],
           metadata={
               'is_round': {'type': 'discrete', 'distribution': Bernoulli},
               'is_smooth': {'type': 'discrete', 'distribution': Bernoulli},
               'color': {'type': 'discrete', 'distribution': Categorical},
               'class_A': {'type': 'discrete', 'distribution': Bernoulli}
           }
       )
   })
   
   # Configure loss functions by concept type using GroupConfig
   loss_config = GroupConfig(
       binary=BCEWithLogitsLoss(),
       categorical=CrossEntropyLoss()
   )
   
   # Automatic routing by concept type
   loss = ConceptLoss(annotations=ann[1], fn_collection=loss_config)
   
   # Use in Lightning training
   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       task_names=['class_A'],
       loss=loss,
       optim_class=torch.optim.AdamW,
       optim_kwargs={'lr': 0.001}
   )
   
   # Manual usage
   predictions = torch.randn(32, 6)  # batch_size=32, 2 binary + 3 categorical + 1 binary
   targets = torch.cat([
       torch.randint(0, 2, (32, 2)),  # binary targets
       torch.randint(0, 3, (32, 1)),  # categorical target (class indices)
       torch.randint(0, 2, (32, 1))   # binary target
   ], dim=1)
   
   loss_value = loss(predictions, targets)


Class Documentation
-------------------

.. autoclass:: ConceptLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WeightedConceptLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WeightedBCEWithLogitsLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WeightedCrossEntropyLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WeightedMSELoss
   :members:
   :undoc-members:
   :show-inheritance:
