High-Level Models
===============================

Ready-to-use concept-based models with automatic or manual training support.

.. currentmodule:: torch_concepts.nn

Summary
-------

**Model Classes**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptBottleneckModel
   ConceptBottleneckModel_Joint
   BlackBox


Overview
--------

High-level models provide two training modes:

- **Manual PyTorch Training**: Initialize without loss for full control
- **Lightning Training**: Initialize with loss/optimizer for automatic training

Quick Start
-----------

.. code-block:: python

   import torch
   from torch.distributions import Bernoulli, Categorical
   from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
   from torch_concepts import Annotations, AxisAnnotation, GroupConfig
   from torch_concepts.nn import ConceptBottleneckModel
   
   # Define annotations with mixed concept types\n   ann = Annotations({
       1: AxisAnnotation(
           labels=['is_round', 'is_smooth', 'color', 'class_A', 'class_B'],
           cardinalities=[1, 1, 3, 1, 1],
           metadata={
               'is_round': {'type': 'discrete', 'distribution': Bernoulli},
               'is_smooth': {'type': 'discrete', 'distribution': Bernoulli},
               'color': {'type': 'discrete', 'distribution': Categorical},
               'class_A': {'type': 'discrete', 'distribution': Bernoulli},
               'class_B': {'type': 'discrete', 'distribution': Bernoulli}
           }
       )
   })
   
   # Manual training mode
   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       task_names=['class_A', 'class_B'],
       latent_encoder_kwargs={'hidden_size': 128, 'n_layers': 2}
   )
   
   # Lightning training mode with loss and metrics
   from torch_concepts.nn import ConceptLoss, ConceptMetrics
   from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
   
   loss_config = GroupConfig(
       binary=BCEWithLogitsLoss(),
       categorical=CrossEntropyLoss()\n   )
   metrics_config = GroupConfig(
       binary={'accuracy': BinaryAccuracy()},
       categorical={'accuracy': MulticlassAccuracy}
   )
   
   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       task_names=['class_A', 'class_B'],
       loss=ConceptLoss(annotations=ann[1], fn_collection=loss_config),
       metrics=ConceptMetrics(annotations=ann[1], fn_collection=metrics_config,
                             summary_metrics=True, perconcept_metrics=True),
       optim_class=torch.optim.AdamW,
       optim_kwargs={'lr': 0.001}
   )


Class Documentation
-------------------

.. autoclass:: ConceptBottleneckModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConceptBottleneckModel_Joint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BlackBox
   :members:
   :undoc-members:
   :show-inheritance:
