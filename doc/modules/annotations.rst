Annotations
============

Containers for model configuration and type information.

.. currentmodule:: torch_concepts.annotations

Summary
-------

**Annotation Classes**

.. autosummary::
   :toctree: generated
   :nosignatures:

   AxisAnnotation
   Annotations


Overview
--------

Annotations store metadata about concepts including names, cardinalities, distribution 
types, and custom attributes. They are required to initialize:

- **Models** (e.g., ConceptBottleneckModel): Specify concept structure and distributions
- **ConceptLoss**: Route to appropriate loss functions based on concept types
- **ConceptMetrics**: Organize metrics by concept and compute per-concept statistics

Distribution information is critical - it tells the model how to represent each concept 
(e.g., Bernoulli for binary, Categorical for multi-class, Normal for continuous).

Distributions can be provided either:

1. **In annotations metadata** (recommended): Include 'distribution' key in metadata
2. **Via model's variable_distributions parameter**: Pass distributions at model initialization

Quick Start
-----------

**Option 1: Distributions in metadata (recommended)**

.. code-block:: python

   from torch_concepts.annotations import AxisAnnotation, Annotations
   from torch.distributions import Bernoulli, Categorical
   
   # Distributions included in annotations
   ann = Annotations({
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
   
   # Use in model (no variable_distributions needed)
   from torch_concepts.nn import ConceptBottleneckModel
   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       task_names=['class_A', 'class_B']
   )
   
   # Use in loss
   from torch_concepts.nn import ConceptLoss
   from torch_concepts import GroupConfig
   from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
   
   loss_config = GroupConfig(
       binary=BCEWithLogitsLoss(),
       categorical=CrossEntropyLoss()
   )
   loss = ConceptLoss(annotations=ann[1], fn_collection=loss_config)
   
   # Use in metrics
   from torch_concepts.nn import ConceptMetrics
   from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
   
   metrics_config = GroupConfig(
       binary={'accuracy': BinaryAccuracy()},
       categorical={'accuracy': MulticlassAccuracy}
   )
   metrics = ConceptMetrics(
       annotations=ann[1],
       fn_collection=metrics_config,
       summary_metrics=True,
       perconcept_metrics=True
   )

**Option 2: Via variable_distributions dictionary**

.. code-block:: python

   # Annotations without distributions
   ann = Annotations({
       1: AxisAnnotation(
           labels=['is_round', 'is_smooth', 'color', 'class_A', 'class_B'],
           cardinalities=[1, 1, 3, 1, 1],
           metadata={
               'is_round': {'type': 'discrete'},
               'is_smooth': {'type': 'discrete'},
               'color': {'type': 'discrete'},
               'class_A': {'type': 'discrete'},
               'class_B': {'type': 'discrete'}
           }
       )
   })
   
   # Provide distributions at model init
   variable_distributions = {
       'is_round': Bernoulli,
       'is_smooth': Bernoulli,
       'color': Categorical,
       'class_A': Bernoulli,
       'class_B': Bernoulli
   }
   
   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       variable_distributions=variable_distributions,
       task_names=['class_A', 'class_B']
   )
   
   # Distributions added internally, then used in loss/metrics
   loss = ConceptLoss(annotations=model.concept_annotations, fn_collection=loss_config)
   metrics = ConceptMetrics(
       annotations=model.concept_annotations,
       fn_collection=metrics_config,
       summary_metrics=True,
       perconcept_metrics=True
   )

**Option 3: Using GroupConfig for automatic type-based assignment**

For models with many concepts of the same types, use ``GroupConfig`` to automatically assign distributions:

.. code-block:: python

   from torch_concepts import GroupConfig
   
   # Annotations with concept types
   ann = Annotations({
       1: AxisAnnotation(
           labels=['is_round', 'is_smooth', 'color', 'shape', 'class_A', 'class_B'],
           cardinalities=[1, 1, 3, 4, 1, 1],
           metadata={
               'is_round': {'type': 'discrete'},    # binary (card=1)
               'is_smooth': {'type': 'discrete'},   # binary (card=1)
               'color': {'type': 'discrete'},       # categorical (card=3)
               'shape': {'type': 'discrete'},       # categorical (card=4)
               'class_A': {'type': 'discrete'},     # binary (card=1)
               'class_B': {'type': 'discrete'}      # binary (card=1)
           }
       )
   })
   
   # GroupConfig automatically assigns by concept type and cardinality
   variable_distributions = GroupConfig(
       binary=Bernoulli,       # for cardinality=1
       categorical=Categorical  # for cardinality>1
   )
   
   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       variable_distributions=variable_distributions,
       task_names=['class_A', 'class_B']
   )

This approach is ideal for large-scale datasets (e.g., CUB-200 with 312 attributes).


Class Documentation
-------------------

.. autoclass:: AxisAnnotation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Annotations
   :members:
   :undoc-members:
   :show-inheritance:
