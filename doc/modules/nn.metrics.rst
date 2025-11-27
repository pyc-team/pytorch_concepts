Metrics
========

Comprehensive guide for evaluating concept-based models with automatic type-aware 
routing and flexible tracking options.

.. currentmodule:: torch_concepts.nn.modules.metrics

Summary
-------

**Metrics Classes**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptMetrics

**Functional Metrics**

.. autosummary::
   :toctree: generated
   :nosignatures:

   completeness_score
   intervention_score
   cace_score


Overview
--------

The :class:`ConceptMetrics` class provides comprehensive evaluation capabilities 
for concept-based models:

- **Automatic type-aware routing**: Routes predictions to appropriate metrics based on concept types
- **Summary metrics**: Aggregate performance across all concepts of each type
- **Per-concept metrics**: Individual tracking for specific concepts  
- **Flexible configuration**: Three ways to specify metrics (pre-instantiated, class+kwargs, class-only)
- **Split-aware tracking**: Independent metrics for train/validation/test splits
- **TorchMetrics integration**: Seamless integration with TorchMetrics library
- **PyTorch Lightning compatible**: Works with PyTorch Lightning training loops

Quick Example
-------------

.. code-block:: python

   import torch
   import torchmetrics
   from torch_concepts import Annotations, AxisAnnotation, GroupConfig
   from torch_concepts.nn import ConceptMetrics
   from torch.distributions import Bernoulli, Categorical

   # Define concept structure
   annotations = Annotations({
       1: AxisAnnotation(
           labels=['is_round', 'is_smooth', 'color'],
           cardinalities=[1, 1, 3],  # binary, binary, categorical
           metadata={
               'is_round': {'type': 'discrete', 'distribution': Bernoulli},
               'is_smooth': {'type': 'discrete', 'distribution': Bernoulli},
               'color': {'type': 'discrete', 'distribution': Categorical}
           }
       )
   })

   # Configure metrics using GroupConfig
   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
           categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy}
       ),
       summary_metrics=True,
       perconcept_metrics=True
   )

   # During training
   predictions = torch.randn(32, 5)  # endogenous space: 1+1+3 logits
   targets = torch.cat([
       torch.randint(0, 2, (32, 2)),  # binary targets
       torch.randint(0, 3, (32, 1))   # categorical target
   ], dim=1)

   metrics.update(preds=predictions, target=targets, split='train')
   results = metrics.compute('train')
   metrics.reset('train')


Metric Configuration
--------------------

There are three ways to specify metrics in ConceptMetrics, each with different trade-offs.

Method 1: Pre-Instantiated Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass already instantiated metric objects for full control:

.. code-block:: python

   from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={
               'accuracy': BinaryAccuracy(threshold=0.6),
               'f1': BinaryF1Score(threshold=0.5),
               'precision': BinaryPrecision(threshold=0.5)
           }
       ),
       summary_metrics=True
   )

**Pros**: Full control over all parameters

**Cons**: Must manually specify all parameters including ``num_classes`` for categorical metrics

Method 2: Class + User kwargs (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a tuple of ``(MetricClass, kwargs_dict)`` to provide custom parameters while letting
ConceptMetrics handle concept-specific parameters:

.. code-block:: python

   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={
               # Custom threshold, other params use defaults
               'accuracy': (BinaryAccuracy, {'threshold': 0.6}),
               'f1': (BinaryF1Score, {'threshold': 0.5})
           },
           categorical={
               # Custom averaging, num_classes added automatically
               'accuracy': (MulticlassAccuracy, {'average': 'macro'}),
               'f1': (MulticlassF1Score, {'average': 'weighted'})
           }
       ),
       summary_metrics=True
   )

**Pros**: Custom parameters + automatic ``num_classes`` handling

**Cons**: Cannot override automatically-set parameters (raises error if you try)

Method 3: Class Only (Simplest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass just the metric class and let ConceptMetrics handle all instantiation:

.. code-block:: python

   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={
               'accuracy': BinaryAccuracy,
               'precision': BinaryPrecision,
               'recall': BinaryRecall
           },
           categorical={
               # num_classes added automatically per concept
               'accuracy': MulticlassAccuracy
           }
       ),
       summary_metrics=True
   )

**Pros**: Simplest syntax, automatic parameter handling

**Cons**: Cannot customize parameters


Mixed Concept Types
-------------------

Working with Binary and Categorical Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ConceptMetrics automatically handles mixed concept types:

.. code-block:: python

   from torch.distributions import Bernoulli, Categorical

   # Mixed concept types
   annotations = Annotations({
       1: AxisAnnotation(
           labels=('binary1', 'binary2', 'color', 'size'),
           cardinalities=[1, 1, 3, 5],  # 2 binary, 2 categorical
           metadata={
               'binary1': {'type': 'discrete', 'distribution': Bernoulli},
               'binary2': {'type': 'discrete', 'distribution': Bernoulli},
               'color': {'type': 'discrete', 'distribution': Categorical},
               'size': {'type': 'discrete', 'distribution': Categorical}
           }
       )
   })

   # Configure metrics for both types
   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={
               'accuracy': BinaryAccuracy,
               'f1': BinaryF1Score
           },
           categorical={
               # Custom averaging for categorical
               'accuracy': (MulticlassAccuracy, {'average': 'macro'})
           }
       ),
       summary_metrics=True,
       perconcept_metrics=True
   )

   # Predictions in endogenous space
   # 2 binary + (3 + 5) categorical = 10 dimensions
   predictions = torch.randn(32, 10)
   
   # Targets in concept space
   targets = torch.cat([
       torch.randint(0, 2, (32, 2)),  # Binary targets
       torch.randint(0, 3, (32, 1)),  # Color (3 classes)
       torch.randint(0, 5, (32, 1))   # Size (5 classes)
   ], dim=1)

   metrics.update(preds=predictions, target=targets, split='train')
   results = metrics.compute('train')

   # Results include both summary and per-concept metrics:
   # 'train/SUMMARY-binary_accuracy'
   # 'train/SUMMARY-binary_f1'
   # 'train/SUMMARY-categorical_accuracy'
   # 'train/binary1_accuracy'
   # 'train/binary2_accuracy'
   # 'train/color_accuracy'
   # 'train/size_accuracy'


Summary vs Per-Concept Metrics
-------------------------------

Summary Metrics Only
~~~~~~~~~~~~~~~~~~~~

Summary metrics aggregate performance across all concepts of each type:

.. code-block:: python

   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={'accuracy': BinaryAccuracy}
       ),
       summary_metrics=True,
       perconcept_metrics=False  # No per-concept tracking
   )

   results = metrics.compute('train')
   # Output: {'train/SUMMARY-binary_accuracy': tensor(0.8542)}

Per-Concept Metrics for All Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track each concept individually:

.. code-block:: python

   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={'accuracy': BinaryAccuracy}
       ),
       summary_metrics=False,  # No summary
       perconcept_metrics=True  # All concepts individually
   )

   results = metrics.compute('train')
   # Output: {
   #     'train/is_round_accuracy': tensor(0.9000),
   #     'train/is_smooth_accuracy': tensor(0.8500),
   #     'train/is_bright_accuracy': tensor(0.8000)
   # }

Selective Per-Concept Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track only specific concepts:

.. code-block:: python

   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           binary={'accuracy': BinaryAccuracy}
       ),
       summary_metrics=True,
       perconcept_metrics=['is_round', 'is_bright']  # Only these two
   )

   results = metrics.compute('train')
   # Output: {
   #     'train/SUMMARY-binary_accuracy': tensor(0.8542),
   #     'train/is_round_accuracy': tensor(0.9000),
   #     'train/is_bright_accuracy': tensor(0.8000)
   #     # Note: is_smooth is not tracked individually
   # }


Multiple Data Splits
---------------------

Train/Validation/Test Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ConceptMetrics maintains independent state for each split:

.. code-block:: python

   # Training loop
   for batch in train_loader:
       predictions = model(batch['inputs'])
       targets = batch['concepts']
       metrics.update(pred=predictions, target=targets, split='train')

   # Validation loop
   for batch in val_loader:
       predictions = model(batch['inputs'])
       targets = batch['concepts']
       metrics.update(pred=predictions, target=targets, split='val')

   # Compute both splits independently
   train_results = metrics.compute('train')
   val_results = metrics.compute('val')

   print(f"Train accuracy: {train_results['train/SUMMARY-binary_accuracy']:.4f}")
   print(f"Val accuracy: {val_results['val/SUMMARY-binary_accuracy']:.4f}")

   # Reset both splits for next epoch
   metrics.reset()  # Resets all splits


Integration with PyTorch Lightning
-----------------------------------

ConceptMetrics integrates seamlessly with PyTorch Lightning:

Basic Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytorch_lightning as pl
   from torch_concepts.nn import ConceptBottleneckModel

   class LitConceptModel(pl.LightningModule):
       def __init__(self, annotations):
           super().__init__()
           
           # Initialize model
           self.model = ConceptBottleneckModel(
               task_names=['task1', 'task2'],
               input_size=128,
               annotations=annotations
           )
           
           # Initialize metrics
           self.metrics = ConceptMetrics(
               annotations=annotations[1],
               fn_collection=GroupConfig(
                   binary={'accuracy': BinaryAccuracy}
               ),
               summary_metrics=True,
               perconcept_metrics=False
           )
       
       def training_step(self, batch, batch_idx):
           # Forward pass
           outputs = self.model(batch['inputs'])
           
           # Update metrics
           self.metrics.update(pred=outputs, target=batch['concepts'], split='train')
           
           # Compute and return loss
           loss = self.compute_loss(outputs, batch)
           return loss
       
       def validation_step(self, batch, batch_idx):
           outputs = self.model(batch['inputs'])
           
           # Update validation metrics
           self.metrics.update(pred=outputs, target=batch['concepts'], split='val')
           
           # Compute validation loss
           loss = self.compute_loss(outputs, batch)
           self.log('val_loss', loss)
       
       def on_train_epoch_end(self):
           # Compute metrics
           train_metrics = self.metrics.compute('train')
           
           # Log to logger (wandb, tensorboard, etc.)
           self.log_dict(train_metrics)
           
           # Reset for next epoch
           self.metrics.reset('train')
       
       def on_validation_epoch_end(self):
           # Compute validation metrics
           val_metrics = self.metrics.compute('val')
           
           # Log metrics
           self.log_dict(val_metrics)
           
           # Reset
           self.metrics.reset('val')


Best Practices
--------------

1. **Choose appropriate metrics**: Select metrics that align with your evaluation goals

   .. code-block:: python

      # For imbalanced datasets
      metrics = ConceptMetrics(
          annotations=annotations[1],
          fn_collection=GroupConfig(
              binary={
                  'f1': BinaryF1Score,  # Better for imbalanced data
                  'auroc': BinaryAUROC
              }
          ),
          summary_metrics=True
      )

2. **Use per-concept metrics selectively**: Track only concepts of interest to reduce logging overhead

   .. code-block:: python

      # Track only important concepts
      metrics = ConceptMetrics(
          annotations=annotations[1],
          fn_collection=GroupConfig(
              binary={'accuracy': BinaryAccuracy}
          ),
          summary_metrics=True,
          perconcept_metrics=['critical_concept1', 'critical_concept2']
      )

3. **Always reset after computing**: Prevents mixing data from different epochs

   .. code-block:: python

      # Good practice
      results = metrics.compute('train')
      log_metrics(results)
      metrics.reset('train')

4. **Use class+kwargs for flexibility**: Recommended approach for most use cases

   .. code-block:: python

      # Flexible and automatic
      metrics = ConceptMetrics(
          annotations=annotations[1],
          fn_collection=GroupConfig(
              binary={
                  'f1': (BinaryF1Score, {'threshold': 0.5})
              }
          ),
          summary_metrics=True
      )

5. **Monitor both summary and per-concept metrics**: Summary for overall performance, per-concept for diagnosing issues


Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue 1: ValueError about num_classes**

.. code-block:: python

   # Wrong: Providing num_classes when it's set automatically
   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           categorical={
               'accuracy': (MulticlassAccuracy, {'num_classes': 5, 'average': 'macro'})
           }
       ),
       summary_metrics=True
   )
   # Error: 'num_classes' should not be provided in metric kwargs

   # Correct: Let ConceptMetrics set num_classes
   metrics = ConceptMetrics(
       annotations=annotations[1],
       fn_collection=GroupConfig(
           categorical={
               'accuracy': (MulticlassAccuracy, {'average': 'macro'})
           }
       ),
       summary_metrics=True
   )

**Issue 2: NotImplementedError for continuous concepts**

Continuous concepts are not yet supported. Ensure all concepts are discrete:

.. code-block:: python

   # Make sure all concepts are discrete
   annotations = Annotations({
       1: AxisAnnotation(
           labels=('concept1', 'concept2'),
           metadata={
               'concept1': {'type': 'discrete'},  # Not 'continuous'
               'concept2': {'type': 'discrete'}
           }
       )
   })

**Issue 3: Shape mismatches**

Ensure predictions are in endogenous space and targets match concept space:

.. code-block:: python

   # Binary concepts: predictions shape matches targets shape
   predictions = torch.randn(32, 3)  # 3 binary concepts
   targets = torch.randint(0, 2, (32, 3))  # Shape must match
   
   # Mixed: predictions in endogenous space, targets in concept space
   predictions = torch.randn(32, 8)  # 2 binary + (3+3) categorical
   targets = torch.cat([
       torch.randint(0, 2, (32, 2)),  # Binary
       torch.randint(0, 3, (32, 1)),  # Cat1
       torch.randint(0, 3, (32, 1))   # Cat2
   ], dim=1)  # Shape: (32, 4)


Class Documentation
-------------------

.. autoclass:: ConceptMetrics
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __repr__

Functional Metrics
------------------

The module also provides functional metrics for specialized evaluation tasks:

.. currentmodule:: torch_concepts.nn.functional

.. autosummary::
   :toctree: generated
   :nosignatures:

   completeness_score
   intervention_score
   cace_score

.. autofunction:: completeness_score
.. autofunction:: intervention_score
.. autofunction:: cace_score

See Also
--------

- :doc:`nn.loss`: Loss functions for concept-based models
- :class:`torch_concepts.nn.modules.utils.GroupConfig`: Configuration helper
- :class:`torch_concepts.annotations.Annotations`: Concept annotations

