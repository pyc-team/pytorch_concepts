Out-of-the-box Models
=====================

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pl_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/lightning.svg
    :width: 20px
    :align: middle

|pyc_logo| PyC provides ready-to-use models for concept-based learning with minimal configuration.
Models support both manual PyTorch training and automatic |pl_logo| PyTorch Lightning training.


Design Principles
-----------------


|pyc_logo| PyC out-of-the-box models handle complexity automatically:

- **Type-Aware Routing**: Predictions automatically routed to correct loss and metric functions based on concept types
- **Minimal Configuration**: Use GroupConfig to specify settings once per type (binary, categorical) rather than per concept
- **Flexible Training**: Choose between manual PyTorch control or automatic Lightning training

Two Training Modes
^^^^^^^^^^^^^^^^^^^

**Manual PyTorch Mode**: Initialize without loss/optimizer for full control

.. code-block:: python

   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       variable_distributions=variable_distributions,
       task_names=['cancer']
   )
   
   # Write your own training loop
   optimizer = torch.optim.Adam(model.parameters())
   for epoch in range(100):
       # Your training code

**Lightning Mode**: Initialize with loss/optimizer for automatic training

.. code-block:: python

   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       task_names=['cancer'],
       loss=concept_loss,         # torch loss or ConceptLoss
       metrics=concept_metrics,   # torchmetrics or ConceptMetrics
       optim_class=torch.optim.AdamW,
       optim_kwargs={'lr': 0.001}
   )
   
   # Automatic training
   trainer = Trainer(max_epochs=100)
   trainer.fit(model, datamodule)


Detailed Guides
^^^^^^^^^^^^^^^

.. dropdown:: Annotations
   :icon: tag
   
   **Concept and Task Metadata**
   
   Annotations store metadata about concepts including names, cardinalities, distribution types,
   and custom attributes. They specify the structure and properties of concepts for models,
   losses, and metrics.
   
   **Quick Start**
   
   .. code-block:: python
   
      from torch_concepts.annotations import AxisAnnotation, Annotations
      from torch.distributions import Bernoulli, Categorical
      
      # Define concept structure with distributions
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
   
   **Key Components**
   
   - **labels**: List of concept and task names
   - **cardinalities**: Number of classes for each (1 for binary, >1 for categorical)
   - **metadata**: Dictionary with concept properties including distribution types
   
   **Distribution Assignment Methods**
   
   Distributions can be provided in three ways:
   
   **Method 1: In annotations metadata (recommended)**
   
   .. code-block:: python
   
      ann = Annotations({
          1: AxisAnnotation(
              labels=['is_round', 'color'],
              cardinalities=[1, 3],
              metadata={
                  'is_round': {'type': 'discrete', 'distribution': Bernoulli},
                  'color': {'type': 'discrete', 'distribution': Categorical}
              }
          )
      })
      
      # Use directly in model
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          task_names=['class_A']
      )
   
   **Method 2: Via variable_distributions dictionary**
   
   .. code-block:: python
   
      # Annotations without distributions
      ann = Annotations({
          1: AxisAnnotation(
              labels=['is_round', 'color'],
              cardinalities=[1, 3],
              metadata={
                  'is_round': {'type': 'discrete'},
                  'color': {'type': 'discrete'}
              }
          )
      })
      
      # Provide distributions separately
      variable_distributions = {
          'is_round': Bernoulli,
          'color': Categorical
      }
      
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          variable_distributions=variable_distributions,
          task_names=['class_A']
      )
   
   **Method 3: Using GroupConfig (for mixed types)**
   
   .. code-block:: python
   
      from torch_concepts import GroupConfig
      
      # Annotations with mixed types
      ann = Annotations({
          1: AxisAnnotation(
              labels=['is_round', 'is_smooth', 'color', 'shape'],
              cardinalities=[1, 1, 3, 4],
              metadata={
                  'is_round': {'type': 'discrete'},   # binary (card=1)
                  'is_smooth': {'type': 'discrete'},  # binary (card=1)
                  'color': {'type': 'discrete'},      # categorical (card=3)
                  'shape': {'type': 'discrete'}       # categorical (card=4)
              }
          )
      })
      
      # GroupConfig automatically assigns by concept type
      variable_distributions = GroupConfig(
          binary=Bernoulli,      # all concepts with cardinality=1
          categorical=Categorical # all concepts with cardinality>1
      )
      
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          variable_distributions=variable_distributions,
          task_names=['class_A']
      )
   
   **Usage with Loss and Metrics**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptLoss, ConceptMetrics
      from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
      from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
      
      # Loss configuration
      loss_config = GroupConfig(
          binary=BCEWithLogitsLoss(),
          categorical=CrossEntropyLoss()
      )
      loss = ConceptLoss(annotations=ann, fn_collection=loss_config)
      
      # Metrics configuration
      metrics_config = GroupConfig(
          binary={'accuracy': BinaryAccuracy()},
          categorical={'accuracy': (MulticlassAccuracy, {'average': 'macro'})}
      )
      metrics = ConceptMetrics(
          annotations=ann,
          fn_collection=metrics_config,
          summary_metrics=True,
          perconcept_metrics=True
      )
   
   **Special Cases**
   
   **Missing distributions**: If distributions are not in metadata and variable_distributions
   is not provided, the model will raise an assertion error.
   
   **Task concepts**: Concepts that are prediction targets (tasks) should be included in
   the annotations and specified via the ``task_names`` parameter.
   
   **Custom metadata**: Add custom fields to metadata for application-specific needs:
   
   .. code-block:: python
   
      metadata={
          'is_round': {
              'type': 'discrete',
              'distribution': Bernoulli,
              'description': 'Object has rounded shape',
              'importance': 0.8
          }
      }

.. dropdown:: GroupConfig
   :icon: gear
   
   **Type-Based Configuration Helper**
   
   GroupConfig simplifies configuration for models with mixed concept types (binary and categorical).
   Instead of configuring each concept individually, configure once per type.
   
   **Quick Start**
   
   .. code-block:: python
   
      from torch_concepts import GroupConfig
      from torch.distributions import Bernoulli, Categorical
      from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
      
      # Configure distributions by type
      variable_distributions = GroupConfig(
          binary=Bernoulli,
          categorical=Categorical
      )
      
      # Configure losses by type
      loss_config = GroupConfig(
          binary=BCEWithLogitsLoss(),
          categorical=CrossEntropyLoss()
      )
      
      # Configure metrics by type
      from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
      
      metrics_config = GroupConfig(
          binary={'accuracy': BinaryAccuracy()},
          categorical={'accuracy': MulticlassAccuracy}
      )
   
   **Automatic Type Detection**
   
   GroupConfig automatically determines concept types based on cardinalities:
   
   - **Binary**: cardinality = 1
   - **Categorical**: cardinality > 1
   - **Continuous**: when type='continuous' in metadata (not yet fully supported)
   
   .. code-block:: python
   
      # Annotations with mixed types
      ann = Annotations({
          1: AxisAnnotation(
              labels=['c1', 'c2', 'c3', 'c4'],
              cardinalities=[1, 1, 3, 5],  # 2 binary + 2 categorical
              metadata={...}
          )
      })
      
      # Single configuration for all binary, another for all categorical
      variable_distributions = GroupConfig(
          binary=Bernoulli,      # Applied to c1, c2 (cardinality=1)
          categorical=Categorical # Applied to c3, c4 (cardinality>1)
      )
   
   **Benefits**
   
   1. **Scalability**: Configure 312 CUB-200 attributes as easily as 5 concepts
   2. **Consistency**: Same settings applied to all concepts of the same type
   3. **Maintainability**: Change one configuration instead of hundreds
   4. **Type Safety**: Validates that all required types are configured
   
   **Usage with Models**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptBottleneckModel
      
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          variable_distributions=GroupConfig(
              binary=Bernoulli,
              categorical=Categorical
          ),
          task_names=['class_A', 'class_B']
      )
   
   **Usage with Loss Functions**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptLoss
      
      loss = ConceptLoss(
          annotations=ann,
          fn_collection=GroupConfig(
              binary=BCEWithLogitsLoss(),
              categorical=CrossEntropyLoss()
          )
      )
   
   **Usage with Metrics**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptMetrics
      
      metrics = ConceptMetrics(
          annotations=ann,
          fn_collection=GroupConfig(
              binary={'accuracy': BinaryAccuracy(), 'f1': BinaryF1Score()},
              categorical={'accuracy': (MulticlassAccuracy, {'average': 'macro'})}
          ),
          summary_metrics=True,
          perconcept_metrics=False
      )
   
   **Special Cases**
   
   **All same type**: GroupConfig works even when all concepts are the same type:
   
   .. code-block:: python
   
      # All binary
      variable_distributions = GroupConfig(binary=Bernoulli)
      
      # All categorical
      variable_distributions = GroupConfig(categorical=Categorical)
   
   **Missing types**: If a required type is not configured, an error is raised:
   
   .. code-block:: python
   
      # ERROR: has categorical concepts but only binary configured
      variable_distributions = GroupConfig(binary=Bernoulli)
      # Will raise error when used with mixed annotations

.. dropdown:: Loss Functions
   :icon: flame
   
   **Type-Aware Loss Computation**
   
   ConceptLoss automatically routes predictions to appropriate loss functions based on
   concept types (binary, categorical). It handles mixed concept types seamlessly.
   
   **Quick Start**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptLoss
      from torch_concepts import GroupConfig
      from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
      
      # Configure losses by type
      loss_config = GroupConfig(
          binary=BCEWithLogitsLoss(),
          categorical=CrossEntropyLoss()
      )
      
      # Create type-aware loss
      loss = ConceptLoss(annotations=ann, fn_collection=loss_config)
      
      # Use in training
      predictions = model(x)
      targets = batch['concepts']
      loss_value = loss(predictions, targets)
   
   **Automatic Routing**
   
   ConceptLoss automatically:
   
   1. Splits predictions and targets by concept type
   2. Routes binary concepts to binary loss
   3. Routes categorical concepts to categorical loss
   4. Aggregates results
   
   .. code-block:: python
   
      # Mixed predictions: 2 binary + 3-class categorical + 1 binary
      predictions = torch.randn(32, 6)  # Shape: [batch, 1+1+3+1]
      
      # Mixed targets: 2 binary + 1 categorical (class indices) + 1 binary
      targets = torch.cat([
          torch.randint(0, 2, (32, 2)),  # Binary targets
          torch.randint(0, 3, (32, 1)),  # Categorical target (indices)
          torch.randint(0, 2, (32, 1))   # Binary target
      ], dim=1)
      
      # Automatic routing to appropriate losses
      loss_value = loss(predictions, targets)
   
   **Weighted Loss**
   
   Use WeightedConceptLoss for custom weighting:
   
   .. code-block:: python
   
      from torch_concepts.nn import WeightedConceptLoss
      
      loss = WeightedConceptLoss(
          annotations=ann,
          fn_collection=loss_config,
          concept_loss_weight=0.5,  # Weight for concept predictions
          task_loss_weight=1.0       # Weight for task predictions
      )
   
   **Integration with Models**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptBottleneckModel
      
      # Lightning training mode
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          task_names=['class_A', 'class_B'],
          loss=loss,  # Automatic loss computation
          optim_class=torch.optim.AdamW,
          optim_kwargs={'lr': 0.001}
      )
      
      # Manual training mode
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          task_names=['class_A', 'class_B']
      )
      
      optimizer = torch.optim.Adam(model.parameters())
      for batch in dataloader:
          predictions = model(batch['inputs'])
          loss_value = loss(predictions, batch['concepts'])
          loss_value.backward()
          optimizer.step()
   
   **Special Cases**
   
   **Target format**: Targets must match the concept space structure:
   
   - Binary concepts: targets are 0 or 1 (shape: [batch, n_binary])
   - Categorical concepts: targets are class indices (shape: [batch, 1] per concept)
   
   **Reduction**: Losses support different reduction modes ('mean', 'sum', 'none'):
   
   .. code-block:: python
   
      loss_config = GroupConfig(
          binary=BCEWithLogitsLoss(reduction='mean'),
          categorical=CrossEntropyLoss(reduction='mean')
      )

.. dropdown:: Metrics
   :icon: graph
   
   **Type-Aware Metric Tracking**
   
   ConceptMetrics automatically routes predictions to appropriate metrics based on concept
   types and provides both summary (aggregate) and per-concept tracking.
   
   **Quick Start**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptMetrics
      from torch_concepts import GroupConfig
      from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
      
      # Configure metrics by type
      metrics_config = GroupConfig(
          binary={'accuracy': BinaryAccuracy()},
          categorical={'accuracy': MulticlassAccuracy}
      )
      
      # Create metrics tracker
      metrics = ConceptMetrics(
          annotations=ann,
          fn_collection=metrics_config,
          summary_metrics=True,      # Aggregate by type
          perconcept_metrics=True     # Individual concept tracking
      )
      
      # During training
      metrics.update(preds=predictions, target=targets, split='train')
      
      # End of epoch
      results = metrics.compute('train')
      metrics.reset('train')
   
   **Summary vs Per-Concept Metrics**
   
   **Summary metrics**: Aggregate performance across all concepts of each type
   
   .. code-block:: python
   
      metrics = ConceptMetrics(
          annotations=ann,
          fn_collection=metrics_config,
          summary_metrics=True,
          perconcept_metrics=False
      )
      
      results = metrics.compute('train')
      # Output: {
      #     'train/SUMMARY-binary_accuracy': tensor(0.8542),
      #     'train/SUMMARY-categorical_accuracy': tensor(0.7621)
      # }
   
   **Per-concept metrics**: Track each concept individually
   
   .. code-block:: python
   
      metrics = ConceptMetrics(
          annotations=ann,
          fn_collection=metrics_config,
          summary_metrics=False,
          perconcept_metrics=True
      )
      
      results = metrics.compute('train')
      # Output: {
      #     'train/is_round_accuracy': tensor(0.9000),
      #     'train/is_smooth_accuracy': tensor(0.8500),
      #     'train/color_accuracy': tensor(0.7621)
      # }
   
   **Selective tracking**: Track only specific concepts
   
   .. code-block:: python
   
      metrics = ConceptMetrics(
          annotations=ann,
          fn_collection=metrics_config,
          summary_metrics=True,
          perconcept_metrics=['is_round', 'color']  # Only these
      )
   
   **Multiple Metrics per Type**
   
   .. code-block:: python
   
      from torchmetrics.classification import BinaryF1Score, BinaryPrecision
      
      metrics_config = GroupConfig(
          binary={
              'accuracy': BinaryAccuracy(),
              'f1': BinaryF1Score(),
              'precision': BinaryPrecision()
          },
          categorical={
              'accuracy': (MulticlassAccuracy, {'average': 'macro'}),
              'f1': (MulticlassF1Score, {'average': 'weighted'})
          }
      )
   
   **Split-Aware Tracking**
   
   Maintain independent metrics for train/validation/test:
   
   .. code-block:: python
   
      # Training loop
      for batch in train_loader:
          predictions = model(batch['inputs'])
          metrics.update(pred=predictions, target=batch['concepts'], split='train')
      
      # Validation loop
      for batch in val_loader:
          predictions = model(batch['inputs'])
          metrics.update(pred=predictions, target=batch['concepts'], split='val')
      
      # Compute separately
      train_results = metrics.compute('train')
      val_results = metrics.compute('val')
      
      # Reset for next epoch
      metrics.reset('train')
      metrics.reset('val')
   
   **Integration with Lightning**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptBottleneckModel
      
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          task_names=['class_A', 'class_B'],
          loss=loss,
          metrics=metrics,  # Automatic metric tracking
          optim_class=torch.optim.AdamW,
          optim_kwargs={'lr': 0.001}
      )
      
      trainer = Trainer(max_epochs=100)
      trainer.fit(model, datamodule)
      # Metrics automatically logged
   
   **Special Cases**
   
   **Metric configuration methods**: Three ways to specify metrics
   
   1. Pre-instantiated: ``{'accuracy': BinaryAccuracy()}``
   2. Class + kwargs: ``{'accuracy': (BinaryAccuracy, {'threshold': 0.6})}``
   3. Class only: ``{'accuracy': BinaryAccuracy}``
   
   **Target format**: Targets must be in concept space:
   
   - Binary: 0 or 1 values
   - Categorical: class indices (0 to num_classes-1)
   
   **num_classes**: For categorical metrics, num_classes is automatically set based on cardinalities

.. dropdown:: Models
   :icon: rocket
   
   **Pre-Built Concept-Based Models**
   
   PyC provides ready-to-use models like ConceptBottleneckModel that support both manual
   PyTorch training and automatic Lightning training.
   
   **Quick Start**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptBottleneckModel
      from torch_concepts import GroupConfig
      from torch.distributions import Bernoulli, Categorical
      
      # Basic model
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          variable_distributions=GroupConfig(
              binary=Bernoulli,
              categorical=Categorical
          ),
          task_names=['class_A', 'class_B']
      )
   
   **Manual PyTorch Training**
   
   .. code-block:: python
   
      # Model without loss/optimizer
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          task_names=['class_A', 'class_B'],
          latent_encoder_kwargs={'hidden_size': 128, 'n_layers': 2}
      )
      
      # Custom training loop
      optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
      loss_fn = nn.BCEWithLogitsLoss()
      
      model.train()
      for epoch in range(100):
          for batch in dataloader:
              optimizer.zero_grad()
              
              # Forward pass - query all concepts and tasks
              predictions = model(
                  batch['inputs']['x'],
                  query=['round', 'smooth', 'bright', 'class_A', 'class_B']
              )
              
              loss = loss_fn(predictions, batch['targets'])
              loss.backward()
              optimizer.step()
   
   **Lightning Training**
   
   .. code-block:: python
   
      from torch_concepts.nn import ConceptLoss, ConceptMetrics
      
      # Model with loss, metrics, and optimizer
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          task_names=['class_A', 'class_B'],
          loss=ConceptLoss(annotations=ann, fn_collection=loss_config),
          metrics=ConceptMetrics(
              annotations=ann,
              fn_collection=metrics_config,
              summary_metrics=True,
              perconcept_metrics=True
          ),
          optim_class=torch.optim.AdamW,
          optim_kwargs={'lr': 0.001}
      )
      
      # Automatic training
      from pytorch_lightning import Trainer
      
      trainer = Trainer(max_epochs=100)
      trainer.fit(model, datamodule)
   
   **Model Architecture**
   
   .. code-block:: python
   
      model = ConceptBottleneckModel(
          input_size=256,                    # After backbone (if any)
          annotations=ann,
          task_names=['class_A', 'class_B'],
          
          # Optional backbone for feature extraction
          backbone=torchvision.models.resnet18(pretrained=True),
          
          # Latent encoder configuration
          latent_encoder_kwargs={
              'hidden_size': 128,    # Hidden dimension
              'n_layers': 2,         # Number of layers
              'activation': 'relu',  # Activation function
              'dropout': 0.1         # Dropout rate
          },
          
          # Distribution configuration
          variable_distributions=GroupConfig(
              binary=Bernoulli,
              categorical=Categorical
          )
      )
   
   **Querying Models**
   
   Models support flexible querying of concepts and tasks:
   
   .. code-block:: python
   
      model.eval()
      with torch.no_grad():
          # Query all variables
          all_preds = model(x, query=['round', 'smooth', 'bright', 'class_A'])
          # Shape: [batch, 4]
          
          # Query only concepts
          concept_preds = model(x, query=['round', 'smooth', 'bright'])
          # Shape: [batch, 3]
          
          # Query only tasks
          task_preds = model(x, query=['class_A', 'class_B'])
          # Shape: [batch, 2]
          
          # Query specific subset
          subset_preds = model(x, query=['round', 'class_A'])
          # Shape: [batch, 2]
   
   **Available Models**
   
   - **ConceptBottleneckModel**: Standard CBM with joint training
   - **ConceptBottleneckModel_Joint**: Explicit joint training variant
   - **BlackBox**: Non-interpretable baseline for comparison
   
   **Special Cases**
   
   **Backbone integration**: For image data, use a backbone for feature extraction
   
   .. code-block:: python
   
      import torchvision.models as models
      
      backbone = models.resnet18(pretrained=True)
      # Remove final classification layer
      backbone = nn.Sequential(*list(backbone.children())[:-1])
      
      model = ConceptBottleneckModel(
          input_size=512,  # ResNet18 output size
          annotations=ann,
          backbone=backbone,
          task_names=['class_A']
      )
   
   **No latent encoder**: For pre-computed features, skip the encoder
   
   .. code-block:: python
   
      model = ConceptBottleneckModel(
          input_size=256,
          annotations=ann,
          task_names=['class_A'],
          latent_encoder_kwargs=None  # Use Identity, no encoding
      )


Complete Example
----------------

Putting it all together:

.. code-block:: python

    import torch
    from torch.distributions import Bernoulli, Categorical
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
    from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
    from pytorch_lightning import Trainer

    from torch_concepts import GroupConfig
    from torch_concepts.nn import (
        ConceptBottleneckModel,
        ConceptLoss,
        ConceptMetrics
    )
    from torch_concepts.data.datamodules import BnLearnDataModule

    # Use the insurance dataset from BnLearn (mixed binary and categorical concepts)
    datamodule = BnLearnDataModule(
        name='insurance',
        root='./data/insurance',
        seed=42,
        n_gen=1000,
        batch_size=32,
        val_size=0.1,
        test_size=0.2
    )

    # Setup the datamodule to load/generate data
    datamodule.setup('fit')

    # Get annotations from the dataset
    ann = datamodule.annotations

    print(f"Dataset concepts: {ann[1].labels}")
    print(f"Concept cardinalities: {ann[1].cardinalities}")

    # 2. Create loss and metrics
    loss = ConceptLoss(
        annotations=ann, 
        fn_collection=GroupConfig(
            binary=BCEWithLogitsLoss(),
            categorical=CrossEntropyLoss()
        )
    )

    metrics = ConceptMetrics(
        annotations=ann,
        fn_collection=GroupConfig(
            binary={'accuracy': BinaryAccuracy()},
            categorical={'accuracy': (MulticlassAccuracy, {'average': 'micro'})}
        ),
        summary_metrics=True,
        perconcept_metrics=True
    )

    # 3. Create model with all configurations
    # Get input size from first batch
    sample_batch = next(iter(datamodule.train_dataloader()))
    # The batch['inputs'] is the tensor directly, not a nested dict
    if isinstance(sample_batch['inputs'], dict):
        input_size = sample_batch['inputs']['x'].shape[1]
    else:
        input_size = sample_batch['inputs'].shape[1]
    print(f"Input size: {input_size}")

    model = ConceptBottleneckModel(
        input_size=input_size,
        annotations=ann,
        variable_distributions=GroupConfig(
            binary=Bernoulli,
            categorical=Categorical
        ),
        task_names=[],  # No task names for this unsupervised example
        loss=loss,
        metrics=metrics,
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.001},
        latent_encoder_kwargs={'hidden_size': 64, 'n_layers': 1}
    )

    print(f"\nModel created successfully!")
    print(f"Number of concepts: {len(ann[1].labels)}")
    print(f"Binary concepts: {sum(1 for c in ann[1].cardinalities if c == 1)}")
    print(f"Categorical concepts: {sum(1 for c in ann[1].cardinalities if c > 1)}")

    # 4. Train with Lightning
    trainer = Trainer(max_epochs=10, enable_checkpointing=False, logger=False)
    trainer.fit(model, datamodule=datamodule)

    # 5. Evaluate
    test_results = trainer.test(model, datamodule=datamodule)

    # 6. Make predictions
    model.eval()
    test_batch = next(iter(datamodule.test_dataloader()))
    # Get the actual tensor from batch
    if isinstance(test_batch['inputs'], dict):
        test_data = test_batch['inputs']['x'][:10]
    else:
        test_data = test_batch['inputs'][:10]

    with torch.no_grad():
        # Query first 3 concepts
        test_predictions = model(test_data, query=ann[1].labels[:3])
        print(f"\n✓ Test predictions shape: {test_predictions.shape}")
        print(f"✓ Queried concepts: {ann[1].labels[:3]}")


Next Steps
----------

- :doc:`/modules/high_level_api` - API reference for out-of-the-box models
- :doc:`/modules/nn.loss` - Loss functions API reference
- :doc:`/modules/nn.metrics` - Metrics API reference
- :doc:`/modules/annotations` - Annotations API reference
- :doc:`using_conceptarium` - No-code experimentation framework
- :doc:`using_mid_level_proba` - Custom probabilistic models
- :doc:`using_low_level` - Custom architectures from scratch
