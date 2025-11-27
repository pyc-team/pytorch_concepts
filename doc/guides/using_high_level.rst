Concept-Based Models
======================================

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

|pyc_logo| PyC provides high-level APIs for quickly building and training concept-based models like Concept Bottleneck Models (CBMs).


Design Principles
-----------------

The |pyc_logo| high-level API simplifies model creation and training through:

- **Pre-built Models**: Ready-to-use models like ``ConceptBottleneckModel``
- **Two Training Modes**: Manual PyTorch or automatic Lightning training
- **Flexible Configuration**: Easy setup of encoders, backbones, losses, and metrics


Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from torch_concepts.nn import ConceptBottleneckModel
   
   model = ConceptBottleneckModel(
       input_size=256,
       annotations=ann,
       variable_distributions=variable_distributions,
       task_names=['cancer']
   )


Two Training Modes
^^^^^^^^^^^^^^^^^^

Models support both manual PyTorch and automatic Lightning training:

- **Manual Mode**: Initialize without loss/optimizer for full control over training loop
- **Lightning Mode**: Initialize with loss/optimizer for automatic training with ``Trainer.fit()``

  - **Loss and Metric Routing**: Predictions automatically routed to correct loss and metric functions based on concept types
  - **Metric Tracking**: Built-in support for tracking aggregate and per-concept metrics during training

Type-Aware Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Using ``GroupConfig``, specify settings once per concept type (binary, categorical, continuous) rather than per concept:

.. code-block:: python

   from torch_concepts import GroupConfig
   from torch.distributions import Bernoulli, Categorical
   
   # Configure distributions by type
   variable_distributions = GroupConfig(
       binary=Bernoulli,      # Applied to all binary concepts
       categorical=Categorical # Applied to all categorical concepts
   )

This scales effortlessly from small datasets (5 concepts) to large ones (312 attributes in CUB-200).


Step 1: Import Libraries
-------------------------

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Prepare Data and Annotations
--------------------------------------

Create sample data with concepts and tasks:

.. code-block:: python

   # Sample dimensions
   batch_size = 32
   input_dim = 64
   
   # Create sample input
   x = torch.randn(batch_size, input_dim)
   
   # Create concept and task labels (binary)
   concept_labels = torch.randint(0, 2, (batch_size, 3)).float()  # round, smooth, bright
   task_labels = torch.randint(0, 2, (batch_size, 2)).float()     # class_A, class_B
   
   # Stack into targets
   targets = torch.cat([concept_labels, task_labels], dim=1)

Annotations describe concepts and tasks. Distributions can be provided in three ways:

**Option 1: In annotations metadata (recommended)**

.. code-block:: python

   from torch.distributions import Bernoulli
   from torch_concepts.annotations import AxisAnnotation, Annotations
   
   ann = Annotations({
       1: AxisAnnotation(
           labels=['round', 'smooth', 'bright', 'class_A', 'class_B'],
           cardinalities=[1, 1, 1, 1, 1],
           metadata={
               'round': {'type': 'discrete', 'distribution': Bernoulli},
               'smooth': {'type': 'discrete', 'distribution': Bernoulli},
               'bright': {'type': 'discrete', 'distribution': Bernoulli},
               'class_A': {'type': 'discrete', 'distribution': Bernoulli},
               'class_B': {'type': 'discrete', 'distribution': Bernoulli}
           }
       )
   })

**Option 2: Via variable_distributions dictionary**

.. code-block:: python

   # Annotations without distributions
   ann = Annotations({
       1: AxisAnnotation(
           labels=['round', 'smooth', 'bright', 'class_A', 'class_B'],
           cardinalities=[1, 1, 1, 1, 1],
           metadata={
               'round': {'type': 'discrete'},
               'smooth': {'type': 'discrete'},
               'bright': {'type': 'discrete'},
               'class_A': {'type': 'discrete'},
               'class_B': {'type': 'discrete'}
           }
       )
   })
   
   # Provide distributions separately
   variable_distributions = {
       'round': Bernoulli,
       'smooth': Bernoulli,
       'bright': Bernoulli,
       'class_A': Bernoulli,
       'class_B': Bernoulli
   }

**Option 3: Using GroupConfig for automatic type-based assignment**

When you have many concepts of the same types, use ``GroupConfig`` to automatically assign distributions based on concept type:

.. code-block:: python

   from torch.distributions import Bernoulli, Categorical
   from torch_concepts import GroupConfig
   
   # Annotations with mixed types
   ann = Annotations({
       1: AxisAnnotation(
           labels=['round', 'smooth', 'bright', 'color', 'shape', 'class_A', 'class_B'],
           cardinalities=[1, 1, 1, 3, 4, 1, 1],
           metadata={
               'round': {'type': 'discrete'},    # binary (card=1)
               'smooth': {'type': 'discrete'},   # binary (card=1)
               'bright': {'type': 'discrete'},   # binary (card=1)
               'color': {'type': 'discrete'},    # categorical (card=3)
               'shape': {'type': 'discrete'},    # categorical (card=4)
               'class_A': {'type': 'discrete'},  # binary (card=1)
               'class_B': {'type': 'discrete'}   # binary (card=1)
           }
       )
   })
   
   # GroupConfig automatically assigns distributions by concept type
   variable_distributions = GroupConfig(
       binary=Bernoulli,      # for all binary concepts (cardinality=1)
       categorical=Categorical # for all categorical concepts (cardinality>1)
   )

This approach is especially useful for:

- Large-scale datasets with many concepts (e.g., CUB-200 with 312 attributes)
- Mixed concept types (binary + categorical)
- Reducing configuration boilerplate

Step 3: Instantiate a Model
----------------------------

.. code-block:: python

   from torch_concepts.nn import ConceptBottleneckModel
   
   # If distributions are in annotations metadata
   model = ConceptBottleneckModel(
       input_size=input_dim,
       annotations=ann,
       task_names=['class_A', 'class_B'],
       latent_encoder_kwargs={'hidden_size': 64, 'n_layers': 2}
   )
   
   # If using variable_distributions dictionary
   model = ConceptBottleneckModel(
       input_size=input_dim,
       annotations=ann,
       variable_distributions=variable_distributions,
       task_names=['class_A', 'class_B'],
       latent_encoder_kwargs={'hidden_size': 64, 'n_layers': 2}
   )
   
   # If using GroupConfig (automatically assigns by concept type)
   model = ConceptBottleneckModel(
       input_size=input_dim,
       annotations=ann,
       variable_distributions=GroupConfig(binary=Bernoulli, categorical=Categorical),
       task_names=['class_A', 'class_B'],
       latent_encoder_kwargs={'hidden_size': 64, 'n_layers': 2}
   )

Step 4: Train - Manual PyTorch
-------------------------------

.. code-block:: python

   import torch.nn as nn
   
   # Manual training loop
   optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
   loss_fn = nn.BCEWithLogitsLoss()
   
   model.train()
   for epoch in range(100):
       optimizer.zero_grad()
       out = model(x, query=['round', 'smooth', 'bright', 'class_A', 'class_B'])
       loss = loss_fn(out, targets)
       loss.backward()
       optimizer.step()
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Step 5: Train - PyTorch Lightning
----------------------------------

.. code-block:: python

   from pytorch_lightning import Trainer
   from torch_concepts.data.base.datamodule import ConceptDataModule
   from torch.utils.data import TensorDataset
   
   # Model with loss and optimizer for Lightning
   model = ConceptBottleneckModel(
       input_size=input_dim,
       annotations=ann,
       task_names=['class_A', 'class_B'],
       loss=nn.BCEWithLogitsLoss(),
       optim_class=torch.optim.AdamW,
       optim_kwargs={'lr': 0.001}
   )
   
   # Create dataset and datamodule
   dataset = TensorDataset(x, targets)
   datamodule = ConceptDataModule(dataset, batch_size=32)
   
   # Train
   trainer = Trainer(max_epochs=100)
   trainer.fit(model, datamodule)

Step 6: Evaluate and Query
---------------------------

After training, query the model for concepts and tasks:

.. code-block:: python

   model.eval()
   with torch.no_grad():
       # Query all variables
       all_predictions = model(x, query=['round', 'smooth', 'bright', 'class_A', 'class_B'])
       
       # Query only concepts
       concept_predictions = model(x, query=['round', 'smooth', 'bright'])
       
       # Query only tasks
       task_predictions = model(x, query=['class_A', 'class_B'])
       
       print(f"All predictions shape: {all_predictions.shape}")  # [32, 5]
       print(f"Concept predictions shape: {concept_predictions.shape}")  # [32, 3]
       print(f"Task predictions shape: {task_predictions.shape}")  # [32, 2]

Advanced: Using GroupConfig for Losses and Metrics
---------------------------------------------------

``GroupConfig`` also works with losses and metrics for mixed concept types:

.. code-block:: python

   import torch.nn as nn
   from torch_concepts import GroupConfig
   from torch_concepts.nn import ConceptLoss, ConceptMetrics
   from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
   from torch.distributions import Bernoulli, Categorical
   
   # Mixed binary and categorical concepts
   ann = Annotations({
       1: AxisAnnotation(
           labels=['is_blue', 'is_large', 'color', 'shape'],
           cardinalities=[1, 1, 3, 4],
           metadata={
               'is_blue': {'type': 'discrete'},   # binary
               'is_large': {'type': 'discrete'},  # binary
               'color': {'type': 'discrete'},     # categorical
               'shape': {'type': 'discrete'}      # categorical
           }
       )
   })
   
   # Configure distributions by type
   variable_distributions = GroupConfig(
       binary=Bernoulli,
       categorical=Categorical
   )
   
   # Configure losses by type
   loss_config = GroupConfig(
       binary=nn.BCEWithLogitsLoss(),
       categorical=nn.CrossEntropyLoss()
   )
   
   # Configure metrics by type
   metrics_config = GroupConfig(
       binary={'accuracy': BinaryAccuracy()},
       categorical={'accuracy': MulticlassAccuracy(num_classes=4)}
   )
   
   # Create loss and metrics
   concept_loss = ConceptLoss(annotations=ann[1], fn_collection=loss_config)
   concept_metrics = ConceptMetrics(
       annotations=ann[1],
       fn_collection=metrics_config,
       summary_metrics=True,
       perconcept_metrics=True
   )
   
   # Create model with all configurations
   model = ConceptBottleneckModel(
       input_size=input_dim,
       annotations=ann,
       variable_distributions=distributions,
       task_names=['class_A', 'class_B'],
       loss=concept_loss,
       metrics=concept_metrics,
       optim_class=torch.optim.AdamW,
       optim_kwargs={'lr': 0.001}
   )

Benefits of GroupConfig:

- **Automatic Assignment**: Distributions/losses/metrics are automatically assigned based on concept type (binary vs categorical)
- **Type Safety**: Validates that required configurations exist for all concept types
- **Reduced Boilerplate**: No need to specify configuration for each concept individually
- **Scalability**: Ideal for datasets with many concepts (e.g., CUB-200 with 312 binary attributes)

Next Steps
----------

- :doc:`Conceptarium Guide </guides/using_conceptarium>` for no-code experimentation
- :doc:`Mid-Level Probabilistic API </guides/using_mid_level_proba>` for custom probabilistic models
- :doc:`Mid-Level Causal API </guides/using_mid_level_causal>` for causal modeling
- :doc:`Low-Level API </guides/using_low_level>` for custom architectures
