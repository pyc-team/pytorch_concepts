High-level API
=====================

High-level API models allow you to quickly build and train concept-based models using pre-configured components and minimal code.

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Documentation
----------------

.. toctree::
   :maxdepth: 1

   nn.base.high
   annotations
   nn.models.high
   nn.loss
   nn.metrics

Design principles
-----------------

Annotations
^^^^^^^^^^^

Annotations define the structure of concepts and tasks in your model by describing their types, cardinalities, and distributions. 

**Basic Annotation Structure**

Annotations consist of axis annotations that describe variables along a dimension:

.. code-block:: python

   import torch_concepts as pyc
   from torch.distributions import Bernoulli, Categorical
   
   # Define concepts and tasks
   labels = ["is_round", "is_smooth", "color", "class_A", "class_B"]
   cardinalities = [1, 1, 3, 1, 1]  # binary, binary, categorical(3), binary, binary
   
   # Metadata with types and distributions
   metadata = {
       'is_round': {'type': 'discrete', 'distribution': Bernoulli},
       'is_smooth': {'type': 'discrete', 'distribution': Bernoulli},
       'color': {'type': 'discrete', 'distribution': Categorical},
       'class_A': {'type': 'discrete', 'distribution': Bernoulli},
       'class_B': {'type': 'discrete', 'distribution': Bernoulli}
   }
   
   annotations = pyc.Annotations({
       1: pyc.AxisAnnotation(
           labels=labels,
           cardinalities=cardinalities,
           metadata=metadata
       )
   })

**GroupConfig for Automatic Configuration**

For models with many concepts, use ``GroupConfig`` to automatically assign configurations based on concept type:

.. code-block:: python

   from torch_concepts import GroupConfig
   
   # Define annotations without individual distributions
   annotations = pyc.Annotations({
       1: pyc.AxisAnnotation(
           labels=["is_round", "is_smooth", "color", "shape"],
           cardinalities=[1, 1, 3, 4],
           metadata={
               'is_round': {'type': 'discrete'},   # binary (card=1)
               'is_smooth': {'type': 'discrete'},  # binary (card=1)
               'color': {'type': 'discrete'},      # categorical (card=3)
               'shape': {'type': 'discrete'}       # categorical (card=4)
           }
       )
   })
   
   # Automatically assign distributions by type
   variable_distributions = GroupConfig(
       binary=Bernoulli,      # for cardinality=1
       categorical=Categorical # for cardinality>1
   )

This approach scales efficiently to datasets with hundreds of concepts (e.g., CUB-200 with 312 attributes).

Out-of-the-box Models
^^^^^^^^^^^^^^^^^^^^^

|pyc_logo| PyC provides ready-to-use models that can be instantiated with minimal configuration:

**Concept Bottleneck Model (CBM)**

A CBM learns interpretable concept representations and uses them to predict tasks:

.. code-block:: python

   from torch_concepts.nn import ConceptBottleneckModel
   
   model = ConceptBottleneckModel(
       input_size=2048,              # e.g., ResNet feature dimension
       annotations=annotations,
       task_names=['class_A', 'class_B'],
       variable_distributions=distributions,  # Optional: GroupConfig or dict
       latent_encoder_kwargs={
           'hidden_size': 128,
           'n_layers': 2,
           'activation': 'relu',
           'dropout': 0.1
       }
   )

**BlackBox Model**

A standard neural network for comparison baselines:

.. code-block:: python

   from torch_concepts.nn import BlackBox
   
   model = BlackBox(
       input_size=2048,
       annotations=annotations,
       task_names=['class_A', 'class_B'],
       latent_encoder_kwargs={
           'hidden_size': 256,
           'n_layers': 3
       }
   )

Losses and Metrics
^^^^^^^^^^^^^^^^^^

Configure losses and metrics using ``GroupConfig`` to automatically handle mixed concept types:

**Concept Loss**

.. code-block:: python

   import torch.nn as nn
   from torch_concepts.nn import ConceptLoss
   from torch_concepts import GroupConfig
   
   # Different loss functions for different concept types
   loss_config = GroupConfig(
       binary=nn.BCEWithLogitsLoss(),
       categorical=nn.CrossEntropyLoss()
   )
   
   concept_loss = ConceptLoss(
       annotations=annotations,
       fn_collection=loss_config
   )

**Concept Metrics**

.. code-block:: python

   from torch_concepts.nn import ConceptMetrics
   from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
   
   # Different metrics for different concept types
   metrics_config = GroupConfig(
       binary={'accuracy': BinaryAccuracy()},
       categorical={'accuracy': MulticlassAccuracy}
   )
   
   concept_metrics = ConceptMetrics(
       annotations=annotations,
       fn_collection=metrics_config,
       summary_metrics=True,      # Compute average across concepts
       perconcept_metrics=True    # Compute per-concept metrics
   )

Training Modes
^^^^^^^^^^^^^^

High-level models support two training approaches:

**Manual PyTorch Training**

.. code-block:: python

   import torch.optim as optim
   
   model = ConceptBottleneckModel(input_size=64, annotations=annotations, 
                                   task_names=['class_A'])
   optimizer = optim.AdamW(model.parameters(), lr=0.001)
   loss_fn = nn.BCEWithLogitsLoss()
   
   for epoch in range(100):
       optimizer.zero_grad()
       predictions = model(x, query=['is_round', 'is_smooth', 'class_A'])
       loss = loss_fn(predictions, targets)
       loss.backward()
       optimizer.step()

**PyTorch Lightning Training**

.. code-block:: python

   from pytorch_lightning import Trainer
   
   # Model with integrated loss and optimizer
   model = ConceptBottleneckModel(
       input_size=64,
       annotations=annotations,
       task_names=['class_A'],
       loss=concept_loss,
       metrics=concept_metrics,
       optim_class=torch.optim.AdamW,
       optim_kwargs={'lr': 0.001}
   )
   
   trainer = Trainer(max_epochs=100)
   trainer.fit(model, datamodule)

Querying Models
^^^^^^^^^^^^^^^

High-level models support flexible querying of concepts and tasks:

.. code-block:: python

   model.eval()
   with torch.no_grad():
       # Query specific variables
       concepts = model(x, query=['is_round', 'is_smooth', 'color'])
       
       # Query tasks only
       tasks = model(x, query=['class_A', 'class_B'])
       
       # Query everything
       all_predictions = model(x, query=['is_round', 'is_smooth', 
                                         'color', 'class_A', 'class_B'])
