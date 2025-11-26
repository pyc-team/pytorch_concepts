Out-of-the-Box Interpretable Models
=======================================

The High-Level API provides pre-built models that work with one line of code.

Step 1: Import Libraries
-------------------------

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Define Annotations
---------------------------

Annotations describe the structure of concepts and tasks:

.. code-block:: python

   # Define concept properties
   concept_labels = ["round", "smooth", "bright"]
   concept_cardinalities = [2, 2, 2]  # Binary concepts

   metadata = {
       'round': {'distribution': torch.distributions.RelaxedBernoulli},
       'smooth': {'distribution': torch.distributions.RelaxedBernoulli},
       'bright': {'distribution': torch.distributions.RelaxedBernoulli},
   }

   # Create annotations
   annotations = pyc.Annotations({
       1: pyc.AxisAnnotation(
           labels=concept_labels,
           cardinalities=concept_cardinalities,
           metadata=metadata
       )
   })

Step 3: Instantiate a Model
----------------------------

Create a Concept Bottleneck Model in one line:

.. code-block:: python

   model = pyc.nn.CBM(
       task_names=['class_A', 'class_B', 'class_C'],
       inference=pyc.nn.DeterministicInference,
       input_size=64,
       annotations=annotations,
       encoder_kwargs={
           'hidden_size': 128,
           'n_layers': 2,
           'activation': 'relu',
           'dropout': 0.1
       }
   )

   print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

Step 4: Forward Pass
---------------------

.. code-block:: python

   batch_size = 32
   input_data = torch.randn(batch_size, 64)

   # Single forward pass
   output = model(input_data)

   print(f"Concepts shape: {output['concepts'].shape}")
   print(f"Task predictions shape: {output['tasks'].shape}")

Step 5: Training with PyTorch Lightning
----------------------------------------

High-level models integrate with PyTorch Lightning:

.. code-block:: python

   import pytorch_lightning as pl
   from torch.utils.data import DataLoader, TensorDataset

   # Create synthetic dataset
   train_x = torch.randn(1000, 64)
   train_concepts = torch.randint(0, 2, (1000, 3)).float()
   train_tasks = torch.randint(0, 3, (1000,))

   dataset = TensorDataset(train_x, train_concepts, train_tasks)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # Create trainer
   trainer = pl.Trainer(max_epochs=5, accelerator='auto')

   # Train
   trainer.fit(model, dataloader)

Step 6: Make Predictions
-------------------------

.. code-block:: python

   model.eval()
   test_data = torch.randn(10, 64)

   with torch.no_grad():
       predictions = model(test_data)
       predicted_classes = torch.argmax(predictions['tasks'], dim=1)
       concept_values = (predictions['concepts'] > 0.5).float()

   print(f"Predicted classes: {predicted_classes}")
   print(f"Active concepts (sample 0): {concept_values[0]}")

Next Steps
----------

- Explore the full :doc:`High-Level API documentation </modules/high_level_api>`
- Try :doc:`Conceptarium </modules/conceptarium>` for no-code experiments
- Check out available :doc:`pre-built models </modules/nn.models.high>`

