Interpretable Layers and Interventions
==================================================

The Low-Level API provides three types of layers: **Encoders**, **Predictors**, and **Special layers**.

Key Principles
--------------

**Three types of objects:**

- **Embedding**: High-dimensional latent representations shared across all concepts
- **Exogenous**: High-dimensional latent representations for a specific concept
- **Logits**: Concept scores before activation

**Three types of layers:**

- **Encoders**: Map latent representations to endogenous
- **Predictors**: Map endogenous to other endogenous
- **Special layers**: Perform operations like memory selection or graph learning

Step 1: Import Libraries
-------------------------

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Create Sample Data
---------------------------

Generate random latent codes and targets for demonstration:

.. code-block:: python

   batch_size = 32
   latent_dim = 64
   n_concepts = 5
   n_tasks = 3

   # Random input latent code
   latent = torch.randn(batch_size, latent_dim)

   # Random concept labels (binary)
   concept_labels = torch.randint(0, 2, (batch_size, n_concepts)).float()

   # Random task labels
   task_labels = torch.randint(0, n_tasks, (batch_size,))

Step 3: Build a Concept Bottleneck Model
-----------------------------------------

Use a ModuleDict to combine encoder and predictor:

.. code-block:: python

   # Create model using ModuleDict
   model = torch.nn.ModuleDict({
       'encoder': pyc.nn.LinearZC(
           in_features_latent=latent_dim,
           out_features=n_concepts
       ),
       'predictor': pyc.nn.LinearCC(
           in_features_endogenous=n_concepts,
           out_features=n_tasks
       ),
   })

Step 4: Forward Pass
---------------------

Compute concept endogenous, then task predictions:

.. code-block:: python

   # Get concept endogenous from latent code
   concept_endogenous = model['encoder'](latent=latent)

   # Get task predictions from concept endogenous
   task_endogenous = model['predictor'](endogenous=concept_endogenous)

   print(f"Concept endogenous shape: {concept_endogenous.shape}")  # [32, 5]
   print(f"Task endogenous shape: {task_endogenous.shape}")        # [32, 3]

Step 5: Compute Loss and Train
-------------------------------

Train with both concept and task supervision:

.. code-block:: python

   import torch.nn.functional as F

   # Compute losses
   concept_loss = F.binary_cross_entropy_with_endogenous(
       concept_endogenous, concept_labels
   )
   task_loss = F.cross_entropy(task_endogenous, task_labels)
   total_loss = task_loss + 0.5 * concept_loss

   # Backpropagation
   total_loss.backward()

   print(f"Concept loss: {concept_loss.item():.4f}")
   print(f"Task loss: {task_loss.item():.4f}")

Step 6: Perform Interventions
------------------------------

Intervene using the ``intervention`` context manager which replaces the encoder layer temporarily.
The context manager takes two main arguments: **strategies** and **policies**.

- Intervention strategies define how the layer behaves during the intervention, e.g., setting concept endogenous to ground truth values.
- Intervention policies define the priority/order of concepts to intervene on.

.. code-block:: python

   from torch_concepts.nn import GroundTruthIntervention, UniformPolicy
   from torch_concepts.nn import intervention

   ground_truth = 10 * torch.rand_like(concept_endogenous)
   strategy = GroundTruthIntervention(model=model['encoder'], ground_truth=ground_truth)
   policy = UniformPolicy(out_features=n_concepts)

   # Apply intervention to encoder
   with intervention(policies=policy,
                     strategies=strategy,
                     target_concepts=[0, 2]) as new_encoder_layer:
       intervened_concepts = new_encoder_layer(latent=latent)
       intervened_tasks = model['predictor'](endogenous=intervened_concepts)

   print(f"Original concept endogenous: {concept_endogenous[0]}")
   print(f"Original task predictions: {task_endogenous[0]}")
   print(f"Intervened concept endogenous: {intervened_concepts[0]}")
   print(f"Intervened task predictions: {intervened_tasks[0]}")

Using Special Layers
--------------------

Add a graph learner to discover concept relationships:

.. code-block:: python

   # Define concept and task names
   concept_names = ['round', 'smooth', 'bright', 'large', 'centered']

   # Create WANDA graph learner
   graph_learner = pyc.nn.WANDAGraphLearner(
       row_labels=concept_names,
       col_labels=concept_names
   )

   print(f"Learned graph shape: {graph_learner.weighted_adj}")

Next Steps
----------

- Explore the full :doc:`Low-Level API documentation </modules/low_level_api>`
- Try the :doc:`Mid-Level API </guides/using_mid_level>` for probabilistic modeling
- Check out :doc:`example notebooks <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`

