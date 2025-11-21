User Guide
==========

Welcome to the PyC User Guide! This guide will help you get started with PyTorch Concepts and build interpretable deep learning models.

PyC is designed with three levels of abstraction to accommodate users with different backgrounds and needs. Choose the level that best fits your experience and use case.

Overview of API Levels
----------------------

PyC provides three complementary APIs:

**Low-Level API**
    Build custom architectures from basic interpretable layers using a PyTorch-like interface. Perfect for users who want fine-grained control over their model architecture.

**Mid-Level API**
    Create probabilistic models with explicit concept representations and causal relationships. Ideal for researchers focused on interpretability and causal reasoning.

**High-Level API**
    Use pre-configured state-of-the-art models with minimal code. Best for quick prototyping and production use cases.

---

Low-Level API: Building with Interpretable Layers
--------------------------------------------------

The Low-Level API provides three types of layers: **Encoders**, **Predictors**, and **Special layers**.

Key Principles
^^^^^^^^^^^^^^

**Three types of objects:**

- **Embedding**: High-dimensional latent representations shared across all concepts
- **Exogenous**: High-dimensional latent representations for a specific concept
- **Logits**: Concept scores before activation

**Three types of layers:**

- **Encoders**: Map latent representations to logits
- **Predictors**: Map logits to other logits
- **Special layers**: Perform operations like memory selection or graph learning

Step 1: Import Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Create Sample Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate random embeddings and targets for demonstration:

.. code-block:: python

   batch_size = 32
   embedding_dim = 64
   n_concepts = 5
   n_tasks = 3

   # Random input embeddings
   embedding = torch.randn(batch_size, embedding_dim)

   # Random concept labels (binary)
   concept_labels = torch.randint(0, 2, (batch_size, n_concepts)).float()

   # Random task labels
   task_labels = torch.randint(0, n_tasks, (batch_size,))

Step 3: Build a Concept Bottleneck Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a ModuleDict to combine encoder and predictor:

.. code-block:: python

   # Create model using ModuleDict
   model = torch.nn.ModuleDict({
       'encoder': pyc.nn.ProbEncoderFromEmb(
           in_features_embedding=embedding_dim,
           out_features=n_concepts
       ),
       'predictor': pyc.nn.ProbPredictor(
           in_features_logits=n_concepts,
           out_features=n_tasks
       ),
   })

Step 4: Forward Pass
^^^^^^^^^^^^^^^^^^^^^

Compute concept logits, then task predictions:

.. code-block:: python

   # Get concept logits from embeddings
   concept_logits = model['encoder'](embedding=embedding)

   # Get task predictions from concept logits
   task_logits = model['predictor'](logits=concept_logits)

   print(f"Concept logits shape: {concept_logits.shape}")  # [32, 5]
   print(f"Task logits shape: {task_logits.shape}")        # [32, 3]

Step 5: Compute Loss and Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train with both concept and task supervision:

.. code-block:: python

   import torch.nn.functional as F

   # Compute losses
   concept_loss = F.binary_cross_entropy_with_logits(
       concept_logits, concept_labels
   )
   task_loss = F.cross_entropy(task_logits, task_labels)
   total_loss = task_loss + 0.5 * concept_loss

   # Backpropagation
   total_loss.backward()

   print(f"Concept loss: {concept_loss.item():.4f}")
   print(f"Task loss: {task_loss.item():.4f}")

Step 6: Perform Interventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intervene using the `intervention` context manager which replaces the encoder layer temporarily.
The context manager takes two main arguments: **strategies** and **policies**.

- Intervention strategies define how the layer behaves during the intervention, e.g., setting concept logits to ground truth values.
- Intervention policies define the priority/order of concepts to intervene on.

.. code-block:: python

   from torch_concepts.nn import GroundTruthIntervention, UniformPolicy
   from torch_concepts.nn import intervention

   ground_truth=10*torch.rand_like(concept_logits)
   strategy = GroundTruthIntervention(model=model['encoder'], ground_truth=ground_truth)
   policy = UniformPolicy(out_features=n_concepts)

   # Apply intervention to encoder
   with intervention(policies=policy,
                     strategies=strategy,
                     target_concepts=[0, 2]) as new_encoder_layer:
       intervened_concepts = new_encoder_layer(embedding=embedding)
       intervened_tasks = model['predictor'](logits=intervened_concepts)

   print(f"Original concept logits: {concept_logits[0]}")
   print(f"Original task predictions: {task_logits[0]}")
   print(f"Intervened concept logits: {intervened_concepts[0]}")
   print(f"Intervened task predictions: {intervened_tasks[0]}")

Using Special Layers
^^^^^^^^^^^^^^^^^^^^

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

---

Mid-Level API: Probabilistic Models
------------------------------------

The Mid-Level API uses **Variables**, **Factors**, and **Probabilistic Models** to build interpretable causal models.

.. warning::

   This API is still under development and interfaces might change in future releases.

Step 1: Import Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Create Sample Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   batch_size = 16
   embedding_dim = 64

   embedding = torch.randn(batch_size, embedding_dim)

Step 3: Define Variables
^^^^^^^^^^^^^^^^^^^^^^^^^

Variables represent random variables in the probabilistic model:

.. code-block:: python

   # Define embedding variable
    embedding_var = pyc.Variable(
         concepts=["embedding"],
         parents=[],
    )

   # Define concept variables
   concepts = pyc.Variable(
       concepts=["round", "smooth", "bright"],
       parents=["embedding"],
       distribution=torch.distributions.RelaxedBernoulli
   )

   # Define task variables
   tasks = pyc.Variable(
       concepts=["class_A", "class_B"],
       parents=["round", "smooth", "bright"],
       distribution=torch.distributions.RelaxedBernoulli
   )

Step 4: Define Factors
^^^^^^^^^^^^^^^^^^^^^^^

Factors are conditional probability distributions parameterized by PyC layers:

.. code-block:: python

   # Factor for embeddings (no parents)
   embedding_factor = pyc.nn.Factor(
        concepts=["embedding"],
        module_class=torch.nn.Identity()
   )

   # Factor for concepts (from embeddings)
   concept_factors = pyc.nn.Factor(
       concepts=["round", "smooth", "bright"],
       module_class=pyc.nn.ProbEncoderFromEmb(
           in_features_embedding=embedding_dim,
           out_features=1
       )
   )

   # Factor for tasks (from concepts)
   task_factors = pyc.nn.Factor(
       concepts=["class_A", "class_B"],
       module_class=pyc.nn.ProbPredictor(
           in_features_logits=3,
           out_features=1
       )
   )

Step 5: Build Probabilistic Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combine variables and factors:

.. code-block:: python

   # Create the probabilistic model
   prob_model = pyc.nn.ProbabilisticModel(
       variables=[embedding_var, *concepts, *tasks],
       factors=[embedding_factor, *concept_factors, *task_factors]
   )

Step 6: Perform Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Query the model using ancestral sampling:

.. code-block:: python

   # Create inference engine
   inference_engine = pyc.nn.AncestralSamplingInference(
       probabilistic_model=prob_model,
       temperature=1.0
   )

   # Query concept predictions
   concept_predictions = inference_engine.query(
       query_concepts=["round", "smooth", "bright"],
       evidence={'embedding': embedding}
   )

   # Query task predictions given concepts
   task_predictions = inference_engine.query(
       query_concepts=["class_A", "class_B"],
       evidence={
           'embedding': embedding,
           'round': concept_predictions[:, 0],
           'smooth': concept_predictions[:, 1],
           'bright': concept_predictions[:, 2]
       }
   )

   print(f"Concept predictions: {concept_predictions}")
   print(f"Task predictions: {task_predictions}")

Step 7: Interventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform do-calculus interventions:

.. code-block:: python

   from torch_concepts.nn import DoIntervention, UniformPolicy
   from torch_concepts.nn import intervention

   strategy = DoIntervention(model=prob_model.factors, constants=100.0)
   policy = UniformPolicy(out_features=prob_model.concept_to_variable["round"].size)

   original_predictions = inference_engine.query(
       query_concepts=["round", "smooth", "bright", "class_A", "class_B"],
       evidence={'embedding': embedding}
   )

   # Apply intervention to encoder
   with intervention(policies=policy,
                     strategies=strategy,
                     target_concepts=["round", "smooth"]):
       intervened_predictions = inference_engine.query(
           query_concepts=["round", "smooth", "bright", "class_A", "class_B"],
           evidence={'embedding': embedding}
       )

   print(f"Original logits: {original_predictions[0]}")
   print(f"Intervened logits: {intervened_predictions[0]}")

---

High-Level API: Out-of-the-Box Models
--------------------------------------

The High-Level API provides pre-built models that work with one line of code.

Step 1: Import Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Define Annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO...


---

Next Steps
----------

**Explore Examples**
    Check out the `examples directory <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`_ for real-world use cases.

**Read API Documentation**
    - :doc:`Low-Level API </modules/low_level_api>` for detailed layer documentation
    - :doc:`Mid-Level API </modules/mid_level_api>` for probabilistic modeling
    - :doc:`High-Level API </modules/high_level_api>` for pre-built models

**Try Conceptarium**
    Use the :doc:`no-code framework </modules/conceptarium>` for running experiments without coding.

Need Help?
----------

- **Issues**: `GitHub Issues <https://github.com/pyc-team/pytorch_concepts/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/pyc-team/pytorch_concepts/discussions>`_
- **Contributing**: :doc:`Contributor Guide </guides/contributing>`
