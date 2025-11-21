Interpretable Probabilistic Models
=====================================

The Mid-Level API uses **Variables**, **Factors**, and **Probabilistic Models** to build interpretable causal models.

.. warning::

   This API is still under development and interfaces might change in future releases.

Step 1: Import Libraries
-------------------------

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Create Sample Data
---------------------------

.. code-block:: python

   batch_size = 16
   embedding_dim = 64

   embedding = torch.randn(batch_size, embedding_dim)

Step 3: Define Variables
-------------------------

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
-----------------------

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
----------------------------------

Combine variables and factors:

.. code-block:: python

   # Create the probabilistic model
   prob_model = pyc.nn.ProbabilisticModel(
       variables=[embedding_var, *concepts, *tasks],
       factors=[embedding_factor, *concept_factors, *task_factors]
   )

Step 6: Perform Inference
--------------------------

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
----------------------

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

Next Steps
----------

- Explore the full :doc:`Mid-Level API documentation </modules/mid_level_api>`
- Try the :doc:`High-Level API </guides/using_high_level>` for out-of-the-box models
- Learn about :doc:`probabilistic inference methods </modules/nn.inference.mid>`

