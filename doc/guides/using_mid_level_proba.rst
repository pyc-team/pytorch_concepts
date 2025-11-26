Interpretable Probabilistic Models
=====================================


.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


|pyc_logo| PyC can be used to build interpretable concept-based probabilisitc models.

.. warning::

   This API is still under development and interfaces might change in future releases.



Design principles
-----------------

Probabilistic Models
^^^^^^^^^^^^^^^^^^^^

At this API level, models are represented as probabilistic models where:

- ``Variable`` objects represent random variables in the probabilistic model. Variables are defined by their name, parents, and distribution type. For instance we can define a list of three concepts as:

  .. code-block:: python

     concepts = pyc.EndogenousVariable(
        concepts=["c1", "c2", "c3"],
        parents=[],
        distribution=torch.distributions.RelaxedBernoulli
     )

- ``ParametricCPD`` objects represent conditional probability distributions (CPDs) between variables in the probabilistic model and are parameterized by |pyc_logo| PyC layers. For instance we can define a list of three parametric CPDs for the above concepts as:

  .. code-block:: python

     concept_cpd = pyc.nn.ParametricCPD(
        concepts=["c1", "c2", "c3"],
        parametrization=pyc.nn.LinearZC(in_features=10, out_features=3)
     )

- ``ProbabilisticModel`` objects are a collection of variables and CPDs. For instance we can define a model as:

  .. code-block:: python

     probabilistic_model = pyc.nn.ProbabilisticModel(
        variables=concepts,
        parametric_cpds=concept_cpd
     )

Inference
^^^^^^^^^

Inference is performed using efficient tensorial probabilistic inference algorithms. For instance, we can perform ancestral sampling as:

.. code-block:: python

   inference_engine = pyc.nn.AncestralSamplingInference(
       probabilistic_model=probabilistic_model,
       graph_learner=wanda,
       temperature=1.
   )
   predictions = inference_engine.query(["c1"], evidence={'input': x})




Step 1: Import Libraries
-------------------------

.. code-block:: python

   import torch
   import torch_concepts as pyc

Step 2: Create Sample Data
---------------------------

.. code-block:: python

   batch_size = 16
   input_dim = 64

   x = torch.randn(batch_size, input_dim)

Step 3: Define Variables
-------------------------

Variables represent random variables in the probabilistic model:

.. code-block:: python

   # Define input variable
   input_var = pyc.InputVariable(
       concepts=["input"],
       parents=[],
   )

   # Define concept variables
   concepts = pyc.EndogenousVariable(
       concepts=["round", "smooth", "bright"],
       parents=["input"],
       distribution=torch.distributions.RelaxedBernoulli
   )

   # Define task variables
   tasks = pyc.EndogenousVariable(
       concepts=["class_A", "class_B"],
       parents=["round", "smooth", "bright"],
       distribution=torch.distributions.RelaxedBernoulli
   )

Step 4: Define ParametricCPDs
-----------------------

ParametricCPDs are conditional probability distributions parameterized by PyC layers:

.. code-block:: python

   # ParametricCPD for input (no parents)
   input_factor = pyc.nn.ParametricCPD(
       concepts=["input"],
       parametrization=torch.nn.Identity()
   )

   # ParametricCPD for concepts (from input)
   concept_cpd = pyc.nn.ParametricCPD(
       concepts=["round", "smooth", "bright"],
       parametrization=pyc.nn.LinearZC(
           in_features=input_dim,
           out_features=1
       )
   )

   # ParametricCPD for tasks (from concepts)
   task_cpd = pyc.nn.ParametricCPD(
       concepts=["class_A", "class_B"],
       parametrization=pyc.nn.LinearCC(
           in_features_endogenous=3,
           out_features=1
       )
   )

Step 5: Build Probabilistic Model
----------------------------------

Combine variables and CPDs:

.. code-block:: python

   # Create the probabilistic model
   prob_model = pyc.nn.ProbabilisticModel(
       variables=[input_var, *concepts, *tasks],
       parametric_cpds=[input_factor, *concept_cpd, *task_cpd]
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
       evidence={'input': x}
   )

   # Query task predictions given concepts
   task_predictions = inference_engine.query(
       query_concepts=["class_A", "class_B"],
       evidence={
           'input': x,
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

   strategy = DoIntervention(model=prob_model.parametric_cpds, constants=100.0)
   policy = UniformPolicy(out_features=prob_model.concept_to_variable["round"].size)

   original_predictions = inference_engine.query(
       query_concepts=["round", "smooth", "bright", "class_A", "class_B"],
       evidence={'input': x}
   )

   # Apply intervention to encoder
   with intervention(
       policies=policy,
       strategies=strategy,
       target_concepts=["round", "smooth"]
   ):
       intervened_predictions = inference_engine.query(
           query_concepts=["round", "smooth", "bright", "class_A", "class_B"],
           evidence={'input': x}
       )

   print(f"Original endogenous: {original_predictions[0]}")
   print(f"Intervened endogenous: {intervened_predictions[0]}")

Next Steps
----------

- Explore the full :doc:`Mid-Level API documentation </modules/mid_level_api>`
- Try the :doc:`High-Level API </guides/using_high_level>` for out-of-the-box models
- Learn about :doc:`probabilistic inference methods </modules/nn.inference.mid>`

