Mid-level API
=============

Mid-level APIs allow you to build custom interpretable and causally transparent probabilistic models.

.. warning::

   This API is still under development and interfaces might change in future releases.

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

   nn.base.mid
   nn.variable
   nn.models
   nn.inference.mid
   nn.constructors


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


Structural Equation Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|pyc_logo| PyC can be used to design Structural Equation Models (SEMs), where:

- ``ExogenousVariable`` and ``EndogenousVariable`` objects represent random variables in the SEM. Variables are defined by their name, parents, and distribution type. For example, in this guide we define variables as:

  .. code-block:: python

     exogenous_var = ExogenousVariable(
         "exogenous",
         parents=[],
         distribution=RelaxedBernoulli
     )
     genotype_var = EndogenousVariable(
         "genotype",
         parents=["exogenous"],
         distribution=RelaxedBernoulli
     )

- ``ParametricCPD`` objects represent the structural equations (causal mechanisms) between variables in the SEM and are parameterized by |pyc_logo| PyC or |pytorch_logo| PyTorch modules. For example:

  .. code-block:: python

     genotype_cpd = ParametricCPD(
         "genotype",
         parametrization=torch.nn.Sequential(
             torch.nn.Linear(1, 1),
             torch.nn.Sigmoid()
         )
     )

- ``ProbabilisticModel`` objects collect all variables and CPDs to define the full SEM. For example:

  .. code-block:: python

     sem_model = ProbabilisticModel(
         variables=[exogenous_var, genotype_var],
         parametric_cpds=[exogenous_cpd, genotype_cpd]
     )

Interventions
^^^^^^^^^^^^^

Interventions allow us to estimate causal effects. For instance, do-interventions allow us to set specific variables
to fixed values and observe the effect on downstream variables simulating a randomized controlled trial.

To perform a do-intervention, use the ``DoIntervention`` strategy and the ``intervention`` context manager.
For example, to set ``smoking`` to 0 (prevent smoking) and query the effect on downstream variables:

.. code-block:: python

   # Intervention: Force smoking to 0 (prevent smoking)
   smoking_strategy_0 = DoIntervention(
       model=sem_model.parametric_cpds,
       constants=0.0
   )

   with intervention(
      policies=UniformPolicy(out_features=1),
      strategies=smoking_strategy_0,
      target_concepts=["smoking"]
   ):
       intervened_results_0 = inference_engine.query(
           query_concepts=["genotype", "smoking", "tar", "cancer"],
           evidence=initial_input
       )
       # Results reflect the effect of setting smoking=0

You can use these interventional results to estimate causal effects, such as the Average Causal Effect (ACE),
as shown in later steps of this guide.
