Structural Equation Models
=====================================

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

|pyc_logo| PyC can be used to build interpretable concept-based causal models and perform causal inference.

.. warning::

   This API is still under development and interfaces might change in future releases.


Design principles
-----------------

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


Detailed Guides
------------------------------


.. dropdown:: Structural Equation Models
    :icon: package

    **Import Libraries**

    Start by importing |pyc_logo| PyC and |pytorch_logo| PyTorch libraries:

    .. code-block:: python

       import torch
       from torch.distributions import RelaxedBernoulli
       import torch_concepts as pyc
       from torch_concepts import EndogenousVariable, ExogenousVariable
       from torch_concepts.nn import ParametricCPD, ProbabilisticModel
       from torch_concepts.nn import AncestralSamplingInference
       from torch_concepts.nn import CallableCC, UniformPolicy, DoIntervention, intervention
       from torch_concepts.nn.functional import cace_score

    **Create Sample Data**

    .. code-block:: python

       n_samples = 1000

       # Create exogenous input (noise/unobserved confounders)
       initial_input = {'exogenous': torch.randn((n_samples, 1))}

    **Define Variables and Causal Structure**

    In Structural Equation Models, we distinguish between exogenous (external) and endogenous (internal) variables.
    Each variable is defined by its name, parents, and distribution type.
    By specifying parents, we define the causal graph structure.

    .. code-block:: python

       # Define exogenous variable (external noise/confounders)
       exogenous_var = ExogenousVariable(
           "exogenous",
           parents=[],
           distribution=RelaxedBernoulli
       )

       # Define endogenous variables (causal chain)
       genotype_var = EndogenousVariable(
           "genotype",
           parents=["exogenous"],
           distribution=RelaxedBernoulli
       )

       smoking_var = EndogenousVariable(
           "smoking",
           parents=["genotype"],
           distribution=RelaxedBernoulli
       )

       tar_var = EndogenousVariable(
           "tar",
           parents=["genotype", "smoking"],
           distribution=RelaxedBernoulli
       )

       cancer_var = EndogenousVariable(
           "cancer",
           parents=["tar"],
           distribution=RelaxedBernoulli
       )

    **Define ParametricCPDs**

    ParametricCPDs define the structural equations (causal mechanisms) between variables.
    We can use |pyc_logo| PyC or |pytorch_logo| PyTorch modules to parameterize these CPDs.
    More specifically, |pyc_logo| PyC provides ``CallableCC`` to define structural equations using arbitrary callables.

    .. code-block:: python

       # CPD for exogenous variable (no parents)
       exogenous_cpd = ParametricCPD(
           "exogenous",
           parametrization=torch.nn.Sigmoid()
       )

       # CPD for genotype (depends on exogenous noise)
       genotype_cpd = ParametricCPD(
           "genotype",
           parametrization=torch.nn.Sequential(
               torch.nn.Linear(1, 1),
               torch.nn.Sigmoid()
           )
       )

       # CPD for smoking (depends on genotype)
       smoking_cpd = ParametricCPD(
           ["smoking"],
           parametrization=CallableCC(
               lambda x: (x > 0.5).float(),
               use_bias=False
           )
       )

       # CPD for tar (depends on genotype and smoking)
       tar_cpd = ParametricCPD(
           "tar",
           parametrization=CallableCC(
               lambda x: torch.logical_or(x[:, 0] > 0.5, x[:, 1] > 0.5).float().unsqueeze(-1),
               use_bias=False
           )
       )

       # CPD for cancer (depends on tar)
       cancer_cpd = ParametricCPD(
           "cancer",
           parametrization=CallableCC(
               lambda x: x,
               use_bias=False
           )
       )

    **Build Structural Equation Model**

    Combine all variables and CPDs into a probabilistic model:

    .. code-block:: python

       # Create the structural equation model
       sem_model = ProbabilisticModel(
           variables=[exogenous_var, genotype_var, smoking_var, tar_var, cancer_var],
           parametric_cpds=[exogenous_cpd, genotype_cpd, smoking_cpd, tar_cpd, cancer_cpd]
       )


.. dropdown:: Observational Inference
    :icon: telescope

    Once the SEM is defined, we can perform observational inference to obtain predictions
    for all endogenous variables given exogenous evidence:

    .. code-block:: python

       # Create inference engine
       inference_engine = AncestralSamplingInference(
           sem_model,
           temperature=1.0,
           log_probs=False
       )

       # Query all endogenous variables
       query_concepts = ["genotype", "smoking", "tar", "cancer"]
       results = inference_engine.query(query_concepts, evidence=initial_input)

       print("Genotype Predictions (first 5 samples):")
       print(results[:, 0][:5])
       print("Smoking Predictions (first 5 samples):")
       print(results[:, 1][:5])
       print("Tar Predictions (first 5 samples):")
       print(results[:, 2][:5])
       print("Cancer Predictions (first 5 samples):")
       print(results[:, 3][:5])


.. dropdown:: Do-Interventions
    :icon: tools

    We can perform do-interventions to set specific variables to fixed values
    and observe the effect on downstream variables, simulating a randomized controlled trial.
    The intervention API is the same we use for probabilistic models and low-level APIs.

    .. code-block:: python

       # Intervention 1: Force smoking to 0 (prevent smoking)
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
           cancer_do_smoking_0 = intervened_results_0[:, 3]

       # Intervention 2: Force smoking to 1 (promote smoking)
       smoking_strategy_1 = DoIntervention(
           model=sem_model.parametric_cpds,
           constants=1.0
       )

       with intervention(
           policies=UniformPolicy(out_features=1),
           strategies=smoking_strategy_1,
           target_concepts=["smoking"]
       ):
           intervened_results_1 = inference_engine.query(
               query_concepts=["genotype", "smoking", "tar", "cancer"],
               evidence=initial_input
           )
           cancer_do_smoking_1 = intervened_results_1[:, 3]


.. dropdown:: Causal Effect Estimation
    :icon: beaker

    Calculate the Average Causal Effect (ACE) using the interventional distributions obtained from the do-interventions:

    .. code-block:: python

       # Compute ACE of smoking on cancer
       ace_cancer_do_smoking = cace_score(cancer_do_smoking_0, cancer_do_smoking_1)
       print(f"ACE of smoking on cancer: {ace_cancer_do_smoking:.3f}")

    This represents the causal effect of smoking on cancer, accounting for the full causal structure.


Next Steps
----------

- Explore the full :doc:`Mid-Level API documentation </modules/mid_level_api>`
- Compare with :doc:`Probabilistic Models </guides/using_mid_level_proba>` for standard probabilistic inference
- Try the :doc:`High-Level API </guides/using_high_level>` for out-of-the-box models
