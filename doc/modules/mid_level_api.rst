Mid-level API
=============

Mid-level APIs allow you to build custom interpretable and causally transparent Probabilistic Models.

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

At this API level, models are represented as Probabilistic Models where:

- **Variables**: represent random variables in the Probabilistic Model. Variables are defined by their name, parents, and distribution type. For instance we can define a list of three concepts as:

  .. code-block:: python

     concepts = pyc.EndogenousVariable(concepts=["c1", "c2", "c3"], parents=[],
                                       distribution=torch.distributions.RelaxedBernoulli)

- **ParametricCPDs**: represent conditional probability distributions (CPDs) between variables in the Probabilistic Model and are parameterized by |pyc_logo| PyC layers. For instance we can define a list of three parametric CPDs for the above concepts as:

  .. code-block:: python

     concept_cpd = pyc.nn.ParametricCPD(concepts=["c1", "c2", "c3"],
                                        parametrization=pyc.nn.ProbEncoderFromEmb(in_features_embedding=10, out_features=3))

- **Probabilistic Model**: a collection of variables and CPDs. For instance we can define a ProbabilisticModel as:

  .. code-block:: python

     probabilistic_model = pyc.nn.ProbabilisticModel(variables=concepts,
                                                     parametric_cpds=concept_cpd)

Inference
^^^^^^^^^

Inference is performed using efficient tensorial probabilistic inference algorithms. For instance, we can perform ancestral sampling as:

.. code-block:: python

   inference_engine = pyc.nn.AncestralSamplingInference(probabilistic_model=probabilistic_model,
                                                        graph_learner=wanda, temperature=1.)
   predictions = inference_engine.query(["c1"], evidence={'embedding': embedding})
