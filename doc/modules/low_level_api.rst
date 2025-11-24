Low-level API
=============

Low-level APIs allow you to assemble custom interpretable architectures from basic interpretable layers in a plain pytorch-like interface.


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

   nn.base.low
   nn.encoders
   nn.predictors
   nn.inference
   nn.policy
   nn.graph
   nn.dense_layers


Design principles
-----------------

Objects
"""""""

In |pyc_logo| PyC there are three types of objects:

- **Embedding**: high-dimensional latent representations shared across all concepts.
- **Exogenous**: high-dimensional latent representations related to a specific concept.
- **Logits**: Concept scores before applying an activation function.

Layers
""""""

There are only three types of layers:

- **Encoders**: layers that map latent representations (latet code or exogenous) to endogenous, e.g.:

  .. code-block:: python

     pyc.nn.LinearZC(in_features_latent=10, out_features=3)

- **Predictors**: layers that map endogenous (plus optionally latent representations) to other endogenous.

  .. code-block:: python

     pyc.nn.HyperLinearCUC(in_features_endogenous=10, in_features_exogenous=7,
                                 embedding_size=24, out_features=3)

- **Special layers**: layers that perform special helpful operations such as memory selection:

  .. code-block:: python

     pyc.nn.SelectorZU(in_features_latent=10, memory_size=5,
                           embedding_size=24, out_features=3)

  and graph learners:

  .. code-block:: python

     wanda = pyc.nn.WANDAGraphLearner(['c1', 'c2', 'c3'], ['task A', 'task B', 'task C'])

Models
""""""

A model is built as in standard PyTorch (e.g., ModuleDict or Sequential) and may include standard |pytorch_logo| PyTorch layers + |pyc_logo| PyC layers:

.. code-block:: python

   concept_bottleneck_model = torch.nn.ModuleDict({
       'encoder': pyc.nn.LinearZC(in_features_latent=10, out_features=3),
       'predictor': pyc.nn.LinearCC(in_features_endogenous=3, out_features=2),
   })

Inference
"""""""""

At this API level, there are two types of inference that can be performed:

- **Standard forward pass**: a standard forward pass using the forward method of each layer in the ModuleDict

  .. code-block:: python

     endogenous_concepts = concept_bottleneck_model['encoder'](latent=latent)
     endogenous_tasks = concept_bottleneck_model['predictor'](endogenous=endogenous_concepts)

- **Interventions**: interventions are context managers that temporarily modify a layer.

  **Intervention strategies**: define how the intervened layer behaves within an intervention context e.g., we can fix the concept endogenous to a constant value:

  .. code-block:: python

     int_strategy = pyc.nn.DoIntervention(model=concept_bottleneck_model["encoder"],
                                          constants=-10)

  **Intervention Policies**: define the order/set of concepts to intervene on e.g., we can intervene on all concepts uniformly:

  .. code-block:: python

     int_policy = pyc.nn.UniformPolicy(out_features=3)

  When a forward pass is performed within an intervention context, the intervened layer behaves differently with a cascading effect on all subsequent layers:

  .. code-block:: python

     with pyc.nn.intervention(policies=int_policy,
                              strategies=int_strategy,
                              target_concepts=[0, 2]) as new_encoder_layer:

         endogenous_concepts = new_encoder_layer(latent=latent)
         endogenous_tasks = concept_bottleneck_model['predictor'](endogenous=endogenous_concepts)

