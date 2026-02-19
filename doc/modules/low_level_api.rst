Low-level API
=============

Low-level APIs allow you to assemble custom interpretable architectures from basic interpretable layers in a plain pytorch-like interface.


.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
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

Overview of Data Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In |pyc_logo| PyC, we distinguish between three types of data representations:

- **Latent/Input**: High-dimensional representations where exogenous and concept information is entangled
- **Exogenous**: Representations that are direct causes of concept variables
- **Concepts**: Representations of observable quantities of interest


Layer Types
^^^^^^^^^^^

In |pyc_logo| PyC you will find three types of layers whose interfaces reflect the distinction between data representations:

- ``Encoder`` layers: Never take as input concept variables
- ``Predictor`` layers: Must take as input a set of concept variables
- Special layers: Perform operations like memory selection or graph learning


Layer Naming Standard
^^^^^^^^^^^^^^^^^^^^^

In order to easily identify the type of layer, |pyc_logo| PyC uses a consistent standard to assign names to layers.
Each layer name follows the format:

``<LayerType><InputType>To<OutputType>``

where:

- ``LayerType``: describes the type of layer (e.g., Linear, HyperLinear, Selector, Transformer, etc...)
- ``InputType`` and ``OutputType``: describe the type of data representations the layer takes as input and produces as output.

For instance, a layer named ``LinearLatentToConcept`` is a linear layer that takes as input a
``Latent`` representation and produces a ``Concepts`` representation. Since it does not take
as input any concept variables, it is an encoder layer.

.. code-block:: python

 pyc.nn.LinearLatentToConcept(in_latent=10, out_features=3)

As another example, a layer named ``HyperlinearConceptExogenousToConcept`` is a hyper-network layer that
takes as input both ``Concepts`` and ``Exogenous`` representations and produces a
``Concepts`` representation. Since it takes as input concept variables, it is a predictor layer.

.. code-block:: python

 pyc.nn.HyperlinearConceptExogenousToConcept(
    in_concepts=10,
    in_exogenous=7,
    embedding_size=24,
    out_features=3
 )

As a final example, graph learners are a special layers that learn relationships between concepts.
They do not follow the standard naming convention of encoders and predictors, but their purpose should be
clear from their name.

.. code-block:: python

 wanda = pyc.nn.WANDAGraphLearner(
    ['c1', 'c2', 'c3'],
    ['task A', 'task B', 'task C']
 )


Models
^^^^^^^^^^^

A model is built as in standard PyTorch (e.g., ModuleDict or Sequential) and may include standard |pytorch_logo| PyTorch layers + |pyc_logo| PyC layers:

.. code-block:: python

   concept_bottleneck_model = torch.nn.ModuleDict({
       'encoder': pyc.nn.LinearLatentToConcept(in_latent=10, out_features=3),
       'predictor': pyc.nn.LinearConceptToConcept(in_concepts=3, out_features=2),
   })

Inference
^^^^^^^^^^^^^^

At this API level, there are two types of inference that can be performed:

- **Standard forward pass**: a standard forward pass using the forward method of each layer in the ModuleDict

  .. code-block:: python

     concepts = concept_bottleneck_model['encoder'](latent=x)
     task_logits = concept_bottleneck_model['predictor'](concepts=concepts)

- **Interventions**: interventions are context managers that temporarily modify a layer.

  **Intervention strategies**: define how the intervened layer behaves within an intervention context e.g., we can fix the concept values to a constant:

  .. code-block:: python

     int_strategy = pyc.nn.DoIntervention(
        model=concept_bottleneck_model["encoder"],
        constants=-10
     )

  **Intervention Policies**: define the order/set of concepts to intervene on e.g., we can intervene on all concepts uniformly:

  .. code-block:: python

     int_policy = pyc.nn.UniformPolicy(out_features=3)

  When a forward pass is performed within an intervention context, the intervened layer behaves differently with a cascading effect on all subsequent layers:

  .. code-block:: python

     with pyc.nn.intervention(
        policies=int_policy,
        strategies=int_strategy,
        target_concepts=[0, 2]
     ) as new_encoder_layer:
         concepts = new_encoder_layer(latent=x)
         task_logits = concept_bottleneck_model['predictor'](
            concepts=concepts
         )

