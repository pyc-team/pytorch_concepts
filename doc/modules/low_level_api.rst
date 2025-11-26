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

Overview of Data Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In |pyc_logo| PyC, we distinguish between three types of data representations:

- **Input**: High-dimensional representations where exogenous and endogenous information is entangled
- **Exogenous**: Representations that are direct causes of endogenous variables
- **Endogenous**: Representations of observable quantities of interest


Layer Types
^^^^^^^^^^^

In |pyc_logo| PyC you will find three types of layers whose interfaces reflect the distinction between data representations:

- ``Encoder`` layers: Never take as input endogenous variables
- ``Predictor`` layers: Must take as input a set of endogenous variables
- Special layers: Perform operations like memory selection or graph learning


Layer Naming Standard
^^^^^^^^^^^^^^^^^^^^^

In order to easily identify the type of layer, |pyc_logo| PyC uses a consistent standard to assign names to layers.
Each layer name follows the format:

``<LayerType><InputType><OutputType>``

where:

- ``LayerType``: describes the type of layer (e.g., Linear, HyperLinear, Selector, Transformer, etc...)
- ``InputType`` and ``OutputType``: describe the type of data representations the layer takes as input and produces as output. |pyc_logo| PyC uses the following abbreviations:

  - ``Z``: Input
  - ``U``: Exogenous
  - ``C``: Endogenous


For instance, a layer named ``LinearZC`` is a linear layer that takes as input an
``Input`` representation and produces an ``Endogenous`` representation. Since it does not take
as input any endogenous variables, it is an encoder layer.

.. code-block:: python

 pyc.nn.LinearZC(in_features=10, out_features=3)

As another example, a layer named ``HyperLinearCUC`` is a hyper-network layer that
takes as input both ``Endogenous`` and ``Exogenous`` representations and produces an
``Endogenous`` representation. Since it takes as input endogenous variables, it is a predictor layer.

.. code-block:: python

 pyc.nn.HyperLinearCUC(
    in_features_endogenous=10,
    in_features_exogenous=7,
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
       'encoder': pyc.nn.LinearZC(in_features=10, out_features=3),
       'predictor': pyc.nn.LinearCC(in_features_endogenous=3, out_features=2),
   })

Inference
^^^^^^^^^^^^^^

At this API level, there are two types of inference that can be performed:

- **Standard forward pass**: a standard forward pass using the forward method of each layer in the ModuleDict

  .. code-block:: python

     endogenous_concepts = concept_bottleneck_model['encoder'](input=x)
     endogenous_tasks = concept_bottleneck_model['predictor'](endogenous=endogenous_concepts)

- **Interventions**: interventions are context managers that temporarily modify a layer.

  **Intervention strategies**: define how the intervened layer behaves within an intervention context e.g., we can fix the concept endogenous to a constant value:

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
         endogenous_concepts = new_encoder_layer(input=x)
         endogenous_tasks = concept_bottleneck_model['predictor'](
            endogenous=endogenous_concepts
         )

