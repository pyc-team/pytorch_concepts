Interpretable Layers and Interventions
==================================================

The Low-Level API provides building blocks to create concept-based models using
interpretable layers and perform interventions using a PyTorch-like interface.

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

Design Principles
--------------

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


Detailed Guides
------------------------------


.. dropdown:: Concept Bottleneck Model
    :icon: package

    **Import Libraries**

    To get started, import |pyc_logo| PyC and |pytorch_logo| PyTorch:

    .. code-block:: python

       import torch
       import torch_concepts as pyc

    **Create Sample Data**

    Generate random inputs and targets for demonstration:

    .. code-block:: python

       batch_size = 32
       input_dim = 64
       n_concepts = 5
       n_tasks = 3

       # Random input
       x = torch.randn(batch_size, input_dim)

       # Random concept labels (binary)
       concept_labels = torch.randint(0, 2, (batch_size, n_concepts)).float()

       # Random task labels
       task_labels = torch.randint(0, n_tasks, (batch_size,))

    **Build a Concept Bottleneck Model**

    Use a ModuleDict to combine encoder and predictor:

    .. code-block:: python

       # Create model using ModuleDict
       model = torch.nn.ModuleDict({
           'encoder': pyc.nn.LinearZC(
               in_features=input_dim,
               out_features=n_concepts
           ),
           'predictor': pyc.nn.LinearCC(
               in_features_endogenous=n_concepts,
               out_features=n_tasks
           ),
       })


.. dropdown:: Inference and Training
    :icon: rocket

    **Inference**

    Once a concept bottleneck model is built, we can perform inference by first obtaining
    concept activations from the encoder, and then task predictions from the predictor:

    .. code-block:: python

       # Get concept endogenous from input
       concept_endogenous = model['encoder'](input=x)

       # Get task predictions from concept endogenous
       task_endogenous = model['predictor'](endogenous=concept_endogenous)

       print(f"Concept endogenous shape: {concept_endogenous.shape}")  # [32, 5]
       print(f"Task endogenous shape: {task_endogenous.shape}")        # [32, 3]

    **Compute Loss and Train**

    Train with both concept and task supervision:

    .. code-block:: python

       import torch.nn.functional as F

       # Compute losses
       concept_loss = F.binary_cross_entropy(torch.sigmoid(concept_endogenous), concept_labels)
       task_loss = F.cross_entropy(task_endogenous, task_labels)
       total_loss = task_loss + 0.5 * concept_loss

       # Backpropagation
       total_loss.backward()

       print(f"Concept loss: {concept_loss.item():.4f}")
       print(f"Task loss: {task_loss.item():.4f}")


.. dropdown:: Interventions
    :icon: tools

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
       with intervention(
           policies=policy,
           strategies=strategy,
           target_concepts=[0, 2]
       ) as new_encoder_layer:
           intervened_concepts = new_encoder_layer(input=x)
           intervened_tasks = model['predictor'](endogenous=intervened_concepts)

       print(f"Original concept endogenous: {concept_endogenous[0]}")
       print(f"Original task predictions: {task_endogenous[0]}")
       print(f"Intervened concept endogenous: {intervened_concepts[0]}")
       print(f"Intervened task predictions: {intervened_tasks[0]}")


.. dropdown:: (Advanced) Graph Learning
    :icon: workflow

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


    The ``graph_learner.weighted_adj`` tensor contains a learnable adjacency matrix representing relationships
    between concepts.


.. dropdown:: (Advanced) Verifiable Concept-Based Models
    :icon: shield-check

    To design more complex concept-based models, you can combine multiple interpretable layers.
    For example, to build a verifiable concept-based model we can use an encoder to predict concept activations,
    a selector to select relevant exogenous information, and a hyper-network predictor to make final predictions
    based on both concept activations and exogenous information.

    .. code-block:: python

       from torch_concepts.nn import LinearZC, SelectorZU, HyperLinearCUC

       memory_size = 7
       exogenous_size = 16
       embedding_size = 5

       # Create model using ModuleDict
       model = torch.nn.ModuleDict({
           'encoder': LinearZC(
               in_features=input_dim,
               out_features=n_concepts
           ),
           'selector': SelectorZU(
               in_features=input_dim,
               memory_size=memory_size,
               exogenous_size=exogenous_size,
               out_features=n_tasks
           ),
           'predictor': HyperLinearCUC(
               in_features_endogenous=n_concepts,
               in_features_exogenous=exogenous_size,
               embedding_size=embedding_size,
           )
       })



Next Steps
----------

- Explore the full :doc:`Low-Level API documentation </modules/low_level_api>`
- Try the :doc:`Mid-Level API </guides/using_mid_level_proba>` for probabilistic modeling
- Try the :doc:`Mid-Level API </guides/using_mid_level_causal>` for causal modeling
- Check out :doc:`example notebooks <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`
