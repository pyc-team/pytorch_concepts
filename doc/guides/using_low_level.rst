Interpretable Layers and Interventions
==================================================

The Low-Level API provides building blocks to create concept-based models using
interpretable layers and perform interventions using a PyTorch-like interface.

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

Design Principles
--------------

Overview of Data Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In |pyc_logo| PyC, we distinguish between three types of data representations:

- **Latent**: High-dimensional (input) representations where exogenous and concept information is entangled
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

``<LayerType><InputType><OutputType>``

where:

- ``LayerType``: describes the type of layer (e.g., Linear, HyperLinear, Selector, Transformer, etc...)
- ``InputType`` and ``OutputType``: describe the type of data representations the layer takes as input and produces as output. 

For instance, a layer named ``LinearLatentToConcept`` is a linear layer that takes as input a
``Latent`` representation and produces a ``Concepts`` representation. Since it does not take
as input any concept variables, it is an encoder layer.

.. code-block:: python

 pyc.nn.LinearLatentToConcept(in_latent=64, out_concepts=3)

As another example, a layer named ``HyperlinearConceptExogenousToConcept`` is a hyper-network layer that
takes as input both ``Concepts`` and ``Exogenous`` representations and produces a
``Concepts`` representation. Since it takes as input concept variables, it is a predictor layer.

.. code-block:: python

 pyc.nn.HyperlinearConceptExogenousToConcept(
    in_concepts=3,
    in_exogenous=8,
    hidden_size=32
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
           'encoder': pyc.nn.LinearLatentToConcept(
               in_latent=input_dim,
               out_concepts=n_concepts
           ),
           'predictor': pyc.nn.LinearConceptToConcept(
               in_concepts=n_concepts,
               out_concepts=n_tasks
           ),
       })


.. dropdown:: Inference and Training
    :icon: rocket

    **Inference**

    Once a concept bottleneck model is built, we can perform inference by first obtaining
    concept activations from the encoder, and then task predictions from the predictor:

    .. code-block:: python

       # Get concept predictions from input
       concept_preds = model['encoder'](latent=x)

       # Get task predictions from concept predictions
       task_preds = model['predictor'](concepts=concept_preds)

       print(f"Concept predictions shape: {concept_preds.shape}")  # [32, 5]
       print(f"Task predictions shape: {task_preds.shape}")        # [32, 3]

    **Compute Loss and Train**

    Train with both concept and task supervision:

    .. code-block:: python

       import torch.nn.functional as F

       # Compute losses
       concept_loss = F.binary_cross_entropy(torch.sigmoid(concept_preds), concept_labels)
       task_loss = F.cross_entropy(task_preds, task_labels)
       total_loss = task_loss + 0.5 * concept_loss

       # Backpropagation
       total_loss.backward()

       print(f"Concept loss: {concept_loss.item():.4f}")
       print(f"Task loss: {task_loss.item():.4f}")


.. dropdown:: Interventions
    :icon: tools

    Intervene using the ``intervention`` context manager which replaces the encoder layer temporarily.
    The context manager takes two main arguments: **strategies** and **policies**.

    - Intervention strategies define how the layer behaves during the intervention, e.g., setting concepts to ground truth values.
    - Intervention policies define the priority/order of concepts to intervene on.

    .. code-block:: python

       from torch_concepts.nn import GroundTruthIntervention, UniformPolicy
       from torch_concepts.nn import intervention

       ground_truth = 10 * torch.rand_like(concept_preds)
       strategy = GroundTruthIntervention(model=model['encoder'], ground_truth=ground_truth)
       policy = UniformPolicy(out_concepts=n_concepts)

       # Apply intervention to encoder
       with intervention(
           policies=policy,
           strategies=strategy,
           target_concepts=[0, 2]
       ) as new_encoder_layer:
           intervened_concepts = new_encoder_layer(latent=x)
           intervened_tasks = model['predictor'](concepts=intervened_concepts)

       print(f"Original concept predictions: {concept_preds[0]}")
       print(f"Original task predictions: {task_preds[0]}")
       print(f"Intervened concept predictions: {intervened_concepts[0]}")
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

       from torch_concepts.nn import LinearLatentToConcept, SelectorLatentToExogenous, HyperlinearConceptExogenousToConcept

       memory_size = 7
       exogenous_size = 16
       hidden_size = 5

       # Create model using ModuleDict
       model = torch.nn.ModuleDict({
           'encoder': LinearLatentToConcept(
               in_latent=input_dim,
               out_concepts=n_concepts
           ),
           'selector': SelectorLatentToExogenous(
               in_latent=input_dim,
               memory_size=memory_size,
               out_exogenous=exogenous_size,
               out_concepts=n_tasks
           ),
           'predictor': HyperlinearConceptExogenousToConcept(
               in_concepts=n_concepts,
               in_exogenous=exogenous_size,
               hidden_size=hidden_size,
           )
       })



Next Steps
----------

- Explore the full :doc:`Low-Level API documentation </modules/low_level_api>`
- Try the :doc:`Mid-Level API </guides/using_mid_level_proba>` for probabilistic modeling
- Try the :doc:`Mid-Level API </guides/using_mid_level_causal>` for causal modeling
- Check out :doc:`example notebooks <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`
