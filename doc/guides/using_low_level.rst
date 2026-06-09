Interpretable Layers and Interventions
=======================================

The Low-Level API provides composable building blocks for concept-based neural
networks, using a pure PyTorch interface.

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

Design Principles
-----------------

Overview of Data Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|pyc_logo| PyC layers operate on two types of data representations:

- **Embeddings**: Vector representations of the input, not yet associated with specific concept variables.
- **Concepts**: Scalar-valued activation scores — one per concept variable. These form the interpretable bottleneck.

Layer Types
^^^^^^^^^^^

- ``Encoder`` layers: map embeddings to concept scores or to a new set of embeddings. They never take concept scores as input.
- ``Predictor`` layers: take concept scores (and optionally embeddings) as input and produce task or concept predictions.
- **Utility layers**: memory selectors, graph learners — specialised operations that do not follow the encoder/predictor interface.


Layer Naming Convention
^^^^^^^^^^^^^^^^^^^^^^^

Layer names follow the pattern ``<OperationType><InputType>To<OutputType>``:

- ``OperationType``: computation performed — ``Linear``, ``Hyperlinear``, ``Stochastic``, ``Selector``, …
- ``InputType``: ``Embedding`` for vector inputs, ``Concept`` for concept scores
- ``OutputType``: ``Concept`` for score outputs, ``Embedding`` for vector outputs

For example, ``LinearEmbeddingToConcept`` is a linear encoder mapping an embedding to concept scores:

.. code-block:: python

   pyc.nn.LinearEmbeddingToConcept(in_embeddings=64, out_concepts=3)

``HyperlinearConceptEmbeddingToConcept`` is a hyper-network predictor taking both concept scores and
per-task embeddings as input:

.. code-block:: python

   pyc.nn.HyperlinearConceptEmbeddingToConcept(
       in_concepts=3,
       in_embeddings=16,
       hidden_size=32,
   )

Graph learners are utility layers that learn relationships between concepts.

.. code-block:: python

   wanda = pyc.nn.WANDAGraphLearner(
       row_labels=['c1', 'c2', 'c3'],
       col_labels=['task A', 'task B', 'task C'],
   )


Detailed Guides
---------------


.. dropdown:: Concept Bottleneck Model
    :icon: package

    A Concept Bottleneck Model (CBM) routes predictions through an interpretable concept layer.
    Use ``ModuleDict`` to combine encoder and predictor:

    .. code-block:: python

       import torch
       import torch.nn.functional as F
       import torch_concepts.nn as pyc_nn
       from torch.nn import ModuleDict

       batch_size, input_dim, n_concepts, n_tasks = 32, 64, 5, 3

       x = torch.randn(batch_size, input_dim)
       concept_labels = torch.randint(0, 2, (batch_size, n_concepts)).float()
       task_labels   = torch.randint(0, 2, (batch_size, n_tasks)).float()

       model = ModuleDict({
           'encoder': pyc_nn.LinearEmbeddingToConcept(
               in_embeddings=input_dim,
               out_concepts=n_concepts,
           ),
           'predictor': pyc_nn.LinearConceptToConcept(
               in_concepts=n_concepts,
               out_concepts=n_tasks,
           ),
       })

    **Inference**

    .. code-block:: python

       concept_preds = model['encoder'](x)                          # (batch, n_concepts) — raw logits
       task_preds    = model['predictor'](concepts=concept_preds)   # (batch, n_tasks) — raw logits

    **Training**

    Outputs are raw logits; use ``binary_cross_entropy_with_logits`` for numerical stability.

    .. code-block:: python

       optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

       optimizer.zero_grad()
       concept_preds = model['encoder'](x)
       task_preds    = model['predictor'](concepts=concept_preds)

       concept_loss = F.binary_cross_entropy_with_logits(concept_preds, concept_labels)
       task_loss    = F.binary_cross_entropy_with_logits(task_preds, task_labels)
       loss = concept_loss + 0.5 * task_loss

       loss.backward()
       optimizer.step()


.. dropdown:: Interventions
    :icon: tools

    The ``intervention`` context manager temporarily replaces an encoder, forcing selected concept
    activations according to a **strategy**. A **policy** determines which concepts are targeted.

    .. code-block:: python

       from torch_concepts.nn import GroundTruthIntervention, UniformPolicy, intervention

       ground_truth = torch.logit(concept_labels, eps=1e-6)
       strategy = GroundTruthIntervention(model=model['encoder'], ground_truth=ground_truth)
       policy   = UniformPolicy(out_concepts=n_concepts)

       with intervention(
           policies=policy,
           strategies=strategy,
           target_concepts=[0, 2],
       ) as intervened_encoder:
           concept_preds = intervened_encoder(x)
           task_preds    = model['predictor'](concepts=concept_preds)


.. dropdown:: (Advanced) Graph Learning
    :icon: workflow

    ``WANDAGraphLearner`` discovers concept relationships via a differentiable weighted adjacency matrix:

    .. code-block:: python

       concept_names = ['round', 'smooth', 'bright', 'large', 'centered']
       graph_learner = pyc_nn.WANDAGraphLearner(
           row_labels=concept_names,
           col_labels=concept_names,
       )

       # weighted_adj: learnable (n_concepts, n_concepts) adjacency tensor
       print(graph_learner.weighted_adj.shape)


.. dropdown:: (Advanced) Verifiable Concept-Based Models
    :icon: shield-check

    Combine a concept encoder, a memory-based selector, and a hyper-network predictor to build
    a model whose task predictions are fully verifiable from concept activations:

    .. code-block:: python

       from torch_concepts.nn import (
           LinearEmbeddingToConcept,
           SelectorEmbeddingEncoder,
           HyperlinearConceptEmbeddingToConcept,
           MLP,
       )
       from torch.nn import ModuleDict

       latent_dim, embedding_size, hidden_size = 64, 16, 32
       memory_size = 7

       model = ModuleDict({
           'encoder': MLP(
               input_size=input_dim,
               hidden_size=latent_dim,
               n_layers=2,
               activation='leaky_relu',
           ),
           'concept_encoder': LinearEmbeddingToConcept(
               in_embeddings=latent_dim,
               out_concepts=n_concepts,
           ),
           'selector': SelectorEmbeddingEncoder(
               in_features=latent_dim,
               out_features=embedding_size,
               n_embeddings=n_tasks,
               memory_size=memory_size,
           ),
           'predictor': HyperlinearConceptEmbeddingToConcept(
               in_concepts=n_concepts,
               in_embeddings=embedding_size,
               hidden_size=hidden_size,
           ),
       })

       latent  = model['encoder'](x)                                   # (batch, latent_dim)
       c_pred  = model['concept_encoder'](latent)                      # (batch, n_concepts)
       emb     = model['selector'](latent, sampling=False)             # (batch, n_tasks, embedding_size)
       y_pred  = model['predictor'](concepts=c_pred, embeddings=emb)  # (batch, n_tasks)


Next Steps
----------

- Explore the full :doc:`Low-Level API documentation </modules/low_level_api>`
- Try the :doc:`Mid-Level API </guides/using_mid_level_proba>` for probabilistic modeling
- Try the :doc:`Mid-Level API </guides/using_mid_level_causal>` for causal modeling
- Check out :doc:`example notebooks <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`
