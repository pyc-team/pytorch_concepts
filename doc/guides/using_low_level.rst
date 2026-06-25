.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Low-Level API
=============

The Low-Level API provides composable building blocks for concept-based neural networks,
using a pure |pytorch_logo| PyTorch interface. It is the right entry point if you are
comfortable with PyTorch and want to assemble interpretable architectures by hand.

.. image:: /_static/img/api_levels/low_level.png
   :alt: Overview of the PyC Low-Level API
   :align: center
   :width: 100%

|

The core building blocks of this API level are:

- **Data representations** — the two kinds of tensors that flow through the layers.
- **Layers** — interpretable encoders and predictors, plus a few special-purpose layers.
- **Models** — standard |pytorch_logo| PyTorch containers that wire the layers together.
- **Interventions** — context managers that temporarily override concept activations.

Each block is described below, followed by an example of how to use it.


Data Representations
--------------------

|pyc_logo| PyC layers operate on two types of data representations:

- **Embeddings**: vector representations of the input, not yet associated with specific
  concept variables.
- **Concepts**: scalar-valued activation scores — one per concept variable. These form the
  interpretable bottleneck.

Keeping the two apart is what makes the architecture interpretable: every concept score
corresponds to a named, human-understandable variable.

**Example.** A small batch with an embedding tensor and a (ground-truth) concept tensor:

.. code-block:: python

   import torch

   batch_size, input_dim, n_concepts = 32, 64, 5

   embeddings = torch.randn(batch_size, input_dim)            # embedding representation
   concept_labels = torch.randint(0, 2, (batch_size, n_concepts)).float()  # concept scores


Layers
------

Layers are the heart of the Low-Level API. They come in two interpretable flavours plus a
few special-purpose layers:

- ``Encoder`` layers map embeddings to concept scores (or to a new embedding). They **never**
  take concept scores as input.
- ``Predictor`` layers take concept scores (and optionally embeddings) as input and produce
  task or concept predictions.
- **Special layers** (e.g. graph learners, memory selectors) perform operations that do not
  follow the encoder/predictor interface.

**Naming convention.** Layer names follow the pattern ``<OperationType><InputType>To<OutputType>``:

- ``OperationType``: the computation performed — ``Linear``, ``Hyperlinear``, ``Mix``, ``Selector``, …
- ``InputType``: ``Embedding`` for vector inputs, ``Concept`` for concept scores.
- ``OutputType``: ``Concept`` for score outputs, ``Embedding`` for vector outputs.

So ``LinearEmbeddingToConcept`` is a linear **encoder** (embedding → concepts) and
``LinearConceptToConcept`` is a linear **predictor** (concepts → concepts/tasks).

**Example.** A linear encoder and a linear predictor:

.. code-block:: python

   import torch_concepts.nn as pyc_nn

   encoder = pyc_nn.LinearEmbeddingToConcept(in_embeddings=64, out_concepts=5)
   predictor = pyc_nn.LinearConceptToConcept(in_concepts=5, out_concepts=3)

   concept_logits = encoder(embeddings)              # (32, 5)
   task_logits = predictor(concepts=concept_logits)  # (32, 3)

**Example (special layer).** Graph learners discover relationships between concepts:

.. code-block:: python

   wanda = pyc_nn.WANDAGraphLearner(
       row_labels=['c1', 'c2', 'c3'],
       col_labels=['task A', 'task B', 'task C'],
   )
   print(wanda.weighted_adj.shape)  # learnable adjacency tensor


Models
------

A model is built as in standard |pytorch_logo| PyTorch — for example with ``ModuleDict`` or
``Sequential`` — and may freely mix standard PyTorch layers with |pyc_logo| PyC layers. A
Concept Bottleneck Model simply routes predictions through an interpretable concept layer.

**Example.** A Concept Bottleneck Model and a manual training step:

.. code-block:: python

   import torch
   import torch.nn.functional as F
   from torch.nn import ModuleDict
   import torch_concepts.nn as pyc_nn

   batch_size, input_dim, n_concepts, n_tasks = 32, 64, 5, 3
   x = torch.randn(batch_size, input_dim)
   concept_labels = torch.randint(0, 2, (batch_size, n_concepts)).float()
   task_labels = torch.randint(0, 2, (batch_size, n_tasks)).float()

   model = ModuleDict({
       'encoder': pyc_nn.LinearEmbeddingToConcept(in_embeddings=input_dim, out_concepts=n_concepts),
       'predictor': pyc_nn.LinearConceptToConcept(in_concepts=n_concepts, out_concepts=n_tasks),
   })

   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

   # Forward pass — outputs are raw logits
   concept_preds = model['encoder'](x)
   task_preds = model['predictor'](concepts=concept_preds)

   # Train with logit-stable losses
   concept_loss = F.binary_cross_entropy_with_logits(concept_preds, concept_labels)
   task_loss = F.binary_cross_entropy_with_logits(task_preds, task_labels)
   loss = concept_loss + 0.5 * task_loss

   loss.backward()
   optimizer.step()


Interventions
-------------

Interventions let you **edit concept activations at inference time** and observe the effect on
downstream predictions. The ``intervention`` context manager temporarily replaces an encoder:

- a **strategy** decides *how* the targeted concepts are overridden (e.g. set to ground truth,
  or to a constant);
- a **policy** decides *which* concepts are targeted.

**Example.** Force two concepts to their ground-truth values:

.. code-block:: python

   import torch
   from torch_concepts.nn import GroundTruthIntervention, UniformPolicy, intervention

   ground_truth = torch.logit(concept_labels, eps=1e-6)
   strategy = GroundTruthIntervention(model=model['encoder'], ground_truth=ground_truth)
   policy = UniformPolicy(out_concepts=n_concepts)

   with intervention(
       policies=policy,
       strategies=strategy,
       target_concepts=[0, 2],   # intervene on the 1st and 3rd concepts
   ) as intervened_encoder:
       concept_preds = intervened_encoder(x)
       task_preds = model['predictor'](concepts=concept_preds)


Next Steps
----------

- Browse the full :doc:`Low-Level API reference </modules/low_level_api>`.
- Move up to the :doc:`Mid-Level API <using_mid_level>` to express models as probabilistic graphs.
- Check out the `example scripts <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`_.
