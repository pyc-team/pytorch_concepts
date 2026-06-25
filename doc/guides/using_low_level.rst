.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Interpretable Layers and Interventions
======================================

The Low-Level API provides composable building blocks for concept-based neural networks,
using a pure |pytorch_logo| PyTorch interface. It is the right entry point if you are
comfortable with PyTorch and want to assemble interpretable architectures by hand.

.. image:: /_static/img/api_levels/low_level.png
   :alt: Overview of the PyC Low-Level API
   :align: center
   :width: 100%

|

The core building blocks of this API level are:

- **Annotations** — give names and types to the concepts on a tensor axis.
- **Annotated Tensors** — tensors that carry their concept annotations along.
- **PyC layers** — interpretable encoders and predictors you wire together.
- **Interventions** — context managers that temporarily override concept activations.
- **Functionals** — stateless operations and evaluation metrics.

Expand each block below for an explanation and an example of how to use it.


.. dropdown:: Annotations
    :icon: tag

    :class:`~torch_concepts.Annotations` give semantic meaning to the concept dimension of a
    tensor. They store the ordered ``labels``, their ``cardinalities`` (``1`` for binary, ``>1``
    for categorical), their ``types``, and optional per-label ``metadata``. This is what lets you
    refer to a concept by name instead of by column index.

    .. code-block:: python

       import torch_concepts as pyc

       annotations = pyc.Annotations(
           labels=["smoking", "genotype", "tar", "cancer"],
           cardinalities=[1, 3, 1, 1],
           types=["binary", "categorical", "continuous", "binary"],
       )

       # Access a single concept's properties by name
       genotype = annotations.concept("genotype")
       print(genotype.cardinality, genotype.type)   # 3 categorical


.. dropdown:: Annotated Tensors
    :icon: package

    An :class:`~torch_concepts.AnnotatedTensor` pairs a plain :class:`torch.Tensor` with an
    :class:`~torch_concepts.Annotations` describing its second axis (axis 1). You can then slice
    columns **by concept name** or **by type**, and any operation that leaves the concept axis
    unchanged keeps the annotation attached.

    .. code-block:: python

       import torch
       import torch_concepts as pyc

       tensor = pyc.AnnotatedTensor(
           data=torch.randn(10, 6),   # (batch_size, sum(cardinalities))
           annotation=annotations,
       )

       smoking = tensor["smoking"]              # slice by concept name
       binary = tensor.split_by_type("binary")  # slice by concept type


.. dropdown:: PyC layers
    :icon: stack

    Layers are the heart of the Low-Level API. They come in two interpretable flavours, plus a
    few special-purpose layers:

    - ``Encoder`` layers map **embeddings** (vector representations of the input) to **concept**
      scores. They never take concept scores as input.
    - ``Predictor`` layers take concept scores (and optionally embeddings) as input and produce
      task or concept predictions.
    - **Special layers** (e.g. graph learners) perform operations that do not follow the
      encoder/predictor interface.

    Layer names follow the pattern ``<OperationType><InputType>To<OutputType>``, so
    ``LinearEmbeddingToConcept`` is a linear encoder (embedding → concepts) and
    ``LinearConceptToConcept`` is a linear predictor (concepts → concepts/tasks). You wire them
    together with standard |pytorch_logo| PyTorch containers such as ``ModuleDict`` — a Concept
    Bottleneck Model is just an encoder feeding a predictor:

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

       # Forward pass — outputs are raw logits
       concept_preds = model['encoder'](x)
       task_preds = model['predictor'](concepts=concept_preds)

       # Train with logit-stable losses
       optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
       loss = F.binary_cross_entropy_with_logits(concept_preds, concept_labels) \
              + 0.5 * F.binary_cross_entropy_with_logits(task_preds, task_labels)
       loss.backward()
       optimizer.step()

    Graph learners are special layers that discover relationships between concepts:

    .. code-block:: python

       wanda = pyc_nn.WANDAGraphLearner(
           row_labels=['c1', 'c2', 'c3'],
           col_labels=['task A', 'task B', 'task C'],
       )
       print(wanda.weighted_adj.shape)   # learnable adjacency tensor


.. dropdown:: Interventions
    :icon: tools

    Interventions let you **edit concept activations at inference time** and observe the effect on
    downstream predictions. The ``intervention`` context manager temporarily replaces an encoder:

    - a **strategy** decides *how* the targeted concepts are overridden (e.g. set to ground truth,
      or to a constant);
    - a **policy** decides *which* concepts are targeted.

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


.. dropdown:: Functionals
    :icon: zap

    :mod:`torch_concepts.nn.functional` collects **stateless operations** used inside and
    alongside PyC layers: interpretable read-outs (linear-equation and logic-rule evaluation),
    differentiable selection, and post-hoc evaluation metrics such as concept completeness and the
    Average Causal Effect.

    .. code-block:: python

       import torch
       from torch_concepts.nn import functional as F

       # Differentiable soft selection over an options axis
       values = torch.randn(32, 5)
       selected = F.soft_select(values, temperature=0.1, dim=1)

       # Post-hoc evaluation: Average Causal Effect of a concept on a task
       # (task predictions with the concept forced to 0 vs to 1)
       ace = F.cace_score(y_pred_c0, y_pred_c1)


Next Steps
----------

- Browse the full :doc:`Low-Level API reference </modules/low_level_api>`.
- Move up to the :doc:`Mid-Level API <using_mid_level>` to express models as probabilistic graphs.
- Check out the `example scripts <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`_.
