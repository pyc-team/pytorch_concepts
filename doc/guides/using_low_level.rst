.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Semantic primitives and Interventions
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
- **PyC layers** — semntics-aware layers that produce concepts.
- **Interventions** — modules to temporarily override layers.
- **Functionals** — evaluation metrics and loss functions.

Expand each block below for an explanation and an example of how to use it.


.. dropdown:: Annotations
    :icon: tag

    :class:`~torch_concepts.Annotations` give semantic meaning to the concept dimension of a
    tensor. They store the ordered ``labels``, their ``cardinalities`` (``1`` for binary, ``>1``
    for categorical), their ``types`` (``binary``, ``categorical`` or ``continuous``), and optional per-label ``metadata``. This is what lets you
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
    columns **by concept name** or **by type**. Any operation that leaves the concept axis
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

    Layers are the heart of the Low-Level API. All layers extend ``BaseConceptLayer``: they
    accept **concept scores** and/or **embeddings** as input (both optional) and always return
    **concept scores**. Passing ``Annotations`` instead of an integer for ``in_concepts`` or
    ``out_concepts`` makes the layer **semantics-aware**: it knows concept names, cardinalities,
    and types, and can annotate its output as an ``AnnotatedTensor``.

    Layer names follow the pattern ``<OperationType><InputType>To<OutputType>``, so
    ``LinearEmbeddingToConcept`` maps embeddings to concepts and
    ``LinearConceptToConcept`` maps concepts to concepts:

    .. code-block:: python

       import torch
       import torch_concepts as pyc

       encoder = pyc.nn.LinearEmbeddingToConcept(in_embeddings=64, out_concepts=5)
       predictor = pyc.nn.LinearConceptToConcept(in_concepts=5, out_concepts=1)

       x = torch.randn(32, 64)
       concepts = encoder(embeddings=x)
       tasks = predictor(concepts=concepts)

    ``MixConceptEmbeddingToConcept`` can be used to mix concept activations with concept embeddings. 
    It requires ``Annotations`` for ``in_concepts`` — this makes
    it inherently **type-aware**, grouping columns by cardinality and concept type:

    .. code-block:: python

       import torch
       import torch_concepts as pyc
       from torch_concepts.nn import MixConceptEmbeddingToConcept

       annotations = pyc.Annotations(
           labels=["smoking", "genotype", "tar", "cancer"],
           cardinalities=[1, 3, 1, 1],
           types=["binary", "categorical", "continuous", "binary"],
       )
       predictor = MixConceptEmbeddingToConcept(
           in_concepts=annotations,   # type-aware: uses cardinalities and types
           in_embeddings=16,
           out_concepts=1,
       )
       concepts   = torch.randn(32, 6)        # (batch, sum(cardinalities))
       embeddings = torch.randn(32, 6, 16)    # (batch, sum(cardinalities), emb_size)
       tasks = predictor(concepts=concepts, embeddings=embeddings)

    ``pyc.nn.Sequential`` extends ``torch.nn.Sequential`` so that its **first module can accept
    multiple inputs** (e.g. ``concepts`` and ``embeddings``). Every subsequent module still
    receives a single tensor. It also accepts an optional ``out_concepts: Annotations`` to
    annotate its output as an ``AnnotatedTensor``:

    .. code-block:: python

       annotations = pyc.Annotations(labels=["c1", "c2"], cardinalities=[1, 1])
       pipeline = pyc.nn.Sequential(
           pyc.nn.MixConceptEmbeddingToConcept(in_concepts=annotations, in_embeddings=16, out_concepts=2),
           torch.nn.ReLU(),
           out_concepts=annotations,   # output is an AnnotatedTensor
       )
       out = pipeline(concepts=concepts, embeddings=embeddings)   # first layer takes both inputs

    Graph learners are special layers that discover relationships between concepts:

    .. code-block:: python

       wanda = pyc.nn.WANDAGraphLearner(
           row_labels=['c1', 'c2', 'c3'],
           col_labels=['task A', 'task B', 'task C'],
       )
       print(wanda.weighted_adj.shape)   # learnable adjacency tensor


.. dropdown:: Putting it Together: Concept Bottleneck Model
    :icon: rocket

    A CBM is an encoder feeding a predictor. During training, supervise both
    concept and task predictions:

    .. code-block:: python

       import torch
       import torch.nn.functional as F
       import torch_concepts as pyc

       encoder   = pyc.nn.LinearEmbeddingToConcept(in_embeddings=64, out_concepts=5)
       predictor = pyc.nn.LinearConceptToConcept(in_concepts=5, out_concepts=1)

       x      = torch.randn(32, 64)
       c_true = torch.randint(0, 2, (32, 5)).float()
       y_true = torch.randint(0, 2, (32, 1)).float()

       concepts = encoder(embeddings=x)
       tasks    = predictor(concepts=concepts)
       loss = F.binary_cross_entropy_with_logits(concepts, c_true) \
            + F.binary_cross_entropy_with_logits(tasks, y_true)


.. dropdown:: Interventions
    :icon: tools

    ``InterventionModule`` wraps any layer as a **drop-in replacement**. The replacement 
    has the same call signature, but concepts are selectively modified at inference time.

    - A **strategy** decides *how* to intervene. Two kinds are supported:

      - **Concept strategies** (``BaseConceptInterventionStrategy``): override the layer's
        *output* concept values — e.g. ``DoIntervention`` (set to a constant) or
        ``GroundTruthIntervention`` (set to ground-truth labels).
      - **Mechanism strategies** (``BaseModuleInterventionStrategy``): modify the layer's
        *weights and connections* — e.g. ``PositiveWeightsIntervention`` (force positive
        weights, making the layer monotonic).

    - A **policy** decides *which* concepts are targeted — e.g. ``UniformPolicy`` (all),
      ``UncertaintyPolicy`` (least certain), ``GradientPolicy`` (most influential).

    .. code-block:: python

       from torch_concepts.nn import InterventionModule, DoIntervention, UniformPolicy

       intervened = InterventionModule(
           original_module=encoder,
           intervention_strategy=DoIntervention(constants=1.0),
           intervention_policy=UniformPolicy(out_concepts=5),
           out_concepts_to_intervene_on=[0, 2],   # target concepts 0 and 2
       )
       concepts = intervened(embeddings=x)   # same interface as encoder
       tasks = predictor(concepts=concepts)


.. dropdown:: Functionals
    :icon: zap

    :mod:`torch_concepts.nn.functional` collects **stateless operations** used inside and
    alongside PyC layers: interpretable read-outs, differentiable selection, and post-hoc
    evaluation metrics.

    .. code-block:: python

       from torch_concepts.nn import functional as F

       # CaCE: causal effect of concept 0 on task predictions
       concepts_c0 = concepts.clone(); concepts_c0[:, 0] = -1e6   # do(C_0 = 0)
       concepts_c1 = concepts.clone(); concepts_c1[:, 0] =  1e6   # do(C_0 = 1)
       cace = F.cace_score(predictor(concepts=concepts_c0), predictor(concepts=concepts_c1))

       # NCC: average number of concepts that explain 95% of each prediction
       ncc = F.number_of_contributing_concepts(predictor.predictor.weight, concepts)


Next Steps
----------

- Browse the full :doc:`Low-Level API reference </modules/low_level_api>`.
- Move up to compose layers into :doc:`Interpretable Probabilistic Models <using_mid_level>`.
- Check out the low-level `example scripts <https://github.com/pyc-team/pytorch_concepts/tree/master/examples/utilization/0_layer>`_.
