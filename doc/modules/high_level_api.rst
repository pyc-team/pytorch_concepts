High-Level API
==============

Ready-to-use concept-based models and the structures that configure them. See the
:doc:`High-Level user guide </guides/using_high_level>` for explanations and examples; the
docstrings of each class below document their parameters and behaviour.

Annotations & Configuration
---------------------------

.. currentmodule:: torch_concepts

.. autosummary::
   :toctree: generated
   :nosignatures:

   Annotations
   AnnotatedTensor
   ConceptGraph

.. currentmodule:: torch_concepts.annotations

.. autosummary::
   :toctree: generated
   :nosignatures:

   Concept

Models
------

.. currentmodule:: torch_concepts.nn

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptBottleneckModel
   ConceptEmbeddingModel
   ConceptBottleneckVAE
   ConditionalVAE
   GraphConceptBottleneckModel
   CausallyReliableConceptBottleneckModel
   BlackBox
   BlackBoxTaskOnly

Externally fitted linear heads
------------------------------

:meth:`ConceptBottleneckModel.set_linear_heads` copies separately trained heads
into an ordinary CBM with scalar deterministic continuous concepts and one
categorical task. This supports workflows where a projection and a classifier
are fitted independently, while retaining the CBM's annotated concept queries
and intervention interface. The method installs parameters; it does not fit
either head or change the backbone.

Supported models and parameter shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct the model with ``variable_distributions={"continuous": Delta}`` and
keep ``plate=None`` (the default) or ``plate=True``. The bottleneck must contain
at least one scalar continuous concept, and the single categorical task must
use the default ``OneHotCategorical`` distribution. Models with default
``Normal`` continuous concepts, ``plate=False``, binary concepts/tasks, multiple
tasks, or custom nonlinear heads are unsupported by this method.

Let ``D = model.latent_size`` be the backbone output width, ``C`` the number of
``model.intermediate_concept_names``, and ``K`` the number of task states.
``D`` is the embedding width, even when the backbone consumes images or other
structured inputs.

.. list-table:: Arguments to set_linear_heads
   :header-rows: 1
   :widths: 20 20 60

   * - Argument
     - Tensor shape
     - Meaning
   * - ``concept_weight``
     - ``(C, D)``
     - Required projection weights; rows follow intermediate concept names.
   * - ``concept_bias``
     - ``(C,)``
     - Projection bias before standardization; omitted means zero.
   * - ``task_weight``
     - ``(K, C)``
     - Required classifier weights; columns follow concept order and rows follow task state order.
   * - ``task_bias``
     - ``(K,)``
     - Classifier bias; omitted means zero.
   * - ``concept_mean``
     - ``(C,)`` or ``(1, C)``
     - Optional training means of projected activations; requires ``concept_std``.
   * - ``concept_std``
     - ``(C,)`` or ``(1, C)``
     - Optional positive training standard deviations; requires ``concept_mean``.

Ordering is the caller's responsibility. For names ``["texture", "length"]``,
row 0 of ``concept_weight`` and column 0 of ``task_weight`` must both refer to
``texture``. For task states ``["small", "large", "other"]``, classifier rows
must follow that order. The method retains the model's annotation schema;
it does not infer names from tensors or reorder weights automatically.

Installing fitted tensors
~~~~~~~~~~~~~~~~~~~~~~~~~

This self-contained example uses a three-dimensional input embedding, two
continuous concepts, and three task classes. Biases are omitted and therefore
installed as zero.

.. code-block:: python

   import torch
   from torch_concepts import Annotations
   from torch_concepts.distributions import Delta
   from torch_concepts.nn import ConceptBottleneckModel

   schema = Annotations(
       labels=["texture", "length", "category"],
       states=[["value"], ["value"], ["small", "large", "other"]],
       types=["continuous", "continuous", "categorical"],
   )
   model = ConceptBottleneckModel(
       input_size=3, annotations=schema, task_names=["category"],
       variable_distributions={"continuous": Delta},
   )
   projection_weight = torch.tensor([[1., 0., 2.], [0., 1., -1.]])
   classifier_weight = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])
   model.set_linear_heads(
       concept_weight=projection_weight,
       task_weight=classifier_weight,
   )
   x = torch.tensor([[2., 3., 1.]])
   output = model.eval()(query=["texture", "length", "category"], input=x)
   concepts = x @ projection_weight.T
   torch.testing.assert_close(output.value[["texture", "length"]].tensor, concepts)
   torch.testing.assert_close(output.logits["category"].tensor,
                              concepts @ classifier_weight.T)

With a backbone, the projection operates on ``backbone(x)`` instead of ``x``.
Task outputs are raw logits; concept outputs are the learned real-valued
activations. Supplying concept evidence or interventions can override those
activations through the usual CBM interface.

Folding standardization into the projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the classifier was fitted on standardized projected activations, supply
their training mean and standard deviation. For projection weights ``W`` and
bias ``b``, the installed encoder uses ``W / std[:, None]`` and
``(b - mean) / std``. An omitted projection bias means ``b = 0``. The classifier
weights and bias are copied without further scaling.

Continuing the example, install the original weights with normalization:

.. code-block:: python

   mean = torch.tensor([1., 2.])
   std = torch.tensor([2., 4.])
   model.set_linear_heads(
       concept_weight=projection_weight,
       task_weight=classifier_weight,
       concept_mean=mean, concept_std=std,
   )
   standardized = (x @ projection_weight.T - mean) / std
   output = model(query=["texture", "length", "category"], input=x)
   torch.testing.assert_close(output.value[["texture", "length"]].tensor, standardized)
   torch.testing.assert_close(output.logits["category"].tensor,
                              standardized @ classifier_weight.T)

Queried concepts are now standardized too, and the saved model contains the
folded parameters. No inference-time scaler is required. Both statistics must
be supplied together; scalar broadcasting is unsupported. Omit both if the
supplied projection already includes normalization, to avoid applying it twice.

For already-fitted public projection and classifier layers, a typical call is:

.. code-block:: python

   model.set_linear_heads(
       concept_weight=projection.encoder.weight,
       concept_bias=projection.encoder.bias,  # None for a bias-free projection
       task_weight=classifier.predictor.weight,
       task_bias=classifier.predictor.bias,
       concept_mean=scaler.mean, concept_std=scaler.std,
   )

Here ``scaler`` is fitted on the training projection outputs, and ``classifier``
is fitted on their standardized values, in the same concept order.

Validation and subsequent training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method raises ``ValueError`` for unsupported architectures, missing weights,
incorrect tensor shapes, complex or nonfinite values, incomplete normalization
statistics, nonpositive standard deviations, or nonfinite installed parameters.
All validation and device/dtype conversion finish before any heads change.
This includes overflow introduced by conversion or normalization. Source
tensors are detached and copied; subsequent changes to the source modules do
not change this CBM, and gradients do not flow back to them.

The method returns ``self``. It preserves parameter ``requires_grad`` flags
and the model's current train/eval mode, so the installed heads can be further
trained normally. To freeze the whole fitted model for inference, use
``model.eval().requires_grad_(False)`` explicitly. Calling the method again
replaces both heads; an omitted bias resets that bias to zero rather than
keeping its previous value.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptBottleneckModel.set_linear_heads

Outputs
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   InferenceOutput

Base Classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseModel
