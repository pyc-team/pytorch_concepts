.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Contributing a New Layer
========================

|pyc_logo| PyC layers are the semantic primitives of the Low-Level API. Every layer extends
:class:`~torch_concepts.nn.BaseConceptLayer` — a thin |pytorch_logo| PyTorch ``nn.Module``
wrapper that standardises how inputs and outputs are described in terms of concepts,
embeddings, and their cardinalities. New layers slot directly into both the Low-Level API and
the mid-level :class:`~torch_concepts.nn.ParametricCPD` system.

Expand each block below for a step-by-step walkthrough.


.. dropdown:: Layer Interface
    :icon: code

    All layers extend ``BaseConceptLayer``. The constructor accepts three optional dimension
    descriptors — pass an :class:`~torch_concepts.Annotations` object instead of an ``int``
    to make the layer *semantics-aware*:

    .. code-block:: python

       class BaseConceptLayer(ABC, torch.nn.Module):
           def __init__(
               self,
               out_concepts: Union[int, Annotations],
               in_concepts:  Union[int, Annotations] = None,
               in_embeddings: Union[int, Annotations] = None,
           ): ...

    After ``super().__init__(...)`` three resolved-integer attributes are available:

    - ``self.in_concepts_shape``  — ``int`` (or ``None`` if not passed)
    - ``self.in_embeddings_shape`` — ``int`` (or ``None`` if not passed)
    - ``self.out_concepts_shape``  — ``int``

    These always hold plain integers regardless of whether you passed an ``Annotations``
    object or an ``int``, so you can use them directly in ``nn.Linear`` / ``nn.Conv``.

    **Naming convention.** Layer class names follow
    ``<OperationType><InputType>To<OutputType>``, e.g.:

    - ``LinearEmbeddingToConcept`` — linear map from embeddings to concept scores
    - ``LinearConceptToConcept`` — linear map from concept scores to concept scores
    - ``MixConceptEmbeddingToConcept`` — mixes concept scores with their embeddings

    **forward keyword arguments** mirror the input types:
    use ``concepts=`` for concept-score tensors and ``embeddings=`` for embedding tensors.
    This lets callers be explicit and lets ``ParametricCPD`` route inputs correctly.


.. dropdown:: Minimal Example
    :icon: stack

    Below is a complete new layer — a concept-score normaliser that applies
    layer-norm before passing concepts through a linear map:

    .. code-block:: python

       # torch_concepts/nn/modules/low/predictors/normed.py
       from typing import Union
       import torch
       import torch.nn as nn
       from torch_concepts import Annotations, AnnotatedTensor
       from torch_concepts.nn.modules.low.base.layer import BaseConceptLayer


       class NormedConceptToConcept(BaseConceptLayer):
           """Layer-normalised linear concept predictor.

           Applies layer-norm to the input concept scores before the linear map.
           Useful when concept activations vary widely in scale.

           Args:
               in_concepts:  Input concept dimension (int or Annotations).
               out_concepts: Output concept dimension (int or Annotations).
           """

           def __init__(
               self,
               in_concepts:  Union[int, Annotations],
               out_concepts: Union[int, Annotations],
           ):
               super().__init__(in_concepts=in_concepts, out_concepts=out_concepts)
               self.norm      = nn.LayerNorm(self.in_concepts_shape)
               self.predictor = nn.Linear(self.in_concepts_shape, self.out_concepts_shape)

           def forward(self, concepts: torch.Tensor) -> torch.Tensor:
               x = self.norm(concepts)
               x = self.predictor(x)
               # If out_concepts is Annotations, wrap output as AnnotatedTensor
               return self.annotate(x)

    Instantiation and forward pass:

    .. code-block:: python

       import torch
       import torch_concepts as pyc

       annotations = pyc.Annotations(
           labels=["smoking", "genotype", "tar"],
           cardinalities=[1, 3, 1],
           types=["binary", "categorical", "continuous"],
       )

       # Plain int — no annotation on output
       layer = NormedConceptToConcept(in_concepts=5, out_concepts=2)
       out = layer(concepts=torch.randn(8, 5))   # (8, 2)

       # Annotations — output is an AnnotatedTensor
       layer = NormedConceptToConcept(in_concepts=annotations, out_concepts=2)
       out = layer(concepts=torch.randn(8, 5))   # AnnotatedTensor (8, 2)
       print(out["genotype"])                    # slice by name


.. dropdown:: Semantics-Aware Layers
    :icon: tag

    Passing an :class:`~torch_concepts.Annotations` object for ``in_concepts`` or
    ``out_concepts`` unlocks semantics-awareness:

    - ``self.in_concepts`` (or ``out_concepts``) holds the full ``Annotations`` object.
    - ``self.in_concepts_shape`` still holds the corresponding integer (``annotations.size``).
    - Call ``self.annotate(x)`` (no arguments) to wrap the output as an
      ``AnnotatedTensor`` — this uses ``self.out_concepts`` if it is an ``Annotations``.

    Use the annotations inside the layer when you need per-concept branching, e.g. grouping
    columns by cardinality or type:

    .. code-block:: python

       class TypeAwareConceptLayer(BaseConceptLayer):
           def __init__(self, in_concepts: Annotations, out_concepts: int):
               super().__init__(in_concepts=in_concepts, out_concepts=out_concepts)
               # Build one head per concept type
               binary_size = sum(
                   c for c, t in zip(in_concepts.cardinalities, in_concepts.types)
                   if t == "binary"
               )
               self.binary_head = nn.Linear(binary_size, out_concepts)
               # ... other heads

           def forward(self, concepts: torch.Tensor) -> torch.Tensor:
               ann: Annotations = self.in_concepts
               binary = concepts[:, [i for i, t in enumerate(ann.types) if t == "binary"]]
               return self.binary_head(binary)

    :class:`~torch_concepts.nn.MixConceptEmbeddingToConcept` is the canonical example —
    it requires ``Annotations`` for ``in_concepts`` so it can split columns by cardinality
    and concept type.


.. dropdown:: Using a Layer inside a ParametricCPD
    :icon: gear

    Layers integrate directly into the Mid-Level API through
    :class:`~torch_concepts.nn.ParametricCPD`. Pass your layer as the value under the
    distribution-parameter key that the target variable's distribution expects:

    .. code-block:: python

       import torch_concepts as pyc
       from torch_concepts import ConceptVariable
       from torch.distributions import Bernoulli
       from torch_concepts.nn import ParametricCPD
       from normed import NormedConceptToConcept   # the layer above

       latent_var = pyc.EmbeddingVariable("latent", distribution=pyc.distributions.Delta, size=64)
       concept    = ConceptVariable("smoking", distribution=Bernoulli)

       cpd = ParametricCPD(
           concept,
           parents=[latent_var],
           parametrization={
               "logits": NormedConceptToConcept(in_concepts=64, out_concepts=1),
           },
       )

    The ``parametrization`` dict key must match the distribution's constructor parameter
    (``logits`` or ``probs`` for ``Bernoulli``/``OneHotCategorical``, ``loc``/``scale`` for
    ``Normal``).


.. dropdown:: Registering the Layer
    :icon: package

    1. **Add the file.** Place your layer under the appropriate subdirectory:

       - ``torch_concepts/nn/modules/low/encoders/`` — if it maps from embeddings
       - ``torch_concepts/nn/modules/low/predictors/`` — if it maps from concept scores
       - Create a new subdirectory if neither fits.

    2. **Export it.** Add the class to ``torch_concepts/nn/__init__.py``:

       .. code-block:: python

          from .modules.low.predictors.normed import NormedConceptToConcept
          __all__ = [..., "NormedConceptToConcept"]

    3. **Document it.** Add an ``autoclass`` entry in ``doc/modules/low_level_api.rst``:

       .. code-block:: rst

          .. autoclass:: torch_concepts.nn.NormedConceptToConcept
             :members:

    4. **Test it.** Add a test in ``tests/nn/modules/low/`` that checks the output shape
       for both ``int`` and ``Annotations`` inputs, and that ``AnnotatedTensor`` is returned
       when ``out_concepts`` is an ``Annotations``.


Next Steps
----------

- See how layers compose into probabilistic models: :doc:`Contributing a New Model <contributing_model>`.
- Browse the Low-Level API reference: :doc:`Low-Level API </modules/low_level_api>`.
- Check existing layer implementations in ``torch_concepts/nn/modules/low/``.
