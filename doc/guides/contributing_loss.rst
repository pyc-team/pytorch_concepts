.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Contributing a New Loss
=======================

This guide explains how to add a new loss term to |pyc_logo| PyC.
Loss terms are plain |pytorch_logo| ``nn.Module`` subclasses that plug directly into
:class:`~torch_concepts.nn.ConceptLoss` without any registration boilerplate.
The class inspects each term's ``forward`` signature at construction time and
passes only the arguments the term asks for, so you include exactly the
parameters you need and nothing else.


.. dropdown:: Loss Term Interface
   :icon: code

   **What ConceptLoss expects**

   A loss term is any ``nn.Module`` whose ``forward`` accepts some subset of the
   following keyword arguments:

   +-------------------+----------------------------------------------------------+
   | Parameter         | Description                                              |
   +===================+==========================================================+
   | ``input``         | Logit tensor for the current concept type.               |
   |                   | Shape: ``(batch * n_concepts, cardinality)`` for         |
   |                   | categorical; ``(batch, n_binary_concepts)`` for binary.  |
   +-------------------+----------------------------------------------------------+
   | ``target``        | Ground-truth labels matching ``input``.                  |
   +-------------------+----------------------------------------------------------+
   | ``padding_mask``  | Boolean tensor, ``True`` for real logit positions,       |
   |                   | ``False`` for padding.  Provided automatically by        |
   |                   | :class:`~torch_concepts.nn.ConceptLoss` when categorical |
   |                   | concepts have mixed cardinalities.                       |
   +-------------------+----------------------------------------------------------+
   | ``weight``        | Optional per-sample weight tensor.                       |
   +-------------------+----------------------------------------------------------+

   You only need to declare the parameters you use. :class:`~torch_concepts.nn.ConceptLoss`
   calls ``inspect.signature`` on your ``forward`` and filters the available
   kwargs to only those your method accepts.  If your term has ``**kwargs`` it
   receives every available argument.

   **Signature inspection and padding_mask**

   Categorical concepts with different cardinalities are padded to a common
   width before being passed to loss terms.  The ``padding_mask`` tensor marks
   which logit positions are real (``True``) and which are padding (``False``).

   - If your term declares ``padding_mask`` in its signature, it will be passed
     the mask and is responsible for ignoring padded positions.
   - If your term does not declare ``padding_mask`` and also does not accept
     ``target``, :class:`~torch_concepts.nn.ConceptLoss` emits a warning
     because the term will see padded logits without knowing which they are.
   - Regularizers that only inspect ``input`` should always declare
     ``padding_mask`` to avoid operating on ``-inf`` padding values.

   **Minimal forward signatures**

   .. code-block:: python

      # Standard supervised loss — needs input and target only.
      def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
          ...

      # Unsupervised regularizer on logits — needs input; optional mask.
      def forward(self, input: torch.Tensor,
                  padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
          ...

   Both signatures are valid.  The return value must be a scalar ``torch.Tensor``.

   **Using multiple terms with weights**

   :class:`~torch_concepts.nn.ConceptLoss` accepts a list of terms per type,
   combined as a weighted sum.  The ``binary_weights`` (or ``categorical_weights``)
   list must have the same length as the ``binary`` (or ``categorical``) list:

   .. code-block:: python

      from torch_concepts.nn import ConceptLoss, L1LogitRegularizer
      from torch.nn import BCEWithLogitsLoss

      loss_fn = ConceptLoss(
          annotations=ann,
          binary=[BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
          binary_weights=[1.0, 0.5],
      )

   When weights are omitted, each term is weighted ``1.0``.


.. dropdown:: Example: Custom Entropy Regularizer
   :icon: flame

   The example below adds a differentiable entropy bonus to binary concept
   logits, encouraging the model to produce confident (low-entropy) predictions.
   It declares ``padding_mask`` so it works safely as a categorical term too.

   .. code-block:: python

      # torch_concepts/nn/modules/loss.py  (add below L1LogitRegularizer)
      import torch
      import torch.nn.functional as F
      from torch import nn
      from typing import Optional


      class EntropyRegularizer(nn.Module):
          """Penalise high-entropy predictions via binary cross-entropy with self.

          Computes the binary entropy ``H(p) = -p log p - (1-p) log(1-p)``
          where ``p = sigmoid(input)``.  Valid (non-padded) positions are
          averaged; the result is multiplied by ``scale``.

          Args:
              scale (float): Multiplicative factor.  Default ``1.0``.
          """

          def __init__(self, scale: float = 1.0):
              super().__init__()
              self.scale = scale

          def forward(
              self,
              input: torch.Tensor,
              padding_mask: Optional[torch.Tensor] = None,
          ) -> torch.Tensor:
              p = torch.sigmoid(input)
              # Binary entropy: H(p) = BCE(p, p)
              entropy = F.binary_cross_entropy(p, p, reduction='none')

              if padding_mask is not None:
                  mask = padding_mask
              else:
                  mask = torch.isfinite(input)

              if mask.any():
                  return self.scale * entropy[mask].mean()
              return torch.tensor(0.0, device=input.device)

   **Verifying the regularizer**

   .. code-block:: python

      import torch
      import torch_concepts as pyc
      from torch.nn import BCEWithLogitsLoss
      from torch_concepts.nn import ConceptLoss
      from torch_concepts.nn.modules.loss import EntropyRegularizer  # before export

      ann = pyc.Annotations(
          labels=["is_round", "color", "label"],
          cardinalities=[1, 3, 1],
          types=["binary", "categorical", "binary"],
      )

      loss_fn = ConceptLoss(
          annotations=ann,
          binary=[BCEWithLogitsLoss(), EntropyRegularizer(scale=0.05)],
          binary_weights=[1.0, 0.5],
          categorical=torch.nn.CrossEntropyLoss(),
      )

      from torch_concepts.nn.modules.outputs import ModelOutput
      batch = 8
      logits = torch.randn(batch, 5)   # 1 + 3 + 1 logits
      target = torch.randint(0, 2, (batch, 3)).float()
      out = ModelOutput(logits=logits, target=target)
      loss = loss_fn(out)
      print(loss)  # scalar tensor

   **Combining with WeightedConceptLoss**

   :class:`~torch_concepts.nn.WeightedConceptLoss` wraps two
   :class:`~torch_concepts.nn.ConceptLoss` instances — one for intermediate
   concepts, one for tasks — and combines them with scalar weights.  Pass
   your custom terms the same way:

   .. code-block:: python

      from torch_concepts.nn import WeightedConceptLoss

      loss_fn = WeightedConceptLoss(
          annotations=ann,
          concept_weight=0.5,
          task_weight=1.0,
          task_names=["label"],
          binary=[BCEWithLogitsLoss(), EntropyRegularizer(scale=0.05)],
          binary_weights=[1.0, 0.5],
      )


.. dropdown:: Registering
   :icon: package

   Once the loss term works locally, register it in two places.

   **1. Module file**

   Add the class to ``torch_concepts/nn/modules/loss.py``.  Place it near
   the existing :class:`~torch_concepts.nn.L1LogitRegularizer` so similar
   terms stay together.

   **2. Public API export**

   Add two lines to ``torch_concepts/nn/__init__.py``:

   .. code-block:: python

      # in torch_concepts/nn/__init__.py
      from .modules.loss import ConceptLoss, WeightedConceptLoss, \
          DepthWeightedConceptLoss, L1LogitRegularizer, EntropyRegularizer

      __all__ = [
          ...
          "EntropyRegularizer",
      ]

   After this, users can do ``from torch_concepts.nn import EntropyRegularizer``.

   **3. API reference page** (optional but recommended)

   Add an ``autoclass`` directive to ``doc/modules/loss_api.rst`` so the
   docstring appears in the rendered documentation:

   .. code-block:: rst

      .. autoclass:: torch_concepts.nn.EntropyRegularizer
         :members:
         :undoc-members:
         :show-inheritance:

   **4. Tests**

   Add a test in ``tests/`` that constructs a :class:`~torch_concepts.nn.ConceptLoss`
   with your term, runs a forward pass, and checks that the output is a scalar.
   Mirror the existing tests in ``tests/test_loss.py`` for the expected structure.


Next Steps
----------

- Read the :class:`~torch_concepts.nn.ConceptLoss` API reference for the full
  list of constructor arguments and the weighted-sum dispatch logic.
- See :class:`~torch_concepts.nn.WeightedConceptLoss` and
  :class:`~torch_concepts.nn.DepthWeightedConceptLoss` for composing losses
  across concept/task splits and graph-structured models.
- Open a pull request to ``dev`` — see :doc:`Contributing <contributing>` for
  the full workflow.
