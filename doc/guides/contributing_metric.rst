.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Contributing a New Metric
=========================

This guide explains how to add a new metric to |pyc_logo| PyC.
:class:`~torch_concepts.nn.ConceptMetrics` wraps a
``torchmetrics.MetricCollection`` and routes predictions to the right metric
based on concept type (binary or categorical) as declared in the
:class:`~torch_concepts.Annotations`.

The recommended path for most metrics is to reuse an existing ``torchmetrics``
metric.  Write a custom subclass only when no existing metric covers your need.


.. dropdown:: Using Existing torchmetrics Metrics
   :icon: graph

   Pass metrics as a dict mapping a name string to a metric spec.  There are
   three supported spec forms, listed in order of preference.

   **Form 1 — Class only**

   Pass the class directly.  :class:`~torch_concepts.nn.ConceptMetrics`
   instantiates it with no extra arguments.  Use this for metrics that need
   no configuration (e.g. ``BinaryAccuracy``):

   .. code-block:: python

      from torchmetrics.classification import BinaryAccuracy
      import torch_concepts as pyc
      from torch_concepts.nn import ConceptMetrics

      ann = pyc.Annotations(
          labels=["is_round", "color", "label"],
          cardinalities=[1, 3, 1],
          types=["binary", "categorical", "binary"],
      )

      metrics = ConceptMetrics(
          annotations=ann,
          binary={"accuracy": BinaryAccuracy},
      )

   **Form 2 — (Class, kwargs) tuple**

   Pass a ``(MetricClass, kwargs)`` tuple to supply extra constructor
   arguments.  For categorical metrics, ``num_classes`` is injected
   automatically from the annotations — do not include it in ``kwargs``:

   .. code-block:: python

      from torchmetrics.classification import (
          BinaryAccuracy, MulticlassAccuracy, MulticlassF1Score,
      )

      metrics = ConceptMetrics(
          annotations=ann,
          binary={"accuracy": BinaryAccuracy},
          categorical={
              "accuracy": (MulticlassAccuracy, {"average": "micro"}),
              "f1":       (MulticlassF1Score,   {"average": "macro"}),
          },
      )

   **Form 3 — Pre-instantiated metric**

   Pass an already-constructed ``torchmetrics.Metric`` object.  You retain
   full control, but you must set ``num_classes`` yourself for categorical
   metrics:

   .. code-block:: python

      from torchmetrics.classification import MulticlassAccuracy

      # num_classes must be set manually — ConceptMetrics cannot inject it.
      cat_acc = MulticlassAccuracy(num_classes=3, average="weighted")

      metrics = ConceptMetrics(
          annotations=ann,
          categorical={"accuracy": cat_acc},
      )

   **Calling the metrics object**

   .. code-block:: python

      import torch

      batch = 8
      logits = torch.randn(batch, 5)        # 1 + 3 + 1 logits
      target = torch.randint(0, 2, (batch, 3)).float()

      metrics.update(logits, target)
      results = metrics.compute()
      # {"SUMMARY-binary_accuracy": tensor(...),
      #  "SUMMARY-categorical_accuracy": tensor(...), ...}
      metrics.reset()

   **Per-concept tracking**

   Set ``per_concept=True`` to get one metric entry per concept, or pass a
   list of concept names to track a subset:

   .. code-block:: python

      metrics = ConceptMetrics(
          annotations=ann,
          binary={"accuracy": BinaryAccuracy},
          categorical={"accuracy": (MulticlassAccuracy, {"average": "micro"})},
          per_concept=["is_round", "color"],  # only these two
      )

      metrics.update(logits, target)
      results = metrics.compute()
      # Includes "is_round_accuracy", "color_accuracy", plus SUMMARY keys.

   **Split tracking**

   Use ``clone(prefix=...)`` to get independent metric objects for train,
   validation, and test splits, each with its own accumulated state:

   .. code-block:: python

      train_metrics = metrics.clone(prefix="train")
      val_metrics   = metrics.clone(prefix="val")
      test_metrics  = metrics.clone(prefix="test")


.. dropdown:: Custom Metric
   :icon: code

   When no existing ``torchmetrics`` metric fits your need, subclass
   ``torchmetrics.Metric`` directly.  The three methods to implement are
   ``__init__``, ``update``, and ``compute``.

   The example below computes **Concept Alignment Score (CAS)** — the
   fraction of concepts whose predicted label matches the ground-truth label
   (macro-averaged binary accuracy per concept, then averaged across
   concepts).  It illustrates the full ``torchmetrics.Metric`` contract.

   .. code-block:: python

      # torch_concepts/nn/modules/metrics.py  (add near end of file)
      import torch
      from torchmetrics import Metric


      class ConceptAlignmentScore(Metric):
          """Mean per-concept accuracy across all binary concepts.

          Accumulates the number of correct predictions and total samples per
          concept, then returns the mean accuracy.

          Args:
              n_concepts (int): Number of binary concepts to track.

          Example::

              cas = ConceptAlignmentScore(n_concepts=3)
              cas.update(preds=torch.tensor([[0.9, -0.3, 0.1]]),
                         target=torch.tensor([[1.0,  0.0, 0.0]]))
              score = cas.compute()  # tensor(0.6667)
          """

          # Higher is better (for logging direction hints in Lightning).
          higher_is_better = True

          def __init__(self, n_concepts: int, **kwargs):
              super().__init__(**kwargs)
              # add_state registers tensors that are reset() on each epoch.
              self.add_state(
                  "correct",
                  default=torch.zeros(n_concepts),
                  dist_reduce_fx="sum",
              )
              self.add_state(
                  "total",
                  default=torch.zeros(n_concepts),
                  dist_reduce_fx="sum",
              )
              self.n_concepts = n_concepts

          def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
              """Accumulate correct predictions.

              Args:
                  preds: Logit tensor, shape ``(batch, n_concepts)``.
                  target: Ground-truth labels, shape ``(batch, n_concepts)``.
                      Values are ``0.0`` or ``1.0``.
              """
              predicted = (preds > 0).float()              # threshold at 0
              correct   = (predicted == target).float()    # (batch, n_concepts)
              self.correct += correct.sum(dim=0)
              self.total   += torch.full_like(self.total, preds.shape[0])

          def compute(self) -> torch.Tensor:
              """Return mean per-concept accuracy (scalar)."""
              per_concept_acc = self.correct / self.total.clamp(min=1)
              return per_concept_acc.mean()

   **Plugging the custom metric into ConceptMetrics**

   Use Form 1 (class only) when ``n_concepts`` matches the number of binary
   concepts in the annotations.  If you need to pass it explicitly, use
   Form 2 (tuple):

   .. code-block:: python

      import torch_concepts as pyc
      from torch_concepts.nn import ConceptMetrics
      from torch_concepts.nn.modules.metrics import ConceptAlignmentScore

      ann = pyc.Annotations(
          labels=["is_round", "shiny", "label"],
          cardinalities=[1, 1, 1],
          types=["binary", "binary", "binary"],
      )

      # n_concepts injected automatically via the annotations — not needed here
      # because ConceptAlignmentScore takes n_concepts, not num_classes.
      # Use Form 3 (pre-instantiated) when auto-injection does not apply.
      n_binary = len(ann.type_groups["binary"]["labels"])
      cas = ConceptAlignmentScore(n_concepts=n_binary)

      metrics = ConceptMetrics(
          annotations=ann,
          binary={"cas": cas},   # Form 3: pre-instantiated
      )

      logits = torch.randn(16, 3)
      target = torch.randint(0, 2, (16, 3)).float()
      metrics.update(logits, target)
      print(metrics.compute())
      # {"SUMMARY-binary_cas": tensor(...)}


.. dropdown:: Registering
   :icon: package

   Once the metric works locally, register it in two places.

   **1. Module file**

   Add the class to ``torch_concepts/nn/modules/metrics.py``.  Place it near
   existing metric classes so related code stays together.

   **2. Public API export**

   Add two lines to ``torch_concepts/nn/__init__.py``:

   .. code-block:: python

      # in torch_concepts/nn/__init__.py
      from .modules.metrics import ConceptMetrics, compute_cace, \
          ConceptAlignmentScore

      __all__ = [
          ...
          "ConceptAlignmentScore",
      ]

   After this, users can do ``from torch_concepts.nn import ConceptAlignmentScore``.

   **3. API reference page** (optional but recommended)

   Add an ``autoclass`` directive to the appropriate reference page so the
   docstring appears in the rendered documentation:

   .. code-block:: rst

      .. autoclass:: torch_concepts.nn.ConceptAlignmentScore
         :members:
         :undoc-members:
         :show-inheritance:

   **4. Tests**

   Add a test in ``tests/`` that constructs a
   :class:`~torch_concepts.nn.ConceptMetrics` with your metric, calls
   ``update`` and ``compute``, and checks the output value.  Mirror the
   existing tests in ``tests/test_metrics.py`` for the expected structure.


Next Steps
----------

- Read the :class:`~torch_concepts.nn.ConceptMetrics` API reference for the
  full list of constructor arguments, the ``clone`` / ``reset`` / ``compute``
  lifecycle, and the output key naming scheme.
- Browse ``torchmetrics`` documentation at https://torchmetrics.readthedocs.io
  to find existing metrics before writing a custom one.
- Open a pull request to ``dev`` — see :doc:`Contributing <contributing>` for
  the full workflow.
