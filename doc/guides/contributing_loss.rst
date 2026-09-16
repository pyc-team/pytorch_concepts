.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Contributing a New Loss
=======================

This guide explains how to add a new loss term to |pyc_logo| PyC. Terms are plain
|pytorch_logo| ``nn.Module`` subclasses with no registration boilerplate: declare
the arguments you need in ``forward`` and return a scalar.

Read :doc:`Losses <using_loss>` first — it covers the term contract (which
arguments exist, how they are matched, where a term belongs). This page is only
about getting a term into the library.


.. dropdown:: Example: a custom entropy regularizer
   :icon: flame

   An entropy bonus on concept logits, pushing the model towards confident
   predictions. It declares ``padding_mask`` so it is also safe as a categorical
   term, where logits are padded with ``-inf``.

   .. code-block:: python

      # torch_concepts/nn/modules/loss.py  (next to L1LogitRegularizer)
      class EntropyRegularizer(nn.Module):
          """Penalise high-entropy predictions.

          Computes the binary entropy ``H(p) = -p log p - (1-p) log(1-p)`` of
          ``p = sigmoid(input)``, averaged over valid positions and scaled.

          Args:
              scale (float): Multiplicative factor. Default ``1.0``.
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
              entropy = F.binary_cross_entropy(p, p, reduction='none')
              mask = padding_mask if padding_mask is not None else torch.isfinite(input)
              if mask.any():
                  return self.scale * entropy[mask].mean()
              return torch.zeros((), device=input.device)

   Check it against a ``ModelOutput`` built by hand — predictions live in
   logit-space (one column per class), the target in concept-space (one column
   per concept), and both carry annotations so they can be aligned by name:

   .. code-block:: python

      import torch
      import torch_concepts as pyc
      from torch_concepts.nn import ConceptLoss
      from torch_concepts.nn.modules.loss import EntropyRegularizer  # before export
      from torch_concepts.nn.modules.outputs import ModelOutput
      from torch_concepts.tensor import AnnotatedTensor

      ann = pyc.Annotations(
          labels=["is_round", "color", "label"],
          cardinalities=[1, 3, 1],
          types=["binary", "categorical", "binary"],
      )
      loss_fn = ConceptLoss(
          binary=[torch.nn.BCEWithLogitsLoss(), EntropyRegularizer(scale=0.05)],
          binary_weights=[1.0, 0.5],
          categorical=torch.nn.CrossEntropyLoss(),
      )

      out = ModelOutput(
          logits=AnnotatedTensor(torch.randn(8, 5), ann),                 # 1 + 3 + 1
          target=AnnotatedTensor(torch.randint(0, 2, (8, 3)).float(),
                                 ann.to_concept_space()),
      )
      print(loss_fn(out))   # scalar tensor


.. dropdown:: Registering
   :icon: package

   1. **Module file** — add the class to ``torch_concepts/nn/modules/loss.py``,
      next to :class:`~torch_concepts.nn.L1LogitRegularizer` so similar terms
      stay together.

   2. **Public API** — add it to the import and to ``__all__`` in
      ``torch_concepts/nn/__init__.py``, so ``from torch_concepts.nn import
      EntropyRegularizer`` works.

   3. **API reference** — add the class name to the autosummary table in
      ``doc/modules/nn.loss.rst``, and a row to the built-in terms table in
      ``doc/guides/using_loss.rst`` if it is generally useful.

   4. **Tests** — add a case to ``tests/nn/modules/test_loss.py`` that builds a
      :class:`~torch_concepts.nn.ConceptLoss` with your term, runs a forward pass,
      and checks the result is a scalar with a gradient.


Next Steps
----------

- :doc:`Losses <using_loss>` — the term contract and how terms compose.
- Open a pull request to ``dev`` — see :doc:`Contributing <contributing>` for the
  full workflow.
