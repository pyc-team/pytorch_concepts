.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Losses
======

A loss in |pyc_logo| PyC takes the model's whole output, not a pair of tensors.
That is what lets it find each concept by name, score each concept *type* with
the right objective, and sum any number of extra terms — without you wiring
anything up by hand.

This page is the whole contract: how the output reaches a term, how to route by
type, how to add and weight extra terms, and what to do when you write a new
model.


How the output reaches a loss
-----------------------------

::

    model(query=..., evidence=...)
        │
        ▼
    ModelOutput
      ├── params        {'logits'|'probs': ..., 'loc'|'scale'|'value': ...}
      │                 one annotated tensor per quantity, all queried variables
      ├── guide_params  same layout, for a variational guide's latents
      ├── target        concept-space ground truth (annotated)
      └── extra         anything else a term needs  →  see `default_extra`
        │
        ▼
    CompositeLoss           each term sees the whole ModelOutput
      ├── ReconstructionLoss / KLDivergenceLoss / OrthogonalityLoss / ...
      ├── ConceptSubset     narrows the output to a named group of concepts
      └── ConceptLoss       routes by concept type:
              binary       ← params['logits'|'probs'].binary()
              categorical  ← params['logits'|'probs'].categorical()  (padded)
              continuous   ← params['loc'|'value']
                │           each paired with target[<the same concept names>]
                ▼
              _filter_kwargs  →  every term gets only the kwargs it declares

Two rules follow from the picture, and they are the only two worth memorising:

1. **Predictions and targets are matched by concept name**, through the
   :class:`~torch_concepts.Annotations` both carry. Nothing depends on column
   order, and a variable the target has no truth for (a reconstructed image, say)
   is skipped rather than looked up.
2. **A term declares what it wants.** ``ConceptLoss`` reads each term's
   ``forward`` signature once, at construction, and passes exactly the arguments
   it names — the same contract as ``torchmetrics.Metric._filter_kwargs``.


What a term may declare
-----------------------

A loss term is any |pytorch_logo| ``nn.Module`` returning a scalar. Declare any
subset of:

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Argument
     - What it is
   * - ``input``
     - Predictions for the current type. ``(batch, n_binary)`` for binary,
       ``(batch * n_concepts, max_cardinality)`` for categorical (concept-major
       rows), ``(batch, n_continuous)`` for continuous.
   * - ``target``
     - Ground truth for the same concepts, in the same order.
   * - ``padding_mask``
     - Categorical only: ``True`` at real class positions, ``False`` at the
       ``-inf`` padding added for concepts below the maximum cardinality.
       Declare it in any term that touches ``input`` without a target — it is
       only built when some term asks for it.
   * - ``scale``
     - Continuous only, when the model reports one.
   * - any key of ``extra``
     - Whatever the model published there, under that exact name.
   * - ``**kwargs``
     - Everything available.

.. code-block:: python

   class L2OnEmbeddings(torch.nn.Module):
       # `embeddings` is matched to out.extra['embeddings'] by name.
       def forward(self, embeddings):
           return embeddings.pow(2).mean()

Terms receive **plain tensors**, not ``AnnotatedTensor`` — by the time a term
runs, the annotation has already done its job of aligning to the target.

A term that needs the *whole* output rather than one type's slice — to read a
guide's latents, or the evidence — subclasses
:class:`~torch_concepts.nn.PyCLoss` instead and takes ``(output, target=None)``.
That is what :class:`~torch_concepts.nn.ReconstructionLoss` and
:class:`~torch_concepts.nn.KLDivergenceLoss` do; add it to a ``CompositeLoss``,
not to a per-type list.


Routing by concept type
-----------------------

Give each type its own objective. Types absent from your data need no entry.

.. code-block:: python

   from torch_concepts.nn import ConceptLoss

   loss_fn = ConceptLoss(
       binary=torch.nn.BCEWithLogitsLoss(),
       categorical=torch.nn.CrossEntropyLoss(),
       continuous=torch.nn.MSELoss(),
   )

Which quantity each type is read from is **inferred** from what the model
reports: the first of ``('logits', 'probs')`` for discrete types, of
``('loc', 'value')`` for continuous. Override it with ``binary_param``,
``categorical_param`` or ``continuous_param`` only when a model reports several
and you want a specific one.

Pass ``annotations=`` to have the configuration checked at construction instead
of at the first training step:

.. code-block:: python

   loss_fn = ConceptLoss(binary=torch.nn.BCEWithLogitsLoss(),
                         annotations=annotations)   # errors now, not at step 1


Stacking terms, and weighting them
----------------------------------

There are two levels, and both take a list of terms with a list of weights:

.. code-block:: python

   from torch_concepts.nn import CompositeLoss, ConceptLoss, L1LogitRegularizer

   concept_term = ConceptLoss(
       binary=[torch.nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.01)],
       binary_weights=[1.0, 0.5],          # per type, on that type's slice
       categorical=torch.nn.CrossEntropyLoss(),
   )

   loss_fn = CompositeLoss(
       terms=[reconstruction, kl, concept_term, orthogonality],
       weights=[1.0, 1.0, 5.0, 1.0],       # shared, on the whole output
   )

::

    total =  Σ_types Σ_i  w_i · term_i(that type's slice)     # ConceptLoss
          +  Σ_j          w_j · term_j(whole output)          # CompositeLoss

**Which level does a term belong to?** Look at what it reads:

- reads a **type's** predictions (a penalty on binary logits) → per-type list;
- reads something **shared** (embeddings, a latent, the evidence) → a
  ``CompositeLoss`` term.

A shared penalty put in a per-type list is charged once per type it is listed
in, on that type's slice — rarely what anyone means. Computing it once outside
the routing keeps it one value, with one weight and one number to read.

**Weighting a group of concepts** — say concepts against tasks, or shallow
against deep — is the same mechanism: wrap a loss in a
:class:`~torch_concepts.nn.ConceptSubset`, name the group, give it a weight.

.. code-block:: python

   from torch_concepts.nn import ConceptSubset

   loss_fn = CompositeLoss(
       terms=[ConceptSubset(ConceptLoss(binary=BCEWithLogitsLoss()), exclude=['cancer']),
              ConceptSubset(ConceptLoss(binary=BCEWithLogitsLoss()), names=['cancer'])],
       weights=[0.5, 1.0],
       names=['concepts', 'tasks'],       # what breakdown() and repr will show
   )

:class:`~torch_concepts.nn.WeightedConceptLoss` and
:class:`~torch_concepts.nn.DepthWeightedConceptLoss` are exactly this, prebuilt.

Weights are a plain mutable list, so a schedule can rewrite one entry during
training — that is how :class:`~torch_concepts.nn.LossWeightWarmup` ramps a KL
term up over the first epochs.


Reading and debugging the objective
-----------------------------------

:meth:`~torch_concepts.nn.CompositeLoss.breakdown` returns each term's weighted
contribution. The values sum to what ``forward`` returns, so use it whenever a
total is not enough — an ELBO whose KL has collapsed still looks fine summed:

.. code-block:: python

   for name, value in loss_fn.breakdown(out, c).items():
       print(f"{name:24s} {value.item():.4f}")
   # ReconstructionLoss       412.8317
   # KLDivergenceLoss          18.4402
   # ConceptLoss                0.9137
   # OrthogonalityLoss          0.0521

Two errors you may meet, and what they mean:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Error
     - Cause
   * - ``ConceptLoss has terms for [...] but scored nothing``
     - The output carries no quantity for any configured type. Check the model's
       ``param_for_discrete_var``, and that the target covers those concepts.
   * - ``TypeError: forward() missing ... 'embeddings'``
     - A term declares a name that nothing published. Add it to ``extra`` (see
       below) — the key and the argument name must match.


Writing a new model
-------------------

Everything a loss needs is already on the output, with one hook for the
exceptions. In most cases there is nothing to do:

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Your model…
     - What you do
   * - reports ``logits`` (``param_for_discrete_var = "logits"``)
     - nothing — inferred
   * - reports ``probs``
     - nothing — inferred; pick a term that expects probabilities, e.g.
       :class:`~torch_concepts.nn.NLLProbLoss`
   * - models continuous concepts as a ``Delta`` (``value``) or a ``Normal``
       (``loc``)
     - nothing — inferred
   * - needs a tensor no quantity carries (embeddings, the evidence)
     - override ``default_extra``; the dict key **is** the term's argument name
   * - has terms beyond concept supervision (an ELBO, a regulariser)
     - wrap them in a ``CompositeLoss``

.. code-block:: python

   class MyGenerativeModel(...):
       def default_extra(self, evidence):
           # ReconstructionLoss reads out.extra['evidence'][variable].
           return {"evidence": evidence}

The learner merges this into ``out.extra`` on every step. Outside |pytorch_logo|
Lightning, set ``out.extra`` yourself after the forward pass.


Built-in terms
--------------

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Term
     - Use
   * - :class:`~torch_concepts.nn.ConceptLoss`
     - Concept supervision, routed by type. The usual starting point.
   * - :class:`~torch_concepts.nn.CompositeLoss`
     - Weighted sum of any terms. The building block for an ELBO.
   * - :class:`~torch_concepts.nn.ConceptSubset`
     - Applies a loss to a named group of concepts, so the group can carry its
       own weight in a ``CompositeLoss``.
   * - :class:`~torch_concepts.nn.WeightedConceptLoss`
     - Concepts and tasks weighted separately. Two ``ConceptSubset`` groups.
   * - :class:`~torch_concepts.nn.DepthWeightedConceptLoss`
     - One ``ConceptSubset`` group per depth level of a
       :class:`~torch_concepts.ConceptGraph`, weighted by ``depth_decay ** d``.
   * - :class:`~torch_concepts.nn.ReconstructionLoss`
     - Negative log-likelihood of an observed variable under its own CPD.
   * - :class:`~torch_concepts.nn.KLDivergenceLoss`
     - ``KL(q ‖ p)`` per latent, with optional ``free_bits``.
   * - :class:`~torch_concepts.nn.OrthogonalityLoss`
     - Pushes concept contexts away from the unsupervised one.
   * - :class:`~torch_concepts.nn.NLLProbLoss`
     - Categorical NLL for a head that emits ``probs`` rather than logits.
   * - :class:`~torch_concepts.nn.L1LogitRegularizer`
     - L1 penalty on logit magnitude, padding-aware.


Next steps
----------

- :doc:`Contributing a New Loss <contributing_loss>` — adding a term to the library.
- :doc:`Out-of-the-box Models <using_high_level>` — training with a loss attached.
- ``examples/utilization/2_model/`` — runnable versions of everything above
  (``7`` per-type, ``13`` stacking and weights, ``14`` kwarg routing, ``15`` an ELBO).
