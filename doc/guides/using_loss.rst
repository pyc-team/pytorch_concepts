.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

.. |pl_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/lightning.svg
    :width: 20px
    :align: middle


Losses
======

A loss in |pyc_logo| PyC is scored on the model's **whole output**, not on a pair
of tensors. That one change is what lets it find each concept by name, give each
concept *type* the objective it deserves, and sum any number of extra terms —
without you wiring anything up by hand.

A loss is built from four pieces:

- **ModelOutput** — the single object every term is scored on.
- :class:`~torch_concepts.nn.ConceptLoss` — Loss for concept supervision.
  Permit to specify one objective per concept **type** (binary, categorical, continuous).
- :class:`~torch_concepts.nn.ConceptSubset` — restricts a loss to a **named
  group** of concepts, so the group can carry its own weight (e.g., a task loss).
- :class:`~torch_concepts.nn.CompositeLoss` — a **weighted sum** of terms that
  each read the whole output (an ELBO, a shared regulariser).

Expand each block below for an explanation and an example.


.. dropdown:: What a loss receives
    :icon: package

    A forward pass returns a :class:`~torch_concepts.nn.ModelOutput`. Every loss
    term — built-in or your own — reads what it needs from it:

    ::

        model(query=..., evidence=...)  →  ModelOutput
          ├── params         predicted distribution parameters {'logits'|'probs': ..., ...}; sliced by quantity or by variable name
          ├── guide_params   same, for a variational guide's latents
          ├── samples        per-variable realisations; sliced by variable name
          ├── probabilities  P(query | evidence); one value per query, not sliceable
          ├── target         concept-space ground truth; sliced by variable name
          └── extra          anything else a term needs; sliced by key   ←  ``default_extra``

    The first four fields are filled by the inference engine. ``extra`` is where the model
    publishes everything else a term might need — the raw input, an intermediate embedding,
    a per-sample weight — by overriding ``default_extra``.

    Each loss term can then read whichever fields of the ModelOutput it needs.


.. dropdown:: ConceptLoss
    :icon: flame

    :class:`~torch_concepts.nn.ConceptLoss` is the loss for concept
    supervision: it scores each queried concept against its target, using the
    objective configured for that concept's **type**.

    .. code-block:: python

       import torch
       from torch_concepts.nn import ConceptLoss

       loss_fn = ConceptLoss(
           binary=torch.nn.BCEWithLogitsLoss(),
           categorical=torch.nn.CrossEntropyLoss(),
           continuous=torch.nn.MSELoss(),
       )

    **Which loss each type is routed to is inferred automatically** from the parameters name 
    in the model's output:``('logits', 'probs')`` for discrete types, ``('loc', 'value')`` 
    for continuous. Override it with ``binary_param``, ``categorical_param`` or ``continuous_param``
    only when a model reports several and you want a specific one — for instance a head that emits
    probabilities:

    .. code-block:: python

       from torch_concepts.nn import NLLProbLoss

       loss_fn = ConceptLoss(categorical=NLLProbLoss(), categorical_param='probs')

    **Stacking several terms on one type.** Each type may take a list instead
    of a single module, summed with per-term weights — here a regularizer
    added to the binary concepts only:

    .. code-block:: python

       from torch_concepts.nn import ConceptLoss, L1LogitRegularizer

       loss_fn = ConceptLoss(
           binary=[torch.nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=0.05)],
           binary_weights=[1.0, 0.5],           # L1 on the binary logits only
           categorical=torch.nn.CrossEntropyLoss(),
       )
       # ConceptLoss(binary=[BCEWithLogitsLoss + 0.5*L1LogitRegularizer],
       #             categorical=CrossEntropyLoss)

    **A term declares what it wants.** ``ConceptLoss`` reads each term's
    ``forward`` signature once, at construction, and passes exactly the
    arguments it names — so two terms on the same type can want different
    things from the very same call:

    .. code-block:: python

       class PenalizeLarge(torch.nn.Module):
           def forward(self, input):          # no `target` declared
               return input.abs().mean()

       loss_fn = ConceptLoss(
           binary=[torch.nn.BCEWithLogitsLoss(), PenalizeLarge()],
       )
       # BCEWithLogitsLoss receives (input, target); PenalizeLarge receives
       # only (input) — both come from the same ConceptLoss.forward() call.


.. dropdown:: ConceptSubset
    :icon: filter

    :class:`~torch_concepts.nn.ConceptSubset` restricts a loss to a **named
    group** of concepts. ``ConceptLoss`` only ever routes by *type*; weighting
    concepts differently from tasks, or shallow concepts from deep ones, needs
    them picked out by *name* instead — that is what this wraps around a loss.

    Exactly one of ``names`` / ``exclude`` is given:

    .. code-block:: python

       from torch_concepts.nn import ConceptLoss, ConceptSubset

       tasks = ConceptSubset(
           ConceptLoss(binary=torch.nn.BCEWithLogitsLoss()),
           names=['PropCost'],
       )


.. dropdown:: CompositeLoss
    :icon: stack

    :class:`~torch_concepts.nn.CompositeLoss` is a **weighted sum** of terms
    that each see the **whole output** — the building block for an objective
    that is not a single concept loss, such as an ELBO or concepts and tasks
    weighted apart:

    .. math::

        \text{total} = \sum_j w_j \cdot \text{term}_j(\text{whole output})

    **Combining independent terms.** A term that does not belong to any single
    concept type — a shared penalty, an ELBO term — is a
    :class:`~torch_concepts.nn.PyCLoss`, summed here rather than folded into a
    per-type list:

    .. code-block:: python

       from torch_concepts.nn import CompositeLoss, ConceptLoss, PyCLoss

       class GlobalLogitL1(PyCLoss):
           """L1 over *every* reported logit at once."""
           def forward(self, output, target=None):
               return 0.01 * output.logits.tensor.abs().mean()

       loss_fn = CompositeLoss(
           terms=[ConceptLoss(binary=torch.nn.BCEWithLogitsLoss(),
                              categorical=torch.nn.CrossEntropyLoss()),
                  GlobalLogitL1()],
           weights=[1.0, 1.0],
           names=['supervision', 'global_l1'],   # what breakdown() and repr show
       )
       # CompositeLoss(supervision + global_l1)

    **Combining two ``ConceptSubset`` groups.** Concepts and tasks, weighted
    apart:

    .. code-block:: python

       from torch_concepts.nn import ConceptSubset

       supervision = dict(binary=torch.nn.BCEWithLogitsLoss(),
                          categorical=torch.nn.CrossEntropyLoss())

       loss_fn = CompositeLoss(
           terms=[ConceptSubset(ConceptLoss(**supervision), exclude=['PropCost']),
                  ConceptSubset(ConceptLoss(**supervision), names=['PropCost'])],
           weights=[0.5, 1.0],
           names=['concepts', 'task'],
       )
       # CompositeLoss(0.5*concepts + task)

    :class:`~torch_concepts.nn.WeightedConceptLoss` (concepts vs. tasks) and
    :class:`~torch_concepts.nn.DepthWeightedConceptLoss` (one group per depth
    level of a :class:`~torch_concepts.ConceptGraph`) build exactly this for
    you.

    **What nests in what.** ``CompositeLoss`` and ``ConceptSubset`` are both a
    :class:`~torch_concepts.nn.PyCLoss`, so they compose freely:

    - a ``CompositeLoss`` term may itself be a ``CompositeLoss``, a
      ``ConceptSubset`` or a ``ConceptLoss``;
    - a ``ConceptSubset`` may wrap a ``ConceptLoss`` **or** a ``CompositeLoss``;
    - a per-type list (inside ``ConceptLoss``) holds plain |pytorch_logo|
      ``nn.Module`` terms only — never a ``PyCLoss``.

    Two conveniences worth knowing:

    - a ``None`` entry in ``terms`` is dropped together with its weight and name,
      so a term can be switched off in place
      (``OrthogonalityLoss(...) if use_unknown else None``);
    - ``weights`` is a plain mutable list, so a schedule can rewrite one entry
      during training — that is how
      :class:`~torch_concepts.nn.LossWeightWarmup` ramps a KL term up over the
      first epochs.

    .. note::

       A shared penalty put in a per-type list instead is charged **once per
       type it is listed in**, on that type's slice — rarely what anyone
       means. Computing it once here keeps it one value, with one weight and
       one number to read.


.. dropdown:: ``default_extra`` — publishing anything else a loss term needs
    :icon: plug

    ``params``, ``guide_params`` and ``target`` cover what the PGM computes.
    Everything else a loss might want — the raw evidence, an intermediate
    embedding, a mask, a per-sample weight — goes through one model hook:

    .. code-block:: python

       class MyModel(ConceptBottleneckModel):
           def default_extra(self, evidence, query=None):
               return {"evidence": evidence}      # or None for nothing

    The returned dict lands **verbatim** on ``out.extra`` on every forward pass —
    it is the model's ``forward`` that calls it, so this works identically in a
    manual |pytorch_logo| PyTorch loop and under |pl_logo| Lightning. It is
    called with the ``evidence`` dict and the ``query`` of that pass, so the
    extras can depend on both.

    There are two ways a term reads it, and which one applies depends on where
    the term sits:

    .. list-table::
       :widths: 34 66
       :header-rows: 1

       * - Term
         - How it gets the extra
       * - inside a ``ConceptLoss`` per-type list
         - ``extra`` is spread as keyword arguments, so **the dict key is the
           argument name**. Declare it in ``forward`` and it arrives.
       * - a ``PyCLoss`` in a ``CompositeLoss``
         - reads ``output.extra[...]`` itself — e.g.
           :class:`~torch_concepts.nn.MSEReconstructionLoss` looks up
           ``output.extra['evidence'][variable]``.

    **Example for case 1 — access extra in a ConceptLoss term.** The key
    ``'embeddings'`` and the argument ``embeddings`` are the same name; that is
    the whole coupling:

    .. code-block:: python

       from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss

       class CBMWithEmbeddings(ConceptBottleneckModel):
           def default_extra(self, evidence, query=None):
               # Runs on every forward pass, so keep it cheap — or cache it.
               return {'embeddings': self.backbone(evidence['input'])}

       class EmbeddingReg(torch.nn.Module):
           # `input` is the binary slice; `embeddings` comes from extra, by name.
           def forward(self, input, embeddings):
               return 0.01 * embeddings.pow(2).mean()

       loss_fn = ConceptLoss(
           binary=[torch.nn.BCEWithLogitsLoss(), EmbeddingReg()],
           binary_weights=[1.0, 0.5],
       )

    **Example for case 2 — access extra in any other loss term.** A
    reconstruction term reads the evidence straight off the output — no
    per-type routing, and no ``CompositeLoss`` needed to show the mechanism:

    .. code-block:: python

       from torch_concepts.nn import ConceptBottleneckModel, PyCLoss

       class CBMWithEvidence(ConceptBottleneckModel):
           def default_extra(self, evidence, query=None):
               return {"evidence": evidence}

       class ReconstructionTerm(PyCLoss):
           # Reads `output.extra['evidence']` itself — no per-type routing.
           def forward(self, output, target=None):
               observed = output.extra['evidence']['input']
               predicted = output.params['value']['input']
               return (predicted - observed).pow(2).mean()

    Two things to keep in mind:

    - Extras are offered to **every** type's terms, so a term that declares
      ``embeddings`` receives it whether it is scoring binary or categorical
      concepts.
    - Avoid the reserved names ``input``, ``target``, ``scale`` and
      ``padding_mask``: extras are merged last and would shadow them.


.. dropdown:: Reading and debugging the objective
    :icon: bug

    For a :class:`~torch_concepts.nn.CompositeLoss`,
    :meth:`~torch_concepts.nn.CompositeLoss.breakdown` returns each term's
    **weighted** contribution. The values sum to exactly what ``forward``
    returns, so use it whenever a total is not enough — an ELBO whose KL has
    collapsed still looks fine summed:

    .. code-block:: python

       for name, value in loss_fn.breakdown(out, c).items():
           print(f"{name:24s} {value.item():.4f}")
       # MSEReconstructionLoss    412.8317
       # KLDivergenceLoss          18.4402
       # ConceptLoss                0.9137
       # OrthogonalityLoss          0.0521

    Under |pl_logo| Lightning this is automatic: a ``CompositeLoss`` logs every
    term separately as ``{split}_{term_name}`` (``train_kl``, ``val_recon``, …)
    alongside the total, which is why passing ``names=`` is worth the keystrokes.

    Two errors you may meet, and what they mean:

    .. list-table::
       :widths: 38 62
       :header-rows: 1

       * - Error
         - Cause
       * - ``ConceptLoss has terms for [...] but scored nothing``
         - The output carries no quantity for any configured type. Check the
           model's ``param_for_discrete_var``, and that the target covers those
           concepts.
       * - ``TypeError: forward() missing ... 'embeddings'``
         - A term declares a name that nothing published. Return it from
           ``default_extra`` — the key and the argument name must match.


.. dropdown:: Putting it together: an ELBO
    :icon: rocket

    A concept bottleneck VAE's objective is four independent terms, each reading
    a different part of the same output — reconstruction from ``extra['evidence']``,
    the KL from ``guide_params``, supervision from the concept slice, and an
    orthogonality penalty from two ``Delta`` variables:

    .. code-block:: python

       import torch.nn as nn
       from pytorch_lightning import Trainer
       from torch_concepts.nn import (
           CompositeLoss, ConceptLoss, KLDivergenceLoss,
           LossWeightWarmup, MSEReconstructionLoss, OrthogonalityLoss,
       )

       loss = CompositeLoss(
           terms=[
               MSEReconstructionLoss(variable='input'),
               KLDivergenceLoss(latents=['z']),
               ConceptLoss(categorical=nn.CrossEntropyLoss()),
               OrthogonalityLoss('mixing', 'unknown', len(concept_names)),
           ],
           weights=[1.0, 1.0, 5.0, 1.0],
           names=['recon', 'kl', 'concepts', 'orth'],
       )
       # CompositeLoss(recon + kl + 5.0*concepts + orth)

       trainer = Trainer(max_epochs=100)
       trainer.fit(model, datamodule=datamodule)

    The model supplies the missing piece by publishing its evidence:

    .. code-block:: python

       def default_extra(self, evidence, query=None):
           return {"evidence": evidence}

    Each term is logged separately as ``train_recon``, ``train_kl``, … so a
    collapsing KL is visible from the first epochs.


Next Steps
----------

- Browse the loss classes in the :doc:`API reference </modules/nn.loss>`.
- :doc:`Contributing a New Loss <contributing_loss>` — adding a term to the library.
- :doc:`Out-of-the-box Models <using_high_level>` — training with a loss attached.
- Check out the `example scripts <https://github.com/pyc-team/pytorch_concepts/tree/master/examples/utilization/2_model>`_:
  ``7`` per-type routing, ``13`` composition and weights, ``14`` kwarg routing
  and ``default_extra``, ``15`` a full ELBO.
