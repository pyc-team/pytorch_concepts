Label-Free Concept Bottleneck Models
===================================

:class:`torch_concepts.nn.LabelFreeCBM` is a high-level model implementing
`Oikarinen et al., ICLR 2023 <https://arxiv.org/abs/2304.06129>`_. A generation
pipeline produces definitions and raw scores; the model fits the projection
and sparse classifier, then builds an ordinary ``ConceptBottleneckModel`` as its
prediction graph.
No manually annotated bottleneck concepts are required. Task labels and their
class order are still needed. Refinement removes concepts after projection
training and never queries an LLM again.

Usage
-----

Install ``pytorch_concepts[data,lfcbm]`` for CLIP annotation and the published
``glm-saga==0.1.2`` sparse solver. The solver is imported only when fitting;
loading and prediction do not require it. See the complete example
:doc:`/auto_examples/high_level/11_label_free_cbm`.

.. code-block:: python

   from torch_concepts.data.generation import ConceptGenerationPipeline
   from torch_concepts.data.generation.generators import FixedConceptGenerator
   from torch_concepts.data.generation.annotators import CLIPAnnotator
   from torch_concepts.nn import LabelFreeCBM

   pipeline = ConceptGenerationPipeline(
       generators=FixedConceptGenerator(
           labels=["striped", "fur", "wheels", "wings"], cardinalities=[1] * 4,
       ),
       annotators=CLIPAnnotator(
           model_name="openai/clip-vit-base-patch16", prompt_template="{}",
       ),
   )
   # dm contains raw tensors and a native categorical task named "class".
   # backbone maps batches of those inputs to feature vectors.
   lfcbm = LabelFreeCBM(pipeline, backbone, "class", class_names)
   lfcbm.fit(dm)
   output = lfcbm(query=["class"], input=x)
   logits = output.logits["class"].tensor
   learned = lfcbm.predict_concepts(x)
   print(learned[lfcbm.concept_names_[:2]])
   model = lfcbm.model_  # ordinary ConceptBottleneckModel
   lfcbm.save("fitted_lfcbm")
   restored = LabelFreeCBM.load("fitted_lfcbm", backbone=backbone)

The pipeline also accepts existing ``LLMConceptGenerator`` instances. Concept
identities and their number are discovered during ``fit``, not at model
construction. Fixed vocabularies need no LLM. The pipeline remains independently
usable, and the fitted prediction graph contains no generation components.

``LabelFreeCBM`` is a PyTorch ``nn.Module``. After fitting, its
``forward(query=..., evidence=..., input=...)`` returns the same
``InferenceOutput`` as other high-level CBMs. The internal ``model_`` remains
independently usable. Module operations such as ``to``, ``eval``, ``parameters``,
and ``state_dict`` include the fitted predictor; normalization statistics are
registered buffers. The backbone stays in evaluation mode even after
``train()`` because LF-CBM freezes it. Training uses ``fit`` for the specialized
Adam/SAGA stages; it does not use the generic Lightning training loop.

Data contracts and splits
------------------------

``fit(datamodule)`` respects its train/validation sets, calling ``setup('fit')``
only when necessary. ``testset`` is never read. Also supported are
``fit(train_dataset, val_dataset)`` and
``fit(dataset, train_indices=..., val_indices=...)``. Explicit train/validation
indices must be disjoint. Each split needs at least two observations.

Samples use PyC's ``sample['inputs']['x']`` tensor format. Task labels default
to the named column of ``sample['concepts']['native']``. Categorical task states
must match ``class_names`` in order. Use ``target_getter(sample)`` with a custom
label source, including datasets with no native annotations. Labels must be
integer indices into ``class_names``. One multiclass task is supported, including
a two-class categorical task.

The pipeline receives only the training dataset for discovery, its local
``generation_indices``, and ``annotation_datasets={'train': ..., 'val': ...}``.
Index arguments to ``fit`` become these separate dataset views. Split-specific
scores are consumed directly; they are not attached to the full dataset. Neither
native nor generated dataset supervision is mutated.

Optional ``embeddings={'train': train_h, 'val': val_h}`` reuses precomputed
features in each split's sample order. They must come from the supplied backbone.
Keep raw inputs for annotation: datasets whose inputs were replaced by
``precompute_embeddings`` are rejected. The fitted predictor retains a frozen
copy of the backbone and accepts new raw inputs independently of training data.

Algorithm and defaults
----------------------

The comparison reference is the `authors' implementation at fa2869d
<https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/fa2869d46c454f406af69b19d3d5427f2d427492>`_,
including train_cbm.py, similarity.py, cbm.py, conceptset_utils.py, and glm_saga.

1. ``CLIPAnnotator`` supplies raw dot products of normalized image/text
   embeddings. Use ``prompt_template="{}"``. Calibrators, annotation filters,
   and aggregation are rejected. Unlike ``data/05_label_free_concepts.py``, this
   method never trains on sigmoid-calibrated pseudo-probabilities. Candidates
   must have unique binary definitions, one finite similarity column each.
2. Training-only mean top-five activations must be strictly greater than
   ``clip_cutoff=.25``. For fewer than five training observations, all are used.
3. A bias-free ``LinearEmbeddingToConcept`` learns a projection of frozen
   backbone features using Adam: lr .001, up to 1000 steps, random batches of
   at most 50000. The objective is negative mean columnwise cosine similarity
   after centering activation patterns and cubing them elementwise. This is
   not the cube of a cosine. Explicit standard-deviation division cancels in
   the final cosine normalization, so the source centers but does not divide.
4. Validation occurs at step zero, every 50 updates, and the final step.
   Training stops at the first non-improvement and restores the best weights.
   Validation alignment must be strictly greater than .45 to retain a concept.
   Names, score columns, and projection rows are selected together. Empty
   vocabularies raise a clear error.
5. Projected training activations supply mean and sample standard deviation
   (``correction=1``), used for both training and validation normalization.
   A retained concept with zero training standard deviation raises an error.
6. A zero-initialized multinomial linear classifier is trained with indexed
   minibatch proximal SAGA. The objective is mean cross-entropy plus
   ``penalty * (alpha * ||W||_1 + (1-alpha) * ||W||_F**2 / 2)``. The intercept
   is not penalized. The joint soft-threshold/shrinkage proximal operator and
   per-observation gradient memory match the reference. Neither projection nor
   backbone is optimized in this stage.

``classifier_lambda=.0007`` follows the authors' CLI parameterization: their
single-point GLM path divides it by ``max(.001, alpha)``. At ``alpha=.99`` the
effective penalty is approximately .0007070707, exposed as
``classifier_regularization_``. SAGA defaults are lr .1, batch 256, 1000 epochs,
and parameter-change tolerance 1e-4. Validation classifier loss is diagnostic;
the single-penalty call returns that path point rather than searching penalties.
Full regularization paths, grouped sparsity, Gaussian GLMs, and sample weighting
are outside this method's scope.

As a numerical stability extension, a zero-norm centered activation pattern
has similarity zero instead of NaN. Nondegenerate inputs agree numerically with
the reference, including gradients. Sampling and initialization use a seed;
identical benchmark trajectories also require matching random state, data
order, precision, device, and pretrained checkpoints.

Classifier fitting uses the MIT-licensed
`MadryLab glm_saga package <https://github.com/MadryLab/glm_saga>`_, credited by
the LF-CBM authors. A small adapter in ``lfcbm.py`` creates a seeded indexed
loader, zeros the classifier, and calls ``train_saga`` at the effective penalty
above. It avoids the published package's regularization-path construction,
which takes ``log10(lambda)`` and cannot represent a zero penalty. No SAGA
updates or upstream few-shot prompt text are reimplemented or vendored.

Predictor and persistence
------------------------

The final CBM models concepts as continuous deterministic ``Delta`` variables
and the task as categorical. It computes:

.. code-block:: text

   h = backbone(x)
   c = W_c h
   z = (c - train_mean) / train_std
   logits = W_F z + b_F

Normalization is folded into the concept encoder as ``W_c / std[:, None]``
and ``-mean / std``. No sigmoid, threshold, or Bernoulli interpretation is
introduced. ``predict`` returns logits in class order; use ``argmax(1)`` for
class predictions. ``predict_concepts`` returns ``z`` as an ``AnnotatedTensor``,
matching the authors' inference output. ``projection_`` retains the unfused
bias-free projection. Calling inference before fitting raises an error.

``LabelFreeCBM`` constructs a standard ``ConceptBottleneckModel`` and copies the
learned weights into its existing concept and task heads through
``model_.pgm.factors``. Normalization is folded into the concept head during this
copy. No changes to the CBM class or its constructor are required.

``concept_annotations_`` is the retained continuous axis; ``concept_names_``
returns its labels. ``generated_annotations_`` holds retained raw train/val
scores. ``activation_scores_`` covers candidates and ``alignment_scores_``
covers concepts surviving activation filtering. Both are explicitly annotated
and addressable by name. ``projection_history_``, ``projection_best_step_``, and
``classifier_diagnostics_`` record projection stopping decisions, solver settings,
and train/validation classifier losses. The external solver does not report
classifier iteration counts or convergence status.

``save(directory)`` writes JSON metadata and a weights-only state dictionary:
backbone, projection, classifier, normalization, class order, concept definitions,
and input/feature shapes. It never serializes a pipeline, LLM, CLIP annotator,
dataset, or arbitrary Python object. Diagnostics are available after fitting
but are not restored for inference.

Identity backbones reconstruct automatically. Otherwise,
``load(directory, backbone=compatible_module, device=...)`` requires the same
architecture and input preprocessing. Its weights are strictly restored from
the checkpoint. No model is downloaded or imported from metadata. The restored
``model_``, predictions, names, and activations work without rediscovery.

Original concepts and reproduction
---------------------------------

Use the published, already-filtered vocabulary with ``FixedConceptGenerator``
to avoid LLM dependencies and generation drift. Custom prompts can be supplied
directly to the existing ``LLMConceptGenerator``. Use the published vocabulary
for exact concept-string parity.

``SemanticConceptFilter`` reproduces the 30-character bound, exact class aliases,
class similarity cutoff .85, and concept similarity cutoff .9. It sorts labels
and combines MPNet/CLIP text similarities as ``(MPNet + 3 * CLIP) / 4``. Pairwise
pruning keeps the concept with greater summed similarity, not simply the first
synonym. The class filter's reference ``class_index != concept_index`` condition
is explicitly preserved despite comparing unrelated axes. Use all-mpnet-base-v2
(optional sentence-transformers package) and CLIP ViT-B/16 encoders for parity.

``examples/high_level/lfcbm_cifar10.json`` records the published CIFAR-10 settings:
the authors' vocabulary, CLIP RN50 backbone, ViT-B/16 annotation, and optimization
defaults. OpenAI CLIP is optional for that RN50 backbone. Keep its preprocessing
inside a reconstructible backbone module, separate from annotation preprocessing.
Hugging Face ViT-B/16 is a compatible converted checkpoint; exact score parity
requires the same weights, tokenizer, resize/crop conventions, and precision.
An OpenAI CLIP-backed existing ``Annotator`` can be supplied for strict parity.

The authors use CIFAR-10's official test partition as ``val`` for projection
selection. The new example uses training observations for validation and holds
official test observations out for evaluation. Choose deliberately when comparing
published numbers. Test data supplied only as ``testset`` is never consumed by
the model. No full image benchmark reproduction was run here.

Verification
------------

Normal tests use deterministic task-only datasets and lightweight fake annotators,
covering dynamic/fixed generation, selection, ordering, middle-column removal,
best-weight restoration, frozen components, training-only normalization, SAGA,
model equivalence, and persistence without network access. Optional comparisons
against a local authors' checkout are run explicitly:

.. code-block:: bash

   LFCBM_REFERENCE=/path/to/Label-free-CBM python -m pytest -o addopts='' tests/nn/modules/high/models/test_lfcbm_reference.py

They compare cosine-cubed values/gradients, GLM-SAGA weights/bias/objective at
several alpha values, and concept-filtering decisions using fixed text embeddings.
No data, remote models, or source code are downloaded by these tests.
