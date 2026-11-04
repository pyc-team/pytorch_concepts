Label-Free Concept Bottleneck Models
===================================

:class:`torch_concepts.nn.LabelFreeCBM` is a high-level model implementing
`Oikarinen et al., ICLR 2023 <https://arxiv.org/abs/2304.06129>`_. A generation
pipeline produces definitions and raw scores; the model fits the projection
and sparse classifier and constructs its own bottleneck graph.
``LabelFreeCBM`` subclasses ``ConceptBottleneckModel``.
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
   lfcbm = LabelFreeCBM(pipeline, backbone)
   dm.setup("fit")
   lfcbm.fit(dm, task_name="class")
   output = lfcbm(query=["class"], input=x)
   logits = output.logits["class"].tensor
   learned = lfcbm.predict_concepts(x)
   print(learned[lfcbm.concept_names_[:2]])
   model = lfcbm  # itself a ConceptBottleneckModel
   lfcbm.save("fitted_lfcbm")
   restored = LabelFreeCBM.load("fitted_lfcbm", backbone=backbone)

The pipeline also accepts existing ``LLMConceptGenerator`` instances. Construction
requires only the pipeline and backbone. Concept identities and their number
are discovered during ``fit``. The caller selects ``task_name`` at fitting time;
ordered class names are inferred from that task's native annotation. Fixed
vocabularies need no LLM. The pipeline remains independently
usable, and the fitted prediction graph contains no generation components.

``LabelFreeCBM`` inherits the high-level CBM interface. Its constructor accepts
and stores the generation pipeline; ``fit`` calls it internally using training
data, then constructs the bottleneck with the discovered vocabulary. Before
fitting there is no prediction graph, and inference raises a clear error.

After fitting, ``forward(query=..., evidence=..., input=...)`` runs the inherited
CBM inference and returns ``InferenceOutput``. The graph, backbone, and inference
engines belong directly to the instance: use ``lfcbm.pgm``, ``lfcbm.backbone``,
and ``lfcbm.concept_annotations``. There is no nested ``model_``. Module operations
such as ``to``, ``eval``, ``parameters``, and ``state_dict`` work directly on the
CBM. Normalization statistics remain registered buffers. ``projection_weight_``
stores the unfused diagnostic matrix, with ``mean_`` and ``std_`` storing training
normalization. The backbone stays in evaluation mode when ``train()`` is called.
Training uses ``fit`` for the Adam/SAGA stages.

Training settings are grouped by stage, following the existing high-level
API's ``inference_kwargs`` convention. Ordinary usage needs no optimizer options:

.. code-block:: python

   lfcbm = LabelFreeCBM(pipeline, backbone)
   dm.setup("fit")
   lfcbm.fit(dm, task_name="class")

   # Optional stage-specific overrides:
   lfcbm = LabelFreeCBM(
       pipeline, backbone,
       projection_kwargs={"steps": 2000, "lr": .001},
       classifier_kwargs={"epochs": 1000, "lam": .0007},
   )

Projection options are ``steps``, ``batch_size``, ``lr``, and
``validation_interval``. Classifier options are ``epochs``, ``batch_size``,
``lr``, ``alpha``, ``lam``, and ``tolerance``. Omitted values use the paper defaults
listed below. Unknown settings fail when their training stage is called.

Reusable building blocks
------------------------

``LabelFreeCBM`` coordinates existing generation and inference infrastructure
with independently usable training layers:

* ``ImageBackbone("RN50", source="openai_clip")`` supplies frozen image features
  using OpenAI CLIP's original preprocessing. No custom loader is needed.
* ``CosineCubedLoss`` and ``nn.functional.cosine_cubed_similarity`` provide the
  projection objective and per-concept alignment scores. The loss also works
  with ``ConceptLoss(continuous=CosineCubedLoss())``.
* ``AlignedEmbeddingToConcept`` extends ``LinearEmbeddingToConcept`` with Adam
  fitting, validation stopping, and best-weight restoration.
* ``StandardScaler`` supplies training-only normalization, replacing zero or
  near-zero standard deviations with one using its existing behavior.
* ``SparseLinearConceptToConcept`` extends ``LinearConceptToConcept`` with the
  optional published SAGA solver and an unpenalized intercept.
* ``ConceptBottleneckModel.set_linear_heads`` installs the fitted heads for
  standard CBM inference.

For example, given frozen embeddings ``train_h`` / ``val_h``, raw similarity
matrices ``train_scores`` / ``val_scores`` surviving the activation cutoff,
and integer task labels, the training layers can be used directly:

.. code-block:: python

   from torch_concepts.data.scalers import StandardScaler
   from torch_concepts.nn import AlignedEmbeddingToConcept, SparseLinearConceptToConcept

   projection = AlignedEmbeddingToConcept(train_h.shape[1], train_scores.shape[1])
   projection.fit(train_h, train_scores, val_h, val_scores, seed=42)
   keep = projection.alignment_scores_ > .45
   train_c = projection(train_h)[:, keep]
   val_c = projection(val_h)[:, keep]
   scaler = StandardScaler().fit(train_c)
   classifier = SparseLinearConceptToConcept(int(keep.sum()), len(class_names))
   classifier.fit(scaler.transform(train_c), train_y,
                  scaler.transform(val_c), val_y, seed=42)
   logits = classifier(scaler.transform(val_c))

The aligned encoder exposes ``alignment_scores_`` (on CPU), ``history_``, and
``best_step_``. The sparse predictor exposes ``diagnostics_``. Both ``fit``
methods return the fitted, frozen layer. Optimizer defaults match the high-level
model; validation data only reports the sparse classifier's validation loss.
``LabelFreeCBM`` additionally handles vocabulary selection and column identity,
input/target validation, graph construction, atomic refitting, and checkpoints.

The example supplies the paper's three few-shot prompt templates once per
class through its local ``concept_prompts`` callable. Pass a custom callable
to ``LLMConceptGenerator(prompt=..., llm=backend)`` to change the prompts.
The existing ``SemanticConceptFilter`` handles the resulting vocabulary.

Data contracts and splits
------------------------

Prepare the dataset and call ``datamodule.setup('fit')`` before fitting.
``fit`` consumes the existing train/validation splits and validates its input
contract; it does not prepare data or call ``setup``. Configure explicit indices
through the datamodule's ``FixedIndicesSplitter``. Each split needs at least two
observations; callers are responsible for choosing disjoint rows. ``testset``
is never read. Unprepared or incomplete splits raise an error before concept
generation.

Feature extraction creates local loaders over the prepared splits with
``shuffle=False, drop_last=False``, preserving score-row order and retaining
the final incomplete batch. These use the datamodule's batch size and workers
and the dataset's collation. The datamodule's ordinary training loader retains
its existing ``drop_last=True`` behavior.

Batches use PyC's ``batch['inputs']['x']`` tensor format. Task labels come from
the column of ``batch['concepts']['native']`` selected by ``fit(..., task_name=...)``.
The declared native schema is read via
``ConceptDataset.native_annotations``, even when generated annotations have
been selected with ``use_as_gt=True``. Its ordered states define class names,
the classifier width, and logit order. Labels must be integer indices into those
states. One multiclass task is supported, including a two-class categorical task.
Prepare custom task labels in the dataset before fitting.

Datasets may already contain concept annotations, contain only task labels, or
start without annotations. No concept or task metadata is required by the
constructor. Full fitting always needs task labels: prepare them externally
before calling ``fit``. An unlabelled dataset raises a clear error before
concept generation. Unlabelled concept learning remains available through the
generation pipeline and projection layer separately.

The pipeline receives only the training dataset for discovery, its local
``generation_indices``, and ``annotation_datasets={'train': ..., 'val': ...}``.
The supplied datasets define the separate split views. Split-specific
scores are consumed directly; they are not attached to the full dataset. Neither
native nor generated dataset supervision is mutated.

Keep raw inputs for annotation: datasets whose inputs were replaced by
``precompute_embeddings`` are rejected. The fitted predictor retains a frozen
copy of the backbone and accepts new raw inputs independently of training data.
For training directly on cached features, compose ``AlignedEmbeddingToConcept``
and ``SparseLinearConceptToConcept`` as shown above.

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
3. A bias-free ``AlignedEmbeddingToConcept`` learns a projection of frozen
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
   The scaler replaces zero or near-zero scales with one to avoid division by
   zero; this safeguard differs from the authors' direct normalization.
6. A zero-initialized multinomial linear classifier is trained with indexed
   minibatch proximal SAGA. The objective is mean cross-entropy plus
   ``penalty * (alpha * ||W||_1 + (1-alpha) * ||W||_F**2 / 2)``. The intercept
   is not penalized. The joint soft-threshold/shrinkage proximal operator and
   per-observation gradient memory match the reference. Neither projection nor
   backbone is optimized in this stage.

``classifier_kwargs={"lam": .0007}`` follows the authors' CLI parameterization: their
single-point GLM path divides it by ``max(.001, alpha)``. At ``alpha=.99`` the
effective penalty is approximately .0007070707, exposed as
``diagnostics_["classifier"]["regularization"]``. SAGA defaults are lr .1, batch 256, 1000 epochs,
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
the LF-CBM authors. ``SparseLinearConceptToConcept.fit`` creates a seeded indexed
loader, zeros the classifier, and calls ``train_saga`` at the effective penalty
above, using the shared positional arguments and supported solver keywords.
The PyPI ``0.1.2`` source inspected for this change accepts both ``verbose`` and
``logger``; the adapter does not depend on the optional ``logger`` keyword.
It avoids the published package's regularization-path construction,
which takes ``log10(lambda)`` and cannot represent a zero penalty. No SAGA
updates are reimplemented. The example credits the authors' few-shot prompt
templates.

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
matching the authors' inference output. ``projection_weight_`` holds the unfused
bias-free projection matrix. Calling inference or saving before a successful
fit raises an error. All training results are computed locally and published
only after predictor construction succeeds. A failed initial fit leaves the
model unfitted; a failed refit preserves the previous predictor, normalization,
vocabulary, and diagnostics. A successful refit replaces all fitted results.

``LabelFreeCBM`` stages construction of a standard CBM and installs the validated
graph on itself only when fitting succeeds. Learned weights are installed using ``ConceptBottleneckModel.set_linear_heads``. This narrow
interface validates dimensions and finite parameters before copying. It supports
plated scalar continuous ``Delta`` concepts and one categorical task, and rejects
unsupported configurations explicitly. Normalization is folded into the concept
head before installation. User module hooks are preserved, and inherited
teacher-forcing query caches are refreshed when refitting changes the vocabulary.
Existing CBM construction and training are unchanged.

``concept_names_`` returns retained labels. Full concept/task definitions live
on ``concept_annotations``. The inherited ``concept_names`` includes the task;
``intermediate_concept_names`` and ``concept_names_`` list only the bottleneck concepts. Training diagnostics are grouped in
``diagnostics_``: ``activation_scores`` covers candidate concepts,
``alignment_scores`` covers concepts surviving activation filtering,
``projection_history`` and ``projection_best_step`` record early stopping, and
``classifier`` records solver settings and train/validation losses. Filtering
scores remain annotated and addressable by name. Raw sample-by-concept score
matrices are discarded after fitting. The external solver does not return
classifier iteration counts or convergence status.

After fitting, ``task_name`` and ``class_names`` are read from the installed CBM
annotations. Candidate task metadata stays local during fitting, so a failed
refit targeting a different column preserves the previous task and class order.

``save(directory)`` writes JSON metadata and the fitted tensor state dictionary,
including the actual inference heads. ``load`` restores these weights directly;
it does not reconstruct them from diagnostic normalization statistics. No
pipeline, LLM, CLIP annotator, dataset, or arbitrary Python object is serialized.
Diagnostics are available after fitting but are not restored for inference.
New checkpoints use format version 3 with direct CBM state keys. ``load`` also
accepts version 2 checkpoints from the earlier wrapper by removing their
``model_.`` key prefix. Version 1 checkpoints must be regenerated.
Floating-point dtype is preserved.

Identity backbones reconstruct automatically. Otherwise,
``load(directory, backbone=compatible_module, device=...)`` requires the same
architecture and input preprocessing. Its weights are strictly restored from
the checkpoint. No model is downloaded or imported from metadata. The restored
model, predictions, names, and activations work without rediscovery.

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
is explicitly preserved by default despite comparing unrelated axes. Set
``reference_class_filter=False`` for general semantic filtering that compares
every class/concept pair, including equal indices. Output remains alphabetical
in both modes. Use all-mpnet-base-v2
(optional sentence-transformers package) and CLIP ViT-B/16 encoders for parity.

``examples/high_level/lfcbm_cifar10.json`` records the published CIFAR-10 settings:
the authors' vocabulary, CLIP RN50 backbone, ViT-B/16 annotation, and optimization
defaults. OpenAI CLIP is optional for that RN50 backbone. Keep its preprocessing
inside a reconstructible backbone module, separate from annotation preprocessing.
Hugging Face ViT-B/16 is a compatible converted checkpoint; exact score parity
requires the same weights, tokenizer, resize/crop conventions, and precision.
An OpenAI CLIP-backed existing ``Annotator`` can be supplied for strict parity.

The authors use CIFAR-10's official test partition as ``val`` for projection
selection. The example follows that protocol: all 50000 training images are used
for fitting and all 10000 official test images for validation and evaluation.
It uses the OpenAI CLIP RN50 image encoder with its original PIL preprocessing
and explicitly configures the published projection and SAGA hyperparameters.
Install OpenAI CLIP with ``python -m pip install git+https://github.com/openai/CLIP.git``.
The reported test accuracy is therefore not an independent held-out estimate.
Test data supplied only as ``testset`` is never consumed by the model.
No full image benchmark reproduction was run here.

Verification
------------

Normal tests use deterministic task-only datasets and lightweight fake annotators,
covering dynamic/fixed generation, selection, ordering, middle-column removal,
best-weight restoration, frozen components, training-only normalization, SAGA,
model equivalence, and persistence without network access. Optional comparisons
against a local authors' checkout require ``LFCBM_REFERENCE`` and otherwise
skip. To install the solver and run the offline tests with coverage:

.. code-block:: bash

   python -m pip install -e ".[lfcbm,tests]"
   python -m pytest -o addopts='' tests/nn/modules/high/models/test_lfcbm.py --cov=torch_concepts.nn.modules.high.models.lfcbm

The CI coverage job installs ``.[data,lfcbm,tests]`` on Python 3.10 and 3.12,
so genuine solver tests execute there rather than being silently skipped.
The direct reference comparisons can be run explicitly:

.. code-block:: bash

   LFCBM_REFERENCE=/path/to/Label-free-CBM python -m pytest -o addopts='' tests/nn/modules/high/models/test_lfcbm_reference.py

They compare cosine-cubed values/gradients, GLM-SAGA weights/bias/objective at
several alpha values, and concept-filtering decisions using fixed text embeddings.
No data, remote models, or source code are downloaded by these tests.

Pre-merge API changes
---------------------

The former ``projection_*`` optimizer arguments move into ``projection_kwargs``
and ``classifier_*`` arguments into ``classifier_kwargs`` (``classifier_lambda``
becomes ``lam``). ``fit`` accepts only a ``ConceptDataModule``. Direct dataset
arguments, ``target_getter``, and the ``embeddings`` argument have been removed.
``task_name`` moves from construction to ``fit``; ``class_names`` is inferred
from the selected native annotation.
Extraction batch size is configured on the datamodule; projection/classifier
minibatch sizes remain optimizer settings. Use ``FixedIndicesSplitter`` for
explicit partitions and prepare native categorical task labels in the dataset.
The ``model_`` wrapper and fitted module aliases have been removed: the
``LabelFreeCBM`` instance owns the CBM directly. Training diagnostics are grouped
under ``diagnostics_``. The prediction/query methods
remain available.
