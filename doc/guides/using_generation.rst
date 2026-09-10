Concept Generation
==================

Concept generation in PyC is built around a simple idea:

1. **Discover which concepts are useful** for the task.
2. **Assign those concepts to the samples** in the dataset.

.. code-block:: text

   [Component] = library pipeline stage; (object) = input/output data

   (Dataset context / prompt)
               ↓
          [Generator] → (Annotations) → [Generator filter]
                                               ↓
   (Dataset images) ────────────────────── [Annotator]
                                               ↓
                                       (AnnotatedTensor)
                                               ↓
                                      [Raw score filter]
                                               ↓
                                         [Calibrator]
                                               ↓
                                   [Calibrated score filter]
                                               ↓
                                [Aggregator: optional callable]
                                               ↓
                                (dict[str, AnnotatedTensor])

The pipeline is modular: each step can be replaced or configured independently.
For example, one component may propose concepts, another may remove unsuitable
ones, another may score how strongly each concept applies to each sample, and
optional post-processing stages can refine those scores.

The :class:`~torch_concepts.data.generation.ConceptGenerationPipeline`
orchestrates these steps.

Pipeline steps
--------------

.. dropdown:: 1. Generate concepts
   :icon: light-bulb

   Each :class:`~torch_concepts.data.generation.Generator` produces a concept
   vocabulary from the available context. The dataset need not have any
   existing concepts: an LLM can propose them from a prompt describing the task.

   .. code-block:: python

      generator = LLMConceptGenerator(
          llm=llm_backend,
          prompt="List visible properties distinguishing {class_names}. One per line.",
      )

   Contract::

      Dataset context -> Generator -> Annotations

   :class:`~torch_concepts.Annotations` describes the concepts and their
   metadata, but contains no sample-level values.

   See :doc:`Annotations and tensors </modules/low_level_api>` and the
   :doc:`concept generation API </modules/generation_api>`.


.. dropdown:: 2. Filter generated concepts
   :icon: filter

   A :class:`~torch_concepts.data.generation.FilterGenerator` can remove or
   reorder generated concept definitions before annotation.

   Contract::

      Annotations -> FilterGenerator -> Annotations

   The default is
   :class:`~torch_concepts.data.generation.filters.DeduplicateConcepts`.
   It keeps the first occurrence of each label and raises ``ValueError`` when
   duplicates have incompatible states, cardinalities, or types.
   Set ``generator_filter=None`` to disable this stage.

   .. code-block:: python

      generator_filter = DeduplicateConcepts()


.. dropdown:: 3. Route concepts to annotators
   :icon: workflow

   A pipeline can use multiple generators and multiple annotators. Routing
   determines which generated vocabularies are passed to which annotators.

   ``merged``
      Merge all generator outputs, filter once, and send the resulting
      vocabulary to every annotator.

   ``cartesian``
      Filter each generator output independently and send every output to every
      annotator.

   ``zip``
      Filter each generator output independently and pair generators and
      annotators in configuration order.

   ``zip`` requires equal numbers of generators and annotators.

   For example, suppose generator G1 proposes ``[red, round]`` and G2 proposes
   ``[round, striped]``, with compatible definitions for ``round``, and two
   annotators A1 and A2. With the default deduplication filter:

   .. list-table:: The same generators and annotators, routed differently
      :header-rows: 1
      :widths: 15 60 25

      * - Mode
        - Annotation jobs
        - Outputs
      * - ``merged``
        - A1 and A2 each score ``[red, round, striped]``.
        - 2 shared-vocabulary tensors
      * - ``cartesian``
        - A1 and A2 each score G1's ``[red, round]`` and, separately,
          G2's ``[round, striped]``.
        - 4 tensors, one per pair
      * - ``zip``
        - A1 scores ``[red, round]``; A2 scores ``[round, striped]``.
        - 2 paired tensors

   Use merged routing for a shared vocabulary, cartesian routing to compare
   every combination, and zip routing for designated generator–annotator pairs.


.. dropdown:: 4. Annotate samples
   :icon: tag

   Each :class:`~torch_concepts.data.generation.Annotator` assigns values for
   the routed concepts to the selected dataset samples.

   Contract::

      Dataset + Annotations -> Annotator -> AnnotatedTensor

   The resulting :class:`~torch_concepts.AnnotatedTensor` contains the
   sample-level values together with their concept metadata.

   .. code-block:: python

      annotator = CLIPAnnotator(model_name="openai/clip-vit-base-patch32", batch_size=64)

   See :doc:`Annotations and tensors </modules/low_level_api>`.


.. dropdown:: 5. Filter raw annotations
   :icon: filter

   ``raw_annotation_filter`` optionally processes annotator outputs before
   calibration.

   Contract::

      AnnotatedTensor -> FilterAnnotator -> AnnotatedTensor

   A :class:`~torch_concepts.data.generation.FilterAnnotator` must preserve
   tensor shape and concept metadata.

   ``ThresholdAnnotationFilter(threshold=0.2)`` sets scores below 0.2 to zero;
   it does not remove concepts or turn the remaining scores into ones.

   .. code-block:: python

      raw_annotation_filter = ThresholdAnnotationFilter(threshold=0.2)


.. dropdown:: 6. Calibrate annotation scores
   :icon: sliders

   A :class:`~torch_concepts.data.generation.Calibrator` optionally transforms
   the raw annotation scores.

   Contract::

      AnnotatedTensor -> Calibrator -> AnnotatedTensor

   Shape and concept metadata are preserved.

   ``SigmoidCalibrator(scale=10.0, bias=-2.5)`` applies
   ``sigmoid(10 * scores - 2.5)``. These example settings transform scores into
   the range (0, 1); they do not guarantee calibrated probabilities.

   .. code-block:: python

      calibrator = SigmoidCalibrator(scale=10.0, bias=-2.5)


.. dropdown:: 7. Filter calibrated annotations
   :icon: filter

   ``calibrated_annotation_filter`` optionally processes the values after
   calibration.

   It uses the same
   :class:`~torch_concepts.data.generation.FilterAnnotator` contract as the raw
   filter.

   If no calibrator is configured, it receives the raw-filtered values.

   .. code-block:: python

      calibrated_annotation_filter = ThresholdAnnotationFilter(threshold=0.5)


.. dropdown:: 8. Aggregate outputs
   :icon: stack

   With ``merged`` routing, an optional ``aggregator`` can combine compatible
   annotator outputs.

   Individual annotator outputs are preserved; aggregation adds an additional
   output named ``aggregated`` (with a numeric suffix on a name collision).
   The callable receives a dictionary of processed tensors with matching rows
   and concept definitions. For example, average their scores:

   .. code-block:: python

      import torch
      from torch_concepts import AnnotatedTensor

      def average_scores(outputs):
          values = list(outputs.values())
          return AnnotatedTensor(
              torch.stack([value.tensor for value in values]).mean(dim=0),
              values[0].annotation,
              axis=1,
          )


.. dropdown:: 9. Return generated supervision
   :icon: package

   The pipeline returns a dictionary of named
   :class:`~torch_concepts.AnnotatedTensor` objects.

   For two samples, two binary concepts, and two annotators with mean
   aggregation, an example output is:

   .. code-block:: python

      import torch
      from torch_concepts import Annotations, AnnotatedTensor

      concepts = Annotations(labels=["round", "striped"], cardinalities=[1, 1])
      outputs = {
          "CLIPAnnotator": AnnotatedTensor(
              torch.tensor([[0.8, 0.2], [0.4, 0.6]]), concepts, axis=1,
          ),
          "CLIPAnnotator_1": AnnotatedTensor(
              torch.tensor([[0.6, 0.4], [0.2, 0.8]]), concepts, axis=1,
          ),
          "aggregated": AnnotatedTensor(
              torch.tensor([[0.7, 0.3], [0.3, 0.7]]), concepts, axis=1,
          ),
      }

   Rows follow sample order; columns follow ``concepts.labels``.

   Names depend on the routing mode and configured component names. Named
   annotation targets also prefix the corresponding output keys.

   See :doc:`concept generation API </modules/generation_api>` for the complete
   naming rules.

Complete example
----------------

The following example uses the ``average_scores`` function above. Install the
repository's data extras with ``python -m pip install -e ".[data]"`` and set
``OPENAI_API_KEY``. It calls an LLM provider and downloads CLIP weights and MNIST
if needed. Only 120 images are annotated to keep the example small.

.. code-block:: python

   from torch_concepts.data import ColorMNISTDataset
   from torch_concepts.data.base import ConceptDataset
   from torch_concepts.data.generation import ConceptGenerationPipeline
   from torch_concepts.data.generation.generators import LiteLLMBackend, LLMConceptGenerator
   from torch_concepts.data.generation.annotators import CLIPAnnotator
   from torch_concepts.data.generation.calibrators import SigmoidCalibrator
   from torch_concepts.data.generation.filters import DeduplicateConcepts, ThresholdAnnotationFilter

   images = ColorMNISTDataset(train=True)
   dataset = ConceptDataset(input_data=images.input_data[:120])
   class_names = [str(i) for i in range(10)]
   pipeline = ConceptGenerationPipeline(
       generators=LLMConceptGenerator(
           llm=LiteLLMBackend(model="openai/gpt-4o-mini"),
           prompt=(
               "List 6 visible binary properties of handwritten digits useful "
               "for distinguishing {class_names}. Return one property per line."
           ),
       ),
       generator_filter=DeduplicateConcepts(),
       routing="merged",
       annotators=[
           CLIPAnnotator(model_name="openai/clip-vit-base-patch32"),
           CLIPAnnotator(model_name="openai/clip-vit-base-patch16"),
       ],
       raw_annotation_filter=ThresholdAnnotationFilter(threshold=0.2),
       calibrator=SigmoidCalibrator(scale=10.0, bias=-2.5),
       calibrated_annotation_filter=ThresholdAnnotationFilter(threshold=0.5),
       aggregator=average_scores,
   )
   outputs = pipeline(dataset, class_names=class_names)
   print(list(outputs))
   # ['CLIPAnnotator', 'CLIPAnnotator_1', 'aggregated']
   print(outputs["aggregated"].annotation.labels)

``class_names`` gives the LLM task context, not the desired concept names.
This string prompt reads no samples. To supply samples, use a callable prompt;
the dataset is passed at generation time, not to the generator constructor.
CLIP's default input getter handles ConceptDataset's ``inputs["x"]`` images.

To annotate named subsets instead, reuse the pipeline as follows. This makes
another generation call and shares its vocabulary across both subsets:

.. code-block:: python

   split_outputs = pipeline(
       dataset,
       class_names=class_names,
       generation_indices=list(range(100)),
       annotation_indices={"train": list(range(100)), "val": list(range(100, 120))},
   )
   train_scores = split_outputs["train_aggregated"]
   val_scores = split_outputs["val_aggregated"]

Use the output in ConceptDataset
-----------------------------------

Attach the full-dataset ``outputs`` already computed above and select one source
as the learner-facing ``dataset.concepts``:

.. code-block:: python

   dataset.set_generated_concepts(
       outputs, use_as_gt=True, generated_gt_name="aggregated",
   )
   assert dataset.concepts is outputs["aggregated"]

Each attached tensor must follow the dataset's row order and contain one row
per sample; split-specific tensors cannot be attached to the full dataset.
Alternatively, ``dataset.generate_concepts(pipeline, class_names=class_names,
use_as_gt=True, generated_gt_name="aggregated")`` generates and attaches in
one call. Omit ``generated_gt_name`` when there is only one source.

Native values remain in ``dataset.native_concepts``. With ``use_as_gt=False``,
they stay selected if present; without native values, generated supervision
is selected automatically. Selecting generated supervision does not append
native task labels, so include those explicitly if the training task needs them.

See :class:`~torch_concepts.data.base.ConceptDataset` and its
:meth:`~torch_concepts.data.base.ConceptDataset.generate_concepts` method for
the full parameter descriptions, including ``class_names``, ``use_as_gt``, and
``generated_gt_name``.

For configuration-based experiments, see :doc:`using_conceptarium`.
