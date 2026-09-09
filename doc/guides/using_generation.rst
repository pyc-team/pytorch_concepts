Concept Generation
==================

The :class:`~torch_concepts.data.generation.ConceptSupervisionPipeline` discovers
concepts and assigns them values for each dataset sample. Use it when you want
concept supervision without manually labeling every image.

A generator returns :class:`~torch_concepts.Annotations`: concept names and their
definitions, with no sample values. An annotator returns an
:class:`~torch_concepts.AnnotatedTensor`: sample-level scores with that metadata
attached to the concept columns. A concept axis is the whole collection of
concept columns, not one individual concept.

.. code-block:: text

   Dataset context → Generator → Annotations → Generator filter
                                                    ↓
   Images ────────────────────────────────────── Annotator
                                                    ↓
   Raw score filter → Calibrator → Calibrated score filter → Aggregator
                                                    ↓
                                   Named AnnotatedTensor outputs

Filtering, calibration, and aggregation are configurable stages. Expand the
sections below for a complete example and the available choices. See the
:doc:`API reference </modules/generation_api>` for all parameters.

.. dropdown:: Discover and annotate image concepts
   :icon: rocket
   :open:

   Install the data dependencies from the repository root:

   .. code-block:: bash

      python -m pip install -e ".[data]"

   Set ``OPENAI_API_KEY`` in your environment. This example calls an OpenAI model
   through LiteLLM and downloads a pretrained CLIP model. It uses ColorMNIST,
   which downloads MNIST if needed. Run the following blocks in order.

   .. code-block:: python

      from torch_concepts.data import ColorMNISTDataset
      from torch_concepts.data.generation import ConceptSupervisionPipeline
      from torch_concepts.data.generation.generators import (
          LiteLLMBackend, LLMConceptGenerator,
      )
      from torch_concepts.data.generation.annotators import CLIPAnnotator
      from torch_concepts.data.generation.calibrators import SigmoidCalibrator
      from torch_concepts.data.generation.filters import (
          DeduplicateConcepts, ThresholdAnnotationFilter,
      )

      dataset = ColorMNISTDataset(train=True)
      generator = LLMConceptGenerator(
          llm=LiteLLMBackend(model="openai/gpt-4o-mini"),
          prompt=(
              "List 6 visible binary properties of handwritten digits useful "
              "for distinguishing {class_names}. Return one property per line."
          ),
      )
      pipeline = ConceptSupervisionPipeline(
          generators=generator,
          annotators=CLIPAnnotator(batch_size=64),
          generator_filter=DeduplicateConcepts(),
          calibrator=SigmoidCalibrator(scale=10.0, bias=-2.5),
          calibrated_annotation_filter=ThresholdAnnotationFilter(threshold=0.5),
          routing="merged",
      )
      outputs = pipeline(dataset, class_names=[str(i) for i in range(10)])
      scores = outputs["CLIPAnnotator"]
      print(scores.shape)              # (number of images, discovered concepts)
      print(scores.annotation.labels)  # the discovered properties

   ``class_names`` supplies task context for the prompt; it does not prescribe
   the generated concept names. This string prompt uses class names only.
   To include dataset samples, supply a callable prompt to the generator; it
   receives ``dataset`` when ``generate()`` or the pipeline is called. The
   generator constructor does not receive a dataset.

   CLIP's default input getter understands the ``inputs["x"]`` image returned
   by ConceptDataset. Supply ``input_getter`` for a different sample structure.
   The sigmoid settings above are illustrative; they do not establish that
   CLIP similarity scores are calibrated probabilities.

.. dropdown:: Filters, calibration, and aggregation
   :icon: gear

   ``DeduplicateConcepts`` is the default generator filter. It accepts and
   returns an ``Annotations`` object, keeping the first occurrence of each
   label. Duplicate definitions with incompatible states, cardinalities, or
   types raise ``ValueError``. Set ``generator_filter=None`` to disable it.

   After annotation, the pipeline applies ``raw_annotation_filter``, then
   ``calibrator``, then ``calibrated_annotation_filter``.
   ``ThresholdAnnotationFilter`` sets scores below the threshold to zero;
   it preserves columns and leaves scores above the threshold unchanged.
   ``SigmoidCalibrator`` applies ``sigmoid(scale * scores + bias)``.

   With merged routing, an optional callable can combine multiple annotators.
   Inputs must have matching rows and concept definitions. Using the generator
   and dataset above:

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

      ensemble = ConceptSupervisionPipeline(
          generators=generator,
          annotators=[
              CLIPAnnotator(model_name="openai/clip-vit-base-patch32"),
              CLIPAnnotator(model_name="openai/clip-vit-base-patch16"),
          ],
          calibrator=SigmoidCalibrator(scale=10.0, bias=-2.5),
          aggregator=average_scores,
          routing="merged",
      )
      ensemble_outputs = ensemble(dataset, class_names=[str(i) for i in range(10)])
      print(list(ensemble_outputs))
      # ['CLIPAnnotator', 'CLIPAnnotator_1', 'aggregated']

   Aggregation adds an output; the individual annotator outputs remain available.

.. dropdown:: Routing and output names
   :icon: workflow

   .. list-table:: Choosing generator–annotator combinations
      :header-rows: 1
      :widths: 15 55 30

      * - Routing
        - Which concepts each annotator receives
        - Output keys
      * - ``merged``
        - All generator outputs combined and filtered once
        - Annotator name
      * - ``cartesian``
        - Every generator output, filtered separately
        - ``<generator>_<annotator>``
      * - ``zip``
        - The corresponding generator output, filtered separately
        - ``<generator>_<annotator>``

   Zip routing requires equal numbers of generators and annotators. Aggregation
   is supported only with merged routing.

   Names come from a component's nonempty ``name`` attribute, otherwise its
   class name. Repeated names receive ``_1``, ``_2``, and so on. An aggregator
   uses ``aggregated``; collisions also receive a numeric suffix. Named
   annotation targets prefix these keys, for example ``train_CLIPAnnotator``.
   Inspect ``outputs.keys()`` for the exact names produced by your configuration.

.. dropdown:: Discover on training data and annotate other splits
   :icon: database

   Generation and annotation selections are independent. If your prompt uses
   samples, restrict discovery to training rows. A callable prompt must honor
   the ``indices`` argument: the generator still receives the full dataset.
   The class-name-only prompt above does not read any rows.

   .. code-block:: python

      train_indices = list(range(100))
      val_indices = list(range(100, 120))
      split_outputs = pipeline(
          dataset,
          class_names=[str(i) for i in range(10)],
          generation_indices=train_indices,
          annotation_indices={"train": train_indices, "val": val_indices},
      )
      train_scores = split_outputs["train_CLIPAnnotator"]
      val_scores = split_outputs["val_CLIPAnnotator"]

   If only ``generation_indices`` is supplied, annotation still covers the full
   dataset. For separate dataset objects, use
   ``annotation_datasets={"train": train_dataset, "val": val_dataset}``
   instead of ``annotation_indices``. Both targets share the vocabulary
   generated in that pipeline call. The caller defines the splits.

.. dropdown:: Attach generated supervision to ConceptDataset
   :icon: package

   To generate and attach full-dataset outputs in one call:

   .. code-block:: python

      generated = dataset.generate_concepts(
          pipeline,
          class_names=[str(i) for i in range(10)],
          use_as_gt=True,
      )
      assert dataset.concepts is generated["CLIPAnnotator"]
      # Original labels remain in dataset.native_concepts.

   A single source is selected automatically. With multiple sources, choose
   an exact output key. To reuse the ensemble outputs already computed above:

   .. code-block:: python

      dataset.set_generated_concepts(
          ensemble_outputs, use_as_gt=True, generated_gt_name="aggregated",
      )

   ``use_as_gt=False`` keeps native concepts selected when they exist. A dataset
   with no native concepts selects generated supervision automatically; multiple
   sources still require ``generated_gt_name``. Generated concepts replace the
   selected supervision as a whole: they do not automatically append native
   task labels. Preserve or combine task supervision explicitly when preparing
   a training experiment.

   ``generate_concepts()`` attaches only outputs aligned one-to-one with the
   complete dataset. Call the pipeline directly for split-specific tensors.

For configuration-based experiments, see :doc:`using_conceptarium`.
