Low-Level API
=============

Composable interpretable layers and intervention utilities. See the
:doc:`Low-Level user guide </guides/using_low_level>` for explanations and examples; the docstrings
of each class below document their parameters and behaviour.

Annotations & Tensors
----------------------

.. currentmodule:: torch_concepts

.. autosummary::
   :toctree: generated
   :nosignatures:

   Annotations
   AnnotatedTensor

.. currentmodule:: torch_concepts.annotations

.. autosummary::
   :toctree: generated
   :nosignatures:

   Concept

.. currentmodule:: torch_concepts

Backbones
---------

Pretrained feature extractors that turn raw images or text into embeddings.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Backbone
   ImageBackbone
   TextBackbone

Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   seed_everything

.. currentmodule:: torch_concepts.nn

Encoders
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearEmbeddingToConcept
   MLPEmbeddingToConcept
   LinearEmbeddingEncoder
   MLPEmbeddingEncoder
   SelectorEmbeddingEncoder
   ConceptWhitening
   WhitenedEmbeddingToConcept
   CAVEmbeddingToConcept

Learning from reference activations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`LinearEmbeddingToConcept` with ``bias=False`` provides a linear concept
projection that can be trained against numeric reference activations using
:class:`CosineCubedLoss`. Reference activations can come from a concept
annotation pipeline. Choose the optimizer, batching, and stopping criterion in
your training code, as with other PyTorch layers.

.. code-block:: python

   import torch
   from torch_concepts.nn import LinearEmbeddingToConcept, CosineCubedLoss
   from torch_concepts.nn.functional import cosine_cubed_similarity

   projection = LinearEmbeddingToConcept(feature_size, concept_annotations, bias=False)
   optimizer = torch.optim.Adam(projection.parameters(), lr=0.001)
   criterion = CosineCubedLoss()
   for _ in range(training_steps):
       optimizer.zero_grad()
       loss = criterion(projection(train_embeddings), train_scores)
       loss.backward()
       optimizer.step()

   with torch.no_grad():
       alignment = cosine_cubed_similarity(projection(val_embeddings), val_scores)
       activations = projection.annotate(projection(new_embeddings))

The embeddings and reference scores are two-dimensional floating-point tensors;
``train_scores`` and ``val_scores`` have one column per concept in the same
order. Alignment scores measure cosine similarity of centered, cubed activation
patterns. Early stopping and checkpoint selection belong in the external
training procedure. The layer has the usual PyTorch ``forward`` interface;
``annotate`` attaches the supplied concept annotations to its output. Integer
concept counts are also accepted when names are unnecessary.

Predictors
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearConceptToConcept
   SparseLinearConceptToConcept
   MLPConceptToConcept
   CallableConceptToConcept
   HyperlinearConceptEmbeddingToConcept
   MixConceptEmbeddingToConcept
   MixConceptEmbeddings
   RuleConceptEmbeddingToConcept
   RuleMemory

Sparse categorical prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`SparseLinearConceptToConcept` fits a multinomial classifier with the
optional, pinned GLM-SAGA solver. Install ``pytorch_concepts[sparse]`` to fit;
forward inference requires only PyTorch. Standardize numeric inputs using
training statistics before fitting.

.. code-block:: python

   from torch_concepts.data.scalers import StandardScaler
   from torch_concepts.nn import SparseLinearConceptToConcept

   scaler = StandardScaler().fit(train_activations)
   classifier = SparseLinearConceptToConcept(n_concepts, n_classes)
   classifier.fit(scaler.transform(train_activations), train_labels,
                  scaler.transform(val_activations), val_labels,
                  alpha=0.9, lam=0.01, epochs=100, seed=7)
   logits = classifier(scaler.transform(new_activations))

Targets are one-dimensional ``torch.long`` class indices. Validation is optional
and reports loss without selecting regularization. The objective is mean
multinomial cross entropy plus
``lam * (alpha * ||W||_1 + (1-alpha)/2 * ||W||_2^2)``; bias is unpenalized.
``alpha=1`` gives L1, ``alpha=0`` gives L2. The fitted predictor is frozen;
``diagnostics_`` records the penalty and split losses. Input columns and output
rows follow the supplied concept and class order. Use :class:`Annotations` in
place of integer counts to preserve names through ``classifier.annotate``.

Dense Layers
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Dense
   MLP
   ResidualMLP
   Sequential

Priors
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LearnablePrior
   FixedPrior
   TiedPrior

Activations
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   TrilActivation
   GlobalScale

Graph Learners
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   WANDAGraphLearner

Interventions
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   intervention
   InterventionModule
   GroundTruthIntervention
   DoIntervention
   DistributionIntervention
   PositiveWeightsIntervention
   UniformPolicy
   RandomPolicy
   UncertaintyInterventionPolicy
   GradientPolicy

Constructors
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LazyConstructor

Base Classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseConceptLayer
   BaseGraphLearner
   InterventionStrategy
   ConceptInterventionStrategy
   ModuleInterventionStrategy
   InterventionPolicy
