Low-Level API
=============

Composable interpretable layers and intervention utilities. See the
:doc:`Low-Level user guide </guides/using_low_level>` for explanations and examples; the docstrings
of each class below document their parameters and behaviour.

.. currentmodule:: torch_concepts.nn

Encoders
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearEmbeddingToConcept
   LinearEmbeddingEncoder
   SelectorEmbeddingEncoder

Predictors
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearConceptToConcept
   CallableConceptToConcept
   HyperlinearConceptEmbeddingToConcept
   MixConceptEmbeddingToConcept

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
   BaseConceptInterventionStrategy
   BaseModuleInterventionStrategy
   BaseInterventionPolicy
   BaseInterventionModule
