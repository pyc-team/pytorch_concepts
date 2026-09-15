Data
====

Datasets and Lightning data modules for concept-based learning. The docstrings of each class
below document their parameters and behaviour.

Base Dataset
------------

See :doc:`concept generation </guides/using_generation>` for attaching generated
supervision to a dataset.

.. currentmodule:: torch_concepts.data.base

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   ConceptDataset

.. currentmodule:: torch_concepts.data

Datasets
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ToyDataset
   ToyDAGDataset
   BnLearnDataset
   CompletenessDataset
   CelebADataset
   PendulumDataset
   MNISTEvenOddDataset
   MNISTAdditionDataset
   ColorMNISTDataset
   MNISTArithmeticDataset
   DSpritesRegressionDataset
   AWA2Dataset
   CUBDataset

Data Modules
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ToyDAGDataModule
   BnLearnDataModule
   CompletenessDataModule
   CelebADataModule
   PendulumDataModule
   ColorMNISTDataModule
   MNISTArithmeticDataModule
   DSpritesRegressionDataModule
   AWA2DataModule
   CUBDataModule
