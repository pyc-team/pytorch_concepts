Data
====

Datasets and Lightning data modules for concept-based learning. The docstrings of each class
below document their parameters and behaviour.

Base Classes
------------

See :doc:`Datasets </guides/using_data>` for how to use them, and
:doc:`concept generation </guides/using_generation>` for attaching generated
supervision to a dataset.

.. currentmodule:: torch_concepts.data.base

.. autosummary::
   :toctree: generated
   :template: generation_class.rst
   :nosignatures:

   ConceptDataset

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptDataModule

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

Environment
-----------

.. module:: torch_concepts.env

Settings shared by the library, its examples and Conceptarium, read from environment
variables when the module is imported.

.. py:data:: CACHE

   Cache folder for datasets, embeddings and checkpoints: ``$PYC_CACHE`` if set, otherwise
   ``$XDG_CACHE_HOME/pyc``, otherwise ``~/.cache/pyc``. It is created on import.

.. py:data:: DATA_ROOT

   Folder where the examples store datasets; the same as :data:`CACHE`.

.. py:data:: HUGGINGFACEHUB_TOKEN

   Hugging Face token, read from ``HF_TOKEN``, ``HUGGINGFACE_HUB_TOKEN`` or
   ``HUGGINGFACEHUB_TOKEN``. Needed for gated models and datasets.

.. py:data:: OPENAI_API_KEY

   OpenAI API key, read from ``OPENAI_API_KEY``, for OpenAI models in concept generation.
