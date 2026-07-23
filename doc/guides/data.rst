Datasets and Data Modules
=================

PyC provides concept-aware datasets and PyTorch Lightning data modules for
loading inputs, concepts and task labels, and optional concept graphs. Datasets
provide direct access to individual samples, whereas data modules additionally
provide train/validation/test splits and ready-to-use data loaders.

The complete API reference is available in :doc:`../modules/data_api`.


Datasets
--------

All PyC datasets follow the
:class:`~torch_concepts.data.base.dataset.ConceptDataset` interface, which extends
``torch.utils.data.Dataset`` with concept-aware fields.

To instantiate a ``ConceptDataset``, provide:

- ``input_data``: the model inputs;
- ``concepts``: the concept (and task) labels.

You can also provide ``annotations`` with the names, cardinalities, and types
of the concepts (and task), and an optional ``graph`` describing their relationships. 
The optional ``concept_names_subset`` argument keeps only selected concepts,
while ``precision`` controls tensor conversion and ``name`` assigns a custom
dataset name.

After construction, the dataset stores:

- ``input_data``: the inputs converted to a tensor;
- ``concepts``: the concept labels as an annotated tensor;
- ``annotations``: concept names, cardinalities, and types (see
  :doc:`Annotations <using_low_level>`);
- ``graph``: the optional concept graph;
- ``name`` and ``precision``: the dataset identifier and numerical precision.

Dataset-specific classes may require additional arguments, such as a data
``root`` (see :doc:`Downloading and Building Datasets <downloading_building_datasets>`) or generation parameters.

Complete instructions for creating a new dataset class are available in :doc:`contributing_dataset`.

Downloading and building datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For built-in datasets, only a ``root`` directory is required. The
dataset class automatically downloads raw files, preprocesses them
to produce ``input_data``, ``concepts``, ``annotations``, and, when available,
``graph``, and uses these data to initialize the dataset instance.

The processed files are also cached under ``root``. When the same dataset is
instantiated again, the class loads the cached files instead of downloading and
processing them again.

More information on the downloading and building process is available in :doc:`contributing_dataset`.

The following example shows how to instantiate the CUB dataset automatically. 


.. code-block:: python

   from torch_concepts.data import CUBDataset

   # construct the dataset automatically, downloading and processing files if necessary
   cub = CUBDataset(root="./data/CUB200")

   # access the cached inputs and concepts for the first sample
   sample = cub[0]
   image = sample["inputs"]["x"]
   concepts = sample["concepts"]["c"]

   print(len(cub), image.shape, concepts.shape)



Data Modules
------------

:class:`~torch_concepts.data.base.datamodule.ConceptDataModule` wraps a
concept dataset in the standard PyTorch Lightning data pipeline. It handles
train/validation/test splitting, optional backbone embedding
precomputation, configurable data scalers, and data loader creation.

To instantiate a ``ConceptDataModule``, provide:

- a concept ``dataset``;
- split settings such as ``val_size``, ``test_size``, ``seed``, or a custom
  ``splitter`` (see :ref:`data-splitters`);
- data-loader settings such as ``batch_size`` and, optionally, ``workers`` and
  ``pin_memory``;
- optionally, a ``scalers`` mapping containing the custom scalers used for
  data normalization (see :ref:`data-scalers`);
- optionally, a ``backbone`` and ``precompute_embs=True`` to cache embeddings
  before training;
- ``force_recompute=True`` to recompute backbone embeddings even when a cached
  file already exists.



The data module must be set up before its dataset splits or data loaders are
accessed. PyTorch Lightning calls ``ConceptDataModule.setup(stage)``
automatically when the data module is passed to a trainer. When the data module
is used directly, call this method explicitly, as in the example below. The
``stage`` argument identifies the operation being prepared (``"fit"``,
``"validate"``, ``"test"``, or ``"predict"``); passing ``None`` prepares the
data module without restricting it to a particular stage. During setup, the
data module:

- computes or loads cached embeddings when ``precompute_embs=True`` and stores
  them in ``dataset.input_data``;
- sets ``dataset.embs_precomputed`` to indicate whether ``input_data`` contains
  precomputed embeddings;
- creates the ``trainset``, ``valset``, and ``testset`` splits.

When ``precompute_embs=False``, ``dataset.input_data`` retains the original
inputs.


The following example shows how to instantiate a data module for the CUB dataset,
computing embeddings from the input images using a ResNet-50, and applying the dataset's
native splitting. The resulting train, validation, and test splits can then be
accessed through the data module's attributes, after calling ``setup("fit")``.

.. code-block:: python

   from torch_concepts.data import CUBDataset
   from torch_concepts.data import CUBDataModule
   from torch_concepts.data.splitters import NativeSplitter

   cub = CUBDataset(root="./data/CUB200")

   datamodule = CUBDataModule(
        dataset=cub,
        splitter=NativeSplitter(),
        batch_size=64,
        backbone="resnet50",
        precompute_embs=True,
        workers=4,
   )

   datamodule.setup("fit")
   trainset = datamodule.trainset
   valset = datamodule.valset
   testset = datamodule.testset

Alternatively, each built-in data module class (e.g., :class:`~torch_concepts.data.CUBDataModule`) can instantiate the dataset internally and configure its native splitter automatically. This
is the shortest option when the dataset does not need to be customized or reused
separately:

.. code-block:: python

   from torch_concepts.data import CUBDataModule

   datamodule = CUBDataModule(
        root="./data/CUB200",
        splitter=NativeSplitter(),
        batch_size=64,
        backbone="resnet50",
        precompute_embs=True,
        workers=4,
   )

   datamodule.setup("fit")
   trainset = datamodule.trainset
   valset = datamodule.valset
   testset = datamodule.testset




.. _data-splitters:

Splitters
---------

Splitters define how a dataset is divided into train, validation, and test
sets. By default, ``ConceptDataModule`` uses ``val_size``, ``test_size``, and
``seed`` to create a random split. When a dataset already provides official
splits, or when you need a custom partition, pass a ``splitter`` object instead.


The following splitters are available:

- :class:`~torch_concepts.data.splitters.RandomSplitter` randomly assigns
  samples to each split. ``seed`` makes the split reproducible. This is the
  default splitter used by ``ConceptDataModule``.

- :class:`~torch_concepts.data.splitters.NativeSplitter` uses the predefined
  splits provided by a dataset.

- :class:`~torch_concepts.data.splitters.FixedIndicesSplitter` uses explicit
  lists of training, validation, and test indices. It is useful for reusing
  exactly the same partition across experiments.

- :class:`~torch_concepts.data.splitters.CustomSplitter` creates the splits
  using user-provided functions. Separate functions and arguments can be
  supplied for the validation and test splits.

- :class:`~torch_concepts.data.splitters.ColoringSplitter` uses a predefined
  coloring file to place samples into training and test distributions. It is
  intended for distribution-shift and out-of-distribution experiments.

For example, pass a ``FixedIndicesSplitter`` to use an explicit, reproducible
partition of CUB instead of its native split:

.. code-block:: python

   from torch_concepts.data import CUBDataset
   from torch_concepts.data.base import CUBDataModule
   from torch_concepts.data.splitters import FixedIndicesSplitter

   cub = CUBDataset(root="./data/CUB200")

   splitter = FixedIndicesSplitter(
       train_idxs=range(0, 8000),
       val_idxs=range(8000, 9000),
       test_idxs=range(9000, len(cub)),
   )

   datamodule = CUBDataModule(
       dataset=cub,
       splitter=splitter,
       batch_size=64,
       backbone="resnet50",
       precompute_embs=True,
       workers=4,
  )

  datamodule.setup("fit")
  trainset = datamodule.trainset
  valset = datamodule.valset
  testset = datamodule.testset


.. _data-scalers:

Scalers
-------

Scalers are used to normalize and denormalize data during training and inference. 
All scaler implementations follow the :class:`~torch_concepts.data.base.scaler.Scaler`
interface and implement the ``fit``, ``transform``, and ``inverse_transform`` methods.
Specifically, ``fit`` computes the statistics needed for normalization, ``transform`` applies the normalization to a tensor, and ``inverse_transform`` reverts the normalization.

By default, PyC provides
:class:`~torch_concepts.data.scalers.StandardScaler` for normalizing data to have zero mean and unit variance. The scaler should be
fitted on the training data only; validation and test data should be transformed
using the same fitted statistics to avoid data leakage. The ``axis`` argument
specifies the dimension along which these statistics are computed. For input
embeddings with shape ``(n_samples, n_features)``, ``axis=0`` computes a
separate mean and standard deviation for each feature across the training
samples.


Scalers can be configured directly when creating a data module by passing a
mapping to the ``scalers`` argument. Each key identifies the batch component to
which the corresponding scaler is applied, allowing inputs and concepts to be
normalized independently. Continuing the CUB example, the ``input`` key applies
``StandardScaler`` only to the input embeddings and leaves the concepts
unchanged:

.. code-block:: python

   from torch_concepts.data.scalers import StandardScaler

   datamodule = CUBDataModule(
       dataset=cub,
       splitter=NativeSplitter(),
       batch_size=64,
       backbone="resnet50",
       precompute_embs=True,
       scalers={"input": StandardScaler(axis=0)},
   )

   datamodule.setup("fit")
   trainset = datamodule.trainset
   


