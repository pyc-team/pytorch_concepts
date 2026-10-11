.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle


Datasets
========

|pyc_logo| PyC datasets pair inputs with concept labels, and describe their concepts with
:class:`~torch_concepts.Annotations`. A **datamodule** wraps a dataset with
train/validation/test splits and data loaders, ready for a PyTorch loop or a Lightning
``Trainer``. The library provides toy datasets with known concepts, image datasets such as
Color-MNIST and CelebA, and datasets sampled from Bayesian networks: see the
:doc:`Data API reference </modules/data_api>` for the full list.

Expand each block below for an explanation and an example.


.. dropdown:: Loading a dataset
    :icon: database

    A dataset is downloaded or generated on first use into the folder given as ``root``, and
    loaded from there afterwards.

    .. code-block:: python

       from torch_concepts.data import ColorMNISTDataModule, ToyDataset
       from torch_concepts.env import DATA_ROOT

       dataset = ToyDataset("xor", n_gen=1000, root=str(DATA_ROOT / "xor"))
       datamodule = ColorMNISTDataModule(root=str(DATA_ROOT / "colormnist"), batch_size=256)

    ``DATA_ROOT`` is the folder shared by all datasets: see *Where data is stored* below.


.. dropdown:: Annotations
    :icon: tag

    The annotations name the concepts and give their number of classes: 1 for a binary
    concept, ``k`` for a categorical concept with ``k`` classes. Models are built from them.

    .. code-block:: python

       annotations = datamodule.annotations
       print(annotations.labels)         # ['parity', 'color', 'digit']
       print(annotations.cardinalities)  # [1, 2, 10]

    Tasks are stored as concepts too: a model is told which ones are tasks with
    ``task_names``, here ``["parity"]``.


.. dropdown:: What a batch contains
    :icon: package

    A batch is a dictionary. The inputs are under ``batch["inputs"]["x"]``, and the labels
    under ``batch["concepts"]``, in three views:

    - ``"c"``: the target that models train on;
    - ``"native"``: the dataset's own labels;
    - ``"generated"``: concepts produced by a :doc:`concept generation <using_generation>`
      pipeline, one entry per pipeline output.

    ``"c"`` holds the native labels unless generated concepts are selected as the target
    (``use_as_gt=True``). Labels have one column per concept: 0 or 1 for a binary concept,
    the class index for a categorical one. Columns can be selected by name.

    .. code-block:: python

       datamodule.setup()
       batch = next(iter(datamodule.train_dataloader()))
       x = batch["inputs"]["x"]    # (256, 3, 28, 28) images
       c = batch["concepts"]["c"]  # (256, 3) labels
       digits = c["digit"]         # (256, 1): one column, selected by name


.. dropdown:: Splits and subsets
    :icon: git-branch

    ``setup()`` splits the dataset into ``datamodule.trainset``, ``valset`` and ``testset``.
    ``max_samples`` keeps a random subset of the dataset, which is handy for quick
    experiments, and ``seed`` makes the subset and the split reproducible. A Lightning
    ``Trainer`` calls ``setup()`` itself.

    .. code-block:: python

       datamodule = ColorMNISTDataModule(
           root=str(DATA_ROOT / "colormnist"),
           max_samples=10000,
           seed=0,
       )
       datamodule.setup()
       print(len(datamodule.trainset), len(datamodule.valset), len(datamodule.testset))
       # 7771 846 1383

    Datasets without an official split are split at random: 10% for validation and 20% for
    testing by default, set with ``val_size`` and ``test_size``. Datasets with one keep it.
    Color-MNIST tests on MNIST's test images and validates on ``val_size`` of its training
    images, as above. CelebA uses its own three partitions, unless ``splitter=None`` asks for
    a random split, which ``max_samples`` requires.


.. dropdown:: Precomputed embeddings
    :icon: cpu

    With a frozen pretrained backbone, each image needs to pass through it only once.
    ``precompute_embeddings`` runs the backbone over the dataset, caches the embeddings next to
    the data, and swaps them in as the inputs, so that models train on the embeddings.

    .. code-block:: python

       from torch_concepts import ImageBackbone
       from torch_concepts.data import CelebADataModule

       datamodule = CelebADataModule(
           root=str(DATA_ROOT / "celeba"),
           max_samples=1000,
           splitter=None,
           seed=42,
       )
       datamodule.precompute_embeddings(ImageBackbone("resnet18"))
       print(datamodule.n_features)  # (512,)


.. dropdown:: Where data is stored
    :icon: file-directory

    The examples keep every dataset in one folder, ``DATA_ROOT`` from
    :mod:`torch_concepts.env`, so that each dataset is downloaded only once:

    - ``~/.cache/pyc`` by default;
    - ``$XDG_CACHE_HOME/pyc`` if ``XDG_CACHE_HOME`` is set;
    - the folder in ``PYC_CACHE``, which overrides both.

    The same module reads credentials from the environment: a Hugging Face token from
    ``HF_TOKEN`` (for gated models and datasets), and ``OPENAI_API_KEY`` for OpenAI models
    in concept generation.


Next Steps
----------

- Browse the full :doc:`Data API reference </modules/data_api>`.
- Label a dataset that has no concepts with :doc:`Concept Generation <using_generation>`.
- See the :doc:`data examples </auto_examples/data/index>`.
