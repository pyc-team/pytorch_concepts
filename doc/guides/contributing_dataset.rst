.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Contributing a New Dataset
==========================

This guide explains how to add a new dataset to |pyc_logo| PyC.
Every dataset in PyC follows the same four-method contract: ``download`` fetches raw
files, ``build`` processes them into tensors, ``load_raw`` reads the processed files
from disk, and ``load`` adds any final preprocessing before handing data to the rest
of the library. The caching layer in the base class means each step runs only once
unless the on-disk files are missing.


.. dropdown:: Dataset Class
   :icon: database

   **Inheritance**

   Every dataset extends :class:`~torch_concepts.data.base.ConceptDataset`, which
   is found in ``torch_concepts.data.base``::

       ConceptDataset  (torch.utils.data.Dataset)
       └── YourDataset

   The base class owns ``input_data``, ``concepts``, ``annotations``, and ``graph``.
   Your job is to populate them by calling ``super().__init__`` at the end of your
   ``__init__``.

   **The four abstract methods**

   +-------------------------+-------------------------------------------------------+
   | Method / property       | What it must do                                       |
   +=========================+=======================================================+
   | ``raw_filenames``       | Return a list of raw file names (empty if nothing     |
   |                         | needs to be downloaded).                              |
   +-------------------------+-------------------------------------------------------+
   | ``processed_filenames`` | Return a list of processed file names written by      |
   |                         | ``build()``. The base class uses these to decide      |
   |                         | whether ``build()`` needs to run.                     |
   +-------------------------+-------------------------------------------------------+
   | ``download()``          | Fetch raw files into ``self.root_dir``. Use           |
   |                         | ``download_url(url, self.root_dir)`` from             |
   |                         | ``torch_concepts.data.io``.                           |
   +-------------------------+-------------------------------------------------------+
   | ``build()``             | Read/generate data, create :class:`~torch_concepts.   |
   |                         | Annotations`, and save everything via                 |
   |                         | ``torch.save`` / ``df.to_hdf``.                       |
   +-------------------------+-------------------------------------------------------+
   | ``load_raw()``          | Call ``self.maybe_build()``, then load every          |
   |                         | processed file and return                             |
   |                         | ``(inputs, concepts, annotations, graph)``.           |
   +-------------------------+-------------------------------------------------------+
   | ``load()``              | Call ``load_raw()``, add any extra preprocessing,     |
   |                         | return the same four-tuple.                           |
   +-------------------------+-------------------------------------------------------+

   The caching pipeline is: ``load()`` → ``load_raw()`` → ``maybe_build()`` →
   ``build()`` → ``maybe_download()`` → ``download()``. You never call these helpers
   yourself; the base class does.

   **Complete worked example**

   The dataset below is a minimal but real example. It generates a synthetic dataset
   with three concept types (binary, categorical, continuous), a downstream task, and
   a causal graph. It can serve as a copy-paste starting point.

   .. code-block:: python

      # torch_concepts/data/datasets/my_dataset.py
      import os
      import logging
      import pandas as pd
      import torch
      from typing import List, Optional

      from torch_concepts.annotations import Annotations
      from torch_concepts.data.base import ConceptDataset

      logger = logging.getLogger(__name__)


      class MyDataset(ConceptDataset):
          """Synthetic dataset with mixed concept types.

          Parameters
          ----------
          root : str, optional
              Directory where processed files are stored. Defaults to
              ``./data/my_dataset`` in the current working directory.
          seed : int, default 42
              Controls data generation. Also baked into processed file names so
              that different seeds produce independent caches on disk.
          n_gen : int, default 5000
              Number of samples to generate.
          concept_subset : list of str, optional
              If provided, only this subset of concepts is kept after loading.
          """

          def __init__(
              self,
              root: str = None,
              seed: int = 42,
              n_gen: int = 5000,
              concept_subset: Optional[List[str]] = None,
          ):
              self.seed = seed
              self.n_gen = n_gen

              if root is None:
                  root = os.path.join(os.getcwd(), "data", "my_dataset")
              self.root = root

              input_data, concepts, annotations, graph = self.load()

              super().__init__(
                  input_data=input_data,
                  concepts=concepts,
                  annotations=annotations,
                  graph=graph,
                  concept_names_subset=concept_subset,
                  name="MyDataset",
              )

          # ------------------------------------------------------------------
          # File lists
          # ------------------------------------------------------------------

          @property
          def raw_filenames(self) -> List[str]:
              # Nothing to download — data is generated programmatically.
              return []

          @property
          def processed_filenames(self) -> List[str]:
              # Encode seed and n_gen so different runs have independent caches.
              return [
                  f"inputs_N_{self.n_gen}_seed_{self.seed}.pt",
                  f"concepts_N_{self.n_gen}_seed_{self.seed}.h5",
                  "annotations.pt",
                  "graph.h5",
              ]

          # ------------------------------------------------------------------
          # Download (nothing to do for synthetic data)
          # ------------------------------------------------------------------

          def download(self):
              pass  # no remote files

          # ------------------------------------------------------------------
          # Build: generate data, create Annotations, save everything
          # ------------------------------------------------------------------

          def build(self):
              logger.info(f"Generating MyDataset (n={self.n_gen}, seed={self.seed})")
              torch.manual_seed(self.seed)

              # --- generate raw tensors -----------------------------------------
              n = self.n_gen

              # binary concept: 0 or 1
              smoker = torch.bernoulli(torch.full((n,), 0.4))

              # categorical concept: 3 genotype states (one-hot stored as int index)
              genotype = torch.multinomial(
                  torch.tensor([0.5, 0.3, 0.2]).expand(n, -1),
                  num_samples=1,
              ).squeeze(1).float()

              # continuous concept: tar level
              tar = torch.randn(n) * 0.5 + smoker * 2.0

              # binary task: cancer risk
              logit = smoker * 1.5 + tar * 0.8 + (genotype == 2).float() * 1.0
              cancer = torch.bernoulli(torch.sigmoid(logit))

              # input features: noisy version of the latent signal
              inputs = torch.stack(
                  [smoker, genotype, tar,
                   torch.randn(n), torch.randn(n), torch.randn(n)],
                  dim=1,
              )

              # concepts tensor: one column per concept/task
              concepts_df = pd.DataFrame({
                  "smoker":   smoker.numpy(),
                  "genotype": genotype.numpy(),
                  "tar":      tar.numpy(),
                  "cancer":   cancer.numpy(),
              })

              # --- build Annotations --------------------------------------------
              # cardinality: 1 = binary or scalar continuous,
              #              K = K-class categorical
              concept_names   = ["smoker", "genotype", "tar", "cancer"]
              cardinalities   = [1, 3, 1, 1]
              types           = ["binary", "categorical", "continuous", "binary"]

              # states: human-readable labels for each state of each concept
              # (None for continuous concepts)
              states = [
                  ["non-smoker", "smoker"],             # smoker
                  ["wild-type", "het", "hom"],          # genotype
                  None,                                  # tar (continuous)
                  ["no cancer", "cancer"],              # cancer
              ]

              # per-concept metadata (optional free-form dict)
              metadata = {
                  "smoker":   {"source": "self-report"},
                  "genotype": {"source": "WGS"},
                  "tar":      {"unit": "mg/cigarette"},
                  "cancer":   {"icd10": "C34"},
              }

              annotations = Annotations(
                  labels=concept_names,
                  cardinalities=cardinalities,
                  types=types,
                  states=states,
                  metadata=metadata,
              )

              # --- causal graph -------------------------------------------------
              # rows = causes, columns = effects; 1 means "causes"
              graph = pd.DataFrame(
                  [[0, 0, 1, 1],   # smoker -> tar, cancer
                   [0, 0, 0, 1],   # genotype -> cancer
                   [0, 0, 0, 1],   # tar -> cancer
                   [0, 0, 0, 0]],  # cancer (sink)
                  index=concept_names,
                  columns=concept_names,
              )

              # --- save all four components -------------------------------------
              os.makedirs(self.root_dir, exist_ok=True)
              torch.save(inputs, self.processed_paths[0])
              concepts_df.to_hdf(self.processed_paths[1], key="concepts", mode="w")
              torch.save(annotations, self.processed_paths[2])
              graph.to_hdf(self.processed_paths[3], key="graph", mode="w")

          # ------------------------------------------------------------------
          # Load
          # ------------------------------------------------------------------

          def load_raw(self):
              self.maybe_build()
              logger.info(f"Loading MyDataset from {self.root_dir}")

              inputs      = torch.load(self.processed_paths[0], weights_only=False)
              concepts    = pd.read_hdf(self.processed_paths[1], "concepts")
              annotations = torch.load(self.processed_paths[2], weights_only=False)
              graph       = pd.read_hdf(self.processed_paths[3], "graph")

              return inputs, concepts, annotations, graph

          def load(self):
              # load_raw already handles all the work for this dataset.
              # Override here if you need to add preprocessing (e.g., normalization,
              # autoencoder embedding extraction) on top of the raw tensors.
              return self.load_raw()

   **Notes on processed file names**

   For synthetic datasets the seed and sample count must be part of the file names
   (e.g., ``inputs_N_5000_seed_42.pt``). This ensures that changing ``seed`` or
   ``n_gen`` triggers a fresh ``build()`` rather than silently re-using an old cache.

   **Downloading a remote file**

   If your dataset comes from a URL, use ``download_url`` from
   ``torch_concepts.data.io`` inside ``download()``:

   .. code-block:: python

      from torch_concepts.data.io import download_url

      def download(self):
          url = "https://example.com/my_data.csv.gz"
          download_url(url, self.root_dir)   # saves to self.root_dir/<filename>

      @property
      def raw_filenames(self) -> List[str]:
          return ["my_data.csv.gz"]

   ``maybe_download()`` (called automatically by ``build()``) skips the download if
   all paths in ``self.raw_paths`` already exist on disk.

   **Verifying the dataset**

   After writing the class, check it interactively before registering it:

   .. code-block:: python

      from torch_concepts.data.datasets.my_dataset import MyDataset

      ds = MyDataset(seed=0, n_gen=200)
      print(ds)
      # MyDataset(n_samples=200, n_features=(6,), n_concepts=4)

      print(ds.concept_names)
      # ['smoker', 'genotype', 'tar', 'cancer']

      print(ds.annotations.types)
      # ('binary', 'categorical', 'continuous', 'binary')

      sample = ds[0]
      print(sample["inputs"]["x"].shape)   # torch.Size([6])
      print(sample["concepts"]["c"].shape) # torch.Size([4])

      print(ds.graph)


.. dropdown:: DataModule
   :icon: package

   A DataModule wraps a dataset and handles splitting, batching, and dataloaders for
   PyTorch Lightning (and plain PyTorch). Extend
   :class:`~torch_concepts.data.base.ConceptDataModule` from
   ``torch_concepts.data.base.datamodule``.

   Your ``__init__`` only needs to instantiate the dataset and call
   ``super().__init__(dataset=dataset, ...)``. Everything else — ``train_dataloader``,
   ``val_dataloader``, ``test_dataloader``, split logic — is inherited.

   .. code-block:: python

      # torch_concepts/data/datamodules/my_dataset.py
      from ..base.datamodule import ConceptDataModule
      from ..datasets.my_dataset import MyDataset


      class MyDataModule(ConceptDataModule):
          """DataModule for MyDataset.

          Parameters
          ----------
          seed : int
              Random seed for the train/val/test split (independent of the
              generation seed passed to the dataset).
          generation_seed : int, default 42
              Seed forwarded to :class:`MyDataset` for data generation.
          n_gen : int, default 5000
              Number of samples forwarded to :class:`MyDataset`.
          val_size : float or int, default 0.1
              Fraction (float) or absolute count (int) for the validation split.
          test_size : float or int, default 0.2
              Fraction (float) or absolute count (int) for the test split.
          batch_size : int, default 256
              Batch size for all dataloaders.
          workers : int, default 0
              Number of dataloader worker processes.
          """

          def __init__(
              self,
              seed: int,
              root: str = None,
              generation_seed: int = 42,
              n_gen: int = 5000,
              val_size: float = 0.1,
              test_size: float = 0.2,
              batch_size: int = 256,
              workers: int = 0,
              **kwargs,
          ):
              dataset = MyDataset(
                  root=root,
                  seed=generation_seed,
                  n_gen=n_gen,
              )

              super().__init__(
                  dataset=dataset,
                  val_size=val_size,
                  test_size=test_size,
                  batch_size=batch_size,
                  workers=workers,
                  seed=seed,
              )

   **Using the DataModule**

   .. code-block:: python

      dm = MyDataModule(seed=1, generation_seed=42, n_gen=5000, batch_size=128)
      dm.setup()

      for batch in dm.train_dataloader():
          x = batch["inputs"]["x"]       # (128, 6)
          c = batch["concepts"]["c"]     # (128, 4)
          break


.. dropdown:: Registering the Dataset
   :icon: code

   Once the dataset and DataModule work locally, register them in three places.

   **1. Module files**

   Place the dataset class in ``torch_concepts/data/datasets/my_dataset.py`` and the
   DataModule in ``torch_concepts/data/datamodules/my_dataset.py``.

   **2. Public API exports**

   Add imports and ``__all__`` entries to ``torch_concepts/data/__init__.py``:

   .. code-block:: python

      # in torch_concepts/data/__init__.py
      from .datasets.my_dataset import MyDataset
      from .datamodules.my_dataset import MyDataModule

      __all__ = [
          ...
          "MyDataset",
          "MyDataModule",
      ]

   After this, users can do ``from torch_concepts.data import MyDataset``.

   **3. API reference page** (optional but recommended)

   Add ``autoclass`` directives to ``doc/modules/data_api.rst`` so the docstrings
   appear in the rendered documentation:

   .. code-block:: rst

      .. autoclass:: torch_concepts.data.MyDataset
         :members:
         :undoc-members:
         :show-inheritance:

      .. autoclass:: torch_concepts.data.MyDataModule
         :members:
         :undoc-members:
         :show-inheritance:

   **4. Tests**

   Add a test in ``tests/`` that instantiates the dataset, checks the output shapes,
   and verifies the annotations. Mirror the existing tests in ``tests/test_data.py``
   for the expected structure.


Next Steps
----------

- Read the :class:`~torch_concepts.data.base.ConceptDataset` API reference for the
  full list of inherited properties (``n_samples``, ``n_features``, ``concept_names``,
  ``graph``, etc.).
- Explore existing datasets in ``torch_concepts/data/datasets/`` to see how different
  data sources (remote files, bnlearn graphs, real image datasets) are handled.
- See :doc:`Contributing <contributing>` for the full pull-request workflow.
