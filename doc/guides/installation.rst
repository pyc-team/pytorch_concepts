Installation
------------

Requirements
^^^^^^^^^^^^

PyC needs Python 3.10 or newer and PyTorch 2.6 or newer. pip installs a recent PyTorch
automatically; for a specific build (CPU-only, or a given CUDA version), install PyTorch
first by following the `PyTorch instructions <https://pytorch.org/get-started/locally/>`_.

Install with pip
^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install --pre "pytorch-concepts[data]"

The core library alone (``pip install --pre pytorch-concepts``) covers annotations, layers,
probabilistic models and ready-made models. Optional extras add the rest:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Adds
   * - ``data``
     - the datasets of ``torch_concepts.data``, pretrained backbones and concept generation
   * - ``conceptarium``
     - Hydra and W&B, to run :doc:`Conceptarium <using_conceptarium>` experiments
   * - ``tests``
     - pytest, to run the test suite
   * - ``docs``
     - Sphinx and its extensions, to build this documentation

Extras can be combined, e.g. ``pip install --pre "pytorch-concepts[data,conceptarium]"``. The
quotes stop shells such as zsh from expanding the brackets.

Development install with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To work on PyC, or to run its examples and Conceptarium, clone the repository and create the
conda environment. It installs PyC in editable mode with the ``data``, ``tests`` and
``conceptarium`` extras:

.. code-block:: bash

   git clone https://github.com/pyc-team/pytorch_concepts.git
   cd pytorch_concepts
   conda env create -f environment.yml
   conda activate pyc

PyTorch comes from PyPI: the CUDA build on Linux, the MPS build on macOS. For a CPU-only or a
specific CUDA build, reinstall it afterwards as shown in the
`PyTorch instructions <https://pytorch.org/get-started/locally/>`_. Without conda, the same
install is ``pip install -e ".[data,tests,conceptarium]"`` from the repository root.

Check the installation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -c "import torch_concepts as pyc; print(pyc.__version__)"

Datasets are downloaded on first use into ``~/.cache/pyc``; see :doc:`Datasets <using_data>` to
change the location.
