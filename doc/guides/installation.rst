Installation
------------

Basic Installation
^^^^^^^^^^^^^^^^^^

You can install PyC with core dependencies from `PyPI <https://pypi.org/project/pytorch-concepts/>`_:

.. code-block:: bash

   pip install --pre pytorch-concepts

This will install the core library without data-related dependencies (opencv-python, pgmpy, bnlearn, pandas, torchvision, datasets, transformers).

Installation with Data Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to use the ``torch_concepts.data`` module, install with the data extras:

.. code-block:: bash

   pip install --pre pytorch-concepts[data]

This will install all dependencies including those required for data loading and preprocessing.

Installation with Full Support (Conda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For full support including all dependencies for development, experiments, and testing, use the provided conda environment:

.. code-block:: bash

   git clone https://github.com/pyc-team/pytorch_concepts.git
   cd pytorch_concepts
   # install and activate conda environment (use environment_silicon.yaml for Apple Silicon chips)
   conda env create -f conceptarium/environment.yaml
   conda activate conceptarium
   # install pyc in editable mode
   pip install -e .

This setup is recommended for contributors and users who want access to all functionalities.

Usage
^^^^^

After installation, you can import it in your Python scripts as:

.. code-block:: python

   import torch_concepts as pyc
