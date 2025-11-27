Installation
------------

Basic Installation
^^^^^^^^^^^^^^^^^^

You can install PyC with core dependencies from `PyPI <https://pypi.org/project/pytorch-concepts/>`_:

.. code-block:: bash

   pip install pytorch-concepts

This will install the core library without data-related dependencies (opencv-python, pgmpy, bnlearn, pandas, torchvision, datasets, transformers).

Installation with Data Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to use the ``torch_concepts.data`` module, install with the data extras:

.. code-block:: bash

   pip install pytorch-concepts[data]

This will install all dependencies including those required for data loading and preprocessing.

Usage
^^^^^

After installation, you can import it in your Python scripts as:

.. code-block:: python

   import torch_concepts as pyc
