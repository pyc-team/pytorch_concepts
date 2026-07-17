.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

Contributing Guide
==================

We welcome contributions to |pyc_logo| PyC! This guide covers the workflow and standards for contributing to the library.
The links below walk you through contributing a specific new component — a model, dataset, loss, or metric.

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`stack;1em;sd-text-primary` New Layer
        :link: contributing_layer
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Add a new semantics-aware Low-Level API layer.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` New Model
        :link: contributing_model
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Add a new concept-based model to the High-Level API.

    .. grid-item-card::  :octicon:`database;1em;sd-text-primary` New Dataset
        :link: contributing_dataset
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Add a new dataset and datamodule with concept annotations.

    .. grid-item-card::  :octicon:`flame;1em;sd-text-primary` New Loss
        :link: contributing_loss
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Add a custom loss term compatible with :class:`~torch_concepts.nn.ConceptLoss`.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` New Metric
        :link: contributing_metric
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Add a custom metric compatible with :class:`~torch_concepts.nn.ConceptMetrics`.

.. toctree::
   :hidden:

   contributing_layer
   contributing_model
   contributing_dataset
   contributing_loss
   contributing_metric


Join Our Community
------------------

Have questions or want to discuss your ideas? Join our Slack community to connect with other contributors and maintainers!

.. image:: https://img.shields.io/badge/Slack-Join%20Us-4A154B?style=for-the-badge&logo=slack
   :target: https://join.slack.com/t/pyc-yu37757/shared_invite/zt-3jdcsex5t-LqkU6Plj5rxFemh5bRhe_Q
   :alt: Slack


How to Contribute
-----------------

1. **Fork the repository** — Create your own fork of the `PyC repository <https://github.com/pyc-team/pytorch_concepts>`_ on GitHub.

2. **Create your branch from** ``dev`` — Clone your fork and create a new branch based on the upstream ``dev`` branch to track the latest changes:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/pytorch_concepts.git
      cd pytorch_concepts
      git remote add upstream https://github.com/pyc-team/pytorch_concepts.git
      git fetch upstream
      git checkout -b my-feature upstream/dev

3. **Make your changes** — Implement your changes locally with clear, descriptive commit messages. Use `Gitmoji <https://gitmoji.dev/>`_ for better clarity (e.g., ``✨ Add new feature``).

4. **Write documentation & tests** — Update docstrings and ``.rst`` files, add unit tests, and verify all tests pass locally.

5. **Submit a Pull Request** — Push your branch and open a PR to the ``dev`` branch:


Development Setup
-----------------

Prerequisites
^^^^^^^^^^^^^

- Python 3.9 or higher
- PyTorch (latest stable version)

Installation
^^^^^^^^^^^^
For development, you may want to install PyC in editable mode and have the complete conda environment with all dependencies to test all functionalities:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/pytorch_concepts.git
   cd pytorch_concepts
   # install and activate conda environment (use environment_silicon for Apple Silicon chips)
   conda env create -f conceptarium/environment.yaml
   conda activate conceptarium
   # install your local pyc
   pip install -e .


Reporting Issues
----------------

If you find a bug or have a feature request, please open an issue on our `GitHub Issues page <https://github.com/pyc-team/pytorch_concepts/issues>`_ using the appropriate issue template.

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, PyTorch version, OS, etc.)


Code Style
----------

Please follow these guidelines when contributing code:

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **PEP 8**
     - Follow `PEP 8 <https://pep8.org/>`_ style guidelines for Python code.
   * - **Type hints**
     - Use type hints where appropriate to improve code clarity.
   * - **Docstrings**
     - Write clear docstrings for all public functions and classes.
   * - **Tests**
     - Write tests for new features and bug fixes.
   * - **Documentation**
     - Update documentation to reflect your changes.


Thank You!
----------

Every contributor helps make PyC better. We appreciate your time and effort!

Thanks to all our contributors! 🧡

.. image:: https://contrib.rocks/image?repo=pyc-team/pytorch_concepts
   :target: https://github.com/pyc-team/pytorch_concepts/graphs/contributors
   :alt: Contributors

External Contributors
^^^^^^^^^^^^^^^^^^^^^^

- `Sonia Laguna <https://sonialagunac.github.io/>`_, ETH Zurich (CH).
- `Moritz Vandenhirtz <https://mvandenhi.github.io/>`_, ETH Zurich (CH).