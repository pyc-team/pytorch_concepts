.. image:: _static/img/pyc_logo_transparent.png
   :class: index-logo-cropped
   :width: 60%
   :align: center

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

.. |hydra_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/hydra-head.svg
   :width: 20px
   :align: middle

.. |pl_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/lightning.svg
    :width: 20px
    :align: middle

.. |wandb_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/wandb.svg
   :width: 20px
   :align: middle

.. |conceptarium_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/conceptarium.svg
   :width: 20px
   :align: middle

|pyc_logo| PyC is a library built upon |pytorch_logo| PyTorch to easily implement **interpretable and causally transparent deep learning models**.
The library provides primitives for layers (encoders, predictors, special layers), Probabilistic Models, and APIs for running experiments at scale.

The name of the library stands for both:

**PyTorch Concepts**
    as concepts are essential building blocks for interpretable deep learning.

**P(y|C)**
    as the main purpose of the library is to support sound probabilistic modeling of the conditional distribution of targets *y* given concepts *C*.


Get Started
-----------

.. grid:: 1 1 2 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`download;1em;sd-text-primary` Installation
        :link: guides/installation
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Learn how to install |pyc_logo| PyC and set up your environment.

    .. grid-item-card::  :octicon:`play;1em;sd-text-primary` Using PyC
        :link: guides/using
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Explore tutorials and examples to get started with |pyc_logo| PyC.

    .. grid-item-card::  :octicon:`code;1em;sd-text-primary` Contributing
        :link: guides/contributing
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Contribute to |pyc_logo| PyC and help improve the library.


Explore Based on Your Background
^^^^^^^^^^^^^^^^^^^^

PyC is designed to accommodate users with different backgrounds and expertise levels.
Pick the best entry point based on your experience:

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`code;1em;sd-text-primary` Pure torch user?
        :link: modules/low_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Low-Level API to build models from basic interpretable layers.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Probabilistic modeling user?
        :link: modules/mid_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Mid-Level API to build custom Probabilistic Models.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Just want to use state-of-the-art models out-of-the-box?
        :link: modules/high_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the High-Level API to use pre-defined models with one line of code.

    .. grid-item-card::  :octicon:`beaker;1em;sd-text-primary` No experience with programming?
        :link: modules/conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use Conceptarium, a no-code framework built on top of |pyc_logo| PyC for running large-scale experiments on concept-based models.


API Reference
-------------

Main Modules
^^^^^^^^^^^^^^^

The main modules of the library are organized into three levels of abstraction: Low-Level API, Mid-Level API, and High-Level API.
These modules allow users with different levels of abstraction to build interptrable models.

.. grid:: 1 1 2 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`graph;1em;sd-text-danger` Mid-Level API
        :link: modules/mid_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-danger

        Build custom interpretable and causally transparent Probabilistic Models.

        .. warning::

           This API is still under development and interfaces might change in future releases.

    .. grid-item-card::  :octicon:`tools;1em;sd-text-primary` Low-Level API
        :link: modules/low_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Build architectures from basic interpretable layers in a plain |pytorch_logo| PyTorch-like interface.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` High-Level API
        :link: modules/high_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use out-of-the-box state-of-the-art models with one line of code.


Shared Modules
^^^^^^^^^^^^^^^^^

The library also includes shared modules that provide additional functionalities such as loss functions, metrics, and utilities.

.. grid:: 1 1 2 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`flame;1em;sd-text-primary` Loss Functions
        :link: modules/other_modules
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Various loss functions for concept-based models.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Metrics
        :link: modules/other_modules
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Evaluation metrics for concept-based models.

    .. grid-item-card::  :octicon:`package;1em;sd-text-primary` Utilities
        :link: modules/other_modules
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Helper utilities and tools for concept-based models.


Extra Modules
^^^^^^^^^^^^^^^^^

Extra modules provide additional APIs for data handling and probability distributions.
These modules have additional dependencies and can be installed separately.

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`database;1em;sd-text-primary` Data API
        :link: modules/data_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Access datasets, dataloaders, preprocessing, and data utilities.

    .. grid-item-card::  :octicon:`infinity;1em;sd-text-primary` Distributions API
        :link: modules/distributions_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Work with probability distributions for probabilistic modeling.


Conceptarium
-------------

Conceptarium is a no-code framework for running large-scale experiments on concept-based models.
The interface is based on configuration files, making it easy to set up and run experiments without writing code.
This framework is intended for benchmarking or researchers in other fields who want to use concept-based models without programming knowledge.

.. grid:: 1
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`beaker;1em;sd-text-primary` Conceptarium
        :link: modules/conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        |conceptarium_logo| Conceptarium is a no-code framework for running large-scale experiments on concept-based models. Built on top of |pyc_logo| PyC with |hydra_logo| Hydra, |pl_logo| PyTorch Lightning, and |wandb_logo| WandB.


Contributing
--------------

- Use the ``dev`` branch to write and test your contributions locally.
- Make small commits and use `Gitmoji <https://gitmoji.dev/>`_ to add emojis to your commit messages.
- Make sure to write documentation and tests for your contributions.
- Make sure all tests pass before submitting the pull request.
- Submit a pull request to the ``main`` branch.

Thanks to all contributors! 🧡

.. image:: https://contrib.rocks/image?repo=pyc-team/pytorch_concepts
   :target: https://github.com/pyc-team/pytorch_concepts/graphs/contributors
   :alt: Contributors



Cite this library
----------------

If you found this library useful for your research article, blog post, or product, we would be grateful if you would cite it using the following bibtex entry:

{% raw %}

.. code-block:: bibtex

   @software{pycteam2025concept,
       author = {Barbiero, Pietro and De Felice, Giovanni and Espinosa Zarlenga, Mateo and Ciravegna, Gabriele and Dominici, Gabriele and De Santis, Francesco and Casanova, Arianna and Debot, David and Giannini, Francesco and Diligenti, Michelangelo and Marra, Giuseppe},
       license = {MIT},
       month = {3},
       title = {{PyTorch Concepts}},
       url = {https://github.com/pyc-team/pytorch_concepts},
       year = {2025}
   }

{% endraw %}

Reference authors: `Pietro Barbiero <http://www.pietrobarbiero.eu/>`_ and `Giovanni De Felice <https://gdefe.github.io/>`_.


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   guides/installation
   guides/using
   guides/contributing
   guides/license

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules/low_level_api
   modules/mid_level_api
   modules/high_level_api
   modules/data_api
   modules/distributions_api
   modules/other_modules

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Indices
   :hidden:

   genindex
   py-modindex
