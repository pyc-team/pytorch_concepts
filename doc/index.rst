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
The library provides primitives for interpretable layers, probabilistic models, causal models, and APIs for running experiments at scale.

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

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`code;1em;sd-text-primary` Pure torch user?
        :link: guides/using_low_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Low-Level API to build models from basic interpretable layers.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Probabilistic modeling user?
        :link: guides/using_mid_level_proba
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Mid-Level API to build custom probabilistic models.

    .. grid-item-card::  :octicon:`workflow;1em;sd-text-primary` Causal modeling user?
        :link: guides/using_mid_level_causal
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Mid-Level API to build Structural Equation Models for causal inference.

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Just want to use state-of-the-art models out-of-the-box?
        :link: guides/using_high_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the High-Level API to use pre-defined models with one line of code.

    .. grid-item-card::  :octicon:`beaker;1em;sd-text-primary` Benchmarking or no experience with programming?
        :link: guides/using_conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use |conceptarium_logo| Conceptarium, a no-code framework built on top of |pyc_logo| PyC for running large-scale experiments on concept-based models.


API Reference
-------------

Main Modules
^^^^^^^^^^^^^^^

The main modules of the library are organized into three levels of abstraction: Low-Level API, Mid-Level API, and High-Level API.
These modules allow users with different levels of abstraction to build interpretable models.

.. grid:: 1 1 2 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`tools;1em;sd-text-primary` Low-Level API
        :link: modules/low_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Build architectures from basic interpretable layers in a plain |pytorch_logo| PyTorch-like interface.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-danger` Mid-Level API
        :link: modules/mid_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-danger

        Build custom interpretable and causally transparent probabilistic models.

        .. warning::

           This API is still under development and interfaces might change in future releases.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` High-Level API
        :link: modules/high_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use out-of-the-box state-of-the-art |pl_logo| PyTorch Lightning models with one line of code.


Shared Modules
^^^^^^^^^^^^^^^^^

The library also includes shared modules that provide additional functionalities such as loss functions, metrics, and utilities.

.. grid:: 1 1 2 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`flame;1em;sd-text-primary` Loss Functions
        :link: modules/nn.loss
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Various loss functions for concept-based models.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Metrics
        :link: modules/nn.metrics
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Evaluation metrics for concept-based models.

    .. grid-item-card::  :octicon:`gear;1em;sd-text-primary` Functional
        :link: modules/nn.functional
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Functional utilities for concept-based models.


Conceptarium
-------------

Conceptarium is a no-code framework for running large-scale experiments on concept-based models.
The interface is based on YAML configuration files, making it easy to set up and run experiments without writing code.
This framework is intended for benchmarking or researchers in other fields who want to use concept-based models without programming knowledge.

.. grid:: 1
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  |conceptarium_logo| Conceptarium
        :link: guides/using_conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Conceptarium is a no-code framework for running large-scale experiments on concept-based models. Built on top of |pyc_logo| PyC, with |pl_logo| PyTorch Lightning, |hydra_logo| Hydra and |wandb_logo| WandB.


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
        :link: modules/distributions
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Work with probability distributions for probabilistic modeling.


Contributing
--------------
We welcome contributions from the community to help improve |pyc_logo| PyC!
Follow the instructions in the `Contributing Guide <guides/contributing.html>`_ to get started.

Thanks to all contributors! 🧡

.. image:: https://contrib.rocks/image?repo=pyc-team/pytorch_concepts
   :target: https://github.com/pyc-team/pytorch_concepts/graphs/contributors
   :alt: Contributors


External Contributors
^^^^^^^^^^^^^^^^^^^^^^

- `Sonia Laguna <https://sonialagunac.github.io/>`_, ETH Zurich (CH).
- `Moritz Vandenhirtz <https://mvandenhi.github.io/>`_, ETH Zurich (CH).



Cite this library
----------------

If you found this library useful for your research article, blog post, or product, we would be grateful if you would cite it using the following bibtex entry:

{% raw %}

.. code-block:: bibtex

   @software{pycteam2025concept,
       author = {Barbiero, Pietro and De Felice, Giovanni and Espinosa Zarlenga, Mateo and Ciravegna, Gabriele and Dominici, Gabriele and De Santis, Francesco and Casanova, Arianna and Debot, David and Giannini, Francesco and Diligenti, Michelangelo and Marra, Giuseppe},
       license = {Apache 2.0},
       month = {3},
       title = {{PyTorch Concepts}},
       url = {https://github.com/pyc-team/pytorch_concepts},
       year = {2025}
   }

{% endraw %}

Reference authors: `Pietro Barbiero <http://www.pietrobarbiero.eu/>`_, `Giovanni De Felice <https://gdefe.github.io/>`_, and `Mateo Espinosa Zarlenga <https://hairyballtheorem.com/>`_.


Funding
-------

This project is supported by the following organizations:

.. raw:: html

   <div class="funding-carousel-container">
      <div class="funding-carousel-track">
         <div class="funding-logo-item">
            <img src="_static/img/funding/fwo_kleur.png" alt="FWO - Research Foundation Flanders">
         </div>
         <div class="funding-logo-item">
            <img src="_static/img/funding/hasler.png" alt="Hasler Foundation">
         </div>
         <div class="funding-logo-item">
            <img src="_static/img/funding/snsf.png" alt="SNSF - Swiss National Science Foundation">
         </div>
         <!-- Duplicate logos for seamless loop -->
         <div class="funding-logo-item">
            <img src="_static/img/funding/fwo_kleur.png" alt="FWO - Research Foundation Flanders">
         </div>
         <div class="funding-logo-item">
            <img src="_static/img/funding/hasler.png" alt="Hasler Foundation">
         </div>
         <div class="funding-logo-item">
            <img src="_static/img/funding/snsf.png" alt="SNSF - Swiss National Science Foundation">
         </div>
      </div>
   </div>


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
   modules/nn.loss
   modules/nn.metrics
   modules/nn.functional
   modules/data_api
   modules/distributions

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Indices
   :hidden:

   genindex
   py-modindex
