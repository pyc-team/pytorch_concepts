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
        :shadow: sm

        Learn how to install |pyc_logo| PyC and set up your environment.

    .. grid-item-card::  :octicon:`play;1em;sd-text-primary` Using PyC
        :link: guides/using
        :link-type: doc
        :shadow: sm

        Explore tutorials and examples to get started with |pyc_logo| PyC.

    .. grid-item-card::  :octicon:`code;1em;sd-text-primary` Contributing
        :link: guides/contributing
        :link-type: doc
        :shadow: sm

        Contribute to |pyc_logo| PyC and help improve the library.



API Reference
-------------

.. grid:: 1
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`beaker;1em;sd-text-primary` Conceptarium
        :link: modules/conceptarium
        :link-type: doc
        :shadow: sm

        |conceptarium_logo| Conceptarium is a no-code framework for running large-scale experiments on concept-based models. Built on top of |pyc_logo| PyC with |hydra_logo| Hydra, |pl_logo| PyTorch Lightning, and |wandb_logo| WandB.

.. grid:: 1 1 2 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`tools;1em;sd-text-primary` Low-Level API
        :link: modules/low_level_api
        :link-type: doc
        :shadow: sm

        Build architectures from basic interpretable layers in a plain |pytorch_logo| PyTorch-like interface.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Mid-Level API
        :link: modules/mid_level_api
        :link-type: doc
        :shadow: sm

        Build custom interpretable and causally transparent Probabilistic Models.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` High-Level API
        :link: modules/high_level_api
        :link-type: doc
        :shadow: sm

        Use out-of-the-box state-of-the-art models with one line of code.


.. grid:: 1 1 2 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`database;1em;sd-text-primary` Data API
        :link: modules/data_api
        :link-type: doc
        :shadow: sm

        Access datasets, dataloaders, preprocessing, and data utilities.

    .. grid-item-card::  :octicon:`infinity;1em;sd-text-primary` Distributions API
        :link: modules/distributions_api
        :link-type: doc
        :shadow: sm

        Work with probability distributions for probabilistic modeling.

    .. grid-item-card::  :octicon:`package;1em;sd-text-primary` Other Modules
        :link: modules/other_modules
        :link-type: doc
        :shadow: sm

        Explore additional utilities and helper modules.


The overall software stack of |pyc_logo| PyC is illustrated below:

.. image:: _static/img/pyc_software_stack.png
   :width: 100%
   :align: center


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