.. image:: _static/img/pyc_logo_transparent.png
   :class: index-logo-cropped
   :width: 60%
   :align: center

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

.. |hydra_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/hydra-head.svg
   :width: 20px
   :align: middle

.. |pl_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/lightning.svg
    :width: 20px
    :align: middle

.. |wandb_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/wandb.svg
   :width: 20px
   :align: middle

.. |conceptarium_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/conceptarium.svg
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

.. grid:: 1 1 1 1
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`download;1em;sd-text-primary` Installation
        :link: guides/installation
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Install |pyc_logo| PyC and set up your environment.


User Guide
----------

|pyc_logo| PyC exposes **three API levels** that build on top of one another.

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`gear;1em;sd-text-primary` Semantic primitives and Interventions
        :link: guides/using_low_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        **(Low-level)**

        Extend |pytorch_logo| PyTorch tensors with concept annotations and build
        semantics-aware layers. Use Interventions to steer concepts and mechanisms.

    .. grid-item-card::  :octicon:`workflow;1em;sd-text-primary` Interpretable Probabilistic Models
        :link: guides/using_mid_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        **(Mid-level)**

        Build interpretable probabilistic graphical models from concept variables
        and neural factors. Run probabilistic inferences over them.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Out-of-the-box Models
        :link: guides/using_high_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        **(High-level)**

        Use state-of-the-art concept-based models with one line of code. These models 
        can be trained with |pytorch_logo| PyTorch loops or automatically with |pl_logo| Lightning.

.. grid:: 1
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  |conceptarium_logo| Benchmarking at scale
        :link: guides/using_conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use |conceptarium_logo| **Conceptarium**, a configuration-based framework built on top of
        |pyc_logo| PyC and |hydra_logo| Hydra for running large-scale experiments.

        **Best for:** no-code benchmarking, large experiment grids.


API Reference
-------------

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`tools;1em;sd-text-primary` Low-Level API
        :link: modules/low_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Layers, annotations, annotated tensors, interventions, functionals.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Mid-Level API
        :link: modules/mid_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Variables, factors, probabilistic models, inference engines.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` High-Level API
        :link: modules/high_level_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Ready-to-use models, losses, and metrics with Lightning support.

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`flame;1em;sd-text-primary` Loss Functions
        :link: modules/nn.loss
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Type-aware concept losses and regularizers.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Metrics
        :link: modules/nn.metrics
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Type-aware concept metrics, torchmetrics-compatible.

    .. grid-item-card::  :octicon:`gear;1em;sd-text-primary` Functional
        :link: modules/nn.functional
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Stateless operations: CaCE, NCC, differentiable selection.

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`database;1em;sd-text-primary` Data API
        :link: modules/data_api
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Datasets, datamodules, scalers, and splitters.

    .. grid-item-card::  :octicon:`infinity;1em;sd-text-primary` Distributions API
        :link: modules/distributions
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Probability distributions for probabilistic modeling.


Contributing
--------------

We welcome contributions! See the :doc:`Contributing Guide <guides/contributing>` to get started.

Thanks to all contributors!

.. image:: https://contrib.rocks/image?repo=pyc-team/pytorch_concepts
   :target: https://github.com/pyc-team/pytorch_concepts/graphs/contributors
   :alt: Contributors


External Contributors
^^^^^^^^^^^^^^^^^^^^^^

- `Sonia Laguna <https://sonialagunac.github.io/>`_, ETH Zurich (CH).
- `Moritz Vandenhirtz <https://mvandenhi.github.io/>`_, ETH Zurich (CH).



Cite this library
-----------------

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
