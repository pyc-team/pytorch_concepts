.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

.. |pl_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/lightning.svg
    :width: 20px
    :align: middle

.. |conceptarium_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/conceptarium.svg
   :width: 20px
   :align: middle


User Guide
==========

Welcome to the |pyc_logo| PyC User Guide! This guide walks you through building
interpretable and causally transparent deep learning models with PyTorch Concepts.


Three API Levels
----------------

|pyc_logo| PyC exposes **three API levels**. They share the same primitives but offer
increasing amounts of abstraction, so you can pick the entry point that matches your
background and how much control you want over the model.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - API level
     - Best for
     - What you work with
   * - :doc:`Low-Level API <using_low_level>`
     - Users comfortable with plain |pytorch_logo| PyTorch
     - Composable interpretable layers (encoders, predictors) wired together by hand.
   * - :doc:`Mid-Level API <using_mid_level>`
     - Users who think in probabilistic and causal models
     - Random variables, factors and inference engines that form a probabilistic graphical model.
   * - :doc:`High-Level API <using_high_level>`
     - Users who want state-of-the-art models out of the box
     - Pre-built models trained in one line with |pl_logo| PyTorch Lightning.

Choose your entry point based on your experience:

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`code;1em;sd-text-primary` Pure PyTorch user?
        :link: using_low_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the **Low-Level API** to build models from basic interpretable layers.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Probabilistic / causal modeling user?
        :link: using_mid_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the **Mid-Level API** to build custom probabilistic and causal models.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Want models out-of-the-box?
        :link: using_high_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the **High-Level API** to use pre-defined models with one line of code.

.. grid:: 1
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  |conceptarium_logo| No experience with programming, or benchmarking at scale?
        :link: using_conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use |conceptarium_logo| **Conceptarium**, a no-code framework built on top of
        |pyc_logo| PyC for running large-scale experiments on concept-based models.


How the levels relate
---------------------

The three levels are layered on top of each other:

- The **High-Level API** builds its models out of **Mid-Level** probabilistic models.
- The **Mid-Level API** parameterises its factors with **Low-Level** interpretable layers.
- The **Low-Level API** is plain |pytorch_logo| PyTorch, so every layer composes with the
  rest of the ecosystem.

Because of this, you can mix and match: drop a low-level layer into a high-level model, or
reuse a mid-level inference engine inside your own training loop.


Each of the following pages opens with a diagram of the API level, explains its core
building blocks, and shows how to use each of them.

.. toctree::
   :maxdepth: 2
   :hidden:

   using_low_level
   using_mid_level
   using_high_level
   using_conceptarium
