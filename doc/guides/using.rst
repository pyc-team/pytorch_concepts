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
increasing amounts of abstraction, and they build on top of one another: the high level is
assembled from mid-level probabilistic models, whose factors are in turn parameterised by
low-level layers. Pick the entry point that matches your background.

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`code;1em;sd-text-primary` Interpretable Layers and Interventions (Low)
        :link: using_low_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Assemble interpretable architectures by hand from composable encoder and predictor
        layers, and edit concept activations at inference time with interventions — all in
        plain |pytorch_logo| PyTorch. These layers are the building blocks the higher levels
        are made of.

        **Best for:** users comfortable with PyTorch who want full control over the architecture.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Interpretable Probabilistic Models (Mid)
        :link: using_mid_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Describe a model as a probabilistic graphical model — random variables, factors and
        inference engines — and run probabilistic or causal queries over it. Each factor is
        parameterised by the low-level layers.

        **Best for:** users who think in probabilistic and causal models.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Out-of-the-box Models (High)
        :link: using_high_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use state-of-the-art concept-based models with one line of code. These models are
        assembled from mid-level probabilistic models and train automatically with
        |pl_logo| PyTorch Lightning.

        **Best for:** users who just want a model that works out of the box.

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


Each of the following pages opens with a diagram of the API level, explains its core
building blocks, and shows how to use each of them.

.. toctree::
   :maxdepth: 2
   :hidden:

   using_low_level
   using_mid_level
   using_high_level
   using_conceptarium
