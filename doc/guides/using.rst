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

.. |hydra_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/hydra-head.svg
   :width: 20px
   :align: middle


User Guide
==========

Welcome to the |pyc_logo| PyC User Guide! This guide walks you through building
interpretable and causally transparent deep learning models with PyTorch Concepts.


Three Levels of Control and Abstraction
----------------

|pyc_logo| PyC exposes **three API levels**. They share the same primitives but offer
increasing amounts of abstraction, and they build on top of one another.

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`gear;1em;sd-text-primary` Semantic primitives and Interventions
        :link: using_low_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        **(Low-level)**

        Extend |pytorch_logo| PyTorch tensors with concept annotations and build semantics-aware layers. 
        Use Interventions to steer concepts and mechanisms.

        **Best for:** pure PyTorch users, research in interpretable modules.

    .. grid-item-card::  :octicon:`workflow;1em;sd-text-primary` Interpretable Probabilistic Models
        :link: using_mid_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        **(Mid-level)**

        Build interpretable probabilistic graphical models from concept variables 
        and neural factors. Run probabilistic inferences over it.

        **Best for:** users who think in probabilistic terms, research in interpretable architectures.

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Out-of-the-box Models
        :link: using_high_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        **(High-level)**

        Use state-of-the-art concept-based models with one line of code. These models 
        can be trained with |pytorch_logo| PyTorch loops or automatically with |pl_logo| Lightning.
    
        **Best for:** users who just want a model that works out of the box.

.. grid:: 1
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  |conceptarium_logo| Benchmarking at scale
        :link: using_conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use |conceptarium_logo| **Conceptarium**, a configuration-based framework built on top of
        |pyc_logo| PyC and |hydra_logo| Hydra for running large-scale experiments.

        **Best for:** no experience with programming, benchmarking with just configurations.


.. toctree::
   :maxdepth: 2
   :hidden:

   using_low_level
   using_mid_level
   using_high_level
   using_conceptarium
