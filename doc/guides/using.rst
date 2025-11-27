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


User Guide
==========

Welcome to the |pyc_logo| PyC User Guide! This guide will help you get started with PyTorch Concepts and build interpretable deep learning models.


Explore Based on Your Background
--------------------------------

|pyc_logo| PyC is designed to accommodate users with different backgrounds and expertise levels.
Pick the best entry point based on your experience:

.. grid:: 1 1 3 3
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`code;1em;sd-text-primary` Pure torch user?
        :link: using_low_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Low-Level API to build models from basic interpretable layers.

    .. grid-item-card::  :octicon:`graph;1em;sd-text-primary` Probabilistic modeling user?
        :link: using_mid_level_proba
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Mid-Level API to build custom probabilistic models.

    .. grid-item-card::  :octicon:`workflow;1em;sd-text-primary` Causal modeling user?
        :link: using_mid_level_causal
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the Mid-Level API to build Structural Equation Models for causal inference.

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Just want to use state-of-the-art models out-of-the-box?
        :link: using_high_level
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Start from the High-Level API to use pre-defined models with one line of code.

    .. grid-item-card::  :octicon:`beaker;1em;sd-text-primary` No experience with programming?
        :link: using_conceptarium
        :link-type: doc
        :shadow: lg
        :class-card: sd-border-primary

        Use |conceptarium_logo| Conceptarium, a no-code framework built on top of |pyc_logo| PyC for running large-scale experiments on concept-based models.



Quick Start Example
-------------------

Here's a minimal example using the low-Level API:

.. code-block:: python

   import torch
   import torch_concepts as pyc

   # Create a concept bottleneck model
   model = torch.nn.ModuleDict({
       'encoder': pyc.nn.LinearZC(
           in_features=64,
           out_features=10
       ),
       'predictor': pyc.nn.LinearCC(
           in_features_endogenous=10,
           out_features=5
       ),
   })

   # Forward pass
   x = torch.randn(32, 64)
   concepts = model['encoder'](input=x)
   predictions = model['predictor'](endogenous=concepts)

For complete examples with training, interventions, and evaluation, see the individual API guides above.

Additional Resources
--------------------

**Examples**
    Check out `complete examples <https://github.com/pyc-team/pytorch_concepts/tree/master/examples>`_ for real-world use cases.

Need Help?
----------

- **Issues**: `GitHub Issues <https://github.com/pyc-team/pytorch_concepts/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/pyc-team/pytorch_concepts/discussions>`_
- **Contributing**: :doc:`Contributor Guide </guides/contributing>`


.. toctree::
   :maxdepth: 2
   :hidden:

   using_low_level
   using_mid_level_proba
   using_mid_level_causal
   using_high_level
   using_conceptarium
