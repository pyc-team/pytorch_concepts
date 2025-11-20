Conceptarium
===================

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




|conceptarium_logo| **Conceptarium** is a high-level experimentation framework for running large-scale experiments on concept-based deep learning models. Built on top of |pyc_logo| PyC, it provides:

- **Configuration-driven experiments**: Use |hydra_logo| `Hydra <https://hydra.cc/>`_ for flexible YAML-based configuration management and run sequential experiments on multiple |pyc_logo| PyC datasets and models with a single command.
- **Automated training**: Leverage |pl_logo| `PyTorch Lightning <https://lightning.ai/pytorch-lightning>`_ for streamlined training loops
- **Experiment tracking**: Integrated |wandb_logo| `Weights & Biases <https://wandb.ai/>`_ logging for monitoring and reproducibility

**Get Started**: Check out the `Conceptarium README <../../conceptarium/README.md>`_ for installation, configuration details, and tutorials on implementing custom models and datasets.

**Quick Example**:

.. code-block:: bash

    # Clone the PyC repository
    git clone https://github.com/pyc-team/pytorch_concepts.git
    cd pytorch_concepts/conceptarium

    # Run a sweep over models and datasets
    python run_experiment.py --config_name your_sweep.yaml
