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


Conceptarium
============

|conceptarium_logo| **Conceptarium** is a no-code framework for running large-scale experiments on concept-based models. 
Built on top of |pyc_logo| PyC, |hydra_logo| Hydra, and |pl_logo| PyTorch Lightning, it enables configuration-driven experimentation
without writing Python code.


Design Principles
-----------------

Configuration-Driven Experimentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conceptarium uses YAML configuration files to define all experiment parameters. No Python coding required:

- **Models**: Select and configure any |pyc_logo| PyC model (CBM, CEM, CGM, BlackBox)
- **Datasets**: Use built-in datasets (CUB-200, CelebA) or add custom ones
- **Training**: Configure optimizer, scheduler, and Lightning Trainer settings
- **Tracking**: Automatic logging to |wandb_logo| W&B for visualization and comparison

Large-Scale Sweeps
^^^^^^^^^^^^^^^^^^

Run multiple experiments with single commands using |hydra_logo| Hydra's multi-run capabilities:

.. code-block:: bash

   # Test 3 datasets × 2 models × 5 seeds = 30 experiments
   python run_experiment.py dataset=celeba,cub,mnist model=cbm,cem seed=1,2,3,4,5

Or by creating custom sweep configuration files:

.. code-block:: yaml

   # conceptarium/conf/my_sweep.yaml
   defaults:
       - _commons    # Inherit standard encoder/optimizer settings
       - _self_      # This file's parameters override
   
   hydra:
       job:
           name: experiment_name
       sweeper:
           # standard grid search
           params:
               seed: 1
               dataset: celeba, cub, mnist, ...
               model: blackbox, cbm, cem, ...

All runs are automatically organized, logged, and tracked.

Hierarchical Composition
^^^^^^^^^^^^^^^^^^^^^^^^

Configurations inherit and override using ``defaults`` for maintainability:

.. code-block:: yaml

   # conceptarium/conf/my_sweep.yaml
   defaults:
     - _commons    # Inherit standard encoder/optimizer settings
     - _self_      # This file's parameters override
   
   # Only specify what's different
   model:
       optim_kwargs:
           lr: 0.05 # Override learning rate

This keeps configurations concise and reduces duplication.


Detailed Guides
^^^^^^^^^^^^^^^

.. dropdown:: Installation and Basic Usage
   :icon: rocket
   
   **Installation**
   
   Clone the repository and set up the environment:
   
   .. code-block:: bash
   
      git clone https://github.com/pyc-team/pytorch_concepts.git
      cd pytorch_concepts/conceptarium
      conda env create -f environment.yml
      conda activate conceptarium
   
   **Basic Usage**
   
   Run a single experiment with default configuration:
   
   .. code-block:: bash
   
      python run_experiment.py
   
   Run a sweep over multiple configurations:
   
   .. code-block:: bash
   
      python run_experiment.py --config-name sweep
   
   Override parameters from command line:
   
   .. code-block:: bash
   
      # Change dataset
      python run_experiment.py dataset=cub
      
      # Change model
      python run_experiment.py model=cbm_joint
      
      # Change multiple parameters
      python run_experiment.py dataset=celeba model=cbm_joint trainer.max_epochs=100
      
      # Run sweep over multiple values
      python run_experiment.py dataset=celeba,cub model=cbm_joint,blackbox seed=1,2,3,4,5

.. dropdown:: Understanding Configurations
   :icon: file-code
   
   **Configuration Structure**
   
   All configurations are stored in ``conceptarium/conf/``:
   
   .. code-block:: text
   
      conf/
      ├── _default.yaml          # Base configuration
      ├── sweep.yaml             # Example sweep configuration
      ├── dataset/               # Dataset configurations
      │   ├── _commons.yaml      # Shared dataset parameters
      │   ├── celeba.yaml        # CelebA dataset
      │   ├── cub.yaml           # CUB-200 dataset
      │   └── ...                # More datasets
      ├── loss/                  # Loss function configs
      │   ├── standard.yaml      # Type-aware losses
      │   └── weighted.yaml      # Weighted losses
      ├── metrics/               # Metric configs
      │   └── standard.yaml      # Type-aware metrics
      └── model/                 # Model configurations
          ├── _commons.yaml      # Shared model parameters
          ├── blackbox.yaml      # Black-box baseline
          ├── cbm.yaml           # Alias for cbm_joint
          └── cbm_joint.yaml     # CBM (joint training)
   
   **Configuration Hierarchy**
   
   Configurations use |hydra_logo| Hydra's composition system with ``defaults`` to inherit and override:
   
   .. code-block:: yaml
   
      # conf/model/cbm_joint.yaml
      defaults:
        - _commons              # Inherit common model parameters
        - _self_                # Current file takes precedence
      
      # Model-specific configuration
      _target_: torch_concepts.nn.ConceptBottleneckModel_Joint
      task_names: ${dataset.default_task_names}
      
      inference:
        _target_: torch_concepts.nn.DeterministicInference
        _partial_: true
   
   **Priority**: Parameters defined later override earlier ones. ``_self_`` controls where current file's parameters fit in the hierarchy.
   
   **Base Configuration**
   
   The ``_default.yaml`` file contains base settings for all experiments:
   
   .. code-block:: yaml
   
      defaults:
        - dataset: cub
        - model: cbm_joint
        - _self_
   
      seed: 42
      
      trainer:
        max_epochs: 500
        patience: 30
        monitor: "val_loss"
        mode: "min"
      
      wandb:
        project: conceptarium
        entity: your-team
        log_model: false
   
   **Key sections**:
   
   - ``defaults``: Which dataset and model configurations to use
   - ``seed``: Random seed for reproducibility
   - ``trainer``: PyTorch Lightning Trainer settings
   - ``wandb``: Weights & Biases logging configuration

.. dropdown:: Working with Datasets
   :icon: database
   
   **Dataset Configuration Files**
   
   Each dataset has a YAML file in ``conf/dataset/`` that specifies:
   
   1. The datamodule class (``_target_``)
   2. Dataset-specific parameters
   3. Backbone architecture (if needed)
   4. Preprocessing settings
   
   **Example - CUB-200 Dataset**
   
   .. code-block:: yaml
   
      # conf/dataset/cub.yaml
      defaults:
        - _commons
        - _self_
   
      _target_: torch_concepts.data.datamodules.CUBDataModule
      
      name: cub
      
      # Backbone for feature extraction
      backbone:
        _target_: torchvision.models.resnet18
        pretrained: true
      
      precompute_embs: true  # Precompute features to speed up training
      
      # Task variables to predict
      default_task_names: [bird_species]
      
      # Concept descriptions (optional, for interpretability)
      label_descriptions:
        - has_wing_color::blue: Wing color is blue
        - has_upperparts_color::blue: Upperparts color is blue
        - has_breast_pattern::solid: Breast pattern is solid
        - has_back_color::brown: Back color is brown
   
   **Example - CelebA Dataset**
   
   .. code-block:: yaml
   
      # conf/dataset/celeba.yaml
      defaults:
        - _commons
        - _self_
   
      _target_: torch_concepts.data.datamodules.CelebADataModule
      
      name: celeba
      
      backbone:
        _target_: torchvision.models.resnet18
        pretrained: true
      
      precompute_embs: true
      
      # Predict attractiveness from facial attributes
      default_task_names: [Attractive]
      
      label_descriptions:
        - Smiling: Person is smiling
        - Male: Person is male
        - Young: Person is young
        - Eyeglasses: Person wears eyeglasses
        - Attractive: Person is attractive
   
   **Common Dataset Parameters**
   
   Defined in ``conf/dataset/_commons.yaml``:
   
   .. code-block:: yaml
   
      batch_size: 256          # Training batch size
      val_size: 0.15           # Validation split fraction
      test_size: 0.15          # Test split fraction
      num_workers: 4           # DataLoader workers
      pin_memory: true         # Pin memory for GPU
      
      # Optional: Subsample concepts
      concept_subset: null     # null = use all concepts
      # concept_subset: [concept1, concept2, concept3]
   
   **Overriding Dataset Parameters**
   
   From command line:
   
   .. code-block:: bash
   
      # Change batch size
      python run_experiment.py dataset.batch_size=512
      
      # Use only specific concepts
      python run_experiment.py dataset.concept_subset=[has_wing_color::blue,has_back_color::brown]
      
      # Change validation split
      python run_experiment.py dataset.val_size=0.2
   
   In a custom sweep file:
   
   .. code-block:: yaml
   
      # conf/my_sweep.yaml
      defaults:
        - _default
        - _self_
      
      dataset:
        batch_size: 512
        val_size: 0.2

.. dropdown:: Working with Models
   :icon: cpu
   
   **Model Configuration Files**
   
   Each model has a YAML file in ``conf/model/`` that specifies:
   
   1. The model class (``_target_``)
   2. Architecture parameters (from ``_commons.yaml``)
   3. Inference strategy
   4. Metric tracking options
   
   **Example - Concept Bottleneck Model**
   
   .. code-block:: yaml
   
      # conf/model/cbm_joint.yaml
      defaults:
        - _commons
        - _self_
      
      _target_: torch_concepts.nn.ConceptBottleneckModel_Joint
      
      # Task variables (from dataset)
      task_names: ${dataset.default_task_names}
      
      # Inference strategy
      inference:
        _target_: torch_concepts.nn.DeterministicInference
        _partial_: true
      
      # Metric tracking
      summary_metrics: true      # Aggregate metrics by concept type
      perconcept_metrics: false  # Per-concept individual metrics
   
   **Example - Black-box Baseline**
   
   .. code-block:: yaml
   
      # conf/model/blackbox.yaml
      defaults:
        - _commons
        - _self_
      
      _target_: torch_concepts.nn.BlackBox
      
      task_names: ${dataset.default_task_names}
      
      # Black-box models don't use concepts
      inference: null
      
      summary_metrics: false
      perconcept_metrics: false
   
   **Common Model Parameters**
   
   Defined in ``conf/model/_commons.yaml``:
   
   .. code-block:: yaml
   
      # Encoder architecture
      encoder_kwargs:
        hidden_size: 128       # Hidden layer dimension
        n_layers: 2            # Number of hidden layers
        activation: relu       # Activation function
        dropout: 0.1           # Dropout probability
      
      # Concept distributions (how concepts are modeled)
      variable_distributions:
        binary: torch.distributions.Bernoulli
        categorical: torch.distributions.Categorical
      
      # Optimizer configuration
      optim_class:
        _target_: torch.optim.AdamW
        _partial_: true
      
      optim_kwargs:
        lr: 0.00075            # Learning rate
        weight_decay: 0.0      # L2 regularization
      
      # Learning rate scheduler
      scheduler_class:
        _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
        _partial_: true
      
      scheduler_kwargs:
        mode: min
        factor: 0.5
        patience: 10
        min_lr: 0.00001
   
   **Loss Configuration**
   
   Loss functions are type-aware, automatically selecting the appropriate loss based on concept types.
   Loss configurations are in ``conf/loss/``:
   
   **Standard losses** (``conf/loss/standard.yaml``):
   
   .. code-block:: yaml
   
      _target_: torch_concepts.nn.ConceptLoss
      _partial_: true
      
      fn_collection:
        discrete:
          binary:
            path: torch.nn.BCEWithLogitsLoss
            kwargs: {}
          categorical:
            path: torch.nn.CrossEntropyLoss
            kwargs: {}
        # continuous:  # Not yet supported
        #   path: torch.nn.MSELoss
        #   kwargs: {}
   
   **Weighted losses** (``conf/loss/weighted.yaml``):
   
   .. code-block:: yaml
   
      _target_: torch_concepts.nn.ConceptLoss
      _partial_: true
      
      fn_collection:
        discrete:
          binary:
            path: torch.nn.BCEWithLogitsLoss
            kwargs:
              reduction: none  # Required for weighting
          categorical:
            path: torch.nn.CrossEntropyLoss
            kwargs:
              reduction: none
      
      concept_loss_weight: 1.0
      task_loss_weight: 1.0
   
   **Metrics Configuration**
   
   Metrics are also type-aware and configured in ``conf/metrics/``:
   
   .. code-block:: yaml
   
      # conf/metrics/standard.yaml
      discrete:
        binary:
          accuracy:
            path: torchmetrics.classification.BinaryAccuracy
            kwargs: {}
        categorical:
          accuracy:
            path: torchmetrics.classification.MulticlassAccuracy
            kwargs:
              average: micro
      
      continuous:
        mae:
          path: torchmetrics.regression.MeanAbsoluteError
          kwargs: {}
        mse:
          path: torchmetrics.regression.MeanSquaredError
          kwargs: {}
   
   **Overriding Model Parameters**
   
   From command line:
   
   .. code-block:: bash
   
      # Change learning rate
      python run_experiment.py model.optim_kwargs.lr=0.001
      
      # Enable per-concept metrics
      python run_experiment.py model.perconcept_metrics=true
      
      # Change encoder architecture
      python run_experiment.py model.encoder_kwargs.hidden_size=256 \
                               model.encoder_kwargs.n_layers=3
      
      # Use weighted loss
      python run_experiment.py loss=weighted
   
   In a custom sweep file:
   
   .. code-block:: yaml
   
      # conf/my_sweep.yaml
      defaults:
        - _default
        - _self_
      
      model:
        encoder_kwargs:
          hidden_size: 256
          n_layers: 3
        optim_kwargs:
          lr: 0.001
        perconcept_metrics: true

.. dropdown:: Running Experiments
   :icon: play
   
   **Single Experiment**
   
   Run with default configuration:
   
   .. code-block:: bash
   
      python run_experiment.py
   
   Specify dataset and model:
   
   .. code-block:: bash
   
      python run_experiment.py dataset=celeba model=cbm_joint
   
   With custom parameters:
   
   .. code-block:: bash
   
      python run_experiment.py \
          dataset=cub \
          model=cem \
          model.optim_kwargs.lr=0.001 \
          trainer.max_epochs=100 \
          seed=42
   
   **Multi-Run Sweeps**
   
   Sweep over multiple values using comma-separated lists:
   
   .. code-block:: bash
   
      # Sweep over datasets
      python run_experiment.py dataset=celeba,cub,mnist
      
      # Sweep over models
      python run_experiment.py model=cbm_joint,cem,cgm
      
      # Sweep over hyperparameters
      python run_experiment.py model.optim_kwargs.lr=0.0001,0.0005,0.001,0.005
      
      # Sweep over seeds for robustness
      python run_experiment.py seed=1,2,3,4,5
      
      # Combined sweeps
      python run_experiment.py \
          dataset=celeba,cub \
          model=cbm_joint,cem \
          seed=1,2,3
   
   This runs 2 × 2 × 3 = 12 experiments.
   
   **Custom Sweep Configuration**
   
   Create a sweep file (``conf/my_sweep.yaml``):
   
   .. code-block:: yaml
   
      defaults:
        - _default
        - _self_
      
      hydra:
        job:
          name: my_sweep
        sweeper:
          params:
            dataset: celeba,cub,mnist
            model: cbm_joint,cem
            seed: 1,2,3,4,5
            model.optim_kwargs.lr: 0.0001,0.001
      
      # Default overrides
      trainer:
        max_epochs: 500
        patience: 50
      
      model:
        summary_metrics: true
        perconcept_metrics: true
   
   Run the sweep:
   
   .. code-block:: bash
   
      python run_experiment.py --config-name my_sweep
   
   **Parallel Execution**
   
   Use Hydra's joblib launcher for parallel execution:
   
   .. code-block:: bash
   
      python run_experiment.py \
          --multirun \
          hydra/launcher=joblib \
          hydra.launcher.n_jobs=4 \
          dataset=celeba,cub \
          model=cbm_joint,cem
   
   Or use SLURM for cluster execution:
   
   .. code-block:: bash
   
      python run_experiment.py \
          --multirun \
          hydra/launcher=submitit_slurm \
          hydra.launcher.partition=gpu \
          hydra.launcher.gpus_per_node=1 \
          dataset=celeba,cub \
          model=cbm_joint,cem

.. dropdown:: Output Structure
   :icon: file-directory
   
   **Directory Organization**
   
   Experiment outputs are organized by timestamp:
   
   .. code-block:: text
   
      outputs/
      └── multirun/
          └── 2025-11-27/
              └── 14-30-15_my_experiment/
                  ├── 0/                          # First run
                  │   ├── .hydra/                 # Hydra configuration
                  │   │   ├── config.yaml         # Full resolved config
                  │   │   ├── hydra.yaml          # Hydra settings
                  │   │   └── overrides.yaml      # CLI overrides
                  │   ├── checkpoints/            # Model checkpoints
                  │   │   ├── best.ckpt           # Best model
                  │   │   └── last.ckpt           # Last epoch
                  │   ├── logs/                   # Training logs
                  │   │   └── version_0/
                  │   │       ├── events.out.tfevents  # TensorBoard
                  │   │       └── hparams.yaml    # Hyperparameters
                  │   └── run.log                 # Console output
                  ├── 1/                          # Second run
                  ├── 2/                          # Third run
                  └── multirun.yaml               # Sweep configuration
   
   **Accessing Results**
   
   Each run directory contains:
   
   - **Checkpoints**: ``checkpoints/best.ckpt`` - Best model based on validation metric
   - **Logs**: ``logs/version_0/`` - TensorBoard logs
   - **Configuration**: ``.hydra/config.yaml`` - Full configuration used for this run
   - **Console output**: ``run.log`` - All printed output
   
   Load a checkpoint:
   
   .. code-block:: python
   
      import torch
      from torch_concepts.nn import ConceptBottleneckModel_Joint
      
      checkpoint = torch.load('outputs/multirun/.../0/checkpoints/best.ckpt')
      model = ConceptBottleneckModel_Joint.load_from_checkpoint(checkpoint)
   
   **Weights & Biases Integration**
   
   All experiments are automatically logged to W&B if configured:
   
   .. code-block:: yaml
   
      # In your config or _default.yaml
      wandb:
        project: my_project
        entity: my_team
        log_model: false        # Set true to save models to W&B
        mode: online            # or 'offline' or 'disabled'
   
   View results at https://wandb.ai/your-team/my_project

.. dropdown:: Creating Custom Configurations
   :icon: pencil
   
   **Adding a New Model**
   
   1. **Implement the model** in |pyc_logo| PyC (see ``examples/contributing/model.md``)
   
   2. **Create configuration file** ``conf/model/my_model.yaml``:
   
      .. code-block:: yaml
   
         defaults:
           - _commons
           - loss: _default
           - metrics: _default
           - _self_
         
         _target_: torch_concepts.nn.MyModel
         
         task_names: ${dataset.default_task_names}
         
         # Model-specific parameters
         my_param: 42
         another_param: hello
   
   3. **Run experiments**:
   
      .. code-block:: bash
   
         python run_experiment.py model=my_model dataset=cub
   
   **Adding a New Dataset**
   
   1. **Implement the dataset and datamodule** (see ``examples/contributing/dataset.md``)
   
   2. **Create configuration file** ``conf/dataset/my_dataset.yaml``:
   
      .. code-block:: yaml
   
         defaults:
           - _commons
           - _self_
         
         _target_: my_package.MyDataModule
         
         name: my_dataset
         
         # Backbone (if needed)
         backbone:
           _target_: torchvision.models.resnet18
           pretrained: true
         
         precompute_embs: false
         
         # Default tasks
         default_task_names: [my_task]
         
         # Dataset-specific parameters
         data_path: /path/to/data
         preprocess: true
   
   3. **Run experiments**:
   
      .. code-block:: bash
   
         python run_experiment.py dataset=my_dataset model=cbm_joint
   
   **Adding Custom Loss/Metrics**
   
   Create ``conf/model/loss/my_loss.yaml``:
   
   .. code-block:: yaml
   
      _target_: torch_concepts.nn.WeightedConceptLoss
      _partial_: true
      
      fn_collection:
        discrete:
          binary:
            path: my_package.MyBinaryLoss
            kwargs:
              alpha: 0.25
              gamma: 2.0
          categorical:
            path: torch.nn.CrossEntropyLoss
            kwargs:
              label_smoothing: 0.1
      
      concept_loss_weight: 0.5
      task_loss_weight: 1.0
   
   Use it:
   
   .. code-block:: bash
   
      python run_experiment.py model/loss=my_loss

.. dropdown:: Advanced Usage
   :icon: gear
   
   **Conditional Configuration**
   
   Use Hydra's variable interpolation:
   
   .. code-block:: yaml
   
      # Automatically adjust batch size based on dataset
      dataset:
        batch_size: ${select:${dataset.name},{celeba:512,cub:256,mnist:1024}}
      
      # Scale learning rate with batch size
      model:
        optim_kwargs:
          lr: ${multiply:0.001,${divide:${dataset.batch_size},256}}
   
   **Configuration Validation**
   
   Add validation to catch errors early:
   
   .. code-block:: yaml
   
      # conf/model/cbm_joint.yaml
      defaults:
        - _commons
        - loss: _default
        - metrics: _default
        - _self_
      
      _target_: torch_concepts.nn.ConceptBottleneckModel_Joint
      
      # Require task names
      task_names: ${dataset.default_task_names}
      ???  # Error if not provided
   
   **Experiment Grouping**
   
   Organize related experiments:
   
   .. code-block:: yaml
   
      # conf/ablation_study.yaml
      hydra:
        job:
          name: ablation_${model.encoder_kwargs.hidden_size}
      
      defaults:
        - _default
        - _self_
      
      model:
        encoder_kwargs:
          hidden_size: ???  # Must be provided
   
   Run:
   
   .. code-block:: bash
   
      python run_experiment.py \
          --config-name ablation_study \
          model.encoder_kwargs.hidden_size=64,128,256,512

.. dropdown:: Best Practices
   :icon: checklist
   
   1. **Use Descriptive Names**
      
      .. code-block:: yaml
   
         hydra:
           job:
             name: ${model._target_}_${dataset.name}_seed${seed}
   
   2. **Keep Configs Small**
      
      - Use ``defaults`` to inherit common parameters
      - Only override what's different
   
   3. **Document Custom Parameters**
      
      .. code-block:: yaml
   
         my_parameter: 42  # Controls X behavior, higher = more Y
   
   4. **Version Control Configurations**
      
      - Commit all YAML files to git
      - Tag important configurations
   
   5. **Use Sweeps for Exploration**
      
      - Start with broad sweeps
      - Narrow down based on results
   
   6. **Monitor with W&B**
      
      - Enable W&B logging for all experiments
      - Use tags to organize runs
   
   7. **Save Important Checkpoints**
      
      - Set ``trainer.save_top_k`` appropriately
      - Copy important checkpoints out of temp directories

.. dropdown:: Troubleshooting
   :icon: tools
   
   **Common Issues**
   
   **Error: "Could not find dataset config"**
   
   - Check that ``conf/dataset/your_dataset.yaml`` exists
   - Verify the filename matches what you're passing to ``dataset=``
   
   **Error: "Missing _target_ in config"**
   
   - Ensure your config has ``_target_`` pointing to the class
   - Check for typos in the class path
   
   **Error: "Validation loss not improving"**
   
   - Check learning rate: try ``model.optim_kwargs.lr=0.0001``
   - Increase patience: ``trainer.patience=50``
   - Check your loss configuration
   
   **Experiments running slowly**
   
   - Enable feature precomputation: ``dataset.precompute_embs=true``
   - Increase batch size: ``dataset.batch_size=512``
   - Use more workers: ``dataset.num_workers=8``
   
   **Out of memory**
   
   - Reduce batch size: ``dataset.batch_size=128``
   - Reduce model size: ``model.encoder_kwargs.hidden_size=64``
   - Enable gradient checkpointing (model-specific)
   
   **Debugging**
   
   Check resolved configuration:
   
   .. code-block:: bash
   
      python run_experiment.py --cfg job
   
   Print config without running:
   
   .. code-block:: bash
   
      python run_experiment.py --cfg all
   
   Validate configuration:
   
   .. code-block:: bash
   
      python run_experiment.py --resolve


See Also
--------

- :doc:`using_high_level` - High-level API for programmatic usage
- `Contributing Guide - Models <https://github.com/pyc-team/pytorch_concepts/tree/master/examples/contributing/model.md>`_ - Implementing custom models
- `Contributing Guide - Datasets <https://github.com/pyc-team/pytorch_concepts/tree/master/examples/contributing/dataset.md>`_ - Implementing custom datasets
- `Conceptarium README <https://github.com/pyc-team/pytorch_concepts/tree/master/conceptarium/README.md>`_ - Additional documentation
- `Hydra Documentation <https://hydra.cc/>`_ - Advanced configuration patterns
- `PyTorch Lightning <https://lightning.ai/>`_ - Training framework documentation
