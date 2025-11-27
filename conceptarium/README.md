<p align="center">
<img src="../doc/_static/img/conceptarium.png" style="width: 30cm">
<br>

# Conceptarium

<img src="../doc/_static/img/logos/conceptarium.svg" width="20px" align="center"/> Conceptarium is a no-code framework for running large-scale experiments on concept-based models. This framework is intended for benchmarking or researchers in other fields who want to use concept-based models without programming knowledge. Conceptarium provides:

- **Configuration-driven experiments**: Use <img src="../doc/_static/img/logos/hydra-head.svg" width="20px" align="center"/> [Hydra](https://hydra.cc/) for flexible YAML-based configuration management and run sequential experiments on multiple <img src="../doc/_static/img/logos/pyc.svg" width="20px" align="center"/> PyC datasets and <img src="../doc/_static/img/logos/pyc.svg" width="20px" align="center"/> PyC models with a single command.

- **Automated training**: Leverage <img src="../doc/_static/img/logos/lightning.svg" width="20px" align="center"/> [PyTorch Lightning](https://lightning.ai/pytorch-lightning) for streamlined training loops

- **Experiment tracking**: Integrated <img src="../doc/_static/img/logos/wandb.svg" width="20px" align="center"/> [Weights & Biases](https://wandb.ai/) logging for monitoring and reproducibility

📚 **Full Documentation**: See the [comprehensive Conceptarium guide](../doc/guides/using_conceptarium.rst) for detailed documentation on:
- Configuration system and hierarchy
- Dataset and model configuration
- Custom losses and metrics
- Advanced usage patterns
- Troubleshooting

---

# Quick Start

## Installation

Clone the <img src="../doc/_static/img/logos/pyc.svg" width="20px" align="center"/> [PyC](https://github.com/pyc-team/pytorch_concepts) repository and navigate to the Conceptarium directory:

```bash
git clone https://github.com/pyc-team/pytorch_concepts.git
cd pytorch_concepts/conceptarium
```

To install all requirements and avoid conflicts, we recommend installing an [Anaconda](https://www.anaconda.com/) environment using the following command:

```bash
conda env create -f environment.yml
```



## Configuration

Configure your experiment by editing `conf/sweep.yaml`:

```yaml
defaults:
  - _default
  - _self_

hydra:
  job:
    name: my_experiment
  sweeper:
    params:
      seed: 1,2,3,4,5               # Sweep over multiple seeds for robustness
      dataset: cub,celeba            # One or more datasets
      model: cbm_joint               # One or more models (blackbox, cbm_joint)

model:
  optim_kwargs:
    lr: 0.01

metrics:
  summary_metrics: true
  perconcept_metrics: true

trainer:
  max_epochs: 200
  patience: 20
```

## Running Experiments

Run a single experiment:
```bash
python run_experiment.py
```

## Custom configurations

You can create as many configuration sweeps as you like. Assign a different name to each, e.g., `conf/your_sweep.yaml`, and run it as follows:

```bash
python run_experiment.py --config-name your_sweep.yaml
```

On top of this, you can also override configurations from command line:
```bash
# Change dataset
python run_experiment.py dataset=cub

# Change learning rate
python run_experiment.py model.optim_kwargs.lr=0.01

# Change multiple configurations
python run_experiment.py model=cbm_joint dataset=cub,celeba seed=1,2,3
```

## Output Structure

Results and logging outputs are saved in `conceptarium/outputs/`:

```
outputs/
└── multirun/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── 0/  # First run
            ├── 1/  # Second run
            └── ...
```

---

# Configuration Details

Conceptarium provides a flexible configuration system based on <img src="../doc/_static/img/logos/hydra-head.svg" width="20px" align="center"/> [Hydra](https://hydra.cc/), enabling easy experimentation across models, datasets, and hyperparameters. All configurations consist of `.yaml` files stored in `conceptarium/conf/`. These can be composed, overridden, and swept over from the command line or other sweep files.


## Configuration Structure

Configuration files are organized in `conceptarium/conf/`:

```
conf/
├── _default.yaml          # Base configuration with defaults
├── sweep.yaml             # Example sweep configuration
├── dataset/               # Dataset configurations
│   ├── _commons.yaml          # Common dataset parameters
│   ├── cub.yaml               # CUB-200-2011 birds dataset
│   ├── celeba.yaml            # CelebA faces dataset
│   └── ...                    # More datasets
├── loss/                  # Loss function configurations
│   ├── standard.yaml          # Standard type-aware losses
│   └── weighted.yaml          # Weighted type-aware losses
├── metrics/               # Metric configurations
│   └── standard.yaml          # Type-aware metrics (Accuracy)
└── model/                 # Model architectures
    ├── _commons.yaml          # Common model parameters
    ├── blackbox.yaml          # Black-box baseline
    ├── cbm.yaml               # Alias for CBM Joint
    └── cbm_joint.yaml         # Concept Bottleneck Model (Joint)
```


## Dataset Configuration (`dataset/*.yaml`)

Dataset configurations specify the dataset class to instantiate, all data-specific parameters, and all necessary preprocessing parameters. An example configuration for the CUB-200-2011 birds dataset is provided below:

```yaml
defaults:
  - _commons
  - _self_

_target_: torch_concepts.data.datamodules.CUBDataModule

name: cub

backbone: 
  _target_: torchvision.models.resnet18
  pretrained: true

precompute_embs: true  # precompute embeddings to speed up training

default_task_names: [bird_species]

label_descriptions:
  - has_wing_color::blue: Wing color is blue or not
  - has_upperparts_color::blue: Upperparts color is blue or not
  - has_breast_pattern::solid: Breast pattern is solid or not
  - has_back_color::brown: Back color is brown or not
  # ... (other visual attributes)
```

### Common Parameters

Default parameters, common to all datasets, are in `_commons.yaml`:

- **`batch_size`**: Training batch size (default: 256)
- **`val_size`**: Validation set fraction (default: 0.15)
- **`test_size`**: Test set fraction (default: 0.15)
- **`concept_subset`**: List of specific concepts to use (optional)

---

## Model Configuration (`model/*.yaml`)

Model configurations specify the architecture, loss, metrics, optimizer, and inference strategy:

```yaml
defaults:
  - _commons
  - _self_
  
_target_: torch_concepts.nn.ConceptBottleneckModel_Joint

task_names: ${dataset.default_task_names}

inference: 
  _target_: torch_concepts.nn.DeterministicInference
  _partial_: true

summary_metrics: true       # enable/disable summary metrics over concepts
perconcept_metrics: false   # enable/disable per-concept metrics
```

### Model Common Parameters

From `_commons.yaml`:

- **`encoder_kwargs`**: Encoder architecture parameters
  - **`hidden_size`**: Hidden layer dimension in encoder
  - **`n_layers`**: Number of hidden layers in encoder
  - **`activation`**: Activation function (relu, tanh, etc.) in encoder
  - **`dropout`**: Dropout probability in encoder
- **`variable_distributions`**: Probability distributions with which concepts are modeled
- **`optim_class`**: Optimizer class
- **`optim_kwargs`**:
  - **`lr`**: 0.00075

and more...

### Loss Configuration (`loss/standard.yaml`)

Type-aware losses automatically select appropriate loss functions based on variable types:

```yaml
_target_: "torch_concepts.nn.ConceptLoss"
_partial_: true

fn_collection:
  discrete:
    binary: 
      path: "torch.nn.BCEWithLogitsLoss"
      kwargs: {}
    categorical: 
      path: "torch.nn.CrossEntropyLoss"
      kwargs: {}
      
  # continuous: 
  # ... not supported yet
```

### Metrics Configuration (`metrics/standard.yaml`)

Type-aware metrics automatically select appropriate metrics based on variable types:

```yaml
discrete:
  binary:
    accuracy: 
      path: "torchmetrics.classification.BinaryAccuracy"
      kwargs: {}
  categorical:
    accuracy: 
      path: "torchmetrics.classification.MulticlassAccuracy"
      kwargs: 
        average: 'micro'
        
continuous: 
  mae: 
    path: "torchmetrics.regression.MeanAbsoluteError"
    kwargs: {}
  mse: 
    path: "torchmetrics.regression.MeanSquaredError"
    kwargs: {}
```

---

# Implementation

Conceptarium is designed to be extensible and accomodate your own experimental setting. You can implement custom models and datasets by following the guidelines below.


## Implementing Your Own Model

Create your model in <img src="../doc/_static/img/logos/pyc.svg" width="20px" align="center"/> PyC by following the guidelines given in [torch_concepts/examples/contributing/model.md](../examples/contributing/model.md).

This involves the following steps:
- Create your model (`your_model.py`).
- Create configuration file in `conceptarium/conf/model/your_model.yaml`, targeting the model class.
- Run experiments using your model. 

If your model is compatible with the default configuration structure, you can run experiments directly as follows:

```bash
python run_experiment.py model=your_model dataset=cub
```

Alternatively, create your own sweep file `conf/your_sweep.yaml` containing your model and run:

```bash
python run_experiment.py --config-name your_sweep
```

---

## Implementing Your Own Dataset

Create your dataset in Conceptarium by following the guidelines given in [torch_concepts/examples/contributing/dataset.md](../examples/contributing/dataset.md).

This involves the following steps:

- Create the dataset (`your_dataset.py`).
- Create the datamodule (`your_datamodule.py`) wrapping the dataset.
- Create configuration file in `conceptarium/conf/dataset/your_dataset.yaml`, targeting the datamodule class.
- Run experiments using your dataset.

If your dataset is compatible with the default configuration structure, you can run experiments directly as follows:

```bash
python run_experiment.py dataset=your_dataset model=cbm_joint
```

Alternatively, create your own sweep file `conf/your_sweep.yaml` containing your dataset and run:

```bash
python run_experiment.py --config-name your_sweep
```

---
