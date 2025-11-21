<p align="center">
<img src="../doc/_static/img/conceptarium.png" style="width: 30cm">
<br>

# Conceptarium

<img src="../doc/_static/img/logos/conceptarium.svg" width="20px" align="center"/> Conceptarium is a no-code framework for running large-scale experiments on concept-based models. This framework is intended for benchmarking or researchers in other fields who want to use concept-based models without programming knowledge. Conceptarium provides:

- **Configuration-driven experiments**: Use <img src="../doc/_static/img/logos/hydra-head.svg" width="20px" align="center"/> [Hydra](https://hydra.cc/) for flexible YAML-based configuration management and run sequential experiments on multiple <img src="../doc/_static/img/logos/pyc.svg" width="20px" align="center"/> PyC datasets and <img src="../doc/_static/img/logos/pyc.svg" width="20px" align="center"/> PyC models with a single command.

- **Automated training**: Leverage <img src="../doc/_static/img/logos/lightning.svg" width="20px" align="center"/> [PyTorch Lightning](https://lightning.ai/pytorch-lightning) for streamlined training loops

- **Experiment tracking**: Integrated <img src="../doc/_static/img/logos/wandb.svg" width="20px" align="center"/> [Weights & Biases](https://wandb.ai/) logging for monitoring and reproducibility

- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running Experiments](#running-experiments)
  - [Custom configurations](#custom-configurations)
  - [Output Structure](#output-structure)
- [Configuration Details](#configuration-details)
  - [Configuration Structure](#configuration-structure)
  - [Dataset Configuration](#dataset-configuration-datasetyaml)
  - [Model Configuration](#model-configuration-modelyaml)
- [Implementation](#implementation)
  - [Implementing Your Own Model](#implementing-your-own-model)
  - [Implementing Your Own Dataset](#implementing-your-own-dataset)
- [Contributing](#contributing)
- [Cite this library](#cite-this-library)

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
      model: cbm            # One or more models (blackbox, cbm, cem, cgm, c2bm, etc.)
      dataset: celeba, cub  # One or more datasets (celeba, cub, MNIST, alarm, etc.)
      seed: 1,2,3,4,5       # sweep over multiple seeds for robustness

model:
  optim_kwargs:
    lr: 0.001
  enable_summary_metrics: true
  enable_perconcept_metrics: false

trainer:
  max_epochs: 500
  patience: 30
  monitor: "val_loss"
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
python run_experiment.py dataset=alarm

# Change learning rate
python run_experiment.py model.optim_kwargs.lr=0.001

# Change multiple configurations
python run_experiment.py model=cbm dataset=asia,alarm seed=1,2,3
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
├── _default.yaml      # Base configuration with defaults
├── sweep.yaml         # Experiment sweep configuration
├── dataset/           # Dataset configurations
│   ├── _commons.yaml      # Common dataset parameters
│   ├── celeba.yaml
│   ├── cub.yaml
│   ├── sachs.yaml
│   └── ...
└── model/             # Model architectures
    ├── loss/              # Loss function configurations
    │   ├── _default.yaml  # Type-aware losses (BCE, CE, MSE)
    │   └── weighted.yaml  # Weighted type-aware losses
    ├── metrics/           # Metric configurations
    │   ├── _default.yaml  # Type-aware metrics (Accuracy, MAE, MSE)
    │   └── ...
    ├── _commons.yaml      # Common model parameters
    ├── blackbox.yaml      # Black-box baseline
    ├── cbm_joint.yaml     # Concept Bottleneck Model (Joint)
    ├── cem.yaml           # Concept Embedding Model
    ├── cgm.yaml           # Concept Graph Model
    └── c2bm.yaml          # Causally Reliable CBM
```
    │   ├── default.yaml   # Type-aware metrics (Accuracy, MAE, MSE)
    │   └── ...
    ├── _commons.yaml      # Common model parameters
    ├── blackbox.yaml      # Black-box baseline
    ├── cbm.yaml           # Concept Bottleneck Model
    ├── cem.yaml           # Concept Embedding Model
    ├── cgm.yaml           # Concept Graph Model
    └── c2bm.yaml          # Causally Reliable CBM
```


## Dataset Configuration (`dataset/*.yaml`)

Dataset configurations specify the dataset class to instantiate, all data-specific parameters, and all necessary preprocessing parameters. An example configuration for the CUB dataset is provided below:

```yaml
defaults:
  - _commons
  - _self_

_target_: torch_concepts.data.datamodules.CUBDataModule   # the path to your datamodule class  

name: cub

backbone: 
  _target_: "path.to.your.backbone.ClassName"
  # ... (backbone arguments)

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

Default parameters, common to all dataset, are in `_commons.yaml`:
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
  - loss: _default
  - metrics: _default
  - _self_
  
_target_: "torch_concepts.nn.ConceptBottleneckModel_Joint"

task_names: ${dataset.default_task_names}

inference: 
  _target_: "torch_concepts.nn.DeterministicInference"
  _partial_: true

enable_summary_metrics: true       # enable/disable summary metrics over concepts
enable_perconcept_metrics: false   # enable/disable per-concept metrics
```

### Common Parameters

From `_commons.yaml`:
- **`encoder_kwargs`**: Encoder architecture parameters
  - **`hidden_size`**: Hidden layer dimension in encoder
  - **`n_layers`**: Number of hidden layers in encoder
  - **`activation`**: Activation function (relu, tanh, etc.) in encoder
  - **`dropout`**: Dropout probability in encoder
- **`variable_distributions`**: Probability distributions with which concepts are modeled:
  - `binary`: Relaxed Bernoulli
  - `categorical`: Relaxed OneHot Categorical
  - `continuous`: Normal distribution
- **`optim_class`**: Optimizer class
- **`optim_kwargs`**:
  - **`lr`**: 0.00075

and more...

### Loss Configuration (`model/loss/_default.yaml`)

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
      
  continuous: 
    path: "torch.nn.MSELoss"
    kwargs: {}
```

### Metrics Configuration (`model/metrics/_default.yaml`)

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
python run_experiment.py model=your_model dataset=...
```
Alernatively, create your own sweep file `conf/your_sweep.yaml` containing your mdoel and run:
```bash
python run_experiment.py --config-file your_sweep.yaml
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
python run_experiment.py dataset=your_dataset model=...
```
Alternatively, create your own sweep file `conf/your_sweep.yaml` containing your dataset and run:
```bash
python run_experiment.py --config-name your_sweep.yaml
```

---
