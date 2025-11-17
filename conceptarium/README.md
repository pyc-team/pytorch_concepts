<p align="center">
<img src="https://github.com/gdefe/conceptarium/blob/main/logo.png" style="width: 30cm">
<br>


# Training with Hydra Configuration

This example demonstrates how to train models using Hydra configuration files. This is the recommended approach for experiments as it provides better organization and reproducibility.

## Quick Start

### Basic Usage

Run with default configuration (asia dataset, CBM model):
```bash
# cd directory where conceptarium is located
python examples/with_hydra.py
```

Run with toy XOR dataset:
```bash
python examples/with_hydra.py dataset=toy_xor
```

### Override Configuration Parameters

You can override any configuration parameter from the command line:

```bash
# Change dataset
python examples/with_hydra.py dataset=alarm

# Change model architecture
python examples/with_hydra.py model.encoder_kwargs.hidden_size=128 model.encoder_kwargs.n_layers=2

# Change training parameters
python examples/with_hydra.py trainer.max_epochs=100 trainer.patience=10

# Change optimizer settings
python examples/with_hydra.py engine.optim_kwargs.lr=0.001

# Change batch size
python examples/with_hydra.py dataset.batch_size=64
```

### Hyperparameter Sweeps

Run multiple experiments with different configurations:

```bash
# Sweep over multiple seeds
python examples/with_hydra.py -m seed=1,2,3,4,5

# Sweep over learning rates
python examples/with_hydra.py -m engine.optim_kwargs.lr=0.0001,0.001,0.01

# Combine multiple sweeps
python examples/with_hydra.py -m seed=1,2,3 trainer.max_epochs=100,200 dataset.batch_size=32,64
```

## Configuration Structure

Configuration files are located in `conceptarium/conf/`:

```
conf/
├── _default.yaml          # Main configuration file with defaults
├── sweep.yaml             # Configuration for hyperparameter sweeps
├── dataset/               # Dataset configurations
│   ├── _commons.yaml      # Common dataset parameters
│   ├── asia.yaml          # Asia Bayesian network
│   ├── alarm.yaml         # Alarm Bayesian network
│   └── toy_xor.yaml       # Toy XOR dataset (NEW)
├── model/                 # Model architectures
│   ├── _commons.yaml      # Common model parameters
│   ├── cbm.yaml           # Concept Bottleneck Model
│   └── blackbox.yaml      # Black-box baseline
└── engine/                # Training configurations
    ├── engine.yaml        # Main engine config
    ├── loss/              # Loss function configurations
    │   └── default.yaml   # BCE, CrossEntropy, MSE
    └── metrics/           # Metric configurations
        └── default.yaml   # Accuracy, MAE, MSE
```

## Configuration Details

### Dataset Configuration (`dataset/*.yaml`)

Specifies the dataset to use and its parameters:

```yaml
defaults:
  - _commons
  - _self_

_target_: conceptarium.data.datamodules.toy.ToyDataModule

name: toy_xor
dataset_name: xor
size: 1000
random_state: 42

default_task_names: [task_xor]

label_descriptions:
  concept_1: "First binary concept for XOR task"
  concept_2: "Second binary concept for XOR task"
  task_xor: "XOR of the two concepts (target variable)"
```

Common parameters (from `_commons.yaml`):
- `batch_size`: Batch size for training
- `val_size`: Validation set fraction
- `test_size`: Test set fraction
- `concept_subset`: Subset of concepts to use

### Model Configuration (`model/*.yaml`)

Specifies the model architecture:

```yaml
defaults:
  - _commons
  - _self_
  
_target_: "conceptarium.nn.models.cbm.CBM"

task_names: ${dataset.default_task_names}

inference: 
  _target_: "torch_concepts.nn.DeterministicInference"
  _partial_: true
```

Common parameters (from `_commons.yaml`):
- `encoder_kwargs.hidden_size`: Hidden layer size
- `encoder_kwargs.n_layers`: Number of layers
- `encoder_kwargs.activation`: Activation function
- `encoder_kwargs.dropout`: Dropout rate
- `variable_distributions`: Probability distributions for concepts

### Engine Configuration (`engine/engine.yaml`)

Specifies training parameters:

```yaml
defaults:
  - metrics: default
  - loss: default
  - _self_
  
_target_: "conceptarium.engines.predictor.Predictor"

optim_class:
  _target_: "hydra.utils.get_class"
  path: "torch.optim.AdamW"
optim_kwargs:
  lr: 0.00075
```

Loss configuration (`engine/loss/default.yaml`):
```yaml
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

Metrics configuration (`engine/metrics/default.yaml`):
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

## Advanced Usage

### Creating Custom Datasets

1. Create a new datamodule in `conceptarium/data/datamodules/`:

```python
from torch_concepts.data import YourDataset
from ..base.datamodule import ConceptDataModule

class YourDataModule(ConceptDataModule):
    def __init__(self, ...):
        dataset = YourDataset(...)
        super().__init__(dataset=dataset, ...)
```

2. Create a configuration file in `conf/dataset/`:

```yaml
defaults:
  - _commons
  - _self_

_target_: conceptarium.data.datamodules.your_module.YourDataModule

name: your_dataset
# Add your dataset-specific parameters here
```

3. Run with your dataset:
```bash
python examples/with_hydra.py dataset=your_dataset
```

### Creating Custom Models

1. Implement your model in `conceptarium/nn/models/`

2. Create a configuration file in `conf/model/`:

```yaml
defaults:
  - _commons
  - _self_
  
_target_: "conceptarium.nn.models.your_model.YourModel"

# Add model-specific parameters here
```

3. Run with your model:
```bash
python examples/with_hydra.py model=your_model
```

## Comparison: with_hydra.py vs no_hydra.ipynb

| Feature | with_hydra.py | no_hydra.ipynb |
|---------|--------------|----------------|
| **Configuration** | Centralized YAML files | Inline Python code |
| **Reproducibility** | Automatic config logging | Manual tracking |
| **Hyperparameter Sweeps** | Built-in support (`-m` flag) | Manual loops |
| **Best for** | Production experiments | Learning & debugging |
| **Flexibility** | Override via command line | Full Python control |
| **Setup** | Requires Hydra knowledge | Straightforward Python |

## Output Structure

After running, outputs are saved in:
```
conceptarium/outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS_job_name/
        ├── .hydra/           # Hydra configuration
        ├── config.yaml       # Resolved configuration
        ├── main.log          # Training logs
        └── checkpoints/      # Model checkpoints
```

For sweeps, each run gets its own subdirectory:
```
outputs/multirun/
└── YYYY-MM-DD/
    └── HH-MM-SS_sweep_name/
        ├── 0/  # First configuration
        ├── 1/  # Second configuration
        └── ...
```

## Tips

1. **Start Simple**: Begin with default configuration, then override specific parameters
2. **Use Sweeps**: Leverage `-m` flag for hyperparameter search
3. **Check Configs**: Look at `outputs/.../hydra/config.yaml` to see resolved configuration
4. **Reuse Configs**: Copy successful configurations to create new presets
5. **Debug with Notebook**: Use `no_hydra.ipynb` for interactive debugging, then move to Hydra for experiments
