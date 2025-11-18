<p align="center">
<img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/doc/_static/img/conceptarium.png" style="width: 30cm">
<br>

# Conceptarium

<img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/conceptarium.svg" width="25px" align="center"/> <em>Conceptarium</em> is a high-level experimentation framework for running large-scale experiments on concept-based deep learning models. Conceptarium is built on top of <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/pytorch.svg" width="25px" align="center"/> [PyTorch](https://pytorch.org/) and <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/pyc.svg" width="25px" align="center"/> [PyC](https://github.com/pyc-team/pytorch_concepts) for model implementation, <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/lightning.svg" width="25px" align="center"/> [PyTorch Lightning](https://lightning.ai/pytorch-lightning) for training automation, <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/hydra-head.svg" width="25px" align="center"/> [Hydra](https://hydra.cc/) for configuration management and <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/wandb.svg" width="25px" align="center"/> [Weights & Biases](https://wandb.ai/) for logging.

- [Quick Start](#quick-start)
- [Configuration Structure](#configuration-structure)
- [Configuration Details](#configuration-details)
  - [Dataset Configuration](#dataset-configuration-datasetyaml)
  - [Model Configuration](#model-configuration-modelyaml)
  - [Engine Configuration](#engine-configuration-engineengineyaml)
- [Implementing Your Own Model](#implementing-your-own-model)
- [Implementing Your Own Dataset](#implementing-your-own-dataset)
- [PyC Book](#pyc-book)
- [Authors](#authors)
- [Licence](#licence)
- [Cite this library](#cite-this-library)

---

# Quick Start

## Installation

Clone the <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/pyc.svg" width="25px" align="center"/> [PyC](https://github.com/pyc-team/pytorch_concepts) repository and navigate to the Conceptarium directory:

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

engine:
  optim_kwargs:
    lr: 0.00075

trainer:
  devices: [0]
  max_epochs: 500
  patience: 30
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

On top of this, you can also override configuration from command line:
```bash
# Change dataset
python run_experiment.py dataset=alarm

# Change learning rate
python run_experiment.py engine.optim_kwargs.lr=0.001

# Change multiple configurations
python run_experiment.py model=cbm,blackbox dataset=asia,alarm seed=1,2,3
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

Conceptarium provides a flexible configuration system based on <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/hydra-head.svg" width="25px" align="center"/> [Hydra](https://hydra.cc/), enabling easy experimentation across models, datasets, and hyperparameters. All configurations are stored in `conceptarium/conf/` and can be composed, overridden, and swept over from the command line or sweep files.


## Configuration Structure

Configuration files are organized in `conceptarium/conf/`:

```
conf/
├── _default.yaml      # Base configuration with defaults
├── sweep.yaml         # Experiment sweep configuration
├── dataset/           # Dataset configurations
│   ├── _commons.yaml      # Common dataset parameters
│   ├── celeba.yaml        # Bayesian network datasets
│   ├── cub.yaml
│   ├── sachs.yaml
│   └── ...
├── model/             # Model architectures
│   ├── _commons.yaml      # Common model parameters
│   ├── blackbox.yaml      # Black-box baseline
│   ├── cbm.yaml           # Concept Bottleneck Model
│   ├── cem.yaml           # Concept Embedding Model
│   ├── cgm.yaml           # Concept Graph Model
│   └── c2bm.yaml          # Causally Reliable CBM
└── engine/            # Training engine configurations
    ├── engine.yaml        # Main engine config
    ├── loss/              # Loss function configurations
    │   └── default.yaml   # Type-aware losses (BCE, CE, MSE)
    └── metrics/           # Metric configurations
        └── default.yaml   # Type-aware metrics (Accuracy, MAE, MSE)
```


## Dataset Configuration (`dataset/*.yaml`)

Dataset configurations specify the dataset class to instantiate and all necessary preprocessing parameters:

```yaml
defaults:
  - _commons
  - _self_

_target_: conceptarium.data.BnLearnDataModule

name: asia

backbone: null # input data is not high-dimensional, so does not require backbone
precompute_embs: false

default_task_names: [dysp]

label_descriptions:
  asia: "Recent trip to Asia"
  tub: "Has tuberculosis"
  smoke: "Is a smoker"
  lung: "Has lung cancer"
  bronc: "Has bronchitis"
  either: "Has tuberculosis or lung cancer"
  xray: "Positive X-ray"
  dysp: "Has dyspnoea (shortness of breath)"
```

### Common Parameters

From `_commons.yaml`:
- **`batch_size`**: Training batch size (default: 256)
- **`val_size`**: Validation set fraction (default: 0.15)
- **`test_size`**: Test set fraction (default: 0.15)
- **`concept_subset`**: List of specific concepts to use (optional)

---

## Model Configuration (`model/*.yaml`)

Model configurations specify the architecture and inference strategy:

```yaml
defaults:
  - _commons
  - _self_
  
_target_: "conceptarium.nn.CBM"

task_names: ${dataset.default_task_names}

inference: 
  _target_: "torch_concepts.nn.DeterministicInference"
  _partial_: true
```

### Common Parameters

From `_commons.yaml`:
- **`encoder_kwargs.hidden_size`**: Hidden layer dimension in encoder
- **`encoder_kwargs.n_layers`**: Number of hidden layers in encoder
- **`encoder_kwargs.activation`**: Activation function (relu, tanh, etc.) in encoder
- **`encoder_kwargs.dropout`**: Dropout probability in encoder
- **`variable_distributions`**: Probability distributions with which concepts are modeled:
  - `binary`: Relaxed Bernoulli
  - `categorical`: Relaxed OneHot Categorical
  - `continuous`: Normal distribution

---

## Engine Configuration (`engine/engine.yaml`)

Engine configurations specify training behavior, losses, and metrics:

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
  
enable_summary_metrics: true       # enable/disable summary metrics over concepts
enable_perconcept_metrics: false   # enable/disable per-concept metrics
```

### Loss Configuration (`engine/loss/default.yaml`)

Type-aware losses automatically select appropriate loss functions based on variable types:

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

### Metrics Configuration (`engine/metrics/default.yaml`)

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

# Implementing Your Own Model

Create your model in Conceptarium by following the guidelines given in [examples/contributing/model.md](examples/contributing/model.md).

This involves the following steps:
- Create your model in `conceptarium/nn/models/your_model.py`.
- Create configuration file in `conf/model/your_model.yaml`.
- Run experiments using your model. 

If your model is compatible with the defualt configuration structure, you can run experiments directly as follows:
```bash
python run_experiment.py model=your_model dataset=...
```
Alernatively, create your own sweep file `conf/your_sweep.yaml` containing your mdoel and run:
```bash
python run_experiment.py --config-file your_sweep.yaml
```

---

# Implementing Your Own Dataset
Create your dataset in Conceptarium by following the guidelines given in [examples/contributing/dataset.md](examples/contributing/dataset.md).

This involves the following steps:
- Create the dataset in `torch_concepts/data/datasets/your_dataset.py`.
- Create the datamodule in `conceptarium/data/datamodules/your_datamodule.py`.
- Create configuration file in `conf/dataset/your_dataset.yaml`.
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

# PyC Book

You can find further reading materials and tutorials in our book [Concept-based Interpretable Deep Learning in Python](https://pyc-team.github.io/pyc-book/).

---

# Contributors

Thanks to all contributors! 🧡

<a href="https://github.com/pyc-team/pytorch_concepts/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pyc-team/pytorch_concepts" />
</a>

---

# Licence

Copyright 2025 PyC Team.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at: <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

---

# Cite this library

If you found this library useful for your research article, blog post, or product, we would be grateful if you would cite it using the following bibtex entry:

```
@software{pycteam2025concept,
    author = {Barbiero, Pietro and De Felice, Giovanni and Espinosa Zarlenga, Mateo and Ciravegna, Gabriele and Dominici, Gabriele and De Santis, Francesco and Casanova, Arianna and Debot, David and Giannini, Francesco and Diligenti, Michelangelo and Marra, Giuseppe},
    license = {MIT},
    month = {3},
    title = {{PyTorch Concepts}},
    url = {https://github.com/pyc-team/pytorch_concepts},
    year = {2025}
}
```
Reference authors: [Pietro Barbiero](http://www.pietrobarbiero.eu/) and [Giovanni De Felice](https://gdefe.github.io/).
