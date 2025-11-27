# Contributing to Conceptarium

This guide shows how to extend Conceptarium with custom models, datasets, configurations, and experiment utilities.

## Table of Contents

- [Adding a Custom Model](#adding-a-custom-model)
- [Adding a Custom Dataset](#adding-a-custom-dataset)
- [Creating Custom Loss Functions](#creating-custom-loss-functions)
- [Creating Custom Metrics](#creating-custom-metrics)
- [Advanced Configuration Patterns](#advanced-configuration-patterns)
- [Extending run_experiment.py](#extending-run_experimentpy)

---

## Adding a Custom Model

### 1. Implement the Model in PyC

First, create your model following the [model contributing guide](./model.md). Your model should be available in the `torch_concepts` package.

### 2. Create Configuration File

Create `conceptarium/conf/model/my_custom_model.yaml`:

```yaml
defaults:
  - _commons                 # Inherit common model parameters
  - loss: _default           # Use default type-aware loss
  - metrics: _default        # Use default type-aware metrics
  - _self_                   # Current config takes precedence

# Target class to instantiate
_target_: torch_concepts.nn.MyCustomModel

# Task variables (inherited from dataset)
task_names: ${dataset.default_task_names}

# Model-specific parameters
my_parameter: 42
another_parameter: "value"

# Architecture configuration
architecture:
  layer_type: "dense"
  num_layers: 3
  hidden_dims: [128, 256, 128]

# Inference strategy
inference:
  _target_: torch_concepts.nn.DeterministicInference
  _partial_: true

# Metric tracking
summary_metrics: true
perconcept_metrics: false
```

### 3. Run Experiments

```bash
# Single run
python run_experiment.py model=my_custom_model dataset=cub

# Sweep over parameters
python run_experiment.py \
    model=my_custom_model \
    model.my_parameter=10,20,30,40 \
    dataset=celeba,cub
```

### Example: Custom CBM Variant

`conceptarium/conf/model/cbm_with_intervention.yaml`:

```yaml
defaults:
  - _commons
  - loss: weighted           # Use weighted loss
  - metrics: _default
  - _self_

_target_: torch_concepts.nn.InterventionalCBM

task_names: ${dataset.default_task_names}

# CBM-specific parameters
intervention_policy:
  _target_: torch_concepts.nn.RandomInterventionPolicy
  intervention_prob: 0.25

concept_bottleneck_type: "sequential"  # or "joint"

# Use per-concept metrics to track intervention effects
perconcept_metrics: true
summary_metrics: true
```

---

## Adding a Custom Dataset

### 1. Implement Dataset and DataModule

Follow the [dataset contributing guide](./dataset.md) to create:
- `MyDataset` class
- `MyDataModule` class (PyTorch Lightning DataModule)

### 2. Create Configuration File

Create `conceptarium/conf/dataset/my_custom_dataset.yaml`:

```yaml
defaults:
  - _commons                 # Inherit common dataset parameters
  - _self_

# Target datamodule class
_target_: my_package.data.MyDataModule

name: my_custom_dataset

# Backbone for feature extraction (if needed)
backbone:
  _target_: torchvision.models.resnet50
  pretrained: true
  # Can also be a custom backbone:
  # _target_: my_package.models.MyBackbone
  # custom_param: value

precompute_embs: true        # Precompute features for faster training

# Default task variables
default_task_names: [primary_task, secondary_task]

# Dataset-specific parameters
data_root: ${oc.env:DATA_ROOT,./data}  # Use env var or default
split_seed: 42
augmentation: true

# Optional: Concept descriptions for interpretability
label_descriptions:
  - concept1: Description of concept 1
  - concept2: Description of concept 2
  - concept3: Description of concept 3

# Optional: Causal structure (for causal datasets)
causal_graph:
  - [concept1, concept2]
  - [concept2, task]
```

### 3. Run Experiments

```bash
# Single run
python run_experiment.py dataset=my_custom_dataset model=cbm_joint

# Test with multiple models
python run_experiment.py dataset=my_custom_dataset model=cbm_joint,cem,cgm
```

### Example: Medical Imaging Dataset

`conceptarium/conf/dataset/medical_xray.yaml`:

```yaml
defaults:
  - _commons
  - _self_

_target_: medical_datasets.XRayDataModule

name: chest_xray

# Pretrained medical imaging backbone
backbone:
  _target_: torchvision.models.densenet121
  pretrained: true
  # Fine-tune on medical images
  checkpoint_path: ${oc.env:PRETRAIN_PATH}/densenet_medical.pth

precompute_embs: false       # Compute on-the-fly for augmentation

# Medical imaging specific
image_size: [224, 224]
normalize: true
augmentation:
  rotation: 10
  horizontal_flip: true
  brightness: 0.2

# Tasks and concepts
default_task_names: [disease_classification]

# Clinical concepts
label_descriptions:
  - has_opacity: Presence of lung opacity
  - has_cardiomegaly: Enlarged heart
  - has_effusion: Pleural effusion present
  - has_consolidation: Lung consolidation

# Concept groups (optional)
concept_groups:
  lung_findings: [has_opacity, has_consolidation]
  cardiac_findings: [has_cardiomegaly]
```

---

## Creating Custom Loss Functions

### 1. Implement Loss in PyC

Create a custom loss class:

```python
# torch_concepts/nn/modules/loss.py or custom module
class FocalConceptLoss(nn.Module):
    """Focal loss for handling class imbalance in concepts."""
    
    def __init__(self, annotations, fn_collection, alpha=0.25, gamma=2.0):
        super().__init__()
        self.annotations = annotations
        self.fn_collection = fn_collection
        self.alpha = alpha
        self.gamma = gamma
        # Implementation...
```

### 2. Create Loss Configuration

Create `conceptarium/conf/model/loss/focal.yaml`:

```yaml
_target_: torch_concepts.nn.FocalConceptLoss
_partial_: true

fn_collection:
  discrete:
    binary:
      path: my_package.losses.FocalBinaryLoss
      kwargs:
        alpha: 0.25
        gamma: 2.0
    categorical:
      path: my_package.losses.FocalCategoricalLoss
      kwargs:
        alpha: 0.25
        gamma: 2.0
```

### 3. Use in Model Configuration

```bash
# Command line
python run_experiment.py model/loss=focal

# Or in model config
python run_experiment.py model=cbm_joint model/loss=focal
```

Or create a model variant:

`conceptarium/conf/model/cbm_focal.yaml`:

```yaml
defaults:
  - cbm_joint                # Inherit from base CBM
  - loss: focal              # Override with focal loss
  - _self_

# Can add other overrides here
```

---

## Creating Custom Metrics

### 1. Create Metrics Configuration

Create `conceptarium/conf/model/metrics/comprehensive.yaml`:

```yaml
discrete:
  binary:
    accuracy:
      path: torchmetrics.classification.BinaryAccuracy
      kwargs: {}
    f1:
      path: torchmetrics.classification.BinaryF1Score
      kwargs: {}
    precision:
      path: torchmetrics.classification.BinaryPrecision
      kwargs: {}
    recall:
      path: torchmetrics.classification.BinaryRecall
      kwargs: {}
    auroc:
      path: torchmetrics.classification.BinaryAUROC
      kwargs: {}
  
  categorical:
    accuracy:
      path: torchmetrics.classification.MulticlassAccuracy
      kwargs:
        average: micro
    f1_macro:
      path: torchmetrics.classification.MulticlassF1Score
      kwargs:
        average: macro
    f1_weighted:
      path: torchmetrics.classification.MulticlassF1Score
      kwargs:
        average: weighted

continuous:
  mae:
    path: torchmetrics.regression.MeanAbsoluteError
    kwargs: {}
  mse:
    path: torchmetrics.regression.MeanSquaredError
    kwargs: {}
  r2:
    path: torchmetrics.regression.R2Score
    kwargs: {}
```

### 2. Use Custom Metrics

```bash
python run_experiment.py model/metrics=comprehensive
```

---

## Advanced Configuration Patterns

### Conditional Configuration Based on Dataset

`conceptarium/conf/model/adaptive_cbm.yaml`:

```yaml
defaults:
  - _commons
  - loss: _default
  - metrics: _default
  - _self_

_target_: torch_concepts.nn.ConceptBottleneckModel_Joint

task_names: ${dataset.default_task_names}

# Conditional batch size based on dataset
dataset:
  batch_size: ${select:${dataset.name},{celeba:512,cub:256,mnist:1024,default:256}}

# Conditional encoder size based on dataset complexity
encoder_kwargs:
  hidden_size: ${select:${dataset.name},{celeba:256,cub:128,mnist:64,default:128}}
  n_layers: ${select:${dataset.name},{celeba:3,cub:2,mnist:1,default:2}}

# Conditional learning rate
optim_kwargs:
  lr: ${multiply:0.001,${divide:${dataset.batch_size},256}}  # Scale with batch size
```

### Experiment-Specific Configuration

`conceptarium/conf/ablation_encoder_size.yaml`:

```yaml
defaults:
  - _default
  - _self_

hydra:
  job:
    name: ablation_encoder_${model.encoder_kwargs.hidden_size}
  sweeper:
    params:
      model.encoder_kwargs.hidden_size: 32,64,128,256,512
      seed: 1,2,3,4,5

# Fixed settings for ablation
dataset: cub
model: cbm_joint

trainer:
  max_epochs: 500
  patience: 50

wandb:
  project: encoder_ablation
  tags: [ablation, encoder_size]
```

Run:

```bash
python run_experiment.py --config-name ablation_encoder_size
```

### Multi-Stage Training Configuration

`conceptarium/conf/two_stage_training.yaml`:

```yaml
defaults:
  - _default
  - _self_

# Stage 1: Train concept encoder
stage1:
  model: cbm_joint
  trainer:
    max_epochs: 200
  model:
    freeze_task_predictor: true
  wandb:
    tags: [stage1, concept_learning]

# Stage 2: Fine-tune task predictor
stage2:
  model: ${stage1.model}
  trainer:
    max_epochs: 100
  model:
    freeze_concept_encoder: true
    optim_kwargs:
      lr: 0.0001  # Lower learning rate
  wandb:
    tags: [stage2, task_learning]
```

---

## Extending run_experiment.py

### Adding Custom Callbacks

Modify `conceptarium/run_experiment.py`:

```python
from pytorch_lightning.callbacks import Callback

class CustomMetricsCallback(Callback):
    """Custom callback for additional metric tracking."""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Custom metric computation
        custom_metric = compute_my_metric(pl_module, trainer.datamodule)
        pl_module.log('custom_metric', custom_metric)

@hydra.main(config_path="conf", config_name="_default", version_base=None)
def main(cfg: DictConfig):
    # ... existing code ...
    
    # Add custom callbacks
    callbacks = [
        # Existing callbacks...
        CustomMetricsCallback(),
    ]
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        # ... other trainer args ...
    )
    
    # ... rest of code ...
```

### Adding Custom Logging

```python
import logging
from pathlib import Path

@hydra.main(config_path="conf", config_name="_default", version_base=None)
def main(cfg: DictConfig):
    # Setup custom logging
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    
    # Log configuration to JSON for easy parsing
    import json
    with open(log_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    
    # Add custom metrics file
    metrics_file = log_dir / "metrics.csv"
    
    # ... training code ...
    
    # Save final metrics
    with open(metrics_file, "w") as f:
        f.write("metric,value\n")
        for k, v in final_metrics.items():
            f.write(f"{k},{v}\n")
```

### Adding Pre/Post Processing Hooks

```python
def preprocess_dataset(cfg, datamodule):
    """Custom preprocessing before training."""
    if cfg.dataset.get('custom_preprocessing', False):
        # Apply custom transformations
        datamodule.setup('fit')
        # Modify data...
    return datamodule

def postprocess_results(cfg, trainer, model):
    """Custom postprocessing after training."""
    # Export model in different format
    if cfg.get('export_onnx', False):
        model.to_onnx(f"model_{cfg.seed}.onnx")
    
    # Generate custom visualizations
    if cfg.get('generate_plots', False):
        plot_concept_activations(model, trainer.datamodule)

@hydra.main(config_path="conf", config_name="_default", version_base=None)
def main(cfg: DictConfig):
    # ... setup code ...
    
    # Preprocess
    datamodule = preprocess_dataset(cfg, datamodule)
    
    # Training
    trainer.fit(model, datamodule=datamodule)
    
    # Postprocess
    postprocess_results(cfg, trainer, model)
```

---

## Best Practices

1. **Keep Configurations Modular**
   - Use `defaults` to compose configurations
   - Create reusable components (losses, metrics, etc.)
   - Avoid duplication

2. **Document Parameters**
   ```yaml
   my_parameter: 42  # Controls X behavior. Higher values = more Y
   ```

3. **Use Type Hints**
   ```yaml
   _target_: my_package.MyClass
   # Ensure MyClass has proper type hints for better IDE support
   ```

4. **Validate Configurations**
   ```yaml
   required_parameter: ???  # Hydra will error if not provided
   ```

5. **Version Control**
   - Commit all YAML configurations
   - Tag important experimental configurations
   - Document breaking changes

6. **Testing**
   ```bash
   # Dry run to validate configuration
   python run_experiment.py --cfg job
   
   # Run quick test
   python run_experiment.py trainer.max_epochs=2 trainer.limit_train_batches=10
   ```

---

## Examples

### Complete Custom Model Pipeline

```bash
# 1. Create model implementation
# torch_concepts/nn/modules/high/models/my_model.py

# 2. Create model config
# conceptarium/conf/model/my_model.yaml

# 3. Create custom loss
# conceptarium/conf/model/loss/my_loss.yaml

# 4. Create custom metrics
# conceptarium/conf/model/metrics/my_metrics.yaml

# 5. Create sweep configuration
# conceptarium/conf/my_experiment.yaml

# 6. Run experiments
python run_experiment.py --config-name my_experiment
```

### Research Workflow

```bash
# 1. Explore hyperparameters
python run_experiment.py \
    --config-name hyperparameter_search \
    model.optim_kwargs.lr=0.0001,0.001,0.01 \
    model.encoder_kwargs.hidden_size=64,128,256

# 2. Run robustness check with best config
python run_experiment.py \
    --config-name best_config \
    seed=1,2,3,4,5,6,7,8,9,10

# 3. Compare models
python run_experiment.py \
    --config-name model_comparison \
    dataset=cub,celeba \
    model=cbm_joint,cem,cgm,blackbox

# 4. Analyze results in W&B
# Visit https://wandb.ai/your-team/your-project
```

---

## See Also

- [Model Contributing Guide](./model.md)
- [Dataset Contributing Guide](./dataset.md)
- [Hydra Documentation](https://hydra.cc/)
- [PyTorch Lightning Documentation](https://lightning.ai/)
