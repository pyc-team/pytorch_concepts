# PyC (PyTorch Concepts) — Comprehensive Contributor Guide

> **Audience:** Experienced Python/PyTorch developers new to this codebase.
> **Version:** 1.0.0a1 · **License:** Apache 2.0 · **Python:** 3.5+

---

## Table of Contents

1. [Library Purpose & Design Philosophy](#1-library-purpose--design-philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Abstractions & Key Concepts](#3-core-abstractions--key-concepts)
4. [Data Flow & Execution Model](#4-data-flow--execution-model)
5. [Extension Points](#5-extension-points)
6. [Dependencies & Integration](#6-dependencies--integration)
7. [Testing & CI](#7-testing--ci)
8. [Coding Conventions & Style](#8-coding-conventions--style)
9. [Known Complexity & Gotchas](#9-known-complexity--gotchas)

---

## 1. Library Purpose & Design Philosophy

### What problem does PyC solve?

PyC (PyTorch Concepts) is a library for building **interpretable and causally transparent deep learning models** using *concepts* as first-class citizens. In standard deep learning, models learn opaque latent representations. PyC forces the model's internal representations to align with human-understandable concepts (e.g., "has_wheels," "is_red," "beak_shape") so that predictions can be traced back to meaningful intermediate variables.

The library name carries a dual meaning:
- **PyTorch Concepts** — concepts are foundational building blocks for interpretable DL.
- **P(y|C)** — the library is built around modeling the conditional distribution of targets *y* given concepts *C*.

### Core design principles

1. **Layered API design.** The library provides four abstraction levels, each building on the one below:

   | Level | Name | Target User | Interface |
   |-------|------|-------------|-----------|
   | 0 | **Low-level** | PyTorch power users | Individual interpretable layers (`nn.Module`) |
   | 1 | **Mid-level** | Probabilistic/causal modelers | Probabilistic graphical models with neural CPDs |
   | 2 | **High-level** | Practitioners | One-line model construction (`ConceptBottleneckModel(...)`) |
   | 3 | **Conceptarium** | Non-programmers | Config-file-driven experiment runner (Hydra + WandB) |

2. **Annotation-driven metadata.** Every tensor dimension can carry semantic annotations (`AxisAnnotation`) describing concept names, cardinalities, types, and distributions. These annotations drive automatic configuration of losses, metrics, and inference engines.

3. **Dual-mode training.** All high-level models work as both plain `nn.Module` (custom training loops) and `pl.LightningModule` (managed training), toggled by a single `lightning=True` flag.

4. **Composability over inheritance.** Encoders, predictors, inference engines, and intervention strategies are composed into models, not baked in. You can swap any piece independently.

5. **Causal reasoning as a first-class concern.** The library natively supports concept interventions (do-calculus), counterfactual evaluation, DAG-aware loss weighting, and graph learning.

---

## 2. Architecture Overview

### Package structure

```
torch_concepts/                    # Main package
├── __init__.py                    # Public API surface
├── _version.py                    # Version: '1.0.0a1'
├── annotations.py                 # AxisAnnotation, Annotations
├── typing.py                      # BackboneType alias
├── utils.py                       # Seeds, validation, temperature schedules
│
├── data/                          # Data subsystem
│   ├── base/                      # Abstract base classes
│   │   ├── dataset.py             #   ConceptDataset(torch Dataset)
│   │   ├── datamodule.py          #   ConceptDataModule(LightningDataModule)
│   │   ├── scaler.py              #   Scaler(ABC)
│   │   └── splitter.py            #   Splitter(ABC)
│   ├── backbone.py                # Backbone (ResNet, DINOv2, CLIP, etc.)
│   ├── datasets/                  # Concrete datasets
│   │   ├── toy.py                 #   ToyDataset, CompletenessDataset
│   │   ├── bnlearn.py             #   BnLearnDataset (Bayesian networks)
│   │   ├── categorical_toy_dag.py #   ToyDAGDataset (custom DAGs)
│   │   ├── celeba.py              #   CelebADataset
│   │   ├── cub.py                 #   CUBDataset
│   │   ├── cebab.py               #   CEBaBDataset
│   │   ├── awa2.py                #   AWA2Dataset
│   │   ├── mnist.py               #   ColorMNIST, MNISTAddition, etc.
│   │   └── traffic.py             #   TrafficLights (synthetic)
│   ├── datamodules/               # Concrete data modules
│   ├── splitters/                 # RandomSplitter, NativeSplitter, ColoringSplitter
│   ├── scalers/                   # StandardScaler
│   ├── preprocessing/             # Autoencoder for embedding extraction
│   ├── io.py                      # Download, extract, pickle utilities
│   └── utils.py                   # Tensor parsing, precision, colorization
│
├── distributions/                 # Custom probability distributions
│   └── delta.py                   # Delta (Dirac) distribution
│
└── nn/                            # Neural network subsystem
    ├── __init__.py                # Unified re-exports
    ├── functional.py              # Stateless functions (metrics, explanations)
    └── modules/
        ├── utils.py               # GroupConfig, check_collection()
        ├── loss.py                # ConceptLoss, WeightedConceptLoss, CompositeLoss
        ├── metrics.py             # ConceptMetrics
        │
        ├── low/                   # Low-level API
        │   ├── base/layer.py      #   BaseConceptLayer, BaseEncoder, BasePredictor
        │   ├── lazy.py            #   LazyConstructor
        │   ├── dense.py           #   Dense, MLP, ResidualMLP
        │   ├── encoders/          #   LinearLatentToConcept, LinearExogenousToConcept, etc.
        │   ├── predictors/        #   LinearConceptToConcept, HyperlinearConcept*, etc.
        │   ├── graph/             #   WANDAGraphLearner (DAG discovery)
        │   ├── inference/         #   BaseInference, BaseIntervention, RewiringIntervention
        │   │   └── intervention.py  GroundTruth/Do/Distribution interventions
        │   └── policy/            #   UniformPolicy, UncertaintyPolicy, RandomPolicy
        │
        ├── mid/                   # Mid-level API [EXPERIMENTAL]
        │   ├── models/
        │   │   ├── variable.py    #   Variable, ConceptVariable, ExogenousVariable, LatentVariable
        │   │   ├── cpd.py         #   ParametricCPD (neural conditional distributions)
        │   │   └── probabilistic_model.py  # ProbabilisticModel (DAG container)
        │   ├── constructors/
        │   │   ├── concept_graph.py  # ConceptGraph
        │   │   ├── bipartite.py     # BipartiteModel constructor
        │   │   └── graph.py         # GraphModel constructor
        │   └── inference/
        │       ├── forward.py       # ForwardInference (topological execution)
        │       ├── deterministic.py # DeterministicInference
        │       ├── ancestral.py     # AncestralSamplingInference
        │       ├── independent.py   # IndependentInference
        │       └── svi.py           # SVIInference
        │
        └── high/                  # High-level API
            ├── base/
            │   ├── model.py       #   BaseModel(nn.Module, ABC)
            │   ├── learner.py     #   BaseLearner(LightningModule)
            │   └── utils.py       #   with_training_mode() dynamic mixin
            └── models/
                ├── cbm.py         #   ConceptBottleneckModel
                ├── cem.py         #   ConceptEmbeddingModel
                ├── c2bm.py        #   CausallyReliableConceptBottleneckModel
                └── blackbox.py    #   BlackBox, BlackBoxTaskOnly
```

### How the pieces relate

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HIGH-LEVEL API                                │
│  ConceptBottleneckModel / ConceptEmbeddingModel / C2BM / BlackBox   │
│  ┌─── BaseModel ──── BaseLearner (optional Lightning mixin) ──────┐ │
│  │         ↓ uses                                                  │ │
│  │  ┌─────────────────────────────────────────────┐                │ │
│  │  │          MID-LEVEL API                      │                │ │
│  │  │  ProbabilisticModel ← Variables + CPDs      │                │ │
│  │  │  ForwardInference (topological execution)   │                │ │
│  │  │  BipartiteModel / GraphModel constructors   │                │ │
│  │  │         ↓ parameterized by                  │                │ │
│  │  │  ┌───────────────────────────────────────┐  │                │ │
│  │  │  │        LOW-LEVEL API                  │  │                │ │
│  │  │  │  Encoders · Predictors · Dense layers │  │                │ │
│  │  │  │  Interventions · Policies             │  │                │ │
│  │  │  │  LazyConstructor (deferred init)      │  │                │ │
│  │  │  └───────────────────────────────────────┘  │                │ │
│  │  └─────────────────────────────────────────────┘                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Annotations ←─── drives ───→ Loss / Metrics routing                │
│  ConceptGraph ←── drives ───→ DAG-aware weighting & topology        │
│  GroupConfig ←─── routes ───→ binary / categorical / continuous      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Abstractions & Key Concepts

### 3.1 Annotations: the semantic backbone

**`AxisAnnotation`** (`torch_concepts/annotations.py`) annotates a single tensor dimension with:

| Field | Type | Purpose |
|-------|------|---------|
| `labels` | `List[str]` | Ordered concept names |
| `cardinalities` | `List[int]` | Cardinality per concept (1 = binary, >1 = categorical) |
| `states` | `List[List[str]]` | State labels for each concept |
| `metadata` | `Dict[str, Dict]` | Per-concept metadata (type, distribution, etc.) |
| `is_nested` | `bool` | True if any cardinality > 1 |

Key design: when neither states nor cardinalities are provided, all concepts are assumed binary (cardinality=1).

```python
from torch_concepts import AxisAnnotation

# Binary concepts
axis = AxisAnnotation(labels=['has_wheels', 'is_red'])
# axis.cardinalities == [1, 1]

# Mixed concepts (binary + categorical)
axis = AxisAnnotation(
    labels=['color', 'shape', 'is_big'],
    cardinalities=[3, 2, 1],
    metadata={
        'color': {'type': 'discrete'},
        'shape': {'type': 'discrete'},
        'is_big': {'type': 'discrete'},
    }
)
# axis.shape == 6  (3 + 2 + 1)
```

**Critical cached properties** for efficient tensor slicing:
- `cumulative_cardinalities` — start positions for each concept in flattened logit space.
- `concept_slices` — `Dict[str, slice]` for direct `tensor[:, axis.concept_slices['color']]` indexing.
- `type_groups` — groups concepts into `'binary'`, `'categorical'`, `'continuous'` with both concept-level and logit-level indices.

**`Annotations`** is a multi-axis container wrapping `Dict[int, AxisAnnotation]`. Axis 1 is the concept axis by convention.

### 3.2 GroupConfig: type-aware routing

`GroupConfig` (`torch_concepts/nn/modules/utils.py`) is a named container with keys `binary`, `categorical`, and `continuous`. It is used throughout the library to specify per-type behavior:

```python
from torch_concepts.nn.modules.utils import GroupConfig

loss_config = GroupConfig(
    binary=nn.BCEWithLogitsLoss(),
    categorical=nn.CrossEntropyLoss(),
)
# Used by ConceptLoss to route predictions by type
```

The companion function `check_collection(fn_collection, ...)` normalizes various input formats (single value, dict, tuple of class + kwargs) into a `GroupConfig`.

### 3.3 Variables: typed random variables

The `Variable` hierarchy (`torch_concepts/nn/modules/mid/models/variable.py`) represents random variables in concept-based probabilistic models:

```
Variable (base)
├── ConceptVariable    — observable, supervisable concepts
├── ExogenousVariable  — high-dimensional representations linked to concepts
├── LatentVariable     — global latent representations (model input)
└── TaskVariable       — supervised target variables
```

Key attributes:
- `concept: str` — variable name
- `parents: List[Variable]` — parent variables (defines DAG edges)
- `distribution: Type[Distribution]` — e.g., `Bernoulli`, `Categorical`, `Delta`
- `size: int` — output dimensionality (1 for binary, K for K-class)

The `__new__` method enables batch construction: passing a list of names returns a list of variables.

```python
from torch_concepts import ConceptVariable, LatentVariable
from torch.distributions import Bernoulli

input_var = LatentVariable("input", parents=[], size=10)
concepts = ConceptVariable(
    ["c1", "c2", "c3"], parents=["input"], distribution=Bernoulli
)
# Returns list of 3 ConceptVariable instances
```

### 3.4 ParametricCPD: neural conditional distributions

`ParametricCPD` (`torch_concepts/nn/modules/mid/models/cpd.py`) links a neural network module (parametrization) to a variable, defining its conditional probability distribution:

```python
from torch_concepts.nn import ParametricCPD, LinearLatentToConcept, LazyConstructor

cpd = ParametricCPD("c1", parametrization=LazyConstructor(LinearLatentToConcept))
# Defers actual layer construction until parent dimensions are known
```

Like `Variable`, batch construction via list of names is supported.

### 3.5 ProbabilisticModel: the DAG container

`ProbabilisticModel` (`torch_concepts/nn/modules/mid/models/probabilistic_model.py`) wraps:
- A list of `Variable` objects (nodes)
- A dict of `ParametricCPD` objects (edges / conditional distributions)
- Automatic parent resolution (string → Variable conversion)
- Lazy layer initialization when `LazyConstructor` is used

It maintains `concept_to_variable: Dict[str, Variable]` for fast lookup.

### 3.6 Inference engines

Inference engines execute the forward pass through a `ProbabilisticModel` in topological order:

| Engine | Strategy | Use case |
|--------|----------|----------|
| `ForwardInference` | Base class; topological sort | — |
| `DeterministicInference` | `sigmoid`/`softmax` on logits | Standard prediction |
| `AncestralSamplingInference` | Sample from distributions | Uncertainty estimation |
| `IndependentInference` | Assumes independent concepts | Fast approximation |
| `SVIInference` | Stochastic variational inference | Amortized inference |

The `activate(pred, variable)` abstract method is the primary extension point:

```python
# From DeterministicInference
def activate(self, pred, variable):
    dist = variable.distribution
    if dist in (Bernoulli, RelaxedBernoulli):
        return torch.sigmoid(pred)
    elif dist in (Categorical, RelaxedOneHotCategorical):
        return torch.softmax(pred, dim=-1)
    elif dist is Delta:
        return pred
```

### 3.7 Intervention framework

Interventions let you override concept predictions at inference time for causal reasoning:

```python
from torch_concepts.nn import DoIntervention, RandomPolicy, intervention

policy = RandomPolicy(out_concepts=2, scale=100)
strategy = DoIntervention(model=encoder, constants=-10)

with intervention(policies=policy, strategies=strategy,
                  target_concepts=["c1"], quantiles=1) as new_encoder:
    c_pred = new_encoder(latent=z)  # c1 forced to -10
```

The `intervention()` context manager wraps modules with `_InterventionWrapper` modules that selectively replace predictions using binary masks.

Concrete strategies:
- `GroundTruthIntervention` — replace with ground truth values.
- `DoIntervention` — set to constant values.
- `DistributionIntervention` — sample from distributions.

### 3.8 ConceptLoss & ConceptMetrics

**`ConceptLoss`** (`torch_concepts/nn/modules/loss.py`) automatically routes predictions to the correct loss function based on concept type:

```python
from torch_concepts.nn.modules.loss import ConceptLoss

loss_fn = ConceptLoss(
    annotations=concept_axis_annotation,
    binary=nn.BCEWithLogitsLoss(),
    categorical=nn.CrossEntropyLoss(),
)
# forward(input, target) → splits by type_groups, applies per-type loss, sums
```

Variants: `WeightedConceptLoss` (separate concept/task weights), `DepthWeightedConceptLoss` (DAG depth-based weighting), `CompositeLoss` (sum of weighted loss terms).

**`ConceptMetrics`** (`torch_concepts/nn/modules/metrics.py`) manages per-type and per-concept metrics with independent tracking across train/val/test splits:

```python
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy

metrics = ConceptMetrics(
    annotations=concept_axis_annotation,
    fn_collection=GroupConfig(
        binary=BinaryAccuracy(),
        categorical=(MulticlassAccuracy, {'average': 'macro'}),
    ),
    summary_metrics=True,
    perconcept_metrics=True,
)
```

### 3.9 High-level models

High-level models compose all the abstractions above into one-line constructors:

```python
from torch_concepts.nn import ConceptBottleneckModel
from torch.distributions import Bernoulli

model = ConceptBottleneckModel(
    input_size=784,
    annotations=annotations,
    variable_distributions={name: Bernoulli for name in all_names},
    task_names=['digit'],
    lightning=True,   # adds BaseLearner mixin
    loss=loss_fn,
    metrics=metrics,
)
# Internally: BipartiteModel → ProbabilisticModel → DeterministicInference
```

| Model | Architecture | Key Difference |
|-------|-------------|----------------|
| `ConceptBottleneckModel` | Input → Encoder → Concepts → Predictor → Tasks | Linear bottleneck |
| `ConceptEmbeddingModel` | Input → Exogenous → Embeddings → Tasks | Concept embeddings with weighted mixing |
| `CausallyReliableConceptBottleneckModel` | CBM + graph learning | WANDA DAG discovery |
| `BlackBox` | Input → MLP → Tasks + Concepts | No bottleneck |

---

## 4. Data Flow & Execution Model

### 4.1 End-to-end data flow (high-level model)

```
Raw input (images, tabular)
    │
    ▼
[Backbone]  (optional: ResNet, DINOv2, CLIP, etc.)
    │  extracts features
    ▼
[Latent Encoder]  (MLP: features → latent_dims)
    │
    ▼
[ProbabilisticModel.forward]
    │
    ├── ForwardInference._topological_sort()
    │       → sorted_variables, levels
    │
    ├── For each level (parallel within level):
    │   ├── Gather parent outputs from evidence/results dict
    │   ├── ParametricCPD.forward(latent=..., concepts=..., exogenous=...)
    │   ├── ForwardInference.activate(logits, variable)
    │   │       → sigmoid (binary) / softmax (categorical) / identity (delta)
    │   └── Store in results dict
    │
    └── _concatenate_results(query_concepts, results)
            → final output tensor [batch, Σ cardinalities]
```

### 4.2 Training loop (Lightning mode)

```
Trainer.fit(model, datamodule)
    │
    ├── ConceptDataModule.setup('fit')
    │   ├── Splitter.fit(dataset) → train/val/test indices
    │   ├── (optional) precompute backbone embeddings
    │   └── (optional) StandardScaler.fit_transform()
    │
    ├── For each batch:
    │   ├── BaseLearner.training_step(batch)
    │   │   └── BaseLearner.shared_step(batch, 'train')
    │   │       ├── unpack_batch(batch) → x, concepts, transforms
    │   │       ├── model.forward(query=..., x=x, evidence=...)
    │   │       ├── model.filter_output_for_loss(output, concepts)
    │   │       ├── loss_fn(endogenous=predictions, target=concepts)
    │   │       │   └── ConceptLoss routes by type_groups
    │   │       ├── model.filter_output_for_metrics(output, concepts)
    │   │       └── metrics.update(predictions, targets)
    │   └── Log losses and metrics
    │
    └── Validation / Test: same flow with val/test metrics
```

### 4.3 Batch structure convention

Datasets return dictionaries, not tuples:

```python
{
    'inputs': {
        'x': torch.Tensor,           # [batch, features]
    },
    'concepts': {
        'c': torch.Tensor,           # [batch, n_concepts]
    },
    'transforms': { ... }            # optional
}
```

### 4.4 Key patterns

**Lazy construction:** `LazyConstructor` defers `nn.Module` instantiation until parent variable dimensions are resolved during `ProbabilisticModel` initialization:

```python
cpd = ParametricCPD("c1", parametrization=LazyConstructor(LinearLatentToConcept))
# Later, ProbabilisticModel calls:
# cpd.parametrization.build(in_latent=10, in_concepts=None, out_concepts=1)
```

**Dynamic mixin injection:** `BaseModel.__new__` uses `with_training_mode(cls, lightning)` to dynamically create a class that inherits from both the model class and `BaseLearner`:

```python
def with_training_mode(klass, lightning: bool):
    if lightning:
        combined = type(
            f"{klass.__name__}WithTraining",
            (klass, BaseLearner),
            {}
        )
        return combined
    return klass
```

**Topological sort + level execution:** `ForwardInference` sorts the DAG topologically and groups variables into levels. Variables at the same level can be computed in parallel (no dependencies between them).

**Annotation-driven routing:** The `type_groups` cached property on `AxisAnnotation` pre-computes which logit indices belong to binary vs. categorical vs. continuous concepts. Loss and metric modules index into these groups at forward time, avoiding runtime type dispatch.

---

## 5. Extension Points

### 5.1 Adding a new encoder

Create a new class inheriting from `BaseEncoder`:

```python
# torch_concepts/nn/modules/low/encoders/my_encoder.py
from torch_concepts.nn.modules.low.base.layer import BaseEncoder

class MyCustomEncoder(BaseEncoder):
    def __init__(self, in_latent: int, out_concepts: int, **kwargs):
        super().__init__(out_concepts=out_concepts, in_latent=in_latent)
        self.layer = nn.Linear(in_latent, out_concepts)

    def forward(self, latent: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layer(latent)
```

Register it by importing in `torch_concepts/nn/__init__.py` and adding to `__all__`.

### 5.2 Adding a new predictor

Inherit from `BasePredictor`:

```python
from torch_concepts.nn.modules.low.base.layer import BasePredictor

class MyPredictor(BasePredictor):
    def __init__(self, in_concepts: int, out_concepts: int, **kwargs):
        super().__init__(out_concepts=out_concepts, in_concepts=in_concepts)
        self.layer = nn.Linear(in_concepts, out_concepts)

    def forward(self, concepts: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layer(concepts)

    def prune(self, mask: torch.Tensor):
        """Optional: zero out connections based on mask."""
        self.layer.weight.data *= mask
```

### 5.3 Adding a new inference engine

Inherit from `ForwardInference` and implement `activate`:

```python
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference

class MyInference(ForwardInference):
    def activate(self, pred: torch.Tensor, variable) -> torch.Tensor:
        # Custom transformation of raw logits
        return my_custom_activation(pred, variable.distribution)
```

### 5.4 Adding a new intervention strategy

Inherit from `RewiringIntervention` and implement `_make_target`:

```python
from torch_concepts.nn.modules.low.inference.intervention import RewiringIntervention

class NoiseIntervention(RewiringIntervention):
    def __init__(self, model, noise_std=0.1):
        super().__init__(model)
        self.noise_std = noise_std

    def _make_target(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return y + torch.randn_like(y) * self.noise_std
```

### 5.5 Adding a new dataset

Inherit from `ConceptDataset`:

```python
from torch_concepts.data.base.dataset import ConceptDataset

class MyDataset(ConceptDataset):
    @property
    def raw_filenames(self):
        return ['my_data.pt']

    @property
    def processed_filenames(self):
        return ['embeddings.pt', 'concepts.pt', 'annotations.pt']

    def download(self):
        # Download raw data
        ...

    def build(self):
        # Process raw → tensors + annotations
        ...
```

### 5.6 Adding a new high-level model

Inherit from `BaseBipartiteModel` (for concept→task) or `BaseModel` (fully custom):

```python
from torch_concepts.nn.modules.high.base.model import BaseModel

class MyModel(BaseModel):
    def __init__(self, input_size, annotations, lightning=False, **kwargs):
        super().__init__(input_size, annotations, lightning=lightning, **kwargs)
        # Build internal ProbabilisticModel...

    def forward(self, query, x=None, evidence=None, **kwargs):
        z = self.maybe_apply_backbone(x)
        z = self.latent_encoder(z)
        return self.inference.query(query, evidence={'input': z}, **kwargs)

    def filter_output_for_loss(self, output, targets):
        return {'endogenous': output, 'target': targets}

    def filter_output_for_metrics(self, output, targets):
        return {'endogenous': output, 'target': targets}
```

### 5.7 Adding a new splitter

```python
from torch_concepts.data.base.splitter import Splitter

class StratifiedSplitter(Splitter):
    def fit(self, dataset):
        # Compute stratified indices
        self.set_indices(train_idx, val_idx, test_idx)
```

---

## 6. Dependencies & Integration

### 6.1 Core dependencies

| Package | Usage |
|---------|-------|
| `torch` | Core tensor operations, `nn.Module` base |
| `pytorch-lightning` | `LightningModule`, `LightningDataModule`, `seed_everything` |
| `scikit-learn` | Accuracy metrics in examples, data preprocessing |
| `scipy` | Sparse matrix operations |
| `pytorch-minimize` | Optimization utilities |
| `networkx` | Graph operations (DAG manipulation) |

### 6.2 Optional dependencies (`[data]` extra)

| Package | Usage |
|---------|-------|
| `torchvision` | Pre-trained backbones (ResNet, VGG, EfficientNet) |
| `transformers` | HuggingFace backbones (DINOv2, CLIP, BERT) |
| `pgmpy` | Bayesian network operations |
| `bnlearn` | Bayesian network datasets |
| `datasets` | HuggingFace datasets (CEBaB) |
| `pandas` | DataFrame operations for concept tables |
| `opencv-python` | Image processing |
| `tables` | HDF5 file I/O |

### 6.3 PyTorch ecosystem integration

- **Models are `nn.Module`:** All layers, encoders, predictors, and models inherit from `torch.nn.Module`. They can be used with any PyTorch optimizer, saved with `torch.save()`, and exported with `torch.jit`.
- **Lightning compatibility:** High-level models optionally inherit from `LightningModule` via dynamic mixin. `ConceptDataModule` inherits from `LightningDataModule`.
- **torchmetrics:** `ConceptMetrics` wraps `torchmetrics.MetricCollection` for automatic train/val/test tracking.
- **torch.distributions:** The library uses PyTorch's distribution classes (`Bernoulli`, `Categorical`, `RelaxedOneHotCategorical`) for specifying variable types and inference activation.

---

## 7. Testing & CI

### 7.1 Test structure

Tests mirror the source tree structure:

```
tests/
├── test_annotations.py             # AxisAnnotation, Annotations
├── test_typing.py                   # Type validation
├── test_utils.py                    # Utility functions
├── data/
│   ├── test_backbone.py
│   ├── base/test_dataset.py, test_datamodule.py, test_scaler.py, test_splitters.py
│   ├── datasets/test_toy.py, test_categorical_toy_dag.py
│   └── preprocessing/test_autoencoder.py
├── distributions/
│   └── test_delta.py
└── nn/
    ├── test_functional.py
    └── modules/
        ├── test_loss.py, test_metrics.py, test_utils_modules.py
        ├── high/
        │   ├── test_integration.py        # Full pipeline integration
        │   ├── base/test_base_model.py, test_base_learner.py
        │   └── models/test_cbm.py, test_cem.py, test_c2bm.py, test_blackbox.py
        ├── low/
        │   ├── test_dense_layers.py, test_lazy.py, test_semantic.py
        │   ├── base/test_layer.py
        │   ├── graph/test_wanda.py
        │   ├── inference/test_intervention.py
        │   └── policy/test_uniform.py, test_uncertainty.py, test_random.py
        └── mid/
            ├── models/test_variable.py, test_cpd.py, test_probabilistic_model.py
            ├── constructors/test_bipartite.py, test_graph.py, test_concept_graph.py
            └── inference/test_forward.py, test_deterministic.py, test_independent.py, ...
```

### 7.2 Running tests locally

```bash
# Install dev dependencies
pip install -e ".[tests]"

# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=torch_concepts

# Run specific test file
pytest tests/nn/modules/test_loss.py -v

# Run with doctest (enabled by default in setup.cfg)
pytest --doctest-modules
```

The `setup.cfg` configures `addopts = --doctest-modules`, meaning doctests in all source files are run as part of the test suite.

### 7.3 Testing conventions

- **Framework:** `unittest.TestCase` classes with `pytest` runner.
- **Pattern:** `setUp()` creates test fixtures (annotations, models, loss functions).
- **Naming:** `test_<thing being tested>` method names.
- **Coverage:** Tests validate both correctness (output shapes, values) and error handling (`assertRaises`).
- **Integration tests:** `test_integration.py` runs complete model → loss → metrics → optimizer pipelines.
- **Regression tests:** Specific regression tests exist for known bugs (e.g., `test_cpd_parent_preservation.py`, `test_exogenous_prefix_matching.py`).

### 7.4 CI configuration

**AppVeyor** (`appveyor.yml`): Windows CI testing across Python 3.5 and 3.6 with multiple NumPy/SciPy/scikit-learn version matrices.

**Codecov** (`codecov.yml`):
- Target coverage range: 70–100%
- Threshold: 1% (PRs must not drop coverage by more than 1%)
- Ignored paths: `tests/`, `examples/`, `doc/`, `conceptarium/`

---

## 8. Coding Conventions & Style

### 8.1 General style

- **PEP 8** compliance is expected.
- **Type hints** are used throughout public APIs (parameters and return types).
- **Imports:** Standard library → third-party → local, with `from` imports preferred for short names.

### 8.2 Naming conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Classes | PascalCase | `ConceptBottleneckModel`, `AxisAnnotation` |
| Functions/methods | snake_case | `seed_everything`, `get_slice` |
| Private methods | `_` prefix | `_topological_sort`, `_make_target` |
| Constants | UPPER_SNAKE | `DISTNAME`, `INSTALL_REQUIRES` |
| Module files | snake_case | `probabilistic_model.py`, `concept_graph.py` |
| Test files | `test_` prefix | `test_annotations.py` |
| Test classes | `Test` prefix | `TestAxisAnnotation` |

### 8.3 Docstring format

Docstrings follow **NumPy/Google hybrid** style with Args/Returns sections:

```python
def query(self, query: List[str], evidence: Dict[str, torch.Tensor],
          debug: bool = False) -> torch.Tensor:
    """Execute forward inference on the probabilistic model.

    Args:
        query: List of variable names to compute.
        evidence: Dict mapping variable names to observed tensors.
        debug: If True, print intermediate results.

    Returns:
        torch.Tensor: Concatenated predictions for query variables.
    """
```

### 8.4 Forward method conventions

Layer `forward()` methods accept keyword arguments with standardized names:

```python
# Encoders receive:
def forward(self, latent=None, exogenous=None, **kwargs) -> torch.Tensor:

# Predictors receive:
def forward(self, concepts=None, latent=None, exogenous=None, **kwargs) -> torch.Tensor:
```

This convention allows the inference engine to pass parent outputs using `**kwargs` without knowing the specific layer type.

### 8.5 Write-once annotations

`AxisAnnotation` attributes (`labels`, `states`, `cardinalities`) are write-once: once set during `__post_init__`, they cannot be reassigned. This prevents accidental mutation of annotation metadata that could desynchronize cached properties.

### 8.6 Commit messages

The project follows [Gitmoji](https://gitmoji.dev/) conventions:
- `✨ Add new feature`
- `🐛 Fix bug`
- `📝 Update documentation`

---

## 9. Known Complexity & Gotchas

### 9.1 Continuous concept support is incomplete

Multiple modules contain `NotImplementedError` or `TODO` markers for continuous concepts:

```python
# In loss.py
elif metadata['type'] == 'continuous' and cardinality == 1:
    raise NotImplementedError("Continuous concepts not supported yet.")
```

Affected modules: `annotations.py` (type routing), `loss.py`, `metrics.py`, `utils.py`, `dataset.py`. This is the most significant contribution opportunity.

### 9.2 The mid-level API is experimental

The `__init__.py` marks mid-level APIs as experimental:

```python
warnings.warn("Mid-level APIs are experimental and may change.")
```

Be cautious when depending on mid-level internals — the `Variable`, `CPD`, and `ProbabilisticModel` interfaces may evolve.

### 9.3 `__new__` magic in Variable, ParametricCPD, and BaseModel

Three classes override `__new__` in non-obvious ways:

- **`Variable.__new__`** — when `concepts` is a list, returns a `List[Variable]` instead of a single instance. This changes the return type of the constructor.
- **`ParametricCPD.__new__`** — same list-splitting behavior.
- **`BaseModel.__new__`** — dynamically creates a subclass that mixes in `BaseLearner` when `lightning=True`. The actual `type()` of the instance differs from the declared class.

This means `isinstance()` checks on Lightning-enabled models may behave unexpectedly:
```python
model = ConceptBottleneckModel(lightning=True, ...)
type(model).__name__  # "ConceptBottleneckModelWithTraining"
isinstance(model, ConceptBottleneckModel)  # True (still works, it's a subclass)
```

### 9.4 Annotation–cardinality mismatches cause shape bugs

Many downstream modules (loss, metrics, inference) compute tensor slicing indices from `AxisAnnotation.cardinalities`. If annotations don't match the actual tensor shapes, you'll get silent wrong results rather than clear errors. Always verify:

```python
expected = axis.shape  # sum of cardinalities
actual = tensor.shape[1]
assert expected == actual, f"Mismatch: annotations say {expected}, tensor has {actual}"
```

### 9.5 `concept_slices` vs. `type_groups` indexing

Two different indexing systems coexist:
- `concept_slices` — per-concept `slice` objects for named access.
- `type_groups['binary']['logits_idx']` — flat index lists grouped by type.

These serve different purposes (named access vs. type-based batching) but can be confusing when both appear in the same module.

### 9.6 The `intervention()` context manager modifies modules in-place

The intervention context manager wraps modules by replacing their `forward` method. This is thread-unsafe and should not be used in multi-threaded/multi-process scenarios without external synchronization.

### 9.7 Forward pass keyword argument routing

The inference engine passes parent outputs to CPD `forward()` methods using keyword argument names (`latent`, `concepts`, `exogenous`). These names are determined by the parent variable types:

| Parent type | kwarg name |
|-------------|------------|
| `LatentVariable` | `latent` |
| `ConceptVariable` | `concepts` |
| `ExogenousVariable` | `exogenous` |

If your custom layer expects different argument names, the inference engine won't pass data correctly. Always match the convention.

### 9.8 `LazyConstructor` must have all needed kwargs at build time

When `ProbabilisticModel` initializes CPDs with `LazyConstructor`, it infers `in_latent`, `in_concepts`, `in_exogenous`, and `out_concepts` from parent variables. If your custom layer requires additional constructor arguments, pass them when creating the `LazyConstructor`:

```python
LazyConstructor(MyLayer, my_custom_arg=42)
# 'my_custom_arg' is stored and passed when .build() is called
```

### 9.9 Doctest mode is always on

`setup.cfg` sets `addopts = --doctest-modules`, meaning every docstring example is executed as a test. Be careful when writing docstrings — all `>>>` blocks must be valid and self-contained, or they'll fail CI.

### 9.10 TODO datasets in the codebase

Several dataset files are prefixed with `TODO_` (e.g., `TODO_colormnist.py`, `TODO_fashionmnist.py`), indicating planned but unfinished implementations. These are contribution targets.

---

## Appendix: Quick reference — Key imports

```python
# Core
from torch_concepts import Annotations, AxisAnnotation, ConceptGraph
from torch_concepts import Variable, ConceptVariable, ExogenousVariable, LatentVariable
from torch_concepts import seed_everything

# Low-level layers
from torch_concepts.nn import (
    LinearLatentToConcept, LinearConceptToConcept,
    LinearExogenousToConcept, LinearLatentToExogenous,
    MixConceptExogegnousToConcept,
)

# Inference & interventions
from torch_concepts.nn import (
    DeterministicInference, AncestralSamplingInference,
    DoIntervention, GroundTruthIntervention,
    RandomPolicy, UniformPolicy, UncertaintyInterventionPolicy,
    intervention,
)

# Mid-level
from torch_concepts.nn import (
    ParametricCPD, ProbabilisticModel,
    BipartiteModel, GraphModel,
    LazyConstructor,
)

# High-level models
from torch_concepts.nn import (
    ConceptBottleneckModel, ConceptEmbeddingModel,
    CausallyReliableConceptBottleneckModel, BlackBox,
)

# Loss & metrics
from torch_concepts.nn.modules.loss import ConceptLoss, WeightedConceptLoss, CompositeLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.nn.modules.utils import GroupConfig

# Data
from torch_concepts.data.datasets import ToyDataset, BnLearnDataset, ToyDAGDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.backbone import Backbone
```
