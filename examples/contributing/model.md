# Contributing a New Model

This guide will help you implement a new model in <img src="../../doc/_static/img/logos/pyc.svg" width="25px" align="center"/> PyC and enable its usage in <img src="../../doc/_static/img/logos/conceptarium.svg" width="25px" align="center"/> Conceptarium.

## Prerequisites

- Understanding of the model architecture (encoder, concept layers, predictor)
- Knowledge of concept dependencies
- Familiarity with inference strategy (deterministic, sampling, etc.)

## Training Modes

PyC models support two training paradigms:

### 1. Standard PyTorch Training (Manual)
- Initialize model **without** loss parameter
- Define optimizer, loss function, and training loop manually
- Full control over forward pass and optimization
- Example: `examples/utilization/2_model/5_torch_training.py`

### 2. PyTorch Lightning Training (Automatic)
- Initialize model **with** loss, optim_class, and optim_kwargs parameters
- Use Lightning Trainer for automatic training/validation/testing
- Inherits training logic from Learner classes (JointLearner, IndependentLearner)
- Example: `examples/utilization/2_model/6_lightning_training.py`

## Implementation Overview

All models extend `BaseModel` from `torch_concepts.nn.modules.high.base.model` and implement:

```python
from typing import Any, Dict, List, Optional, Union, Mapping
import torch
from torch import nn

from torch_concepts import Annotations
from torch_concepts.nn import (
    BipartiteModel, 
    LinearZC, 
    LinearCC, 
    LazyConstructor,
    BaseInference
)

from ..base.model import BaseModel


class YourModel(BaseModel):
    """High-level implementation of Your Model using BipartiteModel.
    
    [Brief description of your model and its key features]
    
    Args:
        task_names: Names of task/target concepts to predict
        inference: Inference module for forward pass and interventions
        input_size: Dimension of input features
        annotations: Concept annotations with metadata
        variable_distributions: Mapping of distribution types to distribution classes
        embs_precomputed: Whether embeddings are pre-computed
        backbone: Optional backbone network
        encoder_kwargs: Configuration for shared encoder MLP
    """
    
    def __init__(
        self,
        task_names: Union[List[str], str, List[int]],
        inference: BaseInference,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Mapping,
        embs_precomputed: bool = False,
        backbone: Optional[callable] = None,
        encoder_kwargs: Dict = None,
        **kwargs
    ) -> None:
        # Initialize BaseModel (sets up encoder, backbone, annotations)
        super().__init__(
            annotations=annotations,
            variable_distributions=variable_distributions,
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder_kwargs=encoder_kwargs,
        )
        
        # Build the model using BipartiteModel
        # This creates a two-layer architecture: embedding -> concepts -> tasks
        model = BipartiteModel(
            task_names=task_names,
            input_size=self.encoder_out_features,
            annotations=annotations,
            encoder=LazyConstructor(LinearZC),
            predictor=LazyConstructor(LinearCC)
        )
        self.pgm = model.pgm
        
        # Initialize inference module
        self.inference = inference(self.pgm)
    
    def forward(
        self,
        x: torch.Tensor,
        query: List[str] = None,
        backbone_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, input_size)
            query: List of concept names to query
            backbone_kwargs: Optional kwargs for backbone
            
        Returns:
            Output endogenous for queried concepts (batch_size, sum(concept_cardinalities))
        """
        # (batch, input_size) -> (batch, backbone_out_features)
        features = self.maybe_apply_backbone(x, backbone_kwargs)
        
        # (batch, backbone_out_features) -> (batch, encoder_out_features)
        features = self.encoder(features)
        
        # Inference: (batch, encoder_out_features) -> (batch, sum(concept_cardinalities))
        out = self.inference.query(query, evidence={'embedding': features})
        return out
    
    def filter_output_for_loss(self, forward_out):
        """Process model output for loss computation.
        
        Default: return output as-is. Override for custom processing.
        """
        return forward_out
    
    def filter_output_for_metrics(self, forward_out):
        """Process model output for metric computation.
        
        Default: return output as-is. Override for custom processing.
        """
        return forward_out
```

### 1.3 Mid-Level API Implementation

For custom architectures using `Variables`, `ParametricCPDs`, and `ProbabilisticGraphicalModel`:

```python
from torch_concepts import Variable, InputVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    ProbabilisticGraphicalModel,
    LinearZC,
    LinearCC,
    BaseInference
)


class YourModel_ParametricCPDs(BaseModel):
    """Mid-level implementation using Variables and ParametricCPDs.
    
    Use this approach when you need:
    - Custom concept dependencies
    - Non-standard graph structures
    - Fine-grained control over layer instantiation
    """

    def __init__(
            self,
            task_names: Union[List[str], str, List[int]],
            inference: BaseInference,
            input_size: int,
            annotations: Annotations,
            variable_distributions: Mapping,
            embs_precomputed: bool = False,
            backbone: Optional[callable] = None,
            encoder_kwargs: Dict = None,
            **kwargs
    ) -> None:
        super().__init__(
            annotations=annotations,
            variable_distributions=variable_distributions,
            input_size=input_size,
            embs_precomputed=embs_precomputed,
            backbone=backbone,
            encoder_kwargs=encoder_kwargs,
        )

        # Step 1: Define embedding variable (latent representation from encoder)
        embedding = InputVariable(
            "embedding",
            parents=[],
            distribution=Delta,
            size=self.encoder_out_features
        )
        embedding_cpd = ParametricCPD("embedding", parametrization=nn.Identity())

        # Step 2: Define concept variables
        concept_names = [c for c in annotations.get_axis_labels(1)
                         if c not in task_names]
        concepts = Variable(
            concept_names,
            parents=['embedding'],  # All concepts depend on embedding
            distribution=[annotations[1].metadata[c]['distribution']
                          for c in concept_names],
            size=[annotations[1].cardinalities[annotations[1].get_index(c)]
                  for c in concept_names]
        )

        # Step 3: Define task variables
        tasks = Variable(
            task_names,
            parents=concept_names,  # Tasks depend on concepts
            distribution=[annotations[1].metadata[c]['distribution']
                          for c in task_names],
            size=[annotations[1].cardinalities[annotations[1].get_index(c)]
                  for c in task_names]
        )

        # Step 4: Define concept encoder CPDs (layers)
        concept_encoders = ParametricCPD(
            concept_names,
            parametrization=[
                LinearZC(
                    in_features=embedding.size,
                    out_features=c.size
                ) for c in concepts
            ]
        )

        # Step 5: Define task predictor CPDs
        task_predictors = ParametricCPD(
            task_names,
            parametrization=[
                LinearCC(
                    in_features_endogenous=sum([c.size for c in concepts]),
                    out_features=t.size
                ) for t in tasks
            ]
        )

        # Step 6: Build Probabilistic Graphical Model
        self.pgm = ProbabilisticGraphicalModel(
            variables=[embedding, *concepts, *tasks],
            parametric_cpds=[embedding_factor, *concept_encoders, *task_predictors]
        )

        # Step 7: Initialize inference
        self.inference = inference(self.pgm)

    def forward(
            self,
            x: torch.Tensor,
            query: List[str] = None,
            backbone_kwargs: Optional[Mapping[str, Any]] = None,
            **kwargs
    ) -> torch.Tensor:
        features = self.maybe_apply_backbone(x, backbone_kwargs)
        features = self.encoder(features)
        out = self.inference.query(query, evidence={'embedding': features})
        return out

    def filter_output_for_loss(self, forward_out):
        return forward_out

    def filter_output_for_metrics(self, forward_out):
        return forward_out
```

### 1.4 Key Components Explained

#### Variables
Represent random variables (concepts) in your model:
- `name`: Variable identifier(s) - string or list of strings
- `parents`: List of parent variable names
- `distribution`: Probability distribution class(es)
- `size`: Dimensionality (cardinality for discrete, feature dim for continuous)

```python
# Binary concept
concept = Variable("smoking", parents=['embedding'], 
                  distribution=Bernoulli, size=1)

# Categorical concept with 5 classes
concept = Variable("diagnosis", parents=['embedding'], 
                  distribution=Categorical, size=5)

# Multiple concepts at once
concepts = Variable(['age', 'gender', 'bmi'], 
                   parents=['embedding'],
                   distribution=[Delta, Bernoulli, Delta],
                   size=[1, 1, 1])
```

#### ParametricCPDs
Represent computational modules (neural network layers):
- `name`: ParametricCPD identifier(s) matching variable names
- `module_class`: PyTorch module(s) that compute the factor

```python
# Single factor
encoder = ParametricCPD("smoking", parametrization=LinearZC(...))

# Multiple CPDs
encoders = ParametricCPD(['age', 'gender'], 
                 parametrization=[LinearZC(...), LinearZC(...)])
```

#### LazyConstructor
Utility for automatically instantiating modules for multiple concepts:

```python
# Creates one LinearZC per concept
encoder = LazyConstructor(LinearZC)
```

#### Inference
Controls how information flows through the model:
- `DeterministicInference`: Standard forward pass
- `AncestralSamplingInference`: Sample from distributions
- Custom inference: Extend `BaseInference` for specialized behavior

### 1.5 Available Layer Types

#### Encoders (Embedding/Exogenous → Logits)
```python
from torch_concepts.nn import (
    LinearZC,      # Linear encoder from embedding
    LinearUC,     # Linear encoder from exogenous
    LinearZU,             # Creates exogenous representations
)
```

#### Predictors (Logits → Logits)
```python
from torch_concepts.nn import (
    LinearCC,           # Linear predictor
    HyperLinearCUC,    # Hypernetwork-based predictor
    MixCUC,    # Mix of endogenous and exogenous
)
```

#### Special Layers
```python
from torch_concepts.nn import (
    SelectorZU,          # Memory-augmented selection
    WANDAGraphLearner,       # Learn concept graph structure
)
```

### 1.6 Custom Output Processing

Override these methods for custom loss/metric computation:

```python
def filter_output_for_loss(self, forward_out):
    """Process output before loss computation.
    
    Example: Split concepts and tasks for weighted loss
    """
    concept_endogenous = forward_out[:, :self.n_concepts]
    task_endogenous = forward_out[:, self.n_concepts:]
    return {
        'concept_input': concept_endogenous,
        'task_input': task_endogenous
    }

def filter_output_for_metrics(self, forward_out):
    """Process output before metric computation.
    
    Example: Apply softmax for probability metrics
    """
    return torch.softmax(forward_out, dim=-1)

def preprocess_batch(self, inputs, concepts):
    """Model-specific preprocessing of batch data.
    
    Example: Add noise or transformations
    """
    # Add your preprocessing logic
    return inputs, concepts
```

## Part 2: Model Configuration File

Create a YAML configuration file at `conceptarium/conf/model/your_model.yaml`.

### 2.1 Basic Configuration

```yaml
defaults:
  - _commons
  - _self_

# Target class for Hydra instantiation
_target_: "torch_concepts.nn.modules.high.models.your_model.YourModel"    # Path to your model class

# Inference configuration
inference:
  _target_: "torch_concepts.nn.DeterministicInference"
  _partial_: true  # Partial instantiation (model will pass pgm)

# Add any model-specific parameters here
```

### 2.2 Common Configuration (`_commons.yaml`)

The `_commons.yaml` file defines shared parameters. Override them in the model config as needed.

```yaml
# Encoder MLP configuration
encoder_kwargs:
  hidden_size: 64
  n_layers: 1
  activation: leaky_relu
  dropout: 0.2

# Variable distributions for different concept types
variable_distributions:
  discrete_card1:  # Binary concepts
    path: "torch.distributions.RelaxedBernoulli"
    kwargs:
      temperature: 0.1
  discrete_cardn:  # Categorical concepts
    path: "torch.distributions.RelaxedOneHotCategorical"
    kwargs:
      temperature: 0.1
  continuous_card1:  # Continuous scalars
    path: "torch_concepts.distributions.Delta"
  continuous_cardn:  # Continuous vectors
    path: "torch_concepts.distributions.Delta"
```

## Part 3: Testing & Verification
Test your model thoroughly before submission. 


## Part 4: Integration & Submission

### 4.1 Contacting the Authors

**Important**: Contact the library authors before submitting to ensure your model fits the library's scope and get guidance on:
    
### 4.2 Documentation

Provide the following documentation:
1. **Model docstring**: Clear description of model architecture, parameters, and usage
2. **Citation**: If based on a paper, include proper citation
3. **Example usage**: If the model is somewhat peculiar, please create example in `torch_concepts/examples/models-usage/your_model.py`
4. **README entry**: Add entry and description to torch_concepts README