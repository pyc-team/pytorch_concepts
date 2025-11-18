# Contributing a New Loss Function

This guide explains how to implement custom loss functions for the `pytorch_concepts` library.

## When to Implement a Custom Loss

Implement a custom loss when:
- You need to weight concept and task losses differently
- You require specialized loss computation (e.g., contrastive, triplet)
- Standard PyTorch losses don't fit your use case
- You need custom regularization terms

**Note**: For standard use cases, PyTorch's built-in losses (`BCEWithLogitsLoss`, `CrossEntropyLoss`, `MSELoss`) work out-of-the-box.

## Implementation

### 1. Create Loss Class

Place your loss in `conceptarium/conceptarium/nn/losses/your_loss.py`:

```python
import torch


class YourCustomLoss(torch.nn.Module):
    """Custom loss function for [specific use case].
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        **kwargs: Additional arguments passed to parent class (if extending)
    """
    
    def __init__(self, param1=default_value, param2=default_value, **kwargs):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            input: Model predictions (logits or probabilities)
            target: Ground truth labels
            
        Returns:
            Scalar loss value
        """
        # Implement your loss computation
        loss = ...  # Your loss calculation
        return loss
```

### 2. Example: Weighted Loss

A common pattern is weighting concept and task losses:

```python
import torch


class WeightedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Weighted BCE loss for concept and task predictions.
    
    Computes separate losses for concepts and tasks, then combines them
    with a weighting factor.
    
    Args:
        concept_loss_weight: Weight for concept loss (0-1). 
                           Task weight = 1 - concept_loss_weight.
                           If None, uses sum of both losses.
    """
    
    def __init__(self, concept_loss_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.concept_loss_weight = concept_loss_weight
    
    def forward(
        self, 
        concept_input: torch.Tensor,
        concept_target: torch.Tensor,
        task_input: torch.Tensor,
        task_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted loss.
        
        Args:
            concept_input: Concept predictions (batch_size, n_concepts)
            concept_target: Concept ground truth
            task_input: Task predictions (batch_size, n_tasks)
            task_target: Task ground truth
            
        Returns:
            Weighted combined loss
        """
        c_loss = super().forward(concept_input, concept_target)
        t_loss = super().forward(task_input, task_target)
        
        if self.concept_loss_weight is not None:
            # Weighted combination
            return (c_loss * self.concept_loss_weight + 
                    t_loss * (1 - self.concept_loss_weight))
        else:
            # Simple sum
            return c_loss + t_loss
```

### 3. Register Loss

Update `conceptarium/conceptarium/nn/losses/__init__.py`:

```python
from .your_loss import YourCustomLoss

__all__ = ['YourCustomLoss']
```

## Configuration

### 1. Create Loss Configuration

Create `conceptarium/conf/engine/loss/your_loss.yaml`:

```yaml
# For models with discrete concepts
discrete:
  binary:  # Binary classification
    path: "conceptarium.nn.losses.YourCustomLoss"
    kwargs:
      param1: value1
      param2: value2
  categorical:  # Multi-class classification
    path: "conceptarium.nn.losses.YourCustomLoss"
    kwargs:
      param1: value1

# For models with continuous concepts
continuous:
  path: "conceptarium.nn.losses.YourCustomLoss"
  kwargs:
    param1: value1
```


## Usage

### Via Configuration

```bash
# Use your custom loss
python train.py engine.loss=your_loss

# Override specific parameters
python train.py engine.loss=weighted \
    engine.loss.discrete.binary.kwargs.concept_loss_weight=0.9
```


## Model Integration

If your loss requires special input format, override `filter_output_for_loss` in your model:

```python
class YourModel(BaseModel):
    def filter_output_for_loss(self, forward_out):
        """Process model output for custom loss.
        
        Example: Split output for weighted loss
        """
        concept_logits = forward_out[:, :self.n_concepts]
        task_logits = forward_out[:, self.n_concepts:]
        
        return {
            'concept_input': concept_logits,
            'task_input': task_logits
        }
```

Then your loss can expect the filtered output. IMPORTANT: this functionality is not yet implemented. We will add it soon in future releases.

```python
def forward(self, concept_input, task_input, concept_target, task_target):
    # Use the filtered inputs
    ...
```

## Testing

```python
import torch
from conceptarium.nn.losses import YourCustomLoss

# Initialize
loss_fn = YourCustomLoss(param1=value1)

# Test with dummy data
batch_size = 16
n_concepts = 5

predictions = torch.randn(batch_size, n_concepts)
targets = torch.randint(0, 2, (batch_size, n_concepts)).float()

# Compute loss
loss = loss_fn(predictions, targets)

print(f"Loss value: {loss.item():.4f}")
print(f"Loss shape: {loss.shape}")  # Should be scalar: torch.Size([])

# Test backward pass
loss.backward()
```

## Summary

**Required steps:**
1. Create loss class in `conceptarium/conceptarium/nn/losses/your_loss.py`
2. Implement `__init__` and `forward` methods
3. Update `__init__.py` to export your loss
4. Create configuration file `conceptarium/conf/engine/loss/your_loss.yaml`
5. Test loss computation and gradients

**Key points:**
- Extend `torch.nn.Module` or existing PyTorch loss
- `forward()` should return a scalar tensor
- Configuration uses `path` (import path) and `kwargs` (parameters)
- Different losses can be specified for binary, categorical, and continuous concepts
- Override model's `filter_output_for_loss()` for custom input formats
