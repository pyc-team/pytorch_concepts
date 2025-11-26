# Contributing a New Metric

This guide explains how to implement custom metrics for the `pytorch_concepts` library.

## When to Implement a Custom Metric

Implement a custom metric when:
- You need domain-specific evaluation measures
- Standard metrics don't capture your model's performance adequately
- You require specialized aggregation across concepts
- You want custom intervention-specific metrics

**Note**: The library integrates with [TorchMetrics](https://torchmetrics.readthedocs.io/), so most standard metrics are already available.

## Recommended Approach: Use TorchMetrics

For most cases, use existing TorchMetrics without custom implementation:

```yaml
# conf/engine/metrics/default.yaml
discrete:
  binary:
    accuracy: 
      path: "torchmetrics.classification.BinaryAccuracy"
      kwargs: {}
    f1: 
      path: "torchmetrics.classification.BinaryF1Score"
      kwargs: {}
    auroc:
      path: "torchmetrics.classification.BinaryAUROC"
      kwargs: {}
```

## Custom Implementation

Only implement custom metrics when TorchMetrics doesn't cover your needs.

### 1. Create Metric Class

Place your metric in `conceptarium/conceptarium/nn/metrics/your_metric.py`:

```python
import torch
from torchmetrics import Metric


class YourCustomMetric(Metric):
    """Custom metric for [specific use case].
    
    This metric computes [description of what it measures].
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        **kwargs: Additional arguments passed to Metric base class
    """
    
    def __init__(self, param1=default_value, param2=default_value, **kwargs):
        super().__init__(**kwargs)
        
        # Parameters
        self.param1 = param1
        self.param2 = param2
        
        # State variables (accumulated across batches)
        self.add_state("state_var1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("state_var2", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state with batch data.
        
        Args:
            preds: Model predictions (batch_size, ...)
            target: Ground truth labels (batch_size, ...)
        """
        # Update your state variables
        # These accumulate across batches
        batch_result = ...  # Compute batch-level result
        self.state_var1 += batch_result
        self.state_var2 += preds.size(0)
    
    def compute(self):
        """Compute final metric value from accumulated state.
        
        Returns:
            Scalar tensor with metric value
        """
        # Compute final metric from state variables
        return self.state_var1 / self.state_var2
```

### 2. Register Metric

Update `conceptarium/conceptarium/nn/metrics/__init__.py`:

```python
from .your_metric import YourCustomMetric

__all__ = ['YourCustomMetric']
```

## Configuration

### 1. Create Metric Configuration

Create or update `conceptarium/conf/engine/metrics/your_metrics.yaml`. Remember that conceptarium supports different metrics for discrete (classification) and continuous (regression) concepts. Also remember that conceptarium implements an option to aggregate metrics across concepts, so concept-specific metrics are not supported right now.

```yaml
# Metrics for discrete (classification) concepts
discrete:
  binary:  # Binary concepts
    accuracy: 
      path: "torchmetrics.classification.BinaryAccuracy"
      kwargs: {}
    custom_metric:
      path: "conceptarium.nn.metrics.YourCustomMetric"
      kwargs:
        param1: value1
        param2: value2
        
  categorical:  # Multi-class concepts
    accuracy: 
      path: "torchmetrics.classification.MulticlassAccuracy"
      kwargs: 
        average: 'micro'
    custom_metric:
      path: "conceptarium.nn.metrics.YourCustomMetric"
      kwargs:
        param1: value1

# Metrics for continuous (regression) concepts
# continuous: 
  # ... not supported yet
```

## Model Integration

If your metric requires special input format, override `filter_output_for_metric` in your model:

```python
class YourModel(BaseModel):
    def filter_output_for_metric(self, forward_out):
        """Process model output for metrics.
        
        Example: Apply activation function
        """
        # Convert endogenous to probabilities
        return torch.sigmoid(forward_out)
```

## Testing

```python
import torch
from conceptarium.nn.metrics import YourCustomMetric

# Initialize
metric = YourCustomMetric(param1=value1)

# Test with dummy data
batch_size = 16
n_concepts = 5

predictions = torch.randn(batch_size, n_concepts)
targets = torch.randint(0, 2, (batch_size, n_concepts)).float()

# Update metric
metric.update(predictions, targets)

# Compute result
result = metric.compute()
print(f"Metric value: {result}")
print(f"Metric shape: {result.shape}")

# Reset
metric.reset()
assert metric.state_var1 == 0  # Check state reset
```

## Summary

**Recommended approach:**
- Use TorchMetrics for standard metrics (no custom code needed)
- Only implement custom metrics for specialized use cases

**If implementing custom metrics:**
1. Create metric class extending `torchmetrics.Metric`
2. Implement `__init__`, `update`, and `compute` methods
3. Use `add_state()` to track values across batches
4. Update `__init__.py` to export your metric
5. Create/update configuration file with metric path and kwargs
6. Test metric with dummy data
