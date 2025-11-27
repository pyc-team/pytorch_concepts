# Contributing a New Metric

This guide will help you implement custom metrics for concept-based models in <img src="../../doc/_static/img/logos/pyc.svg" width="25px" align="center"/> PyC and use them in <img src="../../doc/_static/img/logos/conceptarium.svg" width="25px" align="center"/> Conceptarium. The library provides a flexible metrics system that integrates seamlessly with TorchMetrics while allowing for custom implementations.

## Prerequisites

Before implementing a custom metric, ensure you:
- Know whether your metric applies to binary, categorical, or continuous concepts
- Determine if the metric requires non-standard inputs (beyond predictions and targets)
- Are familiar with TorchMetrics if using their metrics

## Recommended Approach: Use TorchMetrics When Possible

**The preferred approach is to use existing [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) whenever possible.** TorchMetrics provides a comprehensive collection of metrics. Only implement custom metrics when:
1. Your metric is not available in TorchMetrics
2. You need specialized behavior for concept-based models
3. You require custom input handling beyond standard `(preds, target)` pairs

## Part 1: Using TorchMetrics Metrics

### 1.1 Understanding GroupConfig

The `GroupConfig` object organizes metrics by concept type (binary, categorical, continuous). This allows PyC to automatically route concept predictions to the appropriate metrics.

```python
from torch_concepts.nn.modules.utils import GroupConfig
from torch_concepts.nn.modules.metrics import ConceptMetrics
import torchmetrics

# Basic usage with GroupConfig
metrics = ConceptMetrics(
    annotations=concept_annotations,
    fn_collection=GroupConfig(
        binary={
            'accuracy': torchmetrics.classification.BinaryAccuracy(),
            'f1': torchmetrics.classification.BinaryF1Score()
        },
        categorical={
            'accuracy': torchmetrics.classification.MulticlassAccuracy
        }
    ),
    summary_metrics=True,
    perconcept_metrics=False
)
```

### 1.2 Three Ways to Specify Metrics

PyC supports three flexible ways to specify metrics in the `GroupConfig`:

#### Method 1: Pre-instantiated Metrics (Full Control)

Use this when you need complete control over metric initialization:

```python
fn_collection=GroupConfig(
    binary={
        'accuracy': torchmetrics.classification.BinaryAccuracy(threshold=0.6),
        'f1': torchmetrics.classification.BinaryF1Score(threshold=0.5)
    },
    categorical={
        # For summary metrics: manually specify the max cardinality
        'accuracy': torchmetrics.classification.MulticlassAccuracy(
            num_classes=4,  # max cardinality across all categorical concepts
            average='micro'
        )
    }
)
```

**Pros**: Full control over all parameters  
**Cons**: Must manually handle `num_classes` for categorical metrics. Not applicable for per-concept metrics since cardinalities vary.

#### Method 2: Class + User kwargs (Recommended)

Use this to provide custom kwargs while letting PyC handle concept-specific parameters:

```python
fn_collection=GroupConfig(
    binary={
        # Provide custom threshold, other params use defaults
        'accuracy': (torchmetrics.classification.BinaryAccuracy, {'threshold': 0.5}),
    },
    categorical={
        # Provide averaging strategy, PyC adds num_classes automatically
        'accuracy': (torchmetrics.classification.MulticlassAccuracy, {'average': 'macro'}),
        'f1': (torchmetrics.classification.MulticlassF1Score, {'average': 'weighted'})
    }
)
```

**Pros**: Custom parameters + automatic `num_classes` handling  
**Cons**: More verbose

#### Method 3: Class Only (Simplest)

Use this when you want all defaults with automatic concept-specific parameters:

```python
fn_collection=GroupConfig(
    binary={
        'accuracy': torchmetrics.classification.BinaryAccuracy,
        'precision': torchmetrics.classification.BinaryPrecision,
        'recall': torchmetrics.classification.BinaryRecall
    },
    categorical={
        # PyC automatically adds num_classes per concept
        'accuracy': torchmetrics.classification.MulticlassAccuracy
    }
)
```

**Pros**: Simplest syntax, automatic parameter handling  
**Cons**: Cannot customize parameters

### 1.3 Summary vs Per-Concept Metrics

Control metric granularity with `summary_metrics` and `perconcept_metrics`:

```python
metrics = ConceptMetrics(
    annotations=annotations,
    fn_collection=GroupConfig(...),
    summary_metrics=True,    # Aggregate metrics across all concepts of each type
    perconcept_metrics=True  # Track each concept individually
)
```

Options for `perconcept_metrics`:
- `False`: No per-concept tracking
- `True`: Track all concepts individually
- `['concept1', 'concept2']`: Track only specified concepts

**Example output structure:**
```python
{
    'train/SUMMARY-binary_accuracy': 0.85,      # All binary concepts
    'train/SUMMARY-categorical_accuracy': 0.72, # All categorical concepts
    'train/concept1_accuracy': 0.90,            # Individual concept
    'train/concept2_accuracy': 0.80,            # Individual concept
}
```

### 1.4 Usage in Conceptarium

Create a config file at `conceptarium/conf/metrics/<your_config>.yaml`:

```yaml
# conceptarium/conf/metrics/standard.yaml
_target_: "torch_concepts.nn.ConceptMetrics"

summary_metrics: true
perconcept_metrics: true  # or list of concept names: ${dataset.default_task_names}

fn_collection:
  _target_: "torch_concepts.nn.modules.utils.GroupConfig"
  
  binary: 
    accuracy: 
      _target_: "torchmetrics.classification.BinaryAccuracy"
    f1:
      - _target_: "hydra.utils.get_class"
        path: "torchmetrics.classification.BinaryF1Score"
      - threshold: 0.5  # User kwargs
  
  categorical: 
    accuracy:
      - _target_: "hydra.utils.get_class"
        path: "torchmetrics.classification.MulticlassAccuracy"
      - average: 'micro'  # User kwargs, num_classes added automatically
    
  # continuous: 
    # ... not supported yet
```

**Run your experiment:**
```bash
python conceptarium/run_experiment.py metrics=standard
```

## Part 2: Custom Metric Implementation

### 2.1 When to Implement a Custom Metric

Implement a custom metric when:
- Your metric is not available in TorchMetrics
- You need specialized computation for concept-based models
- Your metric requires non-standard inputs (e.g., causal effects, interventions)

### 2.2 Custom Metric Structure

Custom metrics should inherit from `torchmetrics.Metric` and implement three methods:

```python
from torchmetrics import Metric
import torch

class YourCustomMetric(Metric):
    """Your custom metric for concept-based models.
    
    Brief description of what the metric measures and when to use it.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Example:
        >>> metric = YourCustomMetric(param1=value)
        >>> metric.update(preds, target)
        >>> result = metric.compute()
    """
    
    def __init__(self, param1=None, param2=None):
        super().__init__()
        
        # Add metric state variables
        # These accumulate values across batches
        self.add_state("state_var1", 
                      default=torch.tensor(0.0), 
                      dist_reduce_fx="sum")
        self.add_state("state_var2", 
                      default=torch.tensor(0), 
                      dist_reduce_fx="sum")
        
        # Store configuration parameters
        self.param1 = param1
        self.param2 = param2
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state with batch predictions and targets.
        
        Args:
            preds: Model predictions, shape (batch_size, ...)
            target: Ground truth labels, shape (batch_size, ...)
        """
        # Validate inputs
        assert preds.shape == target.shape, "Predictions and targets must have same shape"
        
        # Update state variables
        self.state_var1 += compute_something(preds, target)
        self.state_var2 += preds.size(0)
    
    def compute(self):
        """Compute final metric value from accumulated state.
        
        Returns:
            torch.Tensor: Final metric value
        """
        return self.state_var1.float() / self.state_var2
```

### 2.3 Add Custom Metric to torch_concepts

Place your custom metric in `torch_concepts/nn/modules/metrics.py`:

```python
# In torch_concepts/nn/modules/metrics.py

class ConceptDependencyScore(Metric):
    """Measure correlation between concept predictions.
    
    Computes pairwise correlation between concept predictions to identify
    potential dependencies in the concept space.
    
    Args:
        n_concepts (int): Number of concepts
        
    Example:
        >>> metric = ConceptDependencyScore(n_concepts=5)
        >>> metric.update(concept_preds, target)
        >>> correlation_matrix = metric.compute()
    """
    
    def __init__(self, n_concepts: int):
        super().__init__()
        self.n_concepts = n_concepts
        self.add_state("sum_products", 
                      default=torch.zeros(n_concepts, n_concepts),
                      dist_reduce_fx="sum")
        self.add_state("sum_preds", 
                      default=torch.zeros(n_concepts),
                      dist_reduce_fx="sum")
        self.add_state("total", 
                      default=torch.tensor(0),
                      dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update correlation statistics.
        
        Args:
            preds: Concept predictions (batch_size, n_concepts)
            target: Ground truth (unused, for interface compatibility)
        """
        batch_size = preds.size(0)
        
        # Compute pairwise products
        self.sum_products += preds.T @ preds
        self.sum_preds += preds.sum(dim=0)
        self.total += batch_size
    
    def compute(self):
        """Compute correlation matrix."""
        mean_preds = self.sum_preds / self.total
        cov = self.sum_products / self.total - torch.outer(mean_preds, mean_preds)
        return cov
```

### 2.4 Usage with GroupConfig

Add your custom metric to the appropriate concept type group:

```python
from torch_concepts.nn.modules.metrics import ConceptMetrics, ConceptDependencyScore
from torch_concepts.nn.modules.utils import GroupConfig

metrics = ConceptMetrics(
    annotations=annotations,
    fn_collection=GroupConfig(
        binary={
            'accuracy': torchmetrics.classification.BinaryAccuracy,
            'dependency': ConceptDependencyScore(n_concepts=len(binary_concepts))
        }
    ),
    summary_metrics=True,
    perconcept_metrics=False
)
```

## Part 3: Advanced Custom Metrics

### 3.1 Metrics with Non-Standard Inputs

If your metric requires inputs beyond standard `(preds, target)` pairs, you need to modify how the model passes data to metrics.

**Step 1: Identify what additional inputs you need**

Examples:
- Causal effect metrics: need predictions under different interventions
- Attention metrics: need attention weights from the model
- Intervention metrics: need pre/post intervention predictions

**Step 2: Override `filter_output_for_metrics` in your model**

The `filter_output_for_metrics` method controls what gets passed to metrics. Override it in your model class:

```python
# In your model class (e.g., in torch_concepts/nn/modules/high/models/your_model.py)

class YourModel(BaseModel, JointLearner):
    def filter_output_for_metrics(self, forward_out, target):
        """Filter model outputs for metric computation.
        
        Args:
            forward_out: Raw model output (dict or tensor)
            target: Ground truth concepts
            
        Returns:
            dict: Arguments to pass to metrics
        """
        # Standard case: return predictions and targets
        # This is what ConceptMetrics expects by default
        return {
            'preds': forward_out['concept_logits'],
            'target': target
        }
        
        # Advanced case: return custom inputs for special metrics
        # return {
        #     'preds': forward_out['concept_logits'],
        #     'target': target,
        #     'attention_weights': forward_out['attention'],
        #     'interventions': forward_out['interventions']
        # }
```

**Step 3: Modify `update_and_log_metrics` in the Learner**

If your metric arguments don't match the standard `(preds, target)` signature, override `update_and_log_metrics`:

```python
# In torch_concepts/nn/modules/high/base/learner.py or your custom learner

def update_and_log_metrics(self, metrics_args: Mapping, step: str, batch_size: int):
    """Update metrics and log them.
    
    Args:
        metrics_args (Mapping): Arguments from filter_output_for_metrics
        step (str): Which split to update ('train', 'val', or 'test')
        batch_size (int): Batch size for logging
    """
    # Standard metrics use 'preds' and 'target'
    if 'preds' in metrics_args and 'target' in metrics_args:
        preds = metrics_args['preds']
        target = metrics_args['target']
        self.update_metrics(preds, target, step)
    
    # Custom metrics with additional inputs
    # You would need to modify ConceptMetrics.update() to handle these
    # or create a separate metric collection for special metrics
    
    # Log computed metrics
    collection = getattr(self, f"{step}_metrics", None)
    if collection is not None:
        self.log_metrics(collection, batch_size=batch_size)
```

### 3.2 Example: Causal Effect Metric

Here's a complete example of a metric requiring custom inputs:

```python
# In torch_concepts/nn/modules/metrics.py

class ConceptCausalEffect(Metric):
    """Concept Causal Effect (CaCE) metric.
    
    Measures the causal effect between concepts or between concepts and tasks
    by comparing predictions under interventions do(C=1) vs do(C=0).
    
    Args:
        None
        
    Example:
        >>> cace = ConceptCausalEffect()
        >>> # Requires special input handling
        >>> cace.update(preds_do_1, preds_do_0)
        >>> effect = cace.compute()
        
    References:
        Goyal et al. "Explaining Classifiers with Causal Concept Effect (CaCE)",
        arXiv 2019. https://arxiv.org/abs/1907.07165
    """
    
    def __init__(self):
        super().__init__()
        self.add_state("preds_do_1", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("preds_do_0", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds_do_1: torch.Tensor, preds_do_0: torch.Tensor):
        """Update with predictions under interventions.
        
        Note: This has a different signature than standard metrics!
        You need to handle this in your model's filter_output_for_metrics.
        
        Args:
            preds_do_1: Predictions when C=1, shape (batch_size, n_classes)
            preds_do_0: Predictions when C=0, shape (batch_size, n_classes)
        """
        assert preds_do_1.shape == preds_do_0.shape
        # Expected value under intervention do(C=1)
        self.preds_do_1 += preds_do_1[:, 1].sum()
        # Expected value under intervention do(C=0)
        self.preds_do_0 += preds_do_0[:, 1].sum()
        self.total += preds_do_1.size(0)
    
    def compute(self):
        """Compute causal effect."""
        return (self.preds_do_1.float() / self.total) - (self.preds_do_0.float() / self.total)
```

**Using this metric requires custom handling:**

```python
# In your model
class YourModelWithCausalMetrics(BaseModel, JointLearner):
    def forward(self, x, query, compute_causal=False):
        # Standard forward pass
        out = self.predict_concepts(x, query)
        
        if compute_causal and self.training is False:
            # Compute predictions under interventions during validation/test
            out['preds_do_1'] = self.intervene(x, query, concept_value=1)
            out['preds_do_0'] = self.intervene(x, query, concept_value=0)
        
        return out
    
    def filter_output_for_metrics(self, forward_out, target):
        """Handle both standard and causal metrics."""
        metrics_args = {
            'preds': forward_out['concept_logits'],
            'target': target
        }
        
        # Add causal effect inputs if available
        if 'preds_do_1' in forward_out:
            metrics_args['preds_do_1'] = forward_out['preds_do_1']
            metrics_args['preds_do_0'] = forward_out['preds_do_0']
        
        return metrics_args
```

## Part 4: Testing Your Metric

### 4.1 Unit Testing

Create tests in `tests/nn/modules/metrics/test_your_metric.py`:

```python
import unittest
import torch
from torch_concepts.nn.modules.metrics import YourCustomMetric

class TestYourCustomMetric(unittest.TestCase):
    def test_initialization(self):
        """Test metric initializes correctly."""
        metric = YourCustomMetric(param1=value)
        self.assertIsNotNone(metric)
    
    def test_update_and_compute(self):
        """Test metric computation."""
        metric = YourCustomMetric()
        
        # Create sample data
        preds = torch.randn(10, 5)
        target = torch.randint(0, 2, (10, 5))
        
        # Update metric
        metric.update(preds, target)
        
        # Compute result
        result = metric.compute()
        
        # Verify result
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.isfinite(result).all())
    
    def test_reset(self):
        """Test metric reset."""
        metric = YourCustomMetric()
        metric.update(torch.randn(5, 3), torch.randint(0, 2, (5, 3)))
        metric.reset()
        
        # After reset, state should be back to defaults
        self.assertEqual(metric.state_var1, 0.0)
```

### 4.2 Integration Testing

Test your metric with ConceptMetrics:

```python
def test_custom_metric_with_concept_metrics(self):
    """Test custom metric integrates with ConceptMetrics."""
    from torch_concepts import Annotations, AxisAnnotation
    from torch_concepts.nn.modules.metrics import ConceptMetrics
    from torch_concepts.nn.modules.utils import GroupConfig
    
    # Create annotations
    annotations = Annotations({
        1: AxisAnnotation(
            labels=['c1', 'c2'],
            metadata={
                'c1': {'type': 'discrete'},
                'c2': {'type': 'discrete'}
            },
            cardinalities=[1, 1]
        )
    })
    
    # Create metrics with your custom metric
    metrics = ConceptMetrics(
        annotations=annotations,
        fn_collection=GroupConfig(
            binary={
                'custom': YourCustomMetric(param1=value)
            }
        ),
        summary_metrics=True
    )
    
    # Test update and compute
    preds = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8, 2))
    
    metrics.update(preds, targets, split='train')
    results = metrics.compute('train')
    
    self.assertIn('train/SUMMARY-binary_custom', results)
```

## Summary

**Recommended workflow:**

1. **Start with TorchMetrics**: Use existing metrics whenever possible
2. **Use GroupConfig**: Organize metrics by concept type (binary/categorical/continuous)
3. **Choose initialization method**: 
   - Pre-instantiated for full control
   - Class + kwargs (tuple) for custom params + automatic handling
   - Class only for simplest usage
4. **Configure in Conceptarium**: Create YAML configs for experiments
5. **Custom metrics only when needed**: Inherit from `torchmetrics.Metric`
6. **Handle non-standard inputs**: Override `filter_output_for_metrics` and `update_and_log_metrics`
7. **Test thoroughly**: Write unit and integration tests

**Key files:**
- Metric implementations: `torch_concepts/nn/modules/metrics.py`
- Conceptarium configs: `conceptarium/conf/metrics/`
- Model output filtering: Override `filter_output_for_metrics` in your model
- Learner metric handling: Modify `update_and_log_metrics` in `BaseLearner` if needed
