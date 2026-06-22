"""
Example: Flexible Metric Initialization in ConceptMetrics

This example demonstrates the three ways to specify metrics in ConceptMetrics:
1. Pre-instantiated metrics
2. Metric class with user-provided kwargs (as tuple)
3. Metric class only (concept-specific params added automatically)

This flexibility allows you to:
- Use pre-configured metrics when you need full control
- Pass custom kwargs while letting ConceptMetrics handle concept-specific params
- Let ConceptMetrics fully handle metric instantiation for simplicity
"""

import torch
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.nn.modules.utils import GroupConfig
from torch.distributions import Bernoulli, Categorical
import torchmetrics

def main():
    print("=" * 60)
    print("Flexible Metric Initialization Example")
    print("=" * 60)
    
    # Create annotations with mixed concept types
    concept_names = ['binary1', 'binary2', 'cat1', 'cat2']
    annotations = Annotations({
        1: AxisAnnotation(
            labels=tuple(concept_names),
            types=['binary', 'binary', 'categorical', 'categorical'],
            cardinalities=[1, 1, 3, 4],
        )
    })
    
    print("\nAnnotations:")
    print(f"  Concepts: {concept_names}")
    print(f"  Types: {annotations[1].types}")
    print(f"  Cardinalities: {annotations[1].cardinalities}")
   
    # Three ways to specify metrics

    # Method 1: Pre-instantiated metrics
    print("\n" + "=" * 60)
    print("Method 1: Pre-instantiated metrics")
    print("=" * 60)
    
    metrics1 = ConceptMetrics(
        annotations=annotations,
        summary=True,
        per_concept=False, # this must be set to False when pre-instantiating metrics
                           # for categorical concepts (read below)
        binary={
            'accuracy': torchmetrics.classification.BinaryAccuracy(),
            'mcc': torchmetrics.classification.BinaryMatthewsCorrCoef(),
            'f1': torchmetrics.classification.BinaryF1Score()
        },
        categorical={
            # Metrics for categorical concepts usually require num_classes. Since this could differ
            # among different concepts, 'per_concept' metrics for categorical concepts cannot be pre-instantiated.
            # Instead, 'summary' metrics can be pre-instantiated using the maximum cardinality
            # across all categorical concepts (in this case = 4) 
            'accuracy': torchmetrics.classification.MulticlassAccuracy(num_classes=4, average='micro')
        }
    )
    print(f"✓ Created metrics with pre-instantiated objects")
    print(f"  {metrics1}")
    
    # Method 2: Just the class (simplest)
    print("\n" + "=" * 60)
    print("Method 2: Metric class only")
    print("=" * 60)
    
    metrics2 = ConceptMetrics(
        annotations=annotations,
        summary=True,
        per_concept=True,
        binary={
            'accuracy': torchmetrics.classification.BinaryAccuracy,
            'precision': torchmetrics.classification.BinaryPrecision,
            'recall': torchmetrics.classification.BinaryRecall
        },
        categorical={
            # If only the (non-instantiated) class if provided, 
            # num_classes will be handled internally and automatically
            'accuracy': torchmetrics.classification.MulticlassAccuracy
        }
    )
    print(f"✓ Created metrics with just metric classes")
    print(f"  ConceptMetrics handles all instantiation")
    print(f"  {metrics2}")

    # Method 3: Class + user kwargs (as tuple)
    print("\n" + "=" * 60)
    print("Method 3: Metric class with user kwargs (tuple)")
    print("=" * 60)
    
    metrics3 = ConceptMetrics(
        annotations=annotations,
        summary=True,
        per_concept=['cat1', 'cat2'],  # Track individual categorical concepts
        binary={
            'accuracy': (torchmetrics.classification.BinaryAccuracy, {'threshold': 0.5}),
        },
        categorical={
            # Again, ConceptMetrics handle 'num_classes' internally
            'accuracy': (torchmetrics.classification.MulticlassAccuracy, {'average': 'macro'})
        }
    )
    print(f"✓ Created metrics with (class, kwargs) tuples")
    print(f"  User provided: threshold=0.5, average='macro'")
    print(f"  ConceptMetrics added: num_classes automatically per concept")
    print(f"  {metrics3}")

    # Mixed approach (most flexible)
    print("\n" + "=" * 60)
    print("Method 4: Mix all three approaches")
    print("=" * 60)
    
    metrics_mixed = ConceptMetrics(
        annotations=annotations,
        summary=True,
        per_concept=True,
        binary={
            # Pre-instantiated
            'accuracy': torchmetrics.classification.BinaryAccuracy(),
            # Class + kwargs
            'f1': (torchmetrics.classification.BinaryF1Score, {'threshold': 0.5}),
            # Class only
            'precision': torchmetrics.classification.BinaryPrecision
        },
        categorical={
            # Class + kwargs
            'accuracy': (torchmetrics.classification.MulticlassAccuracy, {'average': 'weighted'}),
            # Class only
            'f1': torchmetrics.classification.MulticlassF1Score
        }
    )
    print(f"✓ Created metrics mixing all three approaches")
    print(f"  This gives maximum flexibility!")
    print(f"  {metrics_mixed}")
    
    # Test with actual data
    print("\n" + "=" * 60)
    print("Testing metrics with sample data")
    print("=" * 60)
    
    batch_size = 16
    # Endogenous: 2 binary + (3 + 4) categorical = 9 dimensions
    endogenous = torch.randn(batch_size, 9)
    targets = torch.cat([
        torch.randint(0, 2, (batch_size, 2)),  # binary concepts
        torch.randint(0, 3, (batch_size, 1)),  # cat1 (3 classes)
        torch.randint(0, 4, (batch_size, 1)),  # cat2 (4 classes)
    ], dim=1)
    
    metrics_mixed.update(endogenous, targets)
    results = metrics_mixed.compute()
    
    print(f"\nComputed metrics ({len(results)} total):")
    for key in sorted(results.keys()):
        value = results[key].item() if hasattr(results[key], 'item') else results[key]
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
