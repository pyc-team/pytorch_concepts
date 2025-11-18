"""Comprehensive test for the Predictor class with actual imports."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch_concepts import AxisAnnotation


def create_mock_annotations(concept_names, cardinalities, tasks, is_nested):
    """Create mock annotations matching the actual structure."""
    # Create a mock annotations object
    class MockAnnotations:
        def __init__(self, concept_names, cardinalities, tasks, is_nested):
            self.labels = concept_names
            self.cardinalities = cardinalities
            self.is_nested = is_nested
            self.metadata = {
                name: {'task': task}
                for name, task in zip(concept_names, tasks)
            }
        
        def get_index(self, name):
            return self.labels.index(name)
    
    return MockAnnotations(concept_names, cardinalities, tasks, is_nested)


def create_mock_model(concept_names, cardinalities, tasks, is_nested):
    """Create a mock model for testing."""
    class MockModel(nn.Module):
        def __init__(self, concept_names, cardinalities, tasks, is_nested):
            super().__init__()
            self.probabilistic_model = None
            self.annotations = type('obj', (object,), {
                'get_axis_annotation': lambda self, axis: create_mock_annotations(
                    concept_names, cardinalities, tasks, is_nested
                )
            })()
            
        def filter_output_for_loss(self, x):
            return x
        
        def filter_output_for_metric(self, x):
            return x
    
    return MockModel(concept_names, cardinalities, tasks, is_nested)


def test_binary_dense_summary_only():
    """Test binary dense format with summary metrics only."""
    print("\n" + "="*70)
    print("TEST 1: Binary Dense - Summary Metrics Only")
    print("="*70)
    
    from conceptarium.engines.predictor import Predictor
    
    concept_names = ['c0', 'c1', 'c2']
    cardinalities = [1, 1, 1]
    tasks = ['classification', 'classification', 'classification']
    is_nested = False
    
    model = create_mock_model(concept_names, cardinalities, tasks, is_nested)
    
    loss_config = {
        'classification': {'binary': 'torch.nn.BCEWithLogitsLoss'},
        'regression': 'torch.nn.MSELoss'
    }
    
    metrics_config = {
        'classification': {
            'binary': {
                'accuracy': 'torchmetrics.classification.BinaryAccuracy',
                'f1': 'torchmetrics.classification.BinaryF1Score'
            }
        }
    }
    
    predictor = Predictor(
        model=model,
        train_inference=lambda x: None,
        loss=loss_config,
        metrics=metrics_config,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        enable_summary_metrics=True,
        enable_perconcept_metrics=False
    )
    
    print(f"Concept names: {predictor.concept_names}")
    print(f"Tasks: {predictor.tasks}")
    print(f"Cardinalities: {predictor.cardinalities}")
    print(f"Is nested: {predictor.is_nested}")
    print(f"Binary concept IDs: {predictor.binary_concept_ids}")
    print(f"Train metrics: {list(predictor.train_metrics.keys())}")
    
    # Test loss computation
    batch_size = 4
    c_hat = torch.randn(batch_size, 3)
    c_true = torch.randint(0, 2, (batch_size, 3)).float()
    
    loss = predictor._compute_loss(c_hat, c_true)
    print(f"\nLoss computed: {loss.item():.4f}")
    
    # Test metrics update
    predictor._update_metrics(c_hat, c_true, predictor.train_metrics)
    results = predictor.train_metrics.compute()
    
    print(f"Metrics computed:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")
    
    assert 'train/binary_accuracy' in results
    assert 'train/binary_f1' in results
    assert len(results) == 2
    print("✓ Test passed!")


def test_binary_dense_perconcept_only():
    """Test binary dense format with per-concept metrics only."""
    print("\n" + "="*70)
    print("TEST 2: Binary Dense - Per-Concept Metrics Only")
    print("="*70)
    
    from conceptarium.engines.predictor import Predictor
    
    concept_names = ['c0', 'c1', 'c2']
    cardinalities = [1, 1, 1]
    tasks = ['classification', 'classification', 'classification']
    is_nested = False
    
    model = create_mock_model(concept_names, cardinalities, tasks, is_nested)
    
    loss_config = {
        'classification': {'binary': 'torch.nn.BCEWithLogitsLoss'},
        'regression': 'torch.nn.MSELoss'
    }
    
    metrics_config = {
        'classification': {
            'binary': {
                'accuracy': 'torchmetrics.classification.BinaryAccuracy',
                'f1': 'torchmetrics.classification.BinaryF1Score'
            }
        }
    }
    
    predictor = Predictor(
        model=model,
        train_inference=lambda x: None,
        loss=loss_config,
        metrics=metrics_config,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        enable_summary_metrics=False,
        enable_perconcept_metrics=True
    )
    
    print(f"Train metrics: {list(predictor.train_metrics.keys())}")
    
    # Test with data
    batch_size = 4
    c_hat = torch.randn(batch_size, 3)
    c_true = torch.randint(0, 2, (batch_size, 3)).float()
    
    predictor._update_metrics(c_hat, c_true, predictor.train_metrics)
    results = predictor.train_metrics.compute()
    
    print(f"Metrics computed:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")
    
    assert 'train/c0_accuracy' in results
    assert 'train/c0_f1' in results
    assert 'train/c1_accuracy' in results
    assert 'train/c2_accuracy' in results
    assert len(results) == 6  # 3 concepts * 2 metrics
    print("✓ Test passed!")


def test_binary_dense_both():
    """Test binary dense format with both metric types."""
    print("\n" + "="*70)
    print("TEST 3: Binary Dense - Both Summary and Per-Concept Metrics")
    print("="*70)
    
    from conceptarium.engines.predictor import Predictor
    
    concept_names = ['c0', 'c1', 'c2']
    cardinalities = [1, 1, 1]
    tasks = ['classification', 'classification', 'classification']
    is_nested = False
    
    model = create_mock_model(concept_names, cardinalities, tasks, is_nested)
    
    loss_config = {
        'classification': {'binary': 'torch.nn.BCEWithLogitsLoss'},
        'regression': 'torch.nn.MSELoss'
    }
    
    metrics_config = {
        'classification': {
            'binary': {
                'accuracy': 'torchmetrics.classification.BinaryAccuracy'
            }
        }
    }
    
    predictor = Predictor(
        model=model,
        train_inference=lambda x: None,
        loss=loss_config,
        metrics=metrics_config,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        enable_summary_metrics=True,
        enable_perconcept_metrics=True
    )
    
    print(f"Train metrics: {list(predictor.train_metrics.keys())}")
    
    batch_size = 4
    c_hat = torch.randn(batch_size, 3)
    c_true = torch.randint(0, 2, (batch_size, 3)).float()
    
    predictor._update_metrics(c_hat, c_true, predictor.train_metrics)
    results = predictor.train_metrics.compute()
    
    print(f"Metrics computed:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")
    
    assert 'train/binary_accuracy' in results  # Summary
    assert 'train/c0_accuracy' in results  # Per-concept
    assert len(results) == 4  # 1 summary + 3 per-concept
    print("✓ Test passed!")


def test_mixed_nested_summary():
    """Test mixed nested format with summary metrics."""
    print("\n" + "="*70)
    print("TEST 4: Mixed Nested - Summary Metrics")
    print("="*70)
    
    from conceptarium.engines.predictor import Predictor
    
    concept_names = ['c0', 'c1', 'c2', 'c3']
    cardinalities = [1, 3, 1, 1]  # binary, categorical, binary, regression
    tasks = ['classification', 'classification', 'classification', 'regression']
    is_nested = True
    
    model = create_mock_model(concept_names, cardinalities, tasks, is_nested)
    
    loss_config = {
        'classification': {
            'binary': 'torch.nn.BCEWithLogitsLoss',
            'categorical': 'torch.nn.NLLLoss'
        },
        'regression': 'torch.nn.MSELoss'
    }
    
    metrics_config = {
        'classification': {
            'binary': {
                'accuracy': 'torchmetrics.classification.BinaryAccuracy'
            },
            'categorical': {
                'accuracy': 'torchmetrics.classification.MulticlassAccuracy'
            }
        },
        'regression': {
            'mae': 'torchmetrics.regression.MeanAbsoluteError'
        }
    }
    
    predictor = Predictor(
        model=model,
        train_inference=lambda x: None,
        loss=loss_config,
        metrics=metrics_config,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        enable_summary_metrics=True,
        enable_perconcept_metrics=False
    )
    
    print(f"Binary concept IDs: {predictor.binary_concept_ids}")
    print(f"Categorical concept IDs: {predictor.categorical_concept_ids}")
    print(f"Regression concept IDs: {predictor.regression_concept_ids}")
    print(f"Train metrics: {list(predictor.train_metrics.keys())}")
    
    # Test loss computation with nested tensors
    batch_size = 4
    c_hat = torch.cat([
        torch.randn(batch_size, 1),
        torch.randn(batch_size, 3),
        torch.randn(batch_size, 1),
        torch.randn(batch_size, 1)
    ], dim=1)
    
    c_true = torch.stack([
        torch.randint(0, 2, (batch_size,)).float(),
        torch.randint(0, 3, (batch_size,)).float(),
        torch.randint(0, 2, (batch_size,)).float(),
        torch.randn(batch_size)
    ], dim=1)
    
    loss = predictor._compute_loss(c_hat, c_true)
    print(f"\nLoss computed: {loss.item():.4f}")
    
    predictor._update_metrics(c_hat, c_true, predictor.train_metrics)
    results = predictor.train_metrics.compute()
    
    print(f"Metrics computed:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")
    
    assert 'train/binary_accuracy' in results
    assert 'train/categorical_accuracy' in results
    assert 'train/regression_mae' in results
    assert len(results) == 3
    print("✓ Test passed!")


def test_mixed_nested_both():
    """Test mixed nested format with both metric types."""
    print("\n" + "="*70)
    print("TEST 5: Mixed Nested - Both Metrics")
    print("="*70)
    
    from conceptarium.engines.predictor import Predictor
    
    concept_names = ['c0', 'c1', 'c2']
    cardinalities = [1, 3, 1]  # binary, categorical, binary
    tasks = ['classification', 'classification', 'classification']
    is_nested = True
    
    model = create_mock_model(concept_names, cardinalities, tasks, is_nested)
    
    loss_config = {
        'classification': {
            'binary': 'torch.nn.BCEWithLogitsLoss',
            'categorical': 'torch.nn.NLLLoss'
        },
        'regression': 'torch.nn.MSELoss'
    }
    
    metrics_config = {
        'classification': {
            'binary': {
                'accuracy': 'torchmetrics.classification.BinaryAccuracy'
            },
            'categorical': {
                'accuracy': 'torchmetrics.classification.MulticlassAccuracy'
            }
        }
    }
    
    predictor = Predictor(
        model=model,
        train_inference=lambda x: None,
        loss=loss_config,
        metrics=metrics_config,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        enable_summary_metrics=True,
        enable_perconcept_metrics=True
    )
    
    print(f"Train metrics: {list(predictor.train_metrics.keys())}")
    
    batch_size = 4
    c_hat = torch.cat([
        torch.randn(batch_size, 1),
        torch.randn(batch_size, 3),
        torch.randn(batch_size, 1)
    ], dim=1)
    
    c_true = torch.stack([
        torch.randint(0, 2, (batch_size,)).float(),
        torch.randint(0, 3, (batch_size,)).float(),
        torch.randint(0, 2, (batch_size,)).float()
    ], dim=1)
    
    loss = predictor._compute_loss(c_hat, c_true)
    print(f"\nLoss computed: {loss.item():.4f}")
    
    predictor._update_metrics(c_hat, c_true, predictor.train_metrics)
    results = predictor.train_metrics.compute()
    
    print(f"Metrics computed:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Summary metrics
    assert 'train/binary_accuracy' in results
    assert 'train/categorical_accuracy' in results
    # Per-concept metrics
    assert 'train/c0_accuracy' in results
    assert 'train/c1_accuracy' in results
    assert 'train/c2_accuracy' in results
    assert len(results) == 5  # 2 summary + 3 per-concept
    print("✓ Test passed!")


def test_regression_dense():
    """Test regression dense format."""
    print("\n" + "="*70)
    print("TEST 6: Regression Dense - Both Metrics")
    print("="*70)
    
    from conceptarium.engines.predictor import Predictor
    
    concept_names = ['c0', 'c1']
    cardinalities = [1, 1]
    tasks = ['regression', 'regression']
    is_nested = False
    
    model = create_mock_model(concept_names, cardinalities, tasks, is_nested)
    
    loss_config = {
        'classification': {'binary': 'torch.nn.BCEWithLogitsLoss'},
        'regression': 'torch.nn.MSELoss'
    }
    
    metrics_config = {
        'regression': {
            'mae': 'torchmetrics.regression.MeanAbsoluteError',
            'mse': 'torchmetrics.regression.MeanSquaredError'
        }
    }
    
    predictor = Predictor(
        model=model,
        train_inference=lambda x: None,
        loss=loss_config,
        metrics=metrics_config,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        enable_summary_metrics=True,
        enable_perconcept_metrics=True
    )
    
    print(f"Train metrics: {list(predictor.train_metrics.keys())}")
    
    batch_size = 4
    c_hat = torch.randn(batch_size, 2)
    c_true = torch.randn(batch_size, 2)
    
    loss = predictor._compute_loss(c_hat, c_true)
    print(f"\nLoss computed: {loss.item():.4f}")
    
    predictor._update_metrics(c_hat, c_true, predictor.train_metrics)
    results = predictor.train_metrics.compute()
    
    print(f"Metrics computed:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")
    
    assert 'train/regression_mae' in results
    assert 'train/regression_mse' in results
    assert 'train/c0_mae' in results
    assert 'train/c1_mse' in results
    assert len(results) == 6  # 2 summary + 4 per-concept (2 concepts * 2 metrics)
    print("✓ Test passed!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("COMPREHENSIVE PREDICTOR TESTING")
    print("="*70)
    
    try:
        test_binary_dense_summary_only()
        test_binary_dense_perconcept_only()
        test_binary_dense_both()
        test_mixed_nested_summary()
        test_mixed_nested_both()
        test_regression_dense()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
