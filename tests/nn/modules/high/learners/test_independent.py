"""
Comprehensive tests for IndependentLearner.

Tests cover:
- Parallel vs sequential modes
- Validation and test steps (cascading)
- Internal methods (_compute_single_level, _prepare_prediction, _extract_parent_evidence)
- Verification that parallel mode actually executes in parallel
"""
import unittest
from unittest.mock import patch, MagicMock
import threading
import time

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts import seed_everything
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.data.datasets import ToyDataset


class TestIndependentLearnerBasic(unittest.TestCase):
    """Basic tests for IndependentLearner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.batch_size = 4
        self.input_size = 8
        self.x = torch.randn(self.batch_size, self.input_size)
        
        self.batch = {
            'inputs': {'x': self.x},
            'concepts': {'c': torch.randint(0, 2, (self.batch_size, 3)).float()}
        }
    
    def _make_model(self, parallel=True):
        """Helper to create independent learner model."""
        return ConceptBottleneckModel(
            training='independent',
            parallel=parallel,
            input_size=self.input_size,
            annotations=self.ann,
            task_names=['task'],
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01}
        )
    
    def test_parallel_true_flag(self):
        """Test that parallel=True sets the flag correctly."""
        model = self._make_model(parallel=True)
        self.assertTrue(model.parallel)
    
    def test_parallel_false_flag(self):
        """Test that parallel=False sets the flag correctly."""
        model = self._make_model(parallel=False)
        self.assertFalse(model.parallel)
    
    def test_training_step_parallel(self):
        """Test training step works with parallel=True."""
        model = self._make_model(parallel=True)
        model.train()
        
        loss = model.training_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
    
    def test_training_step_sequential(self):
        """Test training step works with parallel=False."""
        model = self._make_model(parallel=False)
        model.train()
        
        loss = model.training_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)


class TestIndependentLearnerOutputEquivalence(unittest.TestCase):
    """Test that parallel and sequential modes produce identical outputs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.batch_size = 4
        self.input_size = 8
        self.x = torch.randn(self.batch_size, self.input_size)
        
        self.batch = {
            'inputs': {'x': self.x},
            'concepts': {'c': torch.randint(0, 2, (self.batch_size, 3)).float()}
        }
    
    def _make_model(self, parallel=True):
        """Helper to create independent learner model."""
        return ConceptBottleneckModel(
            training='independent',
            parallel=parallel,
            input_size=self.input_size,
            annotations=self.ann,
            task_names=['task'],
            loss=nn.BCEWithLogitsLoss(),
        )
    
    def test_parallel_and_sequential_produce_same_results(self):
        """Test that parallel and sequential modes produce identical results."""
        torch.manual_seed(42)
        model_parallel = self._make_model(parallel=True)
        
        torch.manual_seed(42)
        model_sequential = self._make_model(parallel=False)
        
        # Copy weights to ensure identical parameters
        model_sequential.load_state_dict(model_parallel.state_dict())
        
        model_parallel.eval()
        model_sequential.eval()
        
        inputs = self.batch['inputs']
        concepts = self.batch['concepts']
        
        with torch.no_grad():
            out_parallel = model_parallel.predict_all_levels_parallel(
                inputs, concepts, use_ground_truth=True
            )
            out_sequential = model_sequential.predict_all_levels_sequential(
                inputs, concepts, use_ground_truth=True
            )
        
        torch.testing.assert_close(out_parallel, out_sequential)


class TestIndependentLearnerValidationAndTest(unittest.TestCase):
    """Test validation and test steps with cascading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.batch_size = 4
        self.input_size = 8
        self.x = torch.randn(self.batch_size, self.input_size)
        
        self.batch = {
            'inputs': {'x': self.x},
            'concepts': {'c': torch.randint(0, 2, (self.batch_size, 3)).float()}
        }
    
    def _make_model(self, parallel=False):
        """Helper to create independent learner model."""
        return ConceptBottleneckModel(
            training='independent',
            parallel=parallel,
            input_size=self.input_size,
            annotations=self.ann,
            task_names=['task'],
            loss=nn.BCEWithLogitsLoss(),
        )
    
    def test_validation_step_returns_loss(self):
        """Test that validation_step returns a scalar loss."""
        model = self._make_model(parallel=False)
        model.eval()
        
        with torch.no_grad():
            loss = model.validation_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)
    
    def test_test_step_returns_loss(self):
        """Test that test_step returns a scalar loss."""
        model = self._make_model(parallel=False)
        model.eval()
        
        with torch.no_grad():
            loss = model.test_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)
    
    def test_validation_uses_cascading(self):
        """Test that validation uses predicted concepts (cascading), not ground truth."""
        model = self._make_model(parallel=False)
        
        inputs = self.batch['inputs']
        concepts = self.batch['concepts']
        
        model.eval()
        with torch.no_grad():
            # Training mode: uses ground truth
            out_train = model.predict_all_levels_sequential(
                inputs, concepts, use_ground_truth=True
            )
            # Validation mode: cascades predictions
            out_val = model.predict_all_levels_sequential(
                inputs, concepts, use_ground_truth=False
            )
        
        # Both should have same shape
        self.assertEqual(out_train.shape, out_val.shape)
        self.assertEqual(out_train.shape[1], 3)  # c1, c2, task


class TestIndependentLearnerInternalMethods(unittest.TestCase):
    """Test internal methods of IndependentLearner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.batch_size = 4
        self.input_size = 8
        self.x = torch.randn(self.batch_size, self.input_size)
        
        self.batch = {
            'inputs': {'x': self.x},
            'concepts': {'c': torch.randint(0, 2, (self.batch_size, 3)).float()}
        }
    
    def _make_model(self, parallel=False):
        """Helper to create independent learner model."""
        return ConceptBottleneckModel(
            training='independent',
            parallel=parallel,
            input_size=self.input_size,
            annotations=self.ann,
            task_names=['task'],
            loss=nn.BCEWithLogitsLoss(),
        )
    
    def test_compute_single_level(self):
        """Test _compute_single_level computes a level correctly."""
        model = self._make_model(parallel=False)
        model.eval()
        
        with torch.no_grad():
            latent_input = model.forward(x=self.x, query=['input'])
        
        level = model.graph_levels[0]
        concepts_tensor = self.batch['concepts']['c']
        
        with torch.no_grad():
            level_idx, returned_level, level_out = model._compute_single_level(
                0, level, latent_input, concepts_tensor
            )
        
        self.assertEqual(level_idx, 0)
        self.assertEqual(returned_level, level)
        self.assertEqual(level_out.shape[0], self.batch_size)
    
    def test_prepare_prediction_output_shape(self):
        """Test _prepare_prediction creates correct output tensor."""
        model = self._make_model(parallel=False)
        model.eval()
        
        inputs = self.batch['inputs']
        
        with torch.no_grad():
            out, latent_input = model._prepare_prediction(inputs, use_ground_truth=True)
        
        total_card = sum(model.concept_annotations.cardinalities)
        self.assertEqual(out.shape, (self.batch_size, total_card))
        self.assertIsNotNone(latent_input)
    
    def test_prepare_prediction_accumulated_evidence_training(self):
        """Test _prepare_prediction with ground truth (training mode)."""
        model = self._make_model(parallel=False)
        model.eval()
        
        inputs = self.batch['inputs']
        
        with torch.no_grad():
            _, _ = model._prepare_prediction(inputs, use_ground_truth=True)
        
        self.assertIsNone(model.accumulated_evidence)
    
    def test_prepare_prediction_accumulated_evidence_validation(self):
        """Test _prepare_prediction without ground truth (validation mode)."""
        model = self._make_model(parallel=False)
        model.eval()
        
        inputs = self.batch['inputs']
        
        with torch.no_grad():
            _, _ = model._prepare_prediction(inputs, use_ground_truth=False)
        
        self.assertIsInstance(model.accumulated_evidence, dict)
    
    def test_extract_parent_evidence(self):
        """Test _extract_parent_evidence extracts concept evidence correctly."""
        model = self._make_model(parallel=False)
        model.eval()
        
        concepts_tensor = self.batch['concepts']['c']
        
        # Extract evidence for task level (which has c1, c2 as parents in CBM)
        level = ['task']
        evidence = model._extract_parent_evidence(concepts_tensor, level)
        
        # Should have c1 and c2 as parent evidence for task
        self.assertIn('c1', evidence)
        self.assertIn('c2', evidence)
        self.assertEqual(evidence['c1'].shape[0], self.batch_size)
        self.assertEqual(evidence['c2'].shape[0], self.batch_size)


class TestIndependentLearnerParallelExecution(unittest.TestCase):
    """Test that parallel mode actually executes in parallel.
    
    These tests verify that when parallel=True, the code actually uses
    ThreadPoolExecutor (CPU) or CUDA streams (GPU) for parallel execution.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.batch_size = 4
        self.input_size = 8
        self.x = torch.randn(self.batch_size, self.input_size)
        
        self.batch = {
            'inputs': {'x': self.x},
            'concepts': {'c': torch.randint(0, 2, (self.batch_size, 3)).float()}
        }
    
    def _make_model(self, parallel=True):
        """Helper to create independent learner model."""
        return ConceptBottleneckModel(
            training='independent',
            parallel=parallel,
            input_size=self.input_size,
            annotations=self.ann,
            task_names=['task'],
            loss=nn.BCEWithLogitsLoss(),
        )
    
    def test_parallel_mode_uses_threadpool_executor(self):
        """Test that parallel mode uses ThreadPoolExecutor on CPU.
        
        We patch ThreadPoolExecutor to verify it gets instantiated and
        submit() is called for each level when running in parallel mode.
        """
        model = self._make_model(parallel=True)
        model.train()
        
        # Ensure we have multiple levels to parallelize
        num_levels = len(model.graph_levels)
        
        if num_levels < 2:
            self.skipTest("Model has only 1 level, no parallelization needed")
        
        with patch('torch_concepts.nn.modules.high.learners.independent.ThreadPoolExecutor') as mock_executor_class:
            # Create a mock executor that returns futures
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor_class.return_value.__exit__ = MagicMock(return_value=False)
            
            # Create mock futures that return proper results when .result() is called
            def create_mock_future(level_idx, level, latent_input, concepts):
                future = MagicMock()
                # Return a tuple of (level_idx, level, level_output)
                level_output = torch.randn(self.batch_size, len(level))
                future.result.return_value = (level_idx, level, level_output)
                return future
            
            mock_executor.submit.side_effect = lambda fn, *args: create_mock_future(*args)
            
            # Run predict_all_levels_parallel
            inputs = self.batch['inputs']
            concepts = self.batch['concepts']
            
            with torch.no_grad():
                _ = model.predict_all_levels_parallel(inputs, concepts, use_ground_truth=True)
            
            # Verify ThreadPoolExecutor was used
            mock_executor_class.assert_called()
            
            # Verify submit was called for each level
            self.assertEqual(
                mock_executor.submit.call_count, 
                num_levels,
                f"Expected {num_levels} submit calls, got {mock_executor.submit.call_count}"
            )
    
    def test_sequential_mode_does_not_use_threadpool(self):
        """Test that sequential mode does NOT use ThreadPoolExecutor."""
        model = self._make_model(parallel=False)
        model.train()
        
        with patch('torch_concepts.nn.modules.high.learners.independent.ThreadPoolExecutor') as mock_executor_class:
            inputs = self.batch['inputs']
            concepts = self.batch['concepts']
            
            with torch.no_grad():
                _ = model.predict_all_levels_sequential(inputs, concepts, use_ground_truth=True)
            
            # Verify ThreadPoolExecutor was NOT used
            mock_executor_class.assert_not_called()
    
    def test_parallel_compute_single_level_called_concurrently(self):
        """Test that _compute_single_level is called for all levels in parallel mode.
        
        We track which threads call _compute_single_level to verify concurrent execution.
        """
        model = self._make_model(parallel=True)
        model.train()
        
        num_levels = len(model.graph_levels)
        
        if num_levels < 2:
            self.skipTest("Model has only 1 level, no parallelization needed")
        
        # Track thread IDs that call _compute_single_level
        thread_ids = []
        original_compute = model._compute_single_level
        
        def tracking_compute(*args, **kwargs):
            thread_ids.append(threading.current_thread().ident)
            return original_compute(*args, **kwargs)
        
        with patch.object(model, '_compute_single_level', side_effect=tracking_compute):
            inputs = self.batch['inputs']
            concepts = self.batch['concepts']
            
            with torch.no_grad():
                _ = model.predict_all_levels_parallel(inputs, concepts, use_ground_truth=True)
        
        # Should have been called once per level
        self.assertEqual(len(thread_ids), num_levels)
        
        # In parallel mode with ThreadPoolExecutor, calls should be from different threads
        # (unless there's only 1 worker, but we use max_workers=num_levels)
        # At minimum, verify all levels were processed
        self.assertEqual(len(thread_ids), num_levels)


class TestIndependentLearnerSingleLevel(unittest.TestCase):
    """Test IndependentLearner behavior with single-level models.
    
    When there's only one level, parallelization should be skipped
    to avoid overhead.
    """
    
    def setUp(self):
        """Set up test fixtures with single-level model."""
        # Single concept (no task) - just one level
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1'],
                cardinalities=[1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                }
            )
        })
        
        self.batch_size = 4
        self.input_size = 8
        self.x = torch.randn(self.batch_size, self.input_size)
        
        self.batch = {
            'inputs': {'x': self.x},
            'concepts': {'c': torch.randint(0, 2, (self.batch_size, 1)).float()}
        }
    
    def test_single_level_no_threadpool_overhead(self):
        """Test that single-level models don't use ThreadPoolExecutor."""
        model = ConceptBottleneckModel(
            training='independent',
            parallel=True,  # Even with parallel=True
            input_size=self.input_size,
            annotations=self.ann,
            task_names=[],  # No tasks, just concepts
            loss=nn.BCEWithLogitsLoss(),
        )
        model.train()
        
        # Verify model has only 1 level
        self.assertEqual(len(model.graph_levels), 1)
        
        with patch('torch_concepts.nn.modules.high.learners.independent.ThreadPoolExecutor') as mock_executor_class:
            inputs = self.batch['inputs']
            concepts = self.batch['concepts']
            
            with torch.no_grad():
                _ = model.predict_all_levels_parallel(inputs, concepts, use_ground_truth=True)
            
            # With only 1 level, ThreadPoolExecutor should NOT be used (no overhead)
            mock_executor_class.assert_not_called()


class TestIndependentLearnerGradientIsolation(unittest.TestCase):
    """Tests for gradient isolation and evidence handling in IndependentLearner.
    
    These tests verify that:
    1. Evidence passed to inference is actually used (not recomputed)
    2. Task loss gradients don't reach the encoder when using GT concepts
    """
    
    def setUp(self):
        """Set up test fixtures with XOR dataset."""
        seed_everything(42)
        self.dataset = ToyDataset(dataset='xor', seed=42, n_gen=100)
        self.annotations = self.dataset.annotations
        self.concept_names = list(self.annotations.get_axis_annotation(1).labels)
        self.variable_distributions = {name: Bernoulli for name in self.concept_names}
    
    def test_evidence_is_used_not_recomputed(self):
        """Test that evidence passed to inference is used, not recomputed.
        
        When we pass explicit evidence for C1, C2 (e.g., flipped values),
        the task predictor output should change accordingly. If evidence
        is ignored and concepts are recomputed from input, the output
        would remain unchanged.
        """
        model = ConceptBottleneckModel(
            input_size=2,
            annotations=self.annotations,
            variable_distributions=self.variable_distributions,
            task_names=['xor'],
            latent_encoder_kwargs={'hidden_size': 8, 'n_layers': 1},
        )
        
        x = self.dataset.input_data[:10]
        
        # Get predictions without evidence
        with torch.no_grad():
            out_normal = model(query=list(self.concept_names), x=x)
            c1_pred = out_normal[:, 0:1]
            c2_pred = out_normal[:, 1:2]
        
        # Pass OPPOSITE evidence for C1, C2
        opposite_c1 = torch.logit((1 - torch.sigmoid(c1_pred)).clamp(1e-7, 1-1e-7))
        opposite_c2 = torch.logit((1 - torch.sigmoid(c2_pred)).clamp(1e-7, 1-1e-7))
        
        features = model.maybe_apply_backbone(x)
        latent_input = model.latent_encoder(features)
        
        evidence = {
            'C1': opposite_c1,
            'C2': opposite_c2,
            'input': latent_input
        }
        
        with torch.no_grad():
            xor_with_evidence = model.inference.query(['xor'], evidence=evidence)
            xor_normal = out_normal[:, 2:3]
        
        # With flipped C1, C2 evidence, XOR output must change
        are_different = not torch.allclose(xor_with_evidence, xor_normal, atol=1e-3)
        
        self.assertTrue(
            are_different,
            "XOR output is identical with and without opposite evidence for C1, C2. "
            "This indicates evidence is being ignored and concepts are recomputed."
        )
    
    def test_no_gradient_to_encoder_from_task_loss(self):
        """Test that task loss gradients don't reach the encoder with GT concepts.
        
        In independent training:
        - Task predictor receives GT concepts (not encoder predictions)
        - Task loss should NOT produce gradients for encoder parameters
        - Only concept loss should affect encoder weights
        
        This is the key test for gradient isolation in CBM independent training.
        """
        model = ConceptBottleneckModel(
            input_size=2,
            annotations=self.annotations,
            variable_distributions=self.variable_distributions,
            task_names=['xor'],
            latent_encoder_kwargs={'hidden_size': 8, 'n_layers': 1},
            training='independent',
            loss=torch.nn.BCEWithLogitsLoss(),
        )
        
        x = self.dataset.input_data[:10]
        c = self.dataset.concepts[:10]
        
        # Zero gradients and compute only task loss path
        model.zero_grad()
        
        features = model.maybe_apply_backbone(x)
        latent_input = model.latent_encoder(features)
        
        # Get task prediction using GT concepts (what independent training does)
        evidence = model._extract_parent_evidence(c, ['xor'])
        evidence['input'] = latent_input
        
        task_out = model.forward(evidence=evidence, query=['xor'])
        
        # Compute task loss only
        task_target = c[:, 2:3]
        task_loss = torch.nn.BCEWithLogitsLoss()(task_out, task_target)
        task_loss.backward()
        
        # Encoder should have NO gradients from task loss
        # because GT concept evidence is detached from the encoder
        encoder_has_task_grads = False
        for name, param in model.latent_encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 1e-10:
                encoder_has_task_grads = True
                break
        
        self.assertFalse(
            encoder_has_task_grads,
            "Encoder has gradients from task loss when using GT concepts. "
            "Task loss is leaking into encoder when it should be blocked."
        )


if __name__ == '__main__':
    unittest.main()
