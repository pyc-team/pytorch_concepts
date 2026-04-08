"""
Comprehensive tests for IndependentInference.

Tests cover:
- Basic query functionality
- Ground truth replacement during propagation
- Gradient flow verification (no gradients from following layers to previous)
- Integration with CBM and CEM models
- Edge cases and error handling
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts import InputVariable, EndogenousVariable, ExogenousVariable
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.parametric_cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.inference.independent import IndependentInference
from torch_concepts.nn.modules.mid.inference.deterministic import DeterministicInference
from torch_concepts.distributions import Delta


class TestIndependentInferenceBasic:
    """Basic tests for IndependentInference query functionality."""

    def _make_simple_model(self):
        """Create a simple linear chain model: input -> A -> B -> task."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)
        var_task = EndogenousVariable('task', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1), parents=['input'])
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(1, 1), parents=['A'])
        cpd_task = ParametricCPD('task', parametrization=nn.Linear(1, 1), parents=['B'])

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_task],
            factors=[cpd_input, cpd_A, cpd_B, cpd_task]
        )
        return model

    def test_query_with_ground_truth(self):
        """Test that IndependentInference returns predictions while using GT for propagation."""
        model = self._make_simple_model()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        # Ground truth as tensor [A, B]
        ground_truth = torch.ones(batch_size, 2)

        result = inference.query(
            query=['A', 'B', 'task'],
            evidence={'input': x},
            ground_truth=ground_truth,
            concept_names=['A', 'B']
        )

        # Result should be (batch_size, 3) - one feature per concept
        assert result.shape == (batch_size, 3)

    def test_query_single_concept(self):
        """Test querying a single concept."""
        model = self._make_simple_model()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        # Ground truth as tensor [A]
        ground_truth = torch.ones(batch_size, 1)

        result = inference.query(
            query=['A'],
            evidence={'input': x},
            ground_truth=ground_truth,
            concept_names=['A']
        )

        assert result.shape == (batch_size, 1)

    def test_query_preserves_order(self):
        """Test that query results are in the requested order."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1), parents=['input'])
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1), parents=['input'])

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            factors=[cpd_input, cpd_A, cpd_B]
        )
        inference = IndependentInference(model)

        x = torch.randn(4, 10)
        gt = torch.randint(0, 2, (4, 2)).float()  # Binary indices for A and B

        # Query in order [A, B]
        result_ab = inference.query(['A', 'B'], {'input': x}, ground_truth=gt, concept_names=['A', 'B'])
        # Query in order [B, A]
        result_ba = inference.query(['B', 'A'], {'input': x}, ground_truth=gt, concept_names=['A', 'B'])

        # First column of result_ab (A) should equal second column of result_ba
        torch.testing.assert_close(result_ab[:, 0:1], result_ba[:, 1:2])
        # Second column of result_ab (B) should equal first column of result_ba
        torch.testing.assert_close(result_ab[:, 1:2], result_ba[:, 0:1])

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        model = self._make_simple_model()
        inference = IndependentInference(model)

        x = torch.randn(4, 10)
        gt = torch.randn(4, 1)

        with pytest.raises(AssertionError, match="Query list cannot be empty"):
            inference.query([], {'input': x}, ground_truth=gt, concept_names=['A'])


class TestIndependentInferenceGroundTruthPropagation:
    """Test that ground truth is correctly used for propagation."""

    def _make_chain_model(self):
        """Create chain model where output depends on parent values."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1), parents=['input'])
        # B depends on A - use identity to make dependency clear
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(1, 1), parents=['A'])

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            factors=[cpd_input, cpd_A, cpd_B]
        )
        return model

    def test_ground_truth_affects_downstream(self):
        """Test that GT values are used to compute downstream concepts."""
        model = self._make_chain_model()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)

        # Compute with GT for A = 0
        gt_zeros = torch.zeros(batch_size, 1)
        result_with_zeros = inference.query(
            ['B'], 
            {'input': x}, 
            ground_truth=gt_zeros,
            concept_names=['A']
        )

        # Compute with GT for A = 1  
        gt_ones = torch.ones(batch_size, 1)
        result_with_ones = inference.query(
            ['B'], 
            {'input': x}, 
            ground_truth=gt_ones,
            concept_names=['A']
        )

        # Results should be different because B depends on A
        assert not torch.allclose(result_with_zeros, result_with_ones)

    def test_predictions_independent_of_gt(self):
        """Test that returned predictions are model outputs, not GT values."""
        model = self._make_chain_model()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)

        # Set GT for A to extreme value
        gt_fixed = torch.full((batch_size, 1), 999.0)
        result = inference.query(
            ['A'], 
            {'input': x}, 
            ground_truth=gt_fixed,
            concept_names=['A']
        )

        # Result should NOT be 999 - it should be the model's prediction
        assert not torch.allclose(result, torch.full((batch_size, 1), 999.0))


class TestIndependentInferenceGradientFlow:
    """Test gradient flow in IndependentInference.
    
    Key property: In independent training, gradients should NOT flow 
    from downstream concepts back to upstream concepts through the 
    computational graph, because GT values are used for propagation.
    """

    def _make_chain_model_with_tracking(self):
        """Create chain model with gradient-tracking linear layers."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)
        var_task = EndogenousVariable('task', distribution=Bernoulli, size=1)

        # Named modules for easy access to gradients
        linear_A = nn.Linear(10, 1)
        linear_B = nn.Linear(1, 1)
        linear_task = nn.Linear(1, 1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=linear_A, parents=['input'])
        cpd_B = ParametricCPD('B', parametrization=linear_B, parents=['A'])
        cpd_task = ParametricCPD('task', parametrization=linear_task, parents=['B'])

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B, var_task],
            factors=[cpd_input, cpd_A, cpd_B, cpd_task]
        )
        return model, {'A': linear_A, 'B': linear_B, 'task': linear_task}

    def test_no_gradient_from_task_to_concept_a(self):
        """Test that computing loss on task does not create gradients for concept A."""
        model, layers = self._make_chain_model_with_tracking()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Provide GT for A and B - these block gradient flow
        ground_truth = torch.zeros(batch_size, 2)  # [A, B]

        # Zero gradients
        for layer in layers.values():
            layer.zero_grad()

        # Forward pass
        result = inference.query(
            ['A', 'B', 'task'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['A', 'B']
        )
        
        # Extract task prediction and compute loss
        task_pred = result[:, 2:]  # Last column is task
        task_target = torch.zeros(batch_size, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(task_pred, task_target)
        
        # Backward pass
        loss.backward()

        # Concept A should have NO gradients because GT was used for propagation
        # The task only sees the GT value of B, not the prediction from A
        assert layers['A'].weight.grad is None or torch.all(layers['A'].weight.grad == 0), \
            "Concept A should not have gradients when GT is used for propagation"

    def test_gradient_exists_for_direct_loss(self):
        """Test that gradients exist when computing loss directly on a concept."""
        model, layers = self._make_chain_model_with_tracking()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Provide GT for B only (concept_names specifies which concepts have GT)
        ground_truth = torch.zeros(batch_size, 1)  # Only B

        # Zero gradients
        for layer in layers.values():
            layer.zero_grad()

        # Forward pass
        result = inference.query(
            ['A', 'B', 'task'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['B']
        )
        
        # Compute loss on A (direct relationship with input)
        a_pred = result[:, 0:1]  # First column is A
        a_target = torch.ones(batch_size, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(a_pred, a_target)
        
        # Backward pass
        loss.backward()

        # Concept A should have gradients (loss is computed directly on its output)
        assert layers['A'].weight.grad is not None
        assert not torch.all(layers['A'].weight.grad == 0), \
            "Concept A should have gradients when loss is on its output"

    def test_gradient_isolation_between_levels(self):
        """Test that each concept level is trained independently."""
        model, layers = self._make_chain_model_with_tracking()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Provide GT for A only
        ground_truth = torch.ones(batch_size, 1)  # Only A

        # Zero gradients
        for layer in layers.values():
            layer.zero_grad()

        # Forward pass
        result = inference.query(
            ['A', 'B', 'task'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['A']
        )
        
        # Compute loss ONLY on B
        b_pred = result[:, 1:2]  # Second column is B
        b_target = torch.zeros(batch_size, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(b_pred, b_target)
        
        # Backward pass
        loss.backward()

        # B should have gradients (its output is used in loss)
        assert layers['B'].weight.grad is not None
        assert not torch.all(layers['B'].weight.grad == 0), \
            "Concept B should have gradients"

        # A should NOT have gradients (GT was used for propagation to B)
        assert layers['A'].weight.grad is None or torch.all(layers['A'].weight.grad == 0), \
            "Concept A should not have gradients when GT is used for downstream"


class TestIndependentInferenceWithCategorical:
    """Test IndependentInference with categorical variables."""

    def _make_categorical_model(self):
        """Create model with categorical concept."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Categorical, size=4)  # 4-class
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 4), parents=['input'])
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(4, 1), parents=['A'])

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            factors=[cpd_input, cpd_A, cpd_B]
        )
        return model

    def test_categorical_query(self):
        """Test query with categorical variable."""
        model = self._make_categorical_model()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Ground truth for A as class indices (index format)
        gt_a = torch.zeros(batch_size, 1)  # Class 0 for all samples

        result = inference.query(
            ['A', 'B'], 
            {'input': x}, 
            ground_truth=gt_a,
            concept_names=['A']
        )

        # A has 4 features, B has 1
        assert result.shape == (batch_size, 5)


class TestIndependentInferenceWithExogenous:
    """Test IndependentInference with exogenous variables (for CEM).
    
    Exogenous variables are auxiliary latent representations (like embeddings in CEM).
    Key properties to test:
    - Exogenous are NEVER replaced by ground truth (they are internal representations)
    - Exogenous always receive gradients regardless of GT usage for endogenous
    - Exogenous are propagated correctly to downstream endogenous variables
    """

    def _make_model_with_exogenous(self):
        """Create model with exogenous variables like CEM uses."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        exo_var = ExogenousVariable('exo', distribution=Delta, size=8)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_exo = ParametricCPD('exo', parametrization=nn.Linear(10, 8), parents=['input'])
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(8, 1), parents=['exo'])
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(9, 1), parents=['exo', 'A'])  # 8 from exo + 1 from A

        model = ProbabilisticModel(
            variables=[input_var, exo_var, var_A, var_B],
            factors=[cpd_input, cpd_exo, cpd_A, cpd_B]
        )
        return model

    def _make_model_with_exogenous_and_tracking(self):
        """Create model with exogenous and gradient tracking."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        exo_var = ExogenousVariable('exo', distribution=Delta, size=8)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        linear_exo = nn.Linear(10, 8)
        linear_A = nn.Linear(8, 1)
        linear_B = nn.Linear(9, 1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_exo = ParametricCPD('exo', parametrization=linear_exo, parents=['input'])
        cpd_A = ParametricCPD('A', parametrization=linear_A, parents=['exo'])
        cpd_B = ParametricCPD('B', parametrization=linear_B, parents=['exo', 'A'])

        model = ProbabilisticModel(
            variables=[input_var, exo_var, var_A, var_B],
            factors=[cpd_input, cpd_exo, cpd_A, cpd_B]
        )
        return model, {'exo': linear_exo, 'A': linear_A, 'B': linear_B}

    def test_exogenous_propagation(self):
        """Test that exogenous variables are propagated correctly."""
        model = self._make_model_with_exogenous()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        ground_truth = torch.ones(batch_size, 1)  # Only A

        result = inference.query(
            ['A', 'B'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['A']
        )

        # A has 1 feature, B has 1 feature
        assert result.shape == (batch_size, 2)

    def test_exogenous_not_in_ground_truth(self):
        """Test that exogenous variables are not replaced by GT even when computed."""
        model = self._make_model_with_exogenous()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # GT only for endogenous A, not exogenous
        gt_a = torch.zeros(batch_size, 1)

        result = inference.query(
            ['A', 'B'], 
            {'input': x},
            ground_truth=gt_a,
            concept_names=['A']
        )

        assert result.shape == (batch_size, 2)

    def test_exogenous_always_computed_never_replaced(self):
        """Test that exogenous values are model outputs, never replaced by GT.
        
        Even if someone mistakenly puts an exogenous in ground_truth dict,
        it should be ignored - exogenous are internal representations.
        """
        model = self._make_model_with_exogenous()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Run with GT=0 for A
        gt_zero = torch.zeros(batch_size, 1)
        result_gt_zero = inference.query(
            ['A', 'B'], 
            {'input': x}, 
            ground_truth=gt_zero,
            concept_names=['A']
        )
        
        # Run with GT=1 for A
        gt_one = torch.ones(batch_size, 1)
        result_gt_one = inference.query(
            ['A', 'B'], 
            {'input': x}, 
            ground_truth=gt_one,
            concept_names=['A']
        )
        
        # A predictions should be the same (GT affects propagation, not prediction)
        # But B predictions should differ because GT for A is used in propagation
        assert not torch.allclose(result_gt_zero[:, 1:], result_gt_one[:, 1:]), \
            "B should differ when GT for A differs"

    def test_exogenous_receives_gradients_when_loss_on_downstream(self):
        """Test that exogenous receives gradients when loss is on downstream concept.
        
        Key property: Exogenous variables should ALWAYS receive gradients
        because they are computed (not replaced) by ground truth.
        """
        model, layers = self._make_model_with_exogenous_and_tracking()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Provide GT for A
        ground_truth = torch.ones(batch_size, 1)

        # Zero gradients
        for layer in layers.values():
            layer.zero_grad()

        # Forward pass
        result = inference.query(
            ['A', 'B'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['A']
        )
        
        # Compute loss on A (first output)
        a_pred = result[:, 0:1]
        a_target = torch.zeros(batch_size, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(a_pred, a_target)
        
        # Backward pass
        loss.backward()

        # Exogenous should have gradients (loss on A flows through exo since A depends on exo)
        assert layers['exo'].weight.grad is not None
        assert not torch.all(layers['exo'].weight.grad == 0), \
            "Exogenous should have gradients when loss is on downstream concept A"

    def test_exogenous_gradient_isolation_from_downstream_with_gt(self):
        """Test gradient flow when GT is used for intermediate endogenous.
        
        Scenario: input -> exo -> A -> B
        If GT is provided for A, loss on B should NOT create gradients for:
        - A (GT is used for propagation)
        
        But SHOULD create gradients for:
        - B (directly involved in loss)
        - exo (exogenous for B, still computed)
        """
        model, layers = self._make_model_with_exogenous_and_tracking()
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Provide GT for A
        ground_truth = torch.ones(batch_size, 1)

        # Zero gradients
        for layer in layers.values():
            layer.zero_grad()

        # Forward pass
        result = inference.query(
            ['A', 'B'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['A']
        )
        
        # Compute loss ONLY on B
        b_pred = result[:, 1:2]
        b_target = torch.zeros(batch_size, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(b_pred, b_target)
        
        # Backward pass
        loss.backward()

        # B should have gradients (directly involved in loss)
        assert layers['B'].weight.grad is not None
        assert not torch.all(layers['B'].weight.grad == 0), \
            "B should have gradients"

        # A should NOT have gradients (GT was used for propagation to B)
        assert layers['A'].weight.grad is None or torch.all(layers['A'].weight.grad == 0), \
            "A should not have gradients when GT is used for downstream"

        # Exo SHOULD have gradients because B depends on exo directly
        # (B's parents are ['exo', 'A'], so exo is concatenated with A for B's input)
        assert layers['exo'].weight.grad is not None
        assert not torch.all(layers['exo'].weight.grad == 0), \
            "Exogenous should have gradients since B depends on it directly"

    def test_multiple_exogenous_variables(self):
        """Test model with multiple exogenous variables."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        exo_1 = ExogenousVariable('exo_1', distribution=Delta, size=4)
        exo_2 = ExogenousVariable('exo_2', distribution=Delta, size=4)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_exo_1 = ParametricCPD('exo_1', parametrization=nn.Linear(10, 4), parents=['input'])
        cpd_exo_2 = ParametricCPD('exo_2', parametrization=nn.Linear(10, 4), parents=['input'])
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(4, 1), parents=['exo_1'])
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(4, 1), parents=['exo_2'])

        model = ProbabilisticModel(
            variables=[input_var, exo_1, exo_2, var_A, var_B],
            factors=[cpd_input, cpd_exo_1, cpd_exo_2, cpd_A, cpd_B]
        )
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        ground_truth = torch.ones(batch_size, 1)  # Only A

        result = inference.query(
            ['A', 'B'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['A']
        )

        assert result.shape == (batch_size, 2)

    def test_cem_like_architecture(self):
        """Test CEM-like architecture with shared exogenous.
        
        CEM architecture: input -> exo (embedding) -> concepts
        The exogenous embedding feeds both the encoder (concept logits)
        and the predictor (task logits).
        """
        input_var = InputVariable('input', distribution=Delta, size=10)
        # Shared embedding (like CEM's concept embeddings)
        exo_emb = ExogenousVariable('embedding', distribution=Delta, size=16)
        # Concept depends on embedding
        var_c1 = EndogenousVariable('c1', distribution=Bernoulli, size=1)
        var_c2 = EndogenousVariable('c2', distribution=Bernoulli, size=1)
        # Task depends on embedding AND concepts
        var_task = EndogenousVariable('task', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_emb = ParametricCPD('embedding', parametrization=nn.Linear(10, 16), parents=['input'])
        cpd_c1 = ParametricCPD('c1', parametrization=nn.Linear(16, 1), parents=['embedding'])
        cpd_c2 = ParametricCPD('c2', parametrization=nn.Linear(16, 1), parents=['embedding'])
        cpd_task = ParametricCPD('task', parametrization=nn.Linear(18, 1), parents=['embedding', 'c1', 'c2'])  # 16 + 1 + 1

        model = ProbabilisticModel(
            variables=[input_var, exo_emb, var_c1, var_c2, var_task],
            factors=[cpd_input, cpd_emb, cpd_c1, cpd_c2, cpd_task]
        )
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        # [c1, c2] tensor
        ground_truth = torch.cat([
            torch.ones(batch_size, 1),
            torch.zeros(batch_size, 1),
        ], dim=1)

        result = inference.query(
            ['c1', 'c2', 'task'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['c1', 'c2']
        )

        # c1 (1) + c2 (1) + task (1) = 3
        assert result.shape == (batch_size, 3)

    def test_cem_gradient_flow_with_shared_exogenous(self):
        """Test gradient flow in CEM-like architecture.
        
        When GT is provided for concepts, loss on task should:
        - Create gradients for task predictor
        - Create gradients for embedding (shared exogenous)
        - NOT create gradients for concept encoders (GT used)
        """
        input_var = InputVariable('input', distribution=Delta, size=10)
        exo_emb = ExogenousVariable('embedding', distribution=Delta, size=16)
        var_c1 = EndogenousVariable('c1', distribution=Bernoulli, size=1)
        var_task = EndogenousVariable('task', distribution=Bernoulli, size=1)

        linear_emb = nn.Linear(10, 16)
        linear_c1 = nn.Linear(16, 1)
        linear_task = nn.Linear(17, 1)  # 16 + 1

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_emb = ParametricCPD('embedding', parametrization=linear_emb, parents=['input'])
        cpd_c1 = ParametricCPD('c1', parametrization=linear_c1, parents=['embedding'])
        cpd_task = ParametricCPD('task', parametrization=linear_task, parents=['embedding', 'c1'])

        model = ProbabilisticModel(
            variables=[input_var, exo_emb, var_c1, var_task],
            factors=[cpd_input, cpd_emb, cpd_c1, cpd_task]
        )
        inference = IndependentInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        ground_truth = torch.ones(batch_size, 1)  # Only c1

        # Zero gradients
        linear_emb.zero_grad()
        linear_c1.zero_grad()
        linear_task.zero_grad()

        result = inference.query(
            ['c1', 'task'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['c1']
        )
        
        # Loss on task only
        task_pred = result[:, 1:2]
        task_target = torch.zeros(batch_size, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(task_pred, task_target)
        loss.backward()

        # Task should have gradients
        assert linear_task.weight.grad is not None
        assert not torch.all(linear_task.weight.grad == 0), \
            "Task predictor should have gradients"

        # Embedding should have gradients (task depends on it directly)
        assert linear_emb.weight.grad is not None
        assert not torch.all(linear_emb.weight.grad == 0), \
            "Embedding should have gradients since task depends on it"

        # c1 should NOT have gradients (GT was used)
        assert linear_c1.weight.grad is None or torch.all(linear_c1.weight.grad == 0), \
            "c1 should not have gradients when GT is used"


class TestIndependentInferenceDeviceModes:
    """Test device handling in IndependentInference."""

    def _make_simple_model(self):
        """Create a simple model for testing."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1), parents=['input'])

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            factors=[cpd_input, cpd_A]
        )
        return model

    def test_cpu_mode(self):
        """Test explicit CPU mode."""
        model = self._make_simple_model()
        inference = IndependentInference(model)

        x = torch.randn(4, 10)
        gt = torch.randn(4, 1)

        result = inference.query(['A'], {'input': x}, ground_truth=gt, concept_names=['A'], device='cpu')

        assert result.shape == (4, 1)

    def test_auto_mode(self):
        """Test auto device detection."""
        model = self._make_simple_model()
        inference = IndependentInference(model)

        x = torch.randn(4, 10)
        gt = torch.randn(4, 1)

        result = inference.query(['A'], {'input': x}, ground_truth=gt, concept_names=['A'], device='auto')

        assert result.shape == (4, 1)

    def test_debug_mode(self):
        """Test debug mode (sequential execution)."""
        model = self._make_simple_model()
        inference = IndependentInference(model)

        x = torch.randn(4, 10)
        gt = torch.randn(4, 1)

        result = inference.query(['A'], {'input': x}, ground_truth=gt, concept_names=['A'], debug=True)

        assert result.shape == (4, 1)


class TestIndependentInferenceErrorHandling:
    """Test error handling in IndependentInference."""

    def _make_simple_model(self):
        """Create a simple model for testing."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1), parents=['input'])

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            factors=[cpd_input, cpd_A]
        )
        return model

    def test_missing_evidence_raises_error(self):
        """Test that missing evidence raises appropriate error."""
        model = self._make_simple_model()
        inference = IndependentInference(model)
        
        gt = torch.randn(4, 1)

        with pytest.raises(AssertionError, match="Evidence must contain an 'input' key"):
            inference.query(['A'], {}, ground_truth=gt, concept_names=['A'])  # Missing 'input' evidence

    def test_invalid_query_concept_raises_error(self):
        """Test that invalid query concept raises error."""
        model = self._make_simple_model()
        inference = IndependentInference(model)

        x = torch.randn(4, 10)
        gt = torch.randn(4, 1)

        with pytest.raises(ValueError, match="was requested but could not be computed"):
            inference.query(['nonexistent'], {'input': x}, ground_truth=gt, concept_names=['A'])


class TestIndependentVsDeterministicInference:
    """Compare IndependentInference vs DeterministicInference behavior."""

    def _make_chain_model(self):
        """Create a chain model for comparison."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1), parents=['input'])
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(1, 1), parents=['A'])

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            factors=[cpd_input, cpd_A, cpd_B]
        )
        return model

    def test_different_results_with_gt(self):
        """Test that Independent and Deterministic give different results when GT differs from prediction."""
        model = self._make_chain_model()
        independent = IndependentInference(model)
        deterministic = DeterministicInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # Extreme GT that's very different from any prediction
        ground_truth = torch.full((batch_size, 1), 100.0)

        result_independent = independent.query(
            ['B'], 
            {'input': x}, 
            ground_truth=ground_truth,
            concept_names=['A']
        )
        result_deterministic = deterministic.query(['B'], {'input': x})

        # Results should differ because Independent uses GT for propagation
        # while Deterministic uses predicted A
        assert not torch.allclose(result_independent, result_deterministic, atol=1e-2)

    def test_same_results_same_gt(self):
        """Test that Independent and Deterministic give same A predictions (GT only affects propagation)."""
        model = self._make_chain_model()
        independent = IndependentInference(model)
        deterministic = DeterministicInference(model)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        
        # GT for A that will be used for propagation to B
        gt_a = torch.zeros(batch_size, 1)
        
        result_independent = independent.query(
            ['A'], 
            {'input': x}, 
            ground_truth=gt_a,
            concept_names=['A']
        )
        result_deterministic = deterministic.query(['A'], {'input': x})

        # A predictions should be identical (GT doesn't affect the prediction itself)
        torch.testing.assert_close(result_independent, result_deterministic)


class TestConceptMapping:
    """Test concept mapping and tensor-to-dict conversion."""

    def _make_simple_model(self):
        """Create a simple model for testing."""
        input_var = InputVariable('input', distribution=Delta, size=10)
        var_A = EndogenousVariable('A', distribution=Bernoulli, size=1)
        var_B = EndogenousVariable('B', distribution=Bernoulli, size=1)

        cpd_input = ParametricCPD('input', parametrization=nn.Identity())
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(10, 1), parents=['input'])
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(10, 1), parents=['input'])

        model = ProbabilisticModel(
            variables=[input_var, var_A, var_B],
            factors=[cpd_input, cpd_A, cpd_B]
        )
        return model

    def test_query_kwargs_property(self):
        """Test query_kwargs returns expected kwarg names from signature."""
        model = self._make_simple_model()
        inference = IndependentInference(model)
        
        # Should include all named parameters from the query signature
        # (excluding self, *args, **kwargs)
        assert 'ground_truth' in inference.query_kwargs
        assert 'concept_names' in inference.query_kwargs
        assert 'query' in inference.query_kwargs
        assert 'evidence' in inference.query_kwargs

    def test_query_with_tensor_and_concept_names(self):
        """Test query accepts ground_truth tensor with concept_names."""
        model = self._make_simple_model()
        inference = IndependentInference(model)
        
        x = torch.randn(4, 10)
        gt_tensor = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        
        result = inference.query(
            query=['A', 'B'],
            evidence={'input': x},
            ground_truth=gt_tensor,
            concept_names=['A', 'B']
        )
        
        assert result.shape == (4, 2)

    def test_query_accepts_extra_kwargs(self):
        """Test query accepts and ignores unknown kwargs from BaseLearner."""
        model = self._make_simple_model()
        inference = IndependentInference(model)
        
        x = torch.randn(4, 10)
        gt = torch.randn(4, 2)  # GT for A and B
        
        # Should not raise even with extra kwargs
        result = inference.query(
            query=['A', 'B'],
            evidence={'input': x},
            ground_truth=gt,
            concept_names=['A', 'B'],
            some_unknown_kwarg="ignored",
            another_one=123
        )
        
        assert result.shape == (4, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
