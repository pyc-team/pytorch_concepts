import torch_concepts.nn.functional as CF
import numpy as np
import unittest
import torch
import pandas as pd
from torch.nn import Linear
from torch_concepts.nn.functional import (
    grouped_concept_exogenous_mixture,
    selection_eval,
    linear_equation_eval,
    linear_equation_expl,
    logic_rule_eval,
    logic_rule_explanations,
    logic_memory_reconstruction,
    selective_calibration,
    confidence_selection,
    soft_select,
    completeness_score,
    intervention_score,
    cace_score,
    residual_concept_causal_effect,
    edge_type,
    custom_hamming_distance,
    prune_linear_layer,
    _default_concept_names,
    minimize_constr,
)
from torch_concepts.nn.modules.low.semantic import CMRSemantic


class TestMinimizeConstr(unittest.TestCase):
    """Test constrained minimization."""

    def test_minimize_unconstrained(self):
        """Test unconstrained minimization."""
        def f(x):
            return ((x - 2) ** 2).sum()

        x0 = torch.zeros(3)
        result = minimize_constr(
            f, x0,
            method='trust-constr',
            max_iter=100,
            tol=1e-6
        )

        self.assertTrue(result['success'])
        self.assertTrue(torch.allclose(result['x'], torch.tensor(2.0), atol=1e-2))

    def test_minimize_with_bounds(self):
        """Test minimization with bounds."""
        def f(x):
            return ((x - 2) ** 2).sum()

        x0 = torch.zeros(3)
        bounds = {'lb': 0.0, 'ub': 1.5}

        result = minimize_constr(
            f, x0,
            bounds=bounds,
            method='trust-constr',
            max_iter=100
        )

        self.assertTrue(result['success'])
        self.assertTrue(torch.all(result['x'] <= 1.5))

    def test_minimize_with_constraints(self):
        """Test minimization with nonlinear constraints."""
        def f(x):
            return ((x - 2) ** 2).sum()

        def constraint_fun(x):
            return x.sum()

        x0 = torch.ones(3)
        constr = {'fun': constraint_fun, 'lb': 0.0, 'ub': 2.0}

        result = minimize_constr(
            f, x0,
            constr=constr,
            method='trust-constr',
            max_iter=100
        )

        self.assertTrue(result['success'])

    def test_minimize_with_tensor_bounds(self):
        """Test with tensor bounds."""
        def f(x):
            return (x ** 2).sum()

        x0 = torch.ones(3)
        lb = torch.tensor([-1.0, -2.0, -3.0])
        ub = torch.tensor([1.0, 2.0, 3.0])
        bounds = {'lb': lb, 'ub': ub}

        result = minimize_constr(f, x0, bounds=bounds, max_iter=50)
        self.assertIsNotNone(result)

    def test_minimize_with_numpy_bounds(self):
        """Test with numpy array bounds."""
        def f(x):
            return (x ** 2).sum()

        x0 = torch.ones(2)
        bounds = {'lb': np.array([-1.0, -1.0]), 'ub': np.array([1.0, 1.0])}

        result = minimize_constr(f, x0, bounds=bounds, max_iter=50)
        self.assertIsNotNone(result)

    def test_minimize_with_callback(self):
        """Test callback functionality."""
        callback_calls = []

        def callback(x, state):
            callback_calls.append(x.clone())

        def f(x):
            return (x ** 2).sum()

        x0 = torch.ones(2)
        result = minimize_constr(f, x0, callback=callback, max_iter=10)
        self.assertGreater(len(callback_calls), 0)

    def test_minimize_with_equality_constraint(self):
        """Test equality constraint (lb == ub)."""
        def f(x):
            return (x ** 2).sum()

        def constraint_fun(x):
            return x[0] + x[1]

        x0 = torch.ones(2)
        constr = {'fun': constraint_fun, 'lb': 1.0, 'ub': 1.0}  # equality

        result = minimize_constr(f, x0, constr=constr, max_iter=50)
        self.assertIsNotNone(result)

    def test_minimize_with_custom_jac_hess(self):
        """Test with custom jacobian and hessian."""
        def f(x):
            return (x ** 2).sum()

        def jac(x):
            return 2 * x

        def hess(x):
            return 2 * torch.eye(x.numel(), dtype=x.dtype, device=x.device)

        x0 = torch.ones(3)
        result = minimize_constr(f, x0, jac=jac, hess=hess, max_iter=50)
        self.assertIsNotNone(result)

    def test_minimize_with_constraint_jac(self):
        """Test constraint with custom jacobian."""
        def f(x):
            return (x ** 2).sum()

        def constraint_fun(x):
            return x.sum()

        def constraint_jac(x):
            return torch.ones_like(x)

        x0 = torch.ones(3)
        constr = {'fun': constraint_fun, 'lb': 0.0, 'ub': 2.0, 'jac': constraint_jac}

        result = minimize_constr(f, x0, constr=constr, max_iter=50)
        self.assertIsNotNone(result)

    def test_minimize_display_options(self):
        """Test different display verbosity levels."""
        def f(x):
            return (x ** 2).sum()

        x0 = torch.ones(2)

        # Test with different disp values
        for disp in [0, 1]:
            result = minimize_constr(f, x0, disp=disp, max_iter=10)
            self.assertIsNotNone(result)

    def test_minimize_tolerance(self):
        """Test with custom tolerance."""
        def f(x):
            return (x ** 2).sum()

        x0 = torch.ones(2)
        result = minimize_constr(f, x0, tol=1e-8, max_iter=50)
        self.assertIsNotNone(result)

    def test_minimize_default_max_iter(self):
        """Test default max_iter value."""
        def f(x):
            return (x ** 2).sum()

        x0 = torch.ones(2)
        result = minimize_constr(f, x0)  # Uses default max_iter=1000
        self.assertIsNotNone(result)


class TestDefaultConceptNames(unittest.TestCase):
    """Test default concept name generation."""

    def test_default_concept_names_single_dim(self):
        """Test with single dimension."""
        names = _default_concept_names([5])
        self.assertEqual(names[1], ['concept_1_0', 'concept_1_1', 'concept_1_2', 'concept_1_3', 'concept_1_4'])

    def test_default_concept_names_multi_dim(self):
        """Test with multiple dimensions."""
        names = _default_concept_names([3, 4])
        self.assertEqual(len(names[1]), 3)
        self.assertEqual(len(names[2]), 4)

    def test_default_concept_names_empty(self):
        """Test with empty shape."""
        names = _default_concept_names([])
        self.assertEqual(names, {})


class TestGroupedConceptExogenousMixture(unittest.TestCase):
    """Test grouped concept exogenous mixture."""

    def test_grouped_mixture_basic(self):
        """Test basic grouped mixture."""
        batch_size = 4
        n_concepts = 10
        emb_size = 20
        groups = [3, 4, 3]

        c_emb = torch.randn(batch_size, n_concepts, emb_size)
        c_scores = torch.rand(batch_size, n_concepts)

        result = grouped_concept_exogenous_mixture(c_emb, c_scores, groups)

        self.assertEqual(result.shape, (batch_size, len(groups), emb_size // 2))

    def test_grouped_mixture_singleton_groups(self):
        """Test with singleton groups (two-half mixture)."""
        batch_size = 2
        n_concepts = 3
        emb_size = 10
        groups = [1, 1, 1]

        c_emb = torch.randn(batch_size, n_concepts, emb_size)
        c_scores = torch.rand(batch_size, n_concepts)

        result = grouped_concept_exogenous_mixture(c_emb, c_scores, groups)
        self.assertEqual(result.shape, (batch_size, 3, emb_size // 2))

    def test_grouped_mixture_invalid_groups(self):
        """Test with invalid group sizes."""
        c_emb = torch.randn(2, 5, 10)
        c_scores = torch.rand(2, 5)
        groups = [2, 2]  # Doesn't sum to 5

        with self.assertRaises(AssertionError):
            grouped_concept_exogenous_mixture(c_emb, c_scores, groups)

    def test_grouped_mixture_odd_exogenous_dim(self):
        """Test with odd exogenous dimension."""
        c_emb = torch.randn(2, 3, 9)  # Odd dimension
        c_scores = torch.rand(2, 3)
        groups = [3]

        with self.assertRaises(AssertionError):
            grouped_concept_exogenous_mixture(c_emb, c_scores, groups)


class TestSelectionEval(unittest.TestCase):
    """Test selection evaluation."""

    def test_selection_eval_basic(self):
        """Test basic selection evaluation."""
        weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        pred1 = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
        pred2 = torch.tensor([[0.9, 0.1], [0.7, 0.3]])

        result = selection_eval(weights, pred1, pred2)
        self.assertEqual(result.shape, (2,))

    def test_selection_eval_single_prediction(self):
        """Test with single prediction."""
        weights = torch.tensor([[1.0, 0.0]])
        pred = torch.tensor([[0.5, 0.5]])

        result = selection_eval(weights, pred)
        self.assertEqual(result.shape, (1,))

    def test_selection_eval_no_predictions(self):
        """Test with no predictions."""
        weights = torch.tensor([[0.5, 0.5]])

        with self.assertRaises(ValueError):
            selection_eval(weights)

    def test_selection_eval_shape_mismatch(self):
        """Test with mismatched shapes."""
        weights = torch.tensor([[0.5, 0.5]])
        pred1 = torch.tensor([[0.8, 0.2]])
        pred2 = torch.tensor([[0.9, 0.1, 0.3]])  # Different shape

        with self.assertRaises(AssertionError):
            selection_eval(weights, pred1, pred2)


class TestLinearEquationEval(unittest.TestCase):
    """Test linear equation evaluation."""

    def test_linear_equation_eval_basic(self):
        """Test basic linear equation evaluation."""
        batch_size = 2
        memory_size = 3
        n_concepts = 4
        n_classes = 2

        concept_weights = torch.randn(batch_size, memory_size, n_concepts, n_classes)
        c_pred = torch.randn(batch_size, n_concepts)

        result = linear_equation_eval(concept_weights, c_pred)
        self.assertEqual(result.shape, (batch_size, n_classes, memory_size))

    def test_linear_equation_eval_with_bias(self):
        """Test with bias term."""
        batch_size = 2
        memory_size = 3
        n_concepts = 4
        n_classes = 2

        concept_weights = torch.randn(batch_size, memory_size, n_concepts, n_classes)
        c_pred = torch.randn(batch_size, n_concepts)
        bias = torch.randn(batch_size, memory_size, n_classes)

        result = linear_equation_eval(concept_weights, c_pred, bias)
        self.assertEqual(result.shape, (batch_size, n_classes, memory_size))

    def test_linear_equation_eval_shape_assertion(self):
        """Test shape assertions."""
        concept_weights = torch.randn(2, 3, 4, 2)
        c_pred = torch.randn(2, 5)  # Wrong number of concepts

        with self.assertRaises(AssertionError):
            linear_equation_eval(concept_weights, c_pred)


class TestLinearEquationExpl(unittest.TestCase):
    """Test linear equation explanation extraction."""

    def test_linear_equation_expl_basic(self):
        """Test basic explanation extraction."""
        batch_size = 2
        memory_size = 2
        n_concepts = 3
        n_tasks = 2

        concept_weights = torch.randn(batch_size, memory_size, n_concepts, n_tasks)

        result = linear_equation_expl(concept_weights)
        self.assertEqual(len(result), batch_size)
        self.assertIsInstance(result[0], dict)

    def test_linear_equation_expl_with_bias(self):
        """Test with bias term."""
        concept_weights = torch.randn(1, 2, 3, 1)
        bias = torch.randn(1, 2, 1)

        result = linear_equation_expl(concept_weights, bias)
        self.assertEqual(len(result), 1)

    def test_linear_equation_expl_with_names(self):
        """Test with custom concept names."""
        concept_weights = torch.randn(1, 2, 3, 1)
        concept_names = {1: ['a', 'b', 'c'], 2: ['task1']}

        result = linear_equation_expl(concept_weights, concept_names=concept_names)
        self.assertIn('task1', result[0])

    def test_linear_equation_expl_invalid_shape(self):
        """Test with invalid shape."""
        concept_weights = torch.randn(2, 3, 4)  # Only 3 dimensions

        with self.assertRaises(ValueError):
            linear_equation_expl(concept_weights)

    def test_linear_equation_expl_with_concept_names_attribute(self):
        """Test with concept_names as tensor attribute."""
        concept_weights = torch.randn(1, 2, 3, 2)
        # Add concept_names as attribute
        concept_weights.concept_names = {1: ['c1', 'c2', 'c3'], 2: ['t1', 't2']}

        result = linear_equation_expl(concept_weights)
        self.assertEqual(len(result), 1)
        self.assertIn('t1', result[0])
        self.assertIn('t2', result[0])

    def test_linear_equation_expl_invalid_concept_names_length(self):
        """Test with invalid concept names length."""
        concept_weights = torch.randn(1, 2, 3, 1)
        concept_names = {1: ['a', 'b'], 2: ['task1']}  # Only 2 concepts instead of 3

        with self.assertRaises(ValueError):
            linear_equation_expl(concept_weights, concept_names=concept_names)


class TestLogicRuleEval(unittest.TestCase):
    """Test logic rule evaluation."""

    def test_logic_rule_eval_basic(self):
        """Test basic logic rule evaluation."""
        batch_size = 2
        memory_size = 3
        n_concepts = 4
        n_tasks = 2
        n_roles = 3

        # Use softmax to ensure weights sum to 1 across roles dimension
        concept_weights = torch.randn(batch_size, memory_size, n_concepts, n_tasks, n_roles)
        concept_weights = torch.softmax(concept_weights, dim=-1)
        c_pred = torch.rand(batch_size, n_concepts)

        result = logic_rule_eval(concept_weights, c_pred)
        self.assertEqual(result.shape, (batch_size, n_tasks, memory_size))
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_logic_rule_eval_with_semantic(self):
        """Test with custom semantic."""
        concept_weights = torch.randn(1, 2, 3, 1, 3)
        concept_weights = torch.softmax(concept_weights, dim=-1)
        c_pred = torch.rand(1, 3)
        semantic = CMRSemantic()

        result = logic_rule_eval(concept_weights, c_pred, semantic=semantic)
        self.assertEqual(result.shape, (1, 1, 2))

    def test_logic_rule_eval_invalid_shape(self):
        """Test with invalid shape."""
        concept_weights = torch.randn(2, 3, 4, 2)  # Only 4 dimensions
        c_pred = torch.rand(2, 4)

        with self.assertRaises(AssertionError):
            logic_rule_eval(concept_weights, c_pred)


class TestLogicRuleExplanations(unittest.TestCase):
    """Test logic rule explanation extraction."""

    def test_logic_rule_explanations_basic(self):
        """Test basic rule extraction."""
        batch_size = 2
        memory_size = 2
        n_concepts = 3
        n_tasks = 1

        # Create weights with clear roles
        concept_logic_weights = torch.zeros(batch_size, memory_size, n_concepts, n_tasks, 3)
        concept_logic_weights[..., 0] = 1.0  # All positive polarity

        result = logic_rule_explanations(concept_logic_weights)
        self.assertEqual(len(result), batch_size)
        self.assertIsInstance(result[0], dict)

    def test_logic_rule_explanations_with_names(self):
        """Test with custom names."""
        concept_logic_weights = torch.zeros(1, 1, 2, 1, 3)
        concept_logic_weights[..., 0] = 1.0
        concept_names = {1: ['concept_a', 'concept_b'], 2: ['task1']}

        result = logic_rule_explanations(concept_logic_weights, concept_names)
        self.assertIn('task1', result[0])

    def test_logic_rule_explanations_invalid_shape(self):
        """Test with invalid shape."""
        concept_logic_weights = torch.randn(1, 2, 3, 1, 4)  # Last dim != 3

        with self.assertRaises(ValueError):
            logic_rule_explanations(concept_logic_weights)

    def test_logic_rule_explanations_with_concept_names_attribute(self):
        """Test with concept_names as tensor attribute."""
        concept_logic_weights = torch.zeros(1, 1, 2, 1, 3)
        concept_logic_weights[..., 0] = 1.0
        concept_logic_weights.concept_names = {1: ['ca', 'cb'], 2: ['task1']}

        result = logic_rule_explanations(concept_logic_weights)
        self.assertIn('task1', result[0])

    def test_logic_rule_explanations_with_negative_polarity(self):
        """Test rule extraction with negative polarity."""
        concept_logic_weights = torch.zeros(1, 1, 2, 1, 3)
        concept_logic_weights[..., 1] = 1.0  # Negative polarity

        result = logic_rule_explanations(concept_logic_weights)
        # Should contain '~' for negation
        rule_str = list(result[0].values())[0]['Rule 0']
        self.assertIn('~', rule_str)

    def test_logic_rule_explanations_with_irrelevance(self):
        """Test rule extraction with irrelevant concepts."""
        concept_logic_weights = torch.zeros(1, 1, 3, 1, 3)
        concept_logic_weights[0, 0, 0, 0, 0] = 1.0  # Positive
        concept_logic_weights[0, 0, 1, 0, 1] = 1.0  # Negative
        concept_logic_weights[0, 0, 2, 0, 2] = 1.0  # Irrelevant - should be skipped

        result = logic_rule_explanations(concept_logic_weights)
        rule_str = list(result[0].values())[0]['Rule 0']
        # Should not contain c_2 (irrelevant concept)
        self.assertNotIn('c_2', rule_str)


class TestLogicMemoryReconstruction(unittest.TestCase):
    """Test logic memory reconstruction."""

    def test_logic_memory_reconstruction_basic(self):
        """Test basic reconstruction."""
        batch_size = 2
        memory_size = 3
        n_concepts = 4
        n_tasks = 2

        concept_weights = torch.randn(batch_size, memory_size, n_concepts, n_tasks, 3)
        concept_weights = torch.softmax(concept_weights, dim=-1)
        c_true = torch.randint(0, 2, (batch_size, n_concepts)).float()
        y_true = torch.randint(0, 2, (batch_size, n_tasks)).float()

        result = logic_memory_reconstruction(concept_weights, c_true, y_true)
        self.assertEqual(result.shape, (batch_size, n_tasks, memory_size))

    def test_logic_memory_reconstruction_with_zeros(self):
        """Test reconstruction with zero concepts."""
        concept_weights = torch.randn(1, 2, 3, 1, 3)
        c_true = torch.zeros(1, 3)
        y_true = torch.zeros(1, 1)

        result = logic_memory_reconstruction(concept_weights, c_true, y_true)
        self.assertEqual(result.shape, (1, 1, 2))

    def test_logic_memory_reconstruction_with_ones(self):
        """Test reconstruction with all-one concepts."""
        concept_weights = torch.randn(1, 2, 3, 1, 3)
        c_true = torch.ones(1, 3)
        y_true = torch.ones(1, 1)

        result = logic_memory_reconstruction(concept_weights, c_true, y_true)
        self.assertEqual(result.shape, (1, 1, 2))


class TestCalibration(unittest.TestCase):
    """Test calibration functions."""

    def test_selective_calibration(self):
        """Test selective calibration."""
        c_confidence = torch.rand(100, 5)
        target_coverage = 0.8

        theta = selective_calibration(c_confidence, target_coverage)
        self.assertEqual(theta.shape, (1, 5))

    def test_confidence_selection(self):
        """Test confidence selection."""
        c_confidence = torch.tensor([[0.9, 0.3, 0.7], [0.2, 0.8, 0.5]])
        theta = torch.tensor([[0.5, 0.5, 0.5]])

        result = confidence_selection(c_confidence, theta)
        self.assertEqual(result.shape, c_confidence.shape)
        self.assertTrue(result[0, 0])  # 0.9 > 0.5
        self.assertFalse(result[0, 1])  # 0.3 < 0.5

    def test_soft_select(self):
        """Test soft selection."""
        values = torch.randn(10, 5)
        temperature = 0.5

        result = soft_select(values, temperature)
        self.assertEqual(result.shape, values.shape)
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_soft_select_different_dim(self):
        """Test soft select with different dimension."""
        values = torch.randn(3, 4, 5)
        result = soft_select(values, 0.5, dim=2)
        self.assertEqual(result.shape, values.shape)


class TestCompletenessScore(unittest.TestCase):
    """Test completeness score."""

    def test_completeness_score_basic(self):
        """Test basic completeness score."""
        y_true = torch.randint(0, 2, (100, 3))
        y_pred_blackbox = torch.rand(100, 3)
        y_pred_whitebox = torch.rand(100, 3)

        from sklearn.metrics import roc_auc_score
        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox,
                                   scorer=roc_auc_score, average='macro')
        self.assertIsInstance(score, float)
        self.assertTrue(score >= 0)


class TestInterventionScore(unittest.TestCase):
    """Test intervention score."""

    def test_intervention_score_basic(self):
        """Test basic intervention score."""
        # Simple predictor
        y_predictor = torch.nn.Linear(5, 2)
        c_pred = torch.rand(20, 5)
        c_true = torch.randint(0, 2, (20, 5)).float()
        y_true = torch.randint(0, 2, (20, 2))
        intervention_groups = [[0], [1], [2]]

        from sklearn.metrics import roc_auc_score
        score = intervention_score(
            y_predictor, c_pred, c_true, y_true, intervention_groups,
            scorer=roc_auc_score, auc=True
        )
        self.assertIsInstance(score, float)

    def test_intervention_score_list_output(self):
        """Test intervention score with list output."""
        y_predictor = torch.nn.Linear(3, 1)
        c_pred = torch.rand(10, 3)
        c_true = torch.randint(0, 2, (10, 3)).float()
        y_true = torch.randint(0, 2, (10, 1))
        intervention_groups = [[0], [1]]

        # Wrap accuracy_score to accept (and ignore) the average parameter
        from sklearn.metrics import accuracy_score
        scores = intervention_score(
            y_predictor, c_pred, c_true, y_true, intervention_groups,
            activation=lambda x: (x > 0).float(),
            scorer=lambda y_true, y_pred, **kwargs: accuracy_score(y_true, y_pred),
            auc=False
        )
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 2)


class TestCACEScore(unittest.TestCase):
    """Test Causal Average Concept Effect score."""

    def test_cace_score_basic(self):
        """Test basic CACE score."""
        y_pred_c0 = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
        y_pred_c1 = torch.tensor([[0.8, 0.2], [0.9, 0.1]])

        result = cace_score(y_pred_c0, y_pred_c1)
        self.assertEqual(result.shape, (2,))

    def test_cace_score_shape_mismatch(self):
        """Test with mismatched shapes."""
        y_pred_c0 = torch.rand(5, 2)
        y_pred_c1 = torch.rand(5, 3)

        with self.assertRaises(RuntimeError):
            cace_score(y_pred_c0, y_pred_c1)

    def test_residual_concept_causal_effect(self):
        """Test residual concept causal effect."""
        cace_before = torch.tensor(0.5)
        cace_after = torch.tensor(0.3)

        result = residual_concept_causal_effect(cace_before, cace_after)
        self.assertEqual(result, 0.6)


class TestGraphMetrics(unittest.TestCase):
    """Test graph similarity metrics."""

    def test_edge_type(self):
        """Test edge type detection."""
        graph = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

        self.assertEqual(edge_type(graph, 0, 1), 'i->j')
        self.assertEqual(edge_type(graph, 1, 0), 'i<-j')
        self.assertEqual(edge_type(graph, 0, 2), '/')

    def test_edge_type_undirected(self):
        """Test undirected edge."""
        graph = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(edge_type(graph, 0, 1), 'i-j')

    def test_hamming_distance(self):
        """Test Hamming distance between graphs."""
        # Create simple graphs
        nodes = ['A', 'B', 'C']
        graph1_data = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        graph2_data = [[0, 1, 0], [0, 0, 0], [0, 1, 0]]

        graph1 = pd.DataFrame(graph1_data, index=nodes, columns=nodes)
        graph2 = pd.DataFrame(graph2_data, index=nodes, columns=nodes)

        cost, count = custom_hamming_distance(graph1, graph2)
        self.assertIsInstance(cost, (int, float))
        self.assertIsInstance(count, int)


class TestPruneLinearLayer(unittest.TestCase):
    """Test linear layer pruning."""

    def test_prune_input_features(self):
        """Test pruning input features."""
        linear = Linear(10, 5)
        mask = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.bool)

        pruned = prune_linear_layer(linear, mask, dim=0)
        self.assertEqual(pruned.in_features, 7)
        self.assertEqual(pruned.out_features, 5)

    def test_prune_output_features(self):
        """Test pruning output features."""
        linear = Linear(10, 8)
        mask = torch.tensor([1, 1, 0, 1, 0, 1, 1, 0], dtype=torch.bool)

        pruned = prune_linear_layer(linear, mask, dim=1)
        self.assertEqual(pruned.in_features, 10)
        self.assertEqual(pruned.out_features, 5)

    def test_prune_with_bias(self):
        """Test pruning with bias."""
        linear = Linear(5, 3, bias=True)
        mask = torch.tensor([1, 0, 1], dtype=torch.bool)

        pruned = prune_linear_layer(linear, mask, dim=1)
        self.assertIsNotNone(pruned.bias)
        self.assertEqual(pruned.bias.shape[0], 2)

    def test_prune_without_bias(self):
        """Test pruning without bias."""
        linear = Linear(5, 3, bias=False)
        mask = torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool)

        pruned = prune_linear_layer(linear, mask, dim=0)
        self.assertIsNone(pruned.bias)

    def test_prune_invalid_mask_length(self):
        """Test with invalid mask length."""
        linear = Linear(10, 5)
        mask = torch.tensor([1, 1, 1], dtype=torch.bool)  # Wrong length

        with self.assertRaises(ValueError):
            prune_linear_layer(linear, mask, dim=0)

    def test_prune_invalid_dim(self):
        """Test with invalid dimension."""
        linear = Linear(5, 3)
        mask = torch.tensor([1, 1, 1], dtype=torch.bool)

        with self.assertRaises(ValueError):
            prune_linear_layer(linear, mask, dim=2)

    def test_prune_non_linear_layer(self):
        """Test with non-Linear layer."""
        conv = torch.nn.Conv2d(3, 5, 3)
        mask = torch.tensor([1, 1, 1], dtype=torch.bool)

        with self.assertRaises(TypeError):
            prune_linear_layer(conv, mask, dim=0)


class TestConceptFunctions(unittest.TestCase):

    def setUp(self):
        self.c_pred = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        self.c_true = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
        self.indexes = torch.tensor([[True, False], [False, True]])
        self.c_confidence = torch.tensor([[0.8, 0.1, 0.6],
                                          [0.9, 0.2, 0.4],
                                          [0.7, 0.3, 0.5]])
        self.target_confidence = 0.5

    def test_selective_calibration(self):
        expected_theta = torch.tensor([[0.8, 0.2, 0.5]])
        expected_result = expected_theta
        result = CF.selective_calibration(self.c_confidence,
                                          self.target_confidence)
        self.assertEqual(torch.all(result == expected_result).item(), True)

    def test_confidence_selection(self):
        theta = torch.tensor([[0.8, 0.3, 0.5]])
        expected_result = torch.tensor([[False, False, True],
                                        [True, False, False],
                                        [False, False, False]])
        result = CF.confidence_selection(self.c_confidence, theta)
        self.assertEqual(torch.all(result == expected_result).item(), True)

    def test_linear_eq_eval(self):
        # batch_size x memory_size x n_concepts x n_classes
        c_imp = torch.tensor([
            [[[0.], [10.]]],
            [[[0.], [-10]]],
            [[[0.], [-10]]],
            [[[0.], [0.]]],
            [[[0.], [0.]]],
        ])
        c_pred = torch.tensor([
            [0., 1.],
            [0., 1.],
            [0., -1.],
            [0., 0.],
            [0., 0.],
        ])
        y_bias = torch.tensor([
            [[.0]],
            [[.0]],
            [[.0]],
            [[.0]],
            [[1.0]],
        ])
        expected_result = torch.tensor([
            [True],
            [False],
            [True],
            [False],
            [True],
        ])
        result = CF.linear_equation_eval(c_imp, c_pred, y_bias)[:, 0]
        # print(result)
        # print((result > 0) == expected_result)
        self.assertEqual(torch.all((result > 0) == expected_result).item(),
                         True)


if __name__ == '__main__':
    unittest.main()
