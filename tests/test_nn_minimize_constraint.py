"""
Comprehensive tests for torch_concepts.nn.minimize_constraint

Tests constrained optimization functionality.
"""
import unittest
import torch
import numpy as np
from torch_concepts.nn.minimize_constraint import minimize_constr


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


if __name__ == '__main__':
    unittest.main()
