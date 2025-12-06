"""
Tests for CPD parent preservation during lazy initialization.

This test module verifies that ParametricCPD parent relationships are preserved
when LazyConstructor builds the parametrization module.

Bug context: When LazyConstructor.build() was called to initialize a lazy module,
ProbabilisticModel created a new ParametricCPD instance but didn't copy the
parents list and variable reference from the original CPD. This caused the CPD
to have an empty parents list during forward pass, leading to incorrect input
feature calculations.

Fix: After LazyConstructor builds the new parametrization, copy the parents
and variable attributes from the old CPD to the new CPD:
    new_parametrization.parents = parametric_cpd.parents
    new_parametrization.variable = parametric_cpd.variable
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.low.lazy import LazyConstructor


class TestCPDParentPreservation(unittest.TestCase):
    """Test that CPD parents are preserved during lazy initialization."""

    def test_cpd_parents_preserved_after_lazy_build(self):
        """Test that parent list is preserved when LazyConstructor builds the CPD.
        
        This is the core bug fix test: when a LazyConstructor is used for a CPD's
        parametrization and build() is called, the resulting CPD must retain its
        parent references.
        """
        # Create parent and child variables
        parent = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)
        
        # Create CPDs with LazyConstructor
        parent_cpd = ParametricCPD(
            concepts='parent',
            parametrization=LazyConstructor(nn.Linear, out_features=1)
        )
        child_cpd = ParametricCPD(
            concepts='child',
            parametrization=LazyConstructor(nn.Linear, out_features=1)
        )
        
        # Set parents and variable on CPDs (this happens in GraphModel._init_predictors)
        parent_cpd.parents = []
        parent_cpd.variable = parent
        child_cpd.parents = [parent]
        child_cpd.variable = child
        
        # Store original parent reference
        original_parents = child_cpd.parents
        self.assertEqual(len(original_parents), 1)
        self.assertEqual(original_parents[0].concepts, ['parent'])
        
        # Create model (this will trigger lazy initialization)
        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        
        # Simulate what happens during forward pass: LazyConstructor.build() is called
        # This happens in ProbabilisticModel when a lazy parametrization is detected
        
        # Get the child CPD from the model
        child_cpd_from_model = model.get_module_of_concept('child')
        
        # Before any forward pass, if it's still lazy, build it manually
        if isinstance(child_cpd_from_model.parametrization, LazyConstructor):
            # Simulate input to trigger build
            dummy_input = torch.randn(4, 1)
            child_cpd_from_model.parametrization.build(
                input=dummy_input,
                endogenous=dummy_input,
                exogenous=None
            )
        
        # After build, check that parents are still present
        self.assertEqual(len(child_cpd_from_model.parents), 1,
                        "Child CPD should still have 1 parent after lazy initialization")
        self.assertEqual(child_cpd_from_model.parents[0].concepts, ['parent'],
                        "Child CPD parent should still reference the parent variable")

    def test_cpd_variable_preserved_after_lazy_build(self):
        """Test that variable reference is preserved when LazyConstructor builds the CPD."""
        # Create variable
        var = Variable(concepts=['test_var'], parents=[], distribution=Bernoulli, size=1)
        
        # Create CPD with LazyConstructor
        cpd = ParametricCPD(
            concepts='test_var',
            parametrization=LazyConstructor(nn.Linear, out_features=1)
        )
        
        # Set variable reference
        cpd.variable = var
        
        # Store original variable reference
        original_variable = cpd.variable
        self.assertIsNotNone(original_variable)
        self.assertEqual(original_variable.concepts, ['test_var'])
        
        # Create model
        model = ProbabilisticModel(
            variables=[var],
            parametric_cpds=[cpd]
        )
        
        # Get CPD from model
        cpd_from_model = model.get_module_of_concept('test_var')
        
        # Build if lazy
        if isinstance(cpd_from_model.parametrization, LazyConstructor):
            dummy_input = torch.randn(4, 10)
            cpd_from_model.parametrization.build(
                input=dummy_input,
                endogenous=None,
                exogenous=None
            )
        
        # Check that variable reference is preserved
        self.assertIsNotNone(cpd_from_model.variable,
                            "CPD should still have variable reference after lazy initialization")
        self.assertEqual(cpd_from_model.variable.concepts, ['test_var'],
                        "CPD variable should still reference the correct variable")

    def test_multiple_parents_preserved(self):
        """Test that multiple parent references are all preserved."""
        # Create parent variables
        parent1 = Variable(concepts=['p1'], parents=[], distribution=Bernoulli, size=1)
        parent2 = Variable(concepts=['p2'], parents=[], distribution=Bernoulli, size=1)
        parent3 = Variable(concepts=['p3'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent1, parent2, parent3], distribution=Bernoulli, size=1)
        
        # Create CPDs
        p1_cpd = ParametricCPD(concepts='p1', parametrization=LazyConstructor(nn.Linear, out_features=1))
        p2_cpd = ParametricCPD(concepts='p2', parametrization=LazyConstructor(nn.Linear, out_features=1))
        p3_cpd = ParametricCPD(concepts='p3', parametrization=LazyConstructor(nn.Linear, out_features=1))
        child_cpd = ParametricCPD(concepts='child', parametrization=LazyConstructor(nn.Linear, out_features=1))
        
        # Set parents
        p1_cpd.parents = []
        p1_cpd.variable = parent1
        p2_cpd.parents = []
        p2_cpd.variable = parent2
        p3_cpd.parents = []
        p3_cpd.variable = parent3
        child_cpd.parents = [parent1, parent2, parent3]
        child_cpd.variable = child
        
        # Create model
        model = ProbabilisticModel(
            variables=[parent1, parent2, parent3, child],
            parametric_cpds=[p1_cpd, p2_cpd, p3_cpd, child_cpd]
        )
        
        # Get child CPD
        child_cpd_from_model = model.get_module_of_concept('child')
        
        # Build if lazy
        if isinstance(child_cpd_from_model.parametrization, LazyConstructor):
            dummy_input = torch.randn(4, 3)
            child_cpd_from_model.parametrization.build(
                input=dummy_input,
                endogenous=dummy_input,
                exogenous=None
            )
        
        # Check all parents are preserved
        self.assertEqual(len(child_cpd_from_model.parents), 3,
                        "Child CPD should have 3 parents after lazy initialization")
        parent_concepts = [p.concepts[0] for p in child_cpd_from_model.parents]
        self.assertIn('p1', parent_concepts)
        self.assertIn('p2', parent_concepts)
        self.assertIn('p3', parent_concepts)

    def test_hierarchical_structure_parents_preserved(self):
        """Test parent preservation in a hierarchical structure.
        
        Tests a more complex hierarchy: A -> B -> C -> D
        Each CPD should maintain its parent references after lazy initialization.
        """
        # Create hierarchical variables
        var_a = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        var_b = Variable(concepts=['B'], parents=[var_a], distribution=Bernoulli, size=1)
        var_c = Variable(concepts=['C'], parents=[var_b], distribution=Bernoulli, size=1)
        var_d = Variable(concepts=['D'], parents=[var_c], distribution=Bernoulli, size=1)
        
        # Create CPDs with lazy constructors
        cpd_a = ParametricCPD(concepts='A', parametrization=LazyConstructor(nn.Linear, out_features=1))
        cpd_b = ParametricCPD(concepts='B', parametrization=LazyConstructor(nn.Linear, out_features=1))
        cpd_c = ParametricCPD(concepts='C', parametrization=LazyConstructor(nn.Linear, out_features=1))
        cpd_d = ParametricCPD(concepts='D', parametrization=LazyConstructor(nn.Linear, out_features=1))
        
        # Set parents and variables
        cpd_a.parents = []
        cpd_a.variable = var_a
        cpd_b.parents = [var_a]
        cpd_b.variable = var_b
        cpd_c.parents = [var_b]
        cpd_c.variable = var_c
        cpd_d.parents = [var_c]
        cpd_d.variable = var_d
        
        # Create model
        model = ProbabilisticModel(
            variables=[var_a, var_b, var_c, var_d],
            parametric_cpds=[cpd_a, cpd_b, cpd_c, cpd_d]
        )
        
        # Check each CPD
        for concept_name, expected_parent_count, expected_parent_name in [
            ('A', 0, None),
            ('B', 1, 'A'),
            ('C', 1, 'B'),
            ('D', 1, 'C')
        ]:
            cpd = model.get_module_of_concept(concept_name)
            
            # Build if lazy
            if isinstance(cpd.parametrization, LazyConstructor):
                input_size = expected_parent_count if expected_parent_count > 0 else 10
                dummy_input = torch.randn(4, input_size)
                cpd.parametrization.build(
                    input=dummy_input,
                    endogenous=dummy_input if expected_parent_count > 0 else None,
                    exogenous=None
                )
            
            self.assertEqual(len(cpd.parents), expected_parent_count,
                            f"CPD {concept_name} should have {expected_parent_count} parents")
            if expected_parent_name:
                self.assertEqual(cpd.parents[0].concepts[0], expected_parent_name,
                                f"CPD {concept_name} parent should be {expected_parent_name}")

    def test_categorical_variables_parents_preserved(self):
        """Test parent preservation with categorical variables.
        
        Categorical variables have cardinality > 2, which affects the input size
        calculations that depend on parent references.
        """
        # Create parent with higher cardinality
        parent = Variable(concepts=['parent'], parents=[], distribution=Categorical, size=5)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)
        
        # Create CPDs
        parent_cpd = ParametricCPD(
            concepts='parent',
            parametrization=LazyConstructor(nn.Linear, out_features=5)
        )
        child_cpd = ParametricCPD(
            concepts='child',
            parametrization=LazyConstructor(nn.Linear, out_features=1)
        )
        
        # Set parents and variables
        parent_cpd.parents = []
        parent_cpd.variable = parent
        child_cpd.parents = [parent]
        child_cpd.variable = child
        
        # Create model
        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        
        # Get child CPD
        child_cpd_from_model = model.get_module_of_concept('child')
        
        # Check parents before build
        self.assertEqual(len(child_cpd_from_model.parents), 1,
                        "Child CPD should have 1 parent before build")
        
        # Build if lazy
        if isinstance(child_cpd_from_model.parametrization, LazyConstructor):
            # Parent has out_features=5, so child input should match
            dummy_input = torch.randn(4, 5)
            child_cpd_from_model.parametrization.build(
                input=dummy_input,
                endogenous=dummy_input,
                exogenous=None
            )
        
        # Check parents after build
        self.assertEqual(len(child_cpd_from_model.parents), 1,
                        "Child CPD should still have 1 parent after build")
        self.assertEqual(child_cpd_from_model.parents[0].out_features, 5,
                        "Parent should have out_features=5")

    def test_non_lazy_cpd_parents_unchanged(self):
        """Test that non-lazy CPDs don't lose parent information.
        
        This is a control test to ensure the fix doesn't break non-lazy CPDs.
        """
        # Create variables
        parent = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)
        
        # Create CPDs with non-lazy parametrization
        parent_cpd = ParametricCPD(
            concepts='parent',
            parametrization=nn.Linear(10, 1)
        )
        child_cpd = ParametricCPD(
            concepts='child',
            parametrization=nn.Linear(1, 1)
        )
        
        # Set parents and variables
        parent_cpd.parents = []
        parent_cpd.variable = parent
        child_cpd.parents = [parent]
        child_cpd.variable = child
        
        # Create model
        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        
        # Get child CPD
        child_cpd_from_model = model.get_module_of_concept('child')
        
        # Check that parents are preserved (no build needed)
        self.assertEqual(len(child_cpd_from_model.parents), 1,
                        "Child CPD should have 1 parent (non-lazy case)")
        self.assertEqual(child_cpd_from_model.parents[0].concepts, ['parent'],
                        "Child CPD parent should reference parent variable")


class TestCPDParentInFeatureCalculation(unittest.TestCase):
    """Test that in_features calculation works correctly with preserved parents."""

    def test_in_features_calculation_with_parents(self):
        """Test that CPD in_features calculation uses preserved parent information.
        
        The in_features of a child CPD should equal the sum of parent out_features.
        This calculation depends on having correct parent references.
        """
        # Create parents with different sizes
        parent1 = Variable(concepts=['p1'], parents=[], distribution=Categorical, size=3)
        parent2 = Variable(concepts=['p2'], parents=[], distribution=Categorical, size=5)
        child = Variable(concepts=['child'], parents=[parent1, parent2], distribution=Bernoulli, size=1)
        
        # Create CPDs
        p1_cpd = ParametricCPD(concepts='p1', parametrization=LazyConstructor(nn.Linear, out_features=3))
        p2_cpd = ParametricCPD(concepts='p2', parametrization=LazyConstructor(nn.Linear, out_features=5))
        child_cpd = ParametricCPD(concepts='child', parametrization=LazyConstructor(nn.Linear, out_features=1))
        
        # Set parents and variables
        p1_cpd.parents = []
        p1_cpd.variable = parent1
        p2_cpd.parents = []
        p2_cpd.variable = parent2
        child_cpd.parents = [parent1, parent2]
        child_cpd.variable = child
        
        # Create model
        model = ProbabilisticModel(
            variables=[parent1, parent2, child],
            parametric_cpds=[p1_cpd, p2_cpd, child_cpd]
        )
        
        # Get child CPD
        child_cpd_from_model = model.get_module_of_concept('child')
        
        # Expected in_features = parent1.out_features + parent2.out_features = 3 + 5 = 8
        expected_in_features = parent1.out_features + parent2.out_features
        
        # Build if lazy
        if isinstance(child_cpd_from_model.parametrization, LazyConstructor):
            dummy_input = torch.randn(4, expected_in_features)
            child_cpd_from_model.parametrization.build(
                input=dummy_input,
                endogenous=dummy_input,
                exogenous=None
            )
        
        # Check that built parametrization has correct in_features
        actual_in_features = child_cpd_from_model.parametrization.in_features
        self.assertEqual(actual_in_features, expected_in_features,
                        f"Child CPD in_features should be {expected_in_features} "
                        f"(sum of parent out_features), got {actual_in_features}")

    def test_forward_pass_with_correct_parent_features(self):
        """Test that forward pass works correctly with preserved parent information.
        
        This integration test verifies that the entire forward pass pipeline works
        when CPD parents are correctly preserved. Note: This is a simplified test
        that doesn't actually run a forward pass, just verifies parent preservation.
        """
        # Create simple parent-child structure
        parent = Variable(concepts=['parent'], parents=[], distribution=Categorical, size=4)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)
        
        # Create CPDs with lazy constructors
        parent_cpd = ParametricCPD(
            concepts='parent',
            parametrization=LazyConstructor(nn.Linear, out_features=4)
        )
        child_cpd = ParametricCPD(
            concepts='child',
            parametrization=LazyConstructor(nn.Linear, out_features=1)
        )
        
        # Set parents and variables
        parent_cpd.parents = []
        parent_cpd.variable = parent
        child_cpd.parents = [parent]
        child_cpd.variable = child
        
        # Create model
        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        
        # Verify child CPD has correct parent reference after model creation
        child_cpd_from_model = model.get_module_of_concept('child')
        self.assertEqual(len(child_cpd_from_model.parents), 1,
                        "Child CPD should still have 1 parent after model creation")


if __name__ == '__main__':
    unittest.main()
