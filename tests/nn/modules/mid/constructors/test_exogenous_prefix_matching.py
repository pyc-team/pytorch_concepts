"""
Tests for exogenous variable prefix matching bug fix.

This test module verifies that exogenous variables are correctly matched to their
corresponding concepts using exact prefix matching, avoiding substring matching bugs.

Bug context: Previously, using substring matching like `"OtherCar" in "exog_OtherCarCost_state_0"`
would incorrectly match, causing concepts to receive exogenous variables from other concepts
with similar names.

Fix: Use exact prefix matching with `startswith(f"exog_{label_name}_state_")` to ensure
concepts only receive their own exogenous variables.
"""
import unittest
import torch
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.nn import BipartiteModel, LinearCC
from torch_concepts.nn import LazyConstructor
from torch_concepts.nn.modules.low.encoders.exogenous import LinearZU
from torch.distributions import Bernoulli, Categorical


class TestExogenousPrefixMatching(unittest.TestCase):
    """Test exact prefix matching for exogenous variables."""

    def test_substring_overlap_concepts(self):
        """Test concepts with substring overlap don't cross-assign exogenous variables.
        
        This is the core bug fix test: concepts like 'Car' and 'CarCost' should not
        have their exogenous variables mixed up due to substring matching.
        """
        # Create concepts where one name is a substring of another
        concept_names = ['Car', 'CarCost', 'Driver', 'Task']
        
        # Create annotations with different cardinalities to make exogenous counts distinct
        metadata = {
            'Car': {'distribution': Categorical, 'type': 'discrete'},
            'CarCost': {'distribution': Categorical, 'type': 'discrete'},
            'Driver': {'distribution': Categorical, 'type': 'discrete'},
            'Task': {'distribution': Bernoulli, 'type': 'discrete'}
        }
        cardinalities = (2, 4, 3, 1)
        
        annotations = Annotations({
            1: AxisAnnotation(
                labels=tuple(concept_names),
                cardinalities=cardinalities,
                metadata=metadata
            )
        })
        
        # Create bipartite model with source_exogenous
        model = BipartiteModel(
            task_names=['Task'],
            input_size=100,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC),
            source_exogenous=LazyConstructor(LinearZU, exogenous_size=16),
            use_source_exogenous=True
        )
        
        # Check that variables were created with correct parent counts
        car_vars = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'Car']
        carcost_vars = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'CarCost']
        driver_vars = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'Driver']
        
        self.assertEqual(len(car_vars), 1)
        self.assertEqual(len(carcost_vars), 1)
        self.assertEqual(len(driver_vars), 1)
        
        car_var = car_vars[0]
        carcost_var = carcost_vars[0]
        driver_var = driver_vars[0]
        
        # Check that each concept has the correct number of parent variables
        # With source_exogenous, each concept should have exogenous variables matching its cardinality
        self.assertEqual(len(car_var.parents), 2,
                        f"Car should have 2 exogenous parent variables, got {len(car_var.parents)}")
        self.assertEqual(len(carcost_var.parents), 4,
                        f"CarCost should have 4 exogenous parent variables, got {len(carcost_var.parents)}")
        self.assertEqual(len(driver_var.parents), 3,
                        f"Driver should have 3 exogenous parent variables, got {len(driver_var.parents)}")
        
        # Verify parent names start with correct prefix (not substrings of other concepts)
        car_parent_names = [p if isinstance(p, str) else p.concepts[0] for p in car_var.parents]
        for name in car_parent_names:
            self.assertTrue(name.startswith('exog_Car_state_'),
                          f"Car parent {name} should start with 'exog_Car_state_'")
            self.assertFalse(name.startswith('exog_CarCost_state_'),
                           f"Car should not have CarCost exogenous variable: {name}")
        
        carcost_parent_names = [p if isinstance(p, str) else p.concepts[0] for p in carcost_var.parents]
        for name in carcost_parent_names:
            self.assertTrue(name.startswith('exog_CarCost_state_'),
                          f"CarCost parent {name} should start with 'exog_CarCost_state_'")

    def test_exact_prefix_matching_with_similar_names(self):
        """Test exact prefix matching with highly similar concept names.
        
        Tests edge cases like 'A', 'AB', 'ABC' to ensure no cross-contamination.
        """
        concept_names = ['A', 'AB', 'ABC', 'Task']
        
        metadata = {
            'A': {'distribution': Categorical, 'type': 'discrete'},
            'AB': {'distribution': Categorical, 'type': 'discrete'},
            'ABC': {'distribution': Categorical, 'type': 'discrete'},
            'Task': {'distribution': Bernoulli, 'type': 'discrete'}
        }
        cardinalities = (2, 3, 4, 1)
        
        annotations = Annotations({
            1: AxisAnnotation(
                labels=tuple(concept_names),
                cardinalities=cardinalities,
                metadata=metadata
            )
        })
        
        model = BipartiteModel(
            task_names=['Task'],
            input_size=50,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC),
            source_exogenous=LazyConstructor(LinearZU, exogenous_size=16),
            use_source_exogenous=True
        )
        
        # Check each concept has only its own exogenous variables
        a_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'A'][0]
        ab_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'AB'][0]
        abc_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'ABC'][0]
        
        self.assertEqual(len(a_var.parents), 2, "A should have 2 exogenous variables")
        self.assertEqual(len(ab_var.parents), 3, "AB should have 3 exogenous variables")
        self.assertEqual(len(abc_var.parents), 4, "ABC should have 4 exogenous variables")
        
        # Verify exact prefix matching - A should not get AB or ABC variables
        a_parent_names = [p if isinstance(p, str) else p.concepts[0] for p in a_var.parents]
        for name in a_parent_names:
            self.assertTrue(name.startswith('exog_A_state_'),
                          f"A parent should start with 'exog_A_state_', got {name}")
            # Make sure it's not 'exog_AB_state_' or 'exog_ABC_state_'
            self.assertFalse('exog_AB' in name or 'exog_ABC' in name,
                           f"A should not have AB/ABC exogenous: {name}")

    def test_underscore_in_concept_names(self):
        """Test that underscores in concept names don't cause matching issues.
        
        Ensures that the '_state_' suffix in exogenous variable names is correctly
        used as part of the matching logic.
        """
        concept_names = ['Age_Group', 'Age_Group_Risk', 'Task']
        
        metadata = {
            'Age_Group': {'distribution': Categorical, 'type': 'discrete'},
            'Age_Group_Risk': {'distribution': Categorical, 'type': 'discrete'},
            'Task': {'distribution': Bernoulli, 'type': 'discrete'}
        }
        cardinalities = (3, 5, 1)
        
        annotations = Annotations({
            1: AxisAnnotation(
                labels=tuple(concept_names),
                cardinalities=cardinalities,
                metadata=metadata
            )
        })
        
        model = BipartiteModel(
            task_names=['Task'],
            input_size=60,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC),
            source_exogenous=LazyConstructor(LinearZU, exogenous_size=16),
            use_source_exogenous=True
        )
        
        age_group_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'Age_Group'][0]
        age_group_risk_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'Age_Group_Risk'][0]
        
        self.assertEqual(len(age_group_var.parents), 3,
                        "Age_Group should have 3 exogenous variables")
        self.assertEqual(len(age_group_risk_var.parents), 5,
                        "Age_Group_Risk should have 5 exogenous variables")
        
        # Verify Age_Group doesn't get Age_Group_Risk's exogenous variables
        age_group_parent_names = [p if isinstance(p, str) else p.concepts[0] for p in age_group_var.parents]
        for name in age_group_parent_names:
            self.assertTrue(name.startswith('exog_Age_Group_state_'),
                          f"Age_Group parent should start with 'exog_Age_Group_state_', got {name}")
            self.assertFalse(name.startswith('exog_Age_Group_Risk_state_'),
                           f"Age_Group should not have Age_Group_Risk exogenous: {name}")

    def test_predictor_exogenous_filtering(self):
        """Test that predictor correctly filters exogenous variables for parent concepts.
        
        The predictor should only receive exogenous variables from its actual parents,
        not from concepts with similar names.
        """
        concept_names = ['Other', 'OtherCar', 'OtherCarCost', 'Task']
        
        metadata = {
            'Other': {'distribution': Categorical, 'type': 'discrete'},
            'OtherCar': {'distribution': Categorical, 'type': 'discrete'},
            'OtherCarCost': {'distribution': Categorical, 'type': 'discrete'},
            'Task': {'distribution': Categorical, 'type': 'discrete'}
        }
        cardinalities = (2, 3, 4, 2)
        
        annotations = Annotations({
            1: AxisAnnotation(
                labels=tuple(concept_names),
                cardinalities=cardinalities,
                metadata=metadata
            )
        })
        
        model = BipartiteModel(
            task_names=['Task'],
            input_size=70,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC),
            source_exogenous=LazyConstructor(LinearZU, exogenous_size=16),
            use_source_exogenous=True
        )
        
        # Check that root concepts have correct exogenous parents
        other_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'Other'][0]
        othercar_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'OtherCar'][0]
        othercarcost_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'OtherCarCost'][0]
        
        self.assertEqual(len(other_var.parents), 2,
                        "Other should have 2 exogenous variables")
        self.assertEqual(len(othercar_var.parents), 3,
                        "OtherCar should have 3 exogenous variables")
        self.assertEqual(len(othercarcost_var.parents), 4,
                        "OtherCarCost should have 4 exogenous variables")
        
        # Verify OtherCar doesn't get OtherCarCost's exogenous (the original bug!)
        othercar_parent_names = [p if isinstance(p, str) else p.concepts[0] for p in othercar_var.parents]
        for name in othercar_parent_names:
            self.assertTrue(name.startswith('exog_OtherCar_state_'),
                          f"OtherCar parent should start with 'exog_OtherCar_state_', got {name}")
            self.assertFalse(name.startswith('exog_OtherCarCost_state_'),
                           f"OtherCar should NOT have OtherCarCost exogenous: {name}")

    def test_no_exogenous_without_source_exogenous_flag(self):
        """Test that exogenous variables are not created when use_source_exogenous=False.
        
        This is a control test to ensure the exogenous feature is opt-in.
        """
        concept_names = ['Car', 'CarCost', 'Task']
        
        metadata = {
            'Car': {'distribution': Categorical, 'type': 'discrete'},
            'CarCost': {'distribution': Categorical, 'type': 'discrete'},
            'Task': {'distribution': Bernoulli, 'type': 'discrete'}
        }
        cardinalities = (2, 4, 1)
        
        annotations = Annotations({
            1: AxisAnnotation(
                labels=tuple(concept_names),
                cardinalities=cardinalities,
                metadata=metadata
            )
        })
        
        # use_source_exogenous=False (default)
        model = BipartiteModel(
            task_names=['Task'],
            input_size=80,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC),
            use_source_exogenous=False
        )
        
        # Encoders should not have exogenous parents when source_exogenous=False
        car_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'Car'][0]
        carcost_var = [v for v in model.probabilistic_model.variables if v.concepts[0] == 'CarCost'][0]

        # Without source exogenous, root concepts should only have 'input' as parent, no exogenous variables
        self.assertEqual(len(car_var.parents), 1,
                        "Car should have 1 parent (input) when use_source_exogenous=False")
        self.assertEqual(type(car_var.parents[0]).__name__, 'InputVariable',
                        "Car's only parent should be InputVariable")
        
        # Verify no exogenous variables exist
        exog_vars = [v for v in model.probabilistic_model.variables if hasattr(v, 'name') and v.name.startswith('exog_')]
        self.assertEqual(len(exog_vars), 0,
                        "No exogenous variables should exist when use_source_exogenous=False")
if __name__ == '__main__':
    unittest.main()
