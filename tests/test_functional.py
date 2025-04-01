import unittest
import torch
import torch_concepts.nn.functional as CF


class TestConceptFunctions(unittest.TestCase):

    def setUp(self):
        self.c_pred = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        self.c_true = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
        self.indexes = torch.tensor([[True, False], [False, True]])
        self.c_confidence = torch.tensor([[0.8, 0.1, 0.6],
                                          [0.9, 0.2, 0.4],
                                          [0.7, 0.3, 0.5]])
        self.target_confidence = 0.5

    def test_intervene(self):
        result = CF.intervene(self.c_pred, self.c_true, self.indexes)
        expected = torch.tensor([[0.9, 0.2], [0.3, 0.6]])
        self.assertTrue(torch.equal(result, expected),
                        f"Expected {expected}, but got {result}")

    def test_concept_embedding_mixture(self):
        c_emb = torch.randn(5, 4, 6)
        c_scores = torch.randint(0, 2, (5, 4))
        result = CF.concept_embedding_mixture(c_emb, c_scores)
        self.assertTrue(result.shape == (5, 4, 3),
                        f"Expected shape (5, 4, 3), but got {result.shape}")

    def test_intervene_on_concept_graph(self):
        # Create a AnnotatedTensor adjacency matrix
        c_adj = torch.tensor([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]], dtype=torch.float)

        # Intervene by zeroing out specific columns
        intervened_c_adj = CF.intervene_on_concept_graph(c_adj, [1])
        # Verify the shape of the output
        self.assertEqual(intervened_c_adj.shape, c_adj.shape)
        # Verify that the specified columns are zeroed out
        expected_data = torch.tensor([[0, 0, 0],
                                      [1, 0, 1],
                                      [0, 0, 0]], dtype=torch.float)
        self.assertTrue(torch.equal(intervened_c_adj, expected_data))

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

    def test_linear_eq_explanations(self):
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
        y_pred = CF.linear_equation_eval(c_imp, c_pred, y_bias)[:, 0]

        concept_names = ['C1', 'C2']
        class_names = ['Y1']

        expected_result = [{'Y1': {'Equation 0': '10.0 * C2'}},
                           {'Y1': {'Equation 0': '-10.0 * C2'}},
                           {'Y1': {'Equation 0': '-10.0 * C2'}},
                           {'Y1': {'Equation 0': ''}},
                           {'Y1': {'Equation 0': '1.0 * bias'}},
                           ]
        result = CF.linear_eq_explanations(c_imp, y_bias, {1: concept_names,
                                                           2: class_names})
        # print(result)
        self.assertEqual(result, expected_result)

        # test global explanation
        from torch_concepts.utils import get_most_common_expl
        global_explanations = get_most_common_expl(result, y_pred)

        expected_global_expl = {
            'Y1': {'10.0 * C2': 1, '-10.0 * C2': 1, '1.0 * bias': 1}
        }
        # print(global_explanations)
        self.assertEqual(global_explanations, expected_global_expl)

    def test_rule_eval(self):
        # here we test the logic_rule_eval function on the classic XOR case
        # we evaluate 5 examples for which concept weights should predict pos.
        # and 4 examples for which the concept weights should predict neg.

        c_pred = torch.tensor([
            [0., 0.],
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
        ])
        # batch_size, memory_size, n_concepts, n_classes, n_roles
        # concept roles pos_polarity, neg_polarity, irrelevance
        c_weights = torch.tensor([
            # both irrelevant
            [[[[0., 0., 1.]],
              [[0., 0., 1.]]]],
            # both neg. imp.
            [[[[0., 1., 0.]],
              [[0., 1., 0.]]]],
            # neg. imp., pos. imp.
            [[[[0., 1., 0.]],
              [[1., 0., 0.]]]],
            # pos. imp., neg. imp.
            [[[[1., 0., 0.]],
              [[0., 1., 0.]]]],
            # both pos. imp.
            [[[[1., 0., 0.]],
              [[1., 0., 0.]]]],
            # both pos. imp.
            [[[[1., 0, 0]],
              [[1., 0, 0]]]],
            # pos. imp., neg. imp.
            [[[[1., 0., 0.]],
              [[0., 1., 0.]]]],
            # neg. imp., pos. imp.
            [[[[0., 1., 0.]],
              [[1., 0., 0.]]]],
            # both neg. imp.
            [[[[0., 1., 0.]],
              [[0., 1., 0.]]]],
        ])

        expected_result = torch.tensor([
            [[True]],
            [[True]],
            [[True]],
            [[True]],
            [[True]],
            [[False]],
            [[False]],
            [[False]],
            [[False]],
        ])

        result = CF.logic_rule_eval(c_weights, c_pred)
        # print(result)
        self.assertEqual(torch.all((result > 0) == expected_result).item(),
                         True)

    def test_rule_explanations(self):
        # check standard XOR predictions and rule extraction
        # batch_size, memory_size, n_concepts, n_classes, n_roles
        c_weights = torch.tensor([
            # neg. imp., pos. imp. for XOR, both neg. imp. for XNOR
            [[[[0., 1., 0.],
               [0., 1., 0.]],
              [[1., 0., 0.],
               [0., 1., 0.]]]],
            # neg. imp., pos. imp. for XOR, both neg. imp. for XNOR
            [[[[0., 1., 0.],
               [0., 1., 0.]],
              [[1., 0., 0.],
               [0., 1., 0.]]]],
            # pos. imp., neg. imp. for XOR, both pos. imp. for XNOR
            [[[[1., 0., 0.],
               [1., 0., 0.]],
              [[0., 1., 0.],
               [1., 0., 0.]]]],
            # pos. imp., neg. imp. for XOR, both pos. imp. for XNOR
            [[[[1., 0., 0.],
               [1., 0., 0.]],
              [[0., 1., 0.],
               [1., 0., 0.]]]],
        ])

        conc_names = ['C1', 'C2']
        cls_names = ['XOR', 'XNOR']

        expected_result = [{'XNOR': {'Rule 0': '~ C1 & ~ C2'},
                            'XOR': {'Rule 0': '~ C1 & C2'}},
                           {'XNOR': {'Rule 0': '~ C1 & ~ C2'},
                            'XOR': {'Rule 0': '~ C1 & C2'}},
                           {'XNOR': {'Rule 0': 'C1 & C2'},
                            'XOR': {'Rule 0': 'C1 & ~ C2'}},
                           {'XNOR': {'Rule 0': 'C1 & C2'},
                            'XOR': {'Rule 0': 'C1 & ~ C2'}}]

        result = CF.logic_rule_explanations(c_weights,
                                            {1: conc_names, 2: cls_names})
        self.assertEqual(result, expected_result)

        # test global explanation
        from torch_concepts.utils import get_most_common_expl
        y_pred = torch.tensor([
            [0., 1.],
            [1., 0.],
            [1., 0.],
            [0., 1.],
        ])
        global_explanations = get_most_common_expl(result, y_pred)
        # print(global_explanations)

        expected_global_expl = {
            'XOR': {'~ C1 & C2': 1, 'C1 & ~ C2': 1},
            'XNOR': {'~ C1 & ~ C2': 1, 'C1 & C2': 1}
        }
        self.assertEqual(global_explanations, expected_global_expl)

    def test_semantics(self):
        from torch_concepts.semantic import (ProductTNorm, GodelTNorm,
                                             CMRSemantic)
        semantics = [ProductTNorm(), GodelTNorm(), CMRSemantic()]

        true_t = torch.tensor([1])
        false_t = torch.tensor([0])

        for semantic in semantics:
            # test the conjunction
            self.assertEqual(semantic.conj(false_t, false_t), false_t)
            self.assertEqual(semantic.conj(false_t, true_t), false_t)
            self.assertEqual(semantic.conj(true_t, false_t), false_t)
            self.assertEqual(semantic.conj(true_t, true_t), true_t)

            # test the disjunction
            self.assertEqual(semantic.disj(false_t, false_t), false_t)
            self.assertEqual(semantic.disj(false_t, true_t), true_t)
            self.assertEqual(semantic.disj(true_t, false_t), true_t)
            # this can never happen in CMR
            if not isinstance(semantic, CF.CMRSemantic):
                self.assertEqual(semantic.disj(true_t, true_t), true_t)

            # test the double implication
            self.assertEqual(semantic.iff(false_t, false_t), true_t)
            self.assertEqual(semantic.iff(false_t, true_t), false_t)
            self.assertEqual(semantic.iff(true_t, false_t), false_t)
            self.assertEqual(semantic.iff(true_t, true_t), true_t)

            # test the negation
            self.assertEqual(semantic.neg(true_t), false_t)
            self.assertEqual(semantic.neg(false_t), true_t)


if __name__ == '__main__':
    unittest.main()
