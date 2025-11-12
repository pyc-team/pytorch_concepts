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


if __name__ == '__main__':
    unittest.main()
