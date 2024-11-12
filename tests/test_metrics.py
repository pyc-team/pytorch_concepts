import unittest
import torch
from sklearn.metrics import f1_score
from torch_concepts.metrics import completeness_score, intervention_score, cace_score


class ANDModel(torch.nn.Module):
    def __init__(self):
        super(ANDModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1, bias=True)

        # Manually set weights and bias to perform AND operation
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0]]))  # Both weights are 1
            self.linear.bias = torch.nn.Parameter(torch.tensor([-1.5]))  # Bias is -1.5

    def forward(self, x):
        return self.linear(x)


class TestCompletenessScore(unittest.TestCase):
    def test_completeness_score_accuracy(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_blackbox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_whitebox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertAlmostEqual(score, 1.0, places=2, msg="Completeness score with f1_score should be 1.0")

    def test_completeness_score_f1(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2])
        y_pred_blackbox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2])
        y_pred_whitebox = torch.tensor([0, 1, 2, 2, 1, 0, 2, 1, 1])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertAlmostEqual(score, 0.3, places=1, msg="Completeness score with f1_score should be 0.0")

    def test_completeness_score_higher_than_1(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_blackbox = torch.tensor([0, 1, 1, 1, 0, 2, 1, 2])
        y_pred_whitebox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertTrue(score > 1, msg="Completeness score should be higher than 1 when the whitebox model is better than the blackbox model")


class TestInterventionScore(unittest.TestCase):

    def test_intervention_score_basic(self):
        y_predictor = ANDModel()
        c_true = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        c_pred = torch.FloatTensor([[.8, .2], [.8, .8], [.8, .2], [.8, .8]])
        y_true = torch.tensor([0, 0, 0, 1])
        intervention_groups = [[], [0], [1]]

        scores = intervention_score(y_predictor, c_pred, c_true, y_true, intervention_groups, auc=False)
        self.assertTrue(isinstance(scores, list))
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[1], 1.0)

        auc_score = intervention_score(y_predictor, c_pred, c_true, y_true, intervention_groups, auc=True)
        self.assertTrue(isinstance(auc_score, float))
        self.assertEqual(round(auc_score*100)/100, 0.89)

class TestCaceScore(unittest.TestCase):
    def test_cace_score_basic(self):
        y_pred_c0 = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        y_pred_c1 = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
        expected_result = torch.tensor([0.15, 0.1, -0.25])
        result = cace_score(y_pred_c0, y_pred_c1)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_cace_score_zero_effect(self):
        y_pred_c0 = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        y_pred_c1 = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        expected_result = torch.tensor([0.0, 0.0, 0.0])
        result = cace_score(y_pred_c0, y_pred_c1)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_cace_score_negative_effect(self):
        y_pred_c0 = torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]])
        y_pred_c1 = torch.tensor([[0.1, 0.1, 0.8], [0.2, 0.1, 0.7]])
        expected_result = torch.tensor([-0.2, -0.25, 0.45])
        result = cace_score(y_pred_c0, y_pred_c1)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_cace_score_different_shapes(self):
        y_pred_c0 = torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]])
        y_pred_c1 = torch.tensor([[0.1, 0.1, 0.8]])
        with self.assertRaises(RuntimeError):
            cace_score(y_pred_c0, y_pred_c1)


if __name__ == '__main__':
    unittest.main()
