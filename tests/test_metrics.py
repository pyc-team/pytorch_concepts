import unittest
import torch
from sklearn.metrics import f1_score
from torch_concepts.metrics import completeness_score


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
        self.assertAlmostEqual(score, 0.0, places=1, msg="Completeness score with f1_score should be 0.0")

    def test_completeness_score_higher_than_1(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_blackbox = torch.tensor([0, 1, 1, 1, 0, 2, 1, 2])
        y_pred_whitebox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertTrue(score > 1, msg="Completeness score should be higher than 1 when the whitebox model is better than the blackbox model")

    def test_completeness_score_negative(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2])
        y_pred_blackbox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2])
        y_pred_whitebox = torch.tensor([0, 1, 0, 2, 1, 0, 2, 1, 1])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertTrue(score < 0, msg="Completeness score should be negative when the model accuracy is worse than random guessing")


if __name__ == '__main__':
    unittest.main()
