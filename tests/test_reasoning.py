import unittest
import torch

from torch_concepts.nn.semantics import GodelTNorm
from torch_concepts.nn.reasoning import DeepConceptReasoner


class TestDeepConceptReasoner(unittest.TestCase):
    def setUp(self):
        self.n_concepts = 5
        self.n_classes = 3
        self.emb_size = 10
        self.logic = GodelTNorm()
        self.temperature = 1.0
        self.set_level_rules = False
        self.model = DeepConceptReasoner(
            n_concepts=self.n_concepts,
            n_classes=self.n_classes,
            emb_size=self.emb_size,
            logic=self.logic,
            temperature=self.temperature,
            set_level_rules=self.set_level_rules
        )

    def test_initialization(self):
        self.assertEqual(self.model.n_concepts, self.n_concepts)
        self.assertEqual(self.model.n_classes, self.n_classes)
        self.assertEqual(self.model.emb_size, self.emb_size)
        self.assertIs(self.model.logic, self.logic)
        self.assertEqual(self.model.temperature, self.temperature)
        self.assertEqual(self.model.set_level_rules, self.set_level_rules)

    def test_softselect(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        temperature = 1.0
        result = self.model._softselect(values, temperature)
        self.assertEqual(result.shape, values.shape)

    def test_forward_output_shape(self):
        x = torch.randn(2, self.emb_size)
        c = torch.randn(2, self.n_concepts)
        preds = self.model(x, c)
        self.assertEqual(preds.shape, (2, self.n_classes))

    def test_forward_with_attention_output_shape(self):
        x = torch.randn(2, self.emb_size)
        c = torch.randn(2, self.n_concepts)
        preds, sign_attn, filter_attn = self.model(x, c, return_attn=True)
        self.assertEqual(preds.shape, (2, self.n_classes))
        self.assertEqual(sign_attn.shape, (2, self.n_concepts, self.n_classes))
        self.assertEqual(filter_attn.shape, (2, self.n_concepts, self.n_classes))

    def test_forward_with_custom_attention(self):
        x = torch.randn(2, self.emb_size)
        c = torch.randn(2, self.n_concepts)
        sign_attn = torch.sigmoid(torch.randn(2, self.n_concepts, self.n_classes))
        filter_attn = torch.sigmoid(torch.randn(2, self.n_concepts, self.n_classes))
        preds = self.model(x, c, sign_attn=sign_attn, filter_attn=filter_attn)
        self.assertEqual(preds.shape, (2, self.n_classes))

    def test_explain_local_mode(self):
        x = torch.randn(2, self.emb_size)
        c = torch.randn(2, self.n_concepts)
        explanations = self.model.explain(x, c, mode='local')
        self.assertTrue(isinstance(explanations, list))
        self.assertTrue(all('explanation' in exp for exp in explanations))

    def test_explain_global_mode(self):
        x = torch.randn(2, self.emb_size)
        c = torch.randn(2, self.n_concepts)
        explanations = self.model.explain(x, c, mode='global')
        self.assertTrue(isinstance(explanations, list))
        self.assertTrue(all('explanation' in exp for exp in explanations))

    def test_explain_exact_mode(self):
        x = torch.randn(2, self.emb_size)
        c = torch.randn(2, self.n_concepts)
        explanations = self.model.explain(x, c, mode='exact')
        self.assertTrue(isinstance(explanations, list))
        self.assertTrue(all('explanation' in exp for exp in explanations))

    def test_explain_invalid_mode(self):
        x = torch.randn(2, self.emb_size)
        c = torch.randn(2, self.n_concepts)
        with self.assertRaises(AssertionError):
            self.model.explain(x, c, mode='invalid_mode')

if __name__ == '__main__':
    unittest.main()
