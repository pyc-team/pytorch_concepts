import unittest
import torch

from torch_concepts.nn.semantics import GodelTNorm
from torch_concepts.nn.reasoning import MLPReasoner, ResidualMLPReasoner, DeepConceptReasoner


class TestMLPReasoner(unittest.TestCase):

    def setUp(self):
        self.n_concepts = 10
        self.n_classes = 5
        self.emb_size = 8
        self.n_layers = 3
        self.batch_size = 2

        self.model = MLPReasoner(self.n_concepts, self.n_classes, self.emb_size, self.n_layers)
        self.x = {
            'c_pred': torch.randn(self.batch_size, self.n_concepts),
            'c_int': torch.randn(self.batch_size, self.n_concepts)
        }

    def test_forward_shape(self):
        output = self.model(self.x)['y_pred']
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))

        output_with_c_int = self.model(self.x, reason_with='c_int')['y_pred']
        self.assertEqual(output_with_c_int.shape, (self.batch_size, self.n_classes))

    def test_no_interventions(self):
        output = self.model(self.x)['y_pred']
        expected_output = self.model.reasoner(self.x['c_pred'])
        self.assertTrue(torch.equal(output, expected_output))

    def test_with_interventions(self):
        output = self.model(self.x, reason_with='c_int')['y_pred']
        expected_output = self.model.reasoner(self.x['c_int'])
        self.assertTrue(torch.equal(output, expected_output))

    def test_linear_layer_configuration(self):
        model = MLPReasoner(self.n_concepts, self.n_classes, self.emb_size, 0)
        layers = [layer for layer in model.reasoner]
        self.assertTrue(isinstance(layers[0], torch.nn.Linear))
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0].in_features, self.n_concepts)
        self.assertEqual(layers[0].out_features, self.n_classes)


class TestResidualMLPReasoner(unittest.TestCase):

    def setUp(self):
        self.n_concepts = 10
        self.n_classes = 5
        self.emb_size = 8
        self.n_layers = 3
        self.n_residuals = 2
        self.batch_size = 2

        self.model = ResidualMLPReasoner(self.n_concepts, self.n_classes, self.emb_size, self.n_layers, self.n_residuals)
        self.x = {
            'c_pred': torch.randn(self.batch_size, self.n_concepts),
            'c_int': torch.randn(self.batch_size, self.n_concepts),
            'residuals': torch.randn(self.batch_size, self.n_residuals)
        }

    def test_forward_shape(self):
        output = self.model(self.x)['y_pred']
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))

        output_with_c_int = self.model(self.x, reason_with='c_int')['y_pred']
        self.assertEqual(output_with_c_int.shape, (self.batch_size, self.n_classes))

    def test_with_interventions(self):
        output = self.model(self.x, reason_with='c_int')['y_pred']
        expected_output = self.model.reasoner(torch.cat([self.x['c_int'], self.x['residuals']], dim=1))
        self.assertTrue(torch.equal(output, expected_output))

    def test_linear_layer_configuration(self):
        model = MLPReasoner(self.n_concepts, self.n_classes, self.emb_size, 0)
        layers = [layer for layer in model.reasoner]
        self.assertTrue(isinstance(layers[0], torch.nn.Linear))
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0].in_features, self.n_concepts)
        self.assertEqual(layers[0].out_features, self.n_classes)


class TestDeepConceptReasoner(unittest.TestCase):
    def setUp(self):
        self.n_concepts = 5
        self.n_classes = 3
        self.emb_size = 10
        self.logic = GodelTNorm()
        self.temperature = 1.0
        self.set_level_rules = False
        self.batch_size = 19
        self.n_residuals = 7
        self.model = DeepConceptReasoner(
            n_concepts=self.n_concepts,
            n_classes=self.n_classes,
            emb_size=self.emb_size,
            n_residuals=self.n_residuals,
            logic=self.logic,
            temperature=self.temperature,
            set_level_rules=self.set_level_rules
        )
        self.x = {
            'c_emb': torch.randn(self.batch_size, self.n_concepts, self.emb_size),
            'c_pred': torch.randn(self.batch_size, self.n_concepts),
            'c_int': torch.randn(self.batch_size, self.n_concepts),
            'residuals': torch.randn(self.batch_size, self.n_residuals)
        }

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
        preds = self.model(self.x)['y_pred']
        self.assertEqual(preds.shape, (self.batch_size, self.n_classes))

    def test_forward_with_c_emb_shape(self):
        model = DeepConceptReasoner(
            n_concepts=self.n_concepts,
            n_classes=self.n_classes,
            emb_size=self.emb_size,
            n_residuals=0,
            logic=self.logic,
            temperature=self.temperature,
            set_level_rules=self.set_level_rules
        )
        preds = model(self.x)['y_pred']
        self.assertEqual(preds.shape, (self.batch_size, self.n_classes))

    def test_forward_with_attention_output_shape(self):
        preds = self.model(self.x)
        y_preds = preds['y_pred']
        sign_attn = preds['sign_attn']
        filter_attn = preds['filter_attn']
        self.assertEqual(y_preds.shape, (self.batch_size, self.n_classes))
        self.assertEqual(sign_attn.shape, (self.batch_size, self.n_concepts, self.n_classes))
        self.assertEqual(filter_attn.shape, (self.batch_size, self.n_concepts, self.n_classes))

    def test_forward_with_custom_attention(self):
        sign_attn = torch.sigmoid(torch.randn(self.batch_size, self.n_concepts, self.n_classes))
        filter_attn = torch.sigmoid(torch.randn(self.batch_size, self.n_concepts, self.n_classes))
        preds = self.model(self.x, sign_attn=sign_attn, filter_attn=filter_attn)['y_pred']
        self.assertEqual(preds.shape, (self.batch_size, self.n_classes))

    def test_explain_local_mode(self):
        explanations = self.model.explain(self.x, mode='local')
        self.assertTrue(isinstance(explanations, list))
        self.assertTrue(all('explanation' in exp for exp in explanations))

    def test_explain_global_mode(self):
        explanations = self.model.explain(self.x, mode='global')
        self.assertTrue(isinstance(explanations, list))
        self.assertTrue(all('explanation' in exp for exp in explanations))

    def test_explain_exact_mode(self):
        explanations = self.model.explain(self.x, mode='exact')
        self.assertTrue(isinstance(explanations, list))
        self.assertTrue(all('explanation' in exp for exp in explanations))

    def test_explain_invalid_mode(self):
        with self.assertRaises(AssertionError):
            self.model.explain(self.x, mode='invalid_mode')


if __name__ == '__main__':
    unittest.main()
