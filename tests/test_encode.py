import unittest
import torch
import torchvision.models as models

from torch_concepts.nn import InputImgEncoder


class TestInputImgEncoder(unittest.TestCase):

    def setUp(self):
        self.original_model = models.resnet18(pretrained=False)
        self.encoder = InputImgEncoder(self.original_model)
        self.input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image

    def test_forward_output_shape(self):
        output = self.encoder(self.input_tensor)
        self.assertEqual(output.shape, (1, 512), "The output shape is incorrect.")  # ResNet18's final conv layer output

    def test_forward_output_type(self):
        output = self.encoder(self.input_tensor)
        self.assertIsInstance(output, torch.Tensor, "The output is not a torch.Tensor.")

    def test_forward_consistency(self):
        # Ensure that the encoder output is consistent with the original model up to the last conv layer
        with torch.no_grad():
            original_features = self.original_model.avgpool(self.original_model.layer4(self.original_model.layer3(
                self.original_model.layer2(self.original_model.layer1(self.original_model.maxpool(
                    self.original_model.relu(
                        self.original_model.bn1(self.original_model.conv1(self.input_tensor)))))))))
            original_features = torch.flatten(original_features, 1)

        encoder_output = self.encoder(self.input_tensor)
        self.assertTrue(torch.allclose(original_features, encoder_output, atol=1e-6),
                        "The encoder output is not consistent with the original model up to the last conv layer.")


if __name__ == '__main__':
    unittest.main()
