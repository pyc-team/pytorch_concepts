import torch


class InputImgEncoder(torch.nn.Module):
    """
    Initialize the input image encoder.

    Attributes:
        original_model: The original model to extract features from.
    """
    def __init__(self, original_model: torch.nn.Module):
        super(InputImgEncoder, self).__init__()
        self.features = torch.nn.Sequential(
            *list(original_model.children())[:-1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the input image encoder.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The output tensor from the last layer of the model.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
