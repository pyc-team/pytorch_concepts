import torch


class ConceptTensor(torch.Tensor):
    """
    ConceptTensor is a subclass of torch.Tensor which ensures that the tensor has at least two dimensions: batch size and number of concepts.
    """

    def __new__(cls, *args, **kwargs):
        # Ensure correct shape before creating the tensor
        if len(args) < 2:
            raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")

        # Create a new instance of ConceptTensor
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # Initialize the ConceptTensor (optional)
        super().__init__()
        self._check_shape()

    def _check_shape(self):
        if len(self.shape) < 2:
            raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")

    def describe(self):
        # Example of an additional method specific to ConceptTensor
        return f"ConceptTensor of shape {self.shape}, dtype {self.dtype}"

    @classmethod
    def concept(cls, tensor):
        # Ensure the tensor has the correct shape
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor.")
        if len(tensor.shape) < 2:
            raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")

        # Convert the existing tensor to ConceptTensor
        return tensor.as_subclass(cls)

    # TODO: check why the following fail
    # def reshape(self, *shape):
    #     # Ensure the reshaped tensor maintains the first two dimensions correctly
    #     if len(shape) < 2:
    #         raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")
    #
    #     return super().reshape(*shape).as_subclass(ConceptTensor)
    #
    # def view(self, *shape):
    #     # Ensure the viewed tensor maintains the first two dimensions correctly
    #     if len(shape) < 2:
    #         raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")
    #
    #     return super().view(*shape).as_subclass(ConceptTensor)
