from typing import List, Union

import torch


class ConceptTensor(torch.Tensor):
    """
    ConceptTensor is a subclass of torch.Tensor which ensures that the tensor has at least two dimensions: batch size and number of concepts.
    Additionally, it can store concept names.

    Attributes:
        data (torch.Tensor): Data tensor.
        concept_names (List[str]): Names of concepts.
    """
    def __new__(cls, data: torch.Tensor, concept_names: List[str] = None, *args, **kwargs) -> 'ConceptTensor':
        # Ensure correct shape before creating the tensor
        if len(data.shape) < 2:
            raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")

        # Create a new instance of ConceptTensor
        instance = super().__new__(cls, data, *args, **kwargs)
        instance.concept_names = cls._check_concept_names(data, concept_names)
        return instance

    @staticmethod
    def _check_concept_names(tensor: torch.Tensor, concept_names: List[str]) -> List[str]:
        if concept_names is not None:
            if len(concept_names) != tensor.shape[1]:
                raise ValueError("Number of concept names must match the number of concepts in the tensor.")
        else:
            concept_names = [f"concept_{i}" for i in range(tensor.shape[1])]
        return concept_names

    def _check_shape(self):
        if len(self.shape) < 2:
            raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")

    def describe(self) -> str:
        """
        Returns a string representation of the ConceptTensor.
        """
        return f"ConceptTensor of shape {self.shape}, dtype {self.dtype}, concepts {self.concept_names}"

    @classmethod
    def concept(cls, tensor: torch.Tensor, concept_names: List[str] = None) -> 'ConceptTensor':
        """
        Create a ConceptTensor from a torch.Tensor.

        Attributes:
            tensor: Input tensor.
            concept_names: Names of concepts.

        Returns:
            ConceptTensor: ConceptTensor instance.
        """
        # Ensure the tensor has the correct shape
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor.")
        if len(tensor.shape) < 2:
            raise ValueError("ConceptTensor must have at least two dimensions: batch size and number of concepts.")

        # Convert the existing tensor to ConceptTensor
        instance = tensor.as_subclass(cls)
        instance.concept_names = cls._check_concept_names(tensor, concept_names)
        return instance

    def assign_concept_names(self, concept_names: List[str]):
        """
        Assign new concept names to the ConceptTensor.

        Attributes:
            concept_names:
        """
        self.concept_names = self._check_concept_names(self, concept_names)

    def extract_by_concept_names(self, target_concepts: List[Union[int, str]]) -> 'ConceptTensor':
        """
        Extract a subset of concepts from the ConceptTensor.

        Attributes:
            target_concepts: List of concept names or indices to extract.

        Returns:
            ConceptTensor: Extracted ConceptTensor.
        """
        if self.concept_names is None:
            raise ValueError("Concept names are not set for this ConceptTensor.")

        indices = [idx for idx, name in enumerate(self.concept_names) if name in target_concepts]
        if isinstance(target_concepts[0], str):
            indices = [self.concept_names.index(name) for name in target_concepts if name in self.concept_names]
            if len(indices) != len(target_concepts):
                raise ValueError("Some concept names are not found in the tensor's concept names.")

        extracted_data = self[:, indices]
        return ConceptTensor(extracted_data, concept_names=target_concepts)

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
