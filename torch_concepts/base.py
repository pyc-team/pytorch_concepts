import torch

from typing import List, Dict, Union


class ConceptTensor(torch.Tensor):
    """
    ConceptTensor is a subclass of torch.Tensor which ensures that the tensor
    has at least two dimensions: batch size and number of concepts.
    Additionally, it can store concept names.

    Attributes:
        data (torch.Tensor): Data tensor.
        concept_names (Dict[int, List[str]]): Names of concepts for each
            dimension.
    """
    def __new__(
        cls,
        data: torch.Tensor,
        concept_names: Dict[int, List[str]] = None,
        *args,
        **kwargs,
    ) -> 'ConceptTensor':
        instance = super().__new__(cls, data, *args, **kwargs)
        instance.concept_names = cls._check_concept_names(data, concept_names)
        return instance

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Perform the torch function as usual
        result = super().__torch_function__(func, types, args, kwargs)

        # Convert the result to a standard torch.Tensor if it's a ConceptTensor
        if isinstance(result, ConceptTensor):
            return result.to_standard_tensor()

        return result

    @staticmethod
    def _generate_default_concept_names(shape):
        concept_names = {}
        for dim in range(len(shape)):
            concept_names[dim] = [
                f"concept_{dim}_{i}" for i in range(shape[dim])
            ]
        return concept_names

    @staticmethod
    def _check_concept_names(
        tensor: torch.Tensor,
        concept_names: Dict[int, List[str]],
    ) -> Dict[int, List[str]]:
        # Initialize concept_names if None
        if concept_names is None:
            concept_names = {}

        # Ensure concept_names is a dictionary
        if not isinstance(concept_names, dict):
            raise ValueError(
                "concept_names must be a dictionary with dimension indices as "
                "keys and lists of names as values."
            )

        # Check that the concept dictionary IDs are within the range of the
        # tensor's dimensions
        for dim in concept_names.keys():
            if dim < 0 or dim >= len(tensor.shape):
                raise ValueError(
                    f"Dimension {dim} is out of range for the tensor with "
                    f"shape {tensor.shape}."
                )

        # Create default concept names for each dimension if not provided
        for dim in range(len(tensor.shape)):
            if dim not in concept_names:
                concept_names[dim] = [
                    f"concept_{dim}_{i}" for i in range(tensor.shape[dim])
                ]
            elif len(concept_names[dim]) != tensor.shape[dim]:
                raise ValueError(
                    f"Number of concept names for dimension {dim} must match "
                    f"the size of that dimension in the tensor."
                )

        return concept_names

    def __str__(self):
        """
        Returns a string representation of the ConceptTensor.
        """
        return (
            f"ConceptTensor of shape {self.shape}, dtype {self.dtype}, "
            f"concepts {self.concept_names}"
        )

    @classmethod
    def concept(
        cls,
        tensor: torch.Tensor,
        concept_names: Dict[int, List[str]] = None,
    ) -> 'ConceptTensor':
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
            raise ValueError(
                "ConceptTensor must have at least two dimensions: batch size "
                "and number of concepts."
            )

        # Convert the existing tensor to ConceptTensor
        instance = tensor.as_subclass(cls)
        instance.concept_names = cls._check_concept_names(tensor, concept_names)
        return instance

    def assign_concept_names(self, concept_names: Dict[int, List[str]]):
        """
        Assign new concept names to the ConceptTensor.

        Attributes:
            concept_names: Dictionary of concept names.
        """
        self.concept_names = self._check_concept_names(self, concept_names)

    def update_concept_names(self, new_concept_names: Dict[int, List[str]]):
        """
        Update the concept names for specified dimensions.

        Attributes:
            new_concept_names: Dictionary with dimension indices as keys and
                lists of new concept names as values.
        """
        for dim, names in new_concept_names.items():
            if dim not in self.concept_names:
                raise ValueError(
                    f"Dimension {dim} is not present in the current concept "
                    f"names."
                )
            if len(names) != len(self.concept_names[dim]):
                raise ValueError(
                    f"Number of new concept names for dimension {dim} must "
                    f"match the size of that dimension."
                )
            self.concept_names[dim] = names

    def extract_by_concept_names(
        self,
        target_concepts: Dict[int, List[Union[int, str]]],
    ) -> 'ConceptTensor':
        """
        Extract a subset of concepts from the ConceptTensor.

        Attributes:
            target_concepts: Dictionary where keys are dimensions and values are
                lists of concept names or indices to extract.

        Returns:
            ConceptTensor: Extracted ConceptTensor.
        """
        if self.concept_names is None:
            raise ValueError(
                "Concept names are not set for this ConceptTensor."
            )

        extracted_data = self
        new_concept_names = self.concept_names.copy()

        for dim, concepts in target_concepts.items():
            if dim not in self.concept_names:
                raise ValueError(
                    f"Concept names are not set for dimension {dim}."
                )

            if isinstance(concepts[0], str):
                indices = [
                    self.concept_names[dim].index(name) for name in concepts
                    if name in self.concept_names[dim]
                ]
                if len(indices) != len(concepts):
                    raise ValueError(
                        "Some concept names are not found in the tensor's "
                        "concept names."
                    )
            else:
                indices = concepts

            extracted_data = extracted_data.index_select(
                dim,
                torch.tensor(indices, device=self.device),
            )
            new_concept_names[dim] = [
                self.concept_names[dim][i] for i in indices
            ]

        return ConceptTensor(extracted_data, concept_names=new_concept_names)

    def new_empty(self, *shape):
        """
        Create a new empty ConceptTensor with the same concept names and given
        shape.

        Attributes:
            shape: Shape of the new tensor.

        Returns:
            ConceptTensor: A new empty ConceptTensor.
        """
        # Create a new empty tensor with the specified shape
        new_tensor = super().new_empty(*shape, device=self.device)

        # Ensure concept names are set correctly
        new_concept_names = {
            dim: self.concept_names[dim][:shape[dim]]
            for dim in range(len(shape))
        }

        return ConceptTensor(new_tensor, concept_names=new_concept_names)

    def to_standard_tensor(self) -> torch.Tensor:
        """
        Convert the ConceptTensor to a standard torch.Tensor while preserving
        gradients.

        Returns:
            torch.Tensor: Standard tensor with gradients.
        """
        return self.as_subclass(torch.Tensor)

    def _update_concept_names_for_new_shape(self, new_shape):
        """
        Update the concept names dictionary for the new tensor shape.
        """
        new_concept_names = {}
        for dim, size in enumerate(new_shape):
            if dim in self.concept_names and (
                len(self.concept_names[dim]) == size
            ):
                new_concept_names[dim] = self.concept_names[dim]
            else:
                new_concept_names[dim] = [
                    f"concept_{dim}_{i}" for i in range(size)
                ]
        return new_concept_names

    def view(self, *shape):
        """
        View the tensor with a new shape and update concept names accordingly.
        """
        new_tensor = super().view(*shape)
        new_tensor = new_tensor.as_subclass(ConceptTensor)
        if hasattr(self, 'concept_names'):
            new_tensor.concept_names = self._update_concept_names_for_new_shape(
                new_tensor.shape
            )
        return new_tensor

    def reshape(self, *shape):
        """
        Reshape the tensor to the specified shape and update concept names
        accordingly.
        """
        new_tensor = super().reshape(*shape)
        new_tensor = new_tensor.as_subclass(ConceptTensor)
        if hasattr(self, 'concept_names'):
            new_tensor.concept_names = self._update_concept_names_for_new_shape(
                new_tensor.shape
            )
        return new_tensor

    def transpose(self, dim0, dim1):
        """
        Transpose two dimensions of the tensor and update concept names
        accordingly.
        """
        new_tensor = super().transpose(dim0, dim1)
        new_concept_names = self.concept_names.copy()

        # Swap the concept names for the transposed dimensions
        if dim0 in self.concept_names and dim1 in self.concept_names:
            new_concept_names[dim0], new_concept_names[dim1] = (
                self.concept_names[dim1],
                self.concept_names[dim0],
            )
        elif dim0 in self.concept_names:
            new_concept_names[dim1] = self.concept_names.pop(dim0)
        elif dim1 in self.concept_names:
            new_concept_names[dim0] = self.concept_names.pop(dim1)

        return ConceptTensor(new_tensor, concept_names=new_concept_names)

    def permute(self, *dims):
        """
        Permute the dimensions of the tensor and update concept names
        accordingly.
        """
        new_tensor = super().permute(*dims)
        new_concept_names = {}

        # Reassign concept names based on the new permutation
        for new_dim, old_dim in enumerate(dims):
            if old_dim in self.concept_names:
                new_concept_names[new_dim] = self.concept_names[old_dim]

        return ConceptTensor(new_tensor, concept_names=new_concept_names)

    def squeeze(self, dim=None):
        """
        Squeeze the tensor and update concept names accordingly.
        """
        if dim is not None:
            new_tensor = super().squeeze(dim)
        else:
            new_tensor = super().squeeze()

        new_tensor = new_tensor.as_subclass(ConceptTensor)
        if hasattr(self, 'concept_names'):
            new_concept_names = {}
            if dim is not None:
                for d, names in self.concept_names.items():
                    if d < dim:
                        new_concept_names[d] = names
                    elif d > dim:
                        new_concept_names[d - 1] = names
            else:
                new_dim = 0
                for d, names in self.concept_names.items():
                    if self.shape[d] != 1:
                        new_concept_names[new_dim] = names
                        new_dim += 1
            new_tensor.concept_names = new_concept_names
        return new_tensor

    def unsqueeze(self, dim):
        """
        Unsqueeze the tensor and update concept names accordingly.
        """
        new_tensor = super().unsqueeze(dim)
        new_tensor = new_tensor.as_subclass(ConceptTensor)
        if hasattr(self, 'concept_names'):
            new_concept_names = {
                i + 1 if i >= dim else i: v
                for i, v in self.concept_names.items()
            }
            new_concept_names[dim] = [f"concept_{dim}_0"]
            new_tensor.concept_names = new_concept_names
        return new_tensor

    def __getitem__(self, key):
        sliced_tensor = super().__getitem__(key)
        if isinstance(sliced_tensor, torch.Tensor) and (
            not isinstance(sliced_tensor, ConceptTensor)
        ):
            sliced_tensor = sliced_tensor.as_subclass(ConceptTensor)

        new_concept_names = {}
        for dim, names in self.concept_names.items():
            if dim < len(sliced_tensor.shape):
                if isinstance(key, tuple):
                    index = key[dim] if dim < len(key) else slice(None)
                else:
                    index = key if dim == 0 else slice(None)

                if isinstance(index, slice):
                    new_concept_names[dim] = names[index]
                elif isinstance(index, int):
                    if index < len(names):
                        new_concept_names[dim] = [names[index]]
                    else:
                        new_concept_names[dim] = [f"concept_{dim}_{index}"]
                elif isinstance(index, list):
                    new_concept_names[dim] = [
                        names[i] for i in index if i < len(names)
                    ]
        sliced_tensor.concept_names = new_concept_names
        return sliced_tensor

    def ravel(self):
        new_tensor = super().ravel()
        return new_tensor.as_subclass(torch.Tensor)


class ConceptDistribution(torch.distributions.Distribution):
    """
    ConceptDistribution is a subclass of torch.distributions.Distribution which
    ensures that the samples are ConceptTensors.
    """

    def __init__(
        self,
        base_dist: torch.distributions.Distribution,
        concept_names: Dict[int, List[str]] = None,
    ):
        self.base_dist = base_dist
        self.concept_names = ConceptTensor._check_concept_names(
            base_dist.mean,
            concept_names.copy(),
        )
        super().__init__()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> ConceptTensor:
        """
        Sample from the distribution.

        Args:
            sample_shape: Shape of the sample.

        Returns:
            ConceptTensor: Sampled ConceptTensor.
        """
        sample = self.base_dist.rsample(sample_shape)
        return ConceptTensor.concept(
            sample,
            concept_names=self.concept_names.copy(),
        )
