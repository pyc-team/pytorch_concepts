import copy
import numpy as np
import torch

from typing import List, Union, Tuple


class AnnotatedTensor(torch.Tensor):
    """
    AnnotatedTensor is a subclass of torch.Tensor which ensures that the tensor
    has at least two dimensions: batch size and at least one
    possibly-semantically annotated dimension at index annotated_axis.

    Attributes:
        data (torch.Tensor): Data tensor.
        annotations (Union[List[str], List[List[str]]): Semantic names for
            each annotated entry/dimension. If this argument is a list of lists,
            then it is expected to have as many elements as annotated_axis.
            Otherwise, if it is a single list of strings, then we will assume
            that only a single dimension is annotated and annotated_axis is
            expected to be a single integer.
        annotated_axis (Union[list[int], int]): Dimension(s) that will be
            annotated using the provided semantics.
            If not provided, it defaults to the last dimension.
    """
    def __new__(
        cls,
        data: torch.Tensor,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
        *args,
        **kwargs,
    ) -> 'AnnotatedTensor':
        instance = super().__new__(cls, data, *args, **kwargs)
        instance.annotations = cls._check_annotations(
            tensor=data,
            annotations=annotations,
            annotated_axis=annotated_axis,
        )
        return instance

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Perform the torch function as usual
        result = super().__torch_function__(func, types, args, kwargs)

        # Convert the result to a standard torch.Tensor if it's a AnnotatedTensor
        if isinstance(result, AnnotatedTensor):
            return result.to_standard_tensor()

        return result

    @staticmethod
    def _generate_default_annotations(shape, annotated_axis=-1):
        return [
            f"dim_{i}" for i in range(shape[annotated_axis])
        ]

    @staticmethod
    def _standarize_arguments(
        tensor: torch.Tensor,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
    ) -> Tuple[List[List[str]], List[int]]:
        if annotations is None:
            annotations = []
        if annotated_axis is None:
            annotated_axis = [i for i in range(len(annotations))]

        if not isinstance(annotations, (list, tuple)):
            raise ValueError(
                f'Expected annotations to be a list of string lists or a '
                f'single list of strings. Instead, we were given '
                f'{annotations}.'
            )
        if len(annotations) and (
            not isinstance(annotations[0], (list, tuple))
        ):
            if not isinstance(annotations[0], str):
                raise ValueError(
                    f'Expected annotations to be a list of string lists or a '
                    f'single list of strings. Instead, we were given '
                    f'{annotations}.'
                )
            # Then this is a single list of annotations, so let's wrap it up
            # to be a list of lists
            annotations = [annotations]

        if not isinstance(annotated_axis, (list, tuple, int)):
            raise ValueError(
                f'Expected annotated_axis to be a list of integers or a '
                f'single integer. Instead, we were given '
                f'{annotated_axis}.'
            )
        if not isinstance(annotated_axis, (list, tuple)):
            annotated_axis = [annotated_axis]

        if len(annotations) != len(annotated_axis):
                raise ValueError(
                    f'We expected to be provided as many sets of axis '
                    f'annotations as annotated axii. Instead, we got '
                    f'{len(annotations)} sets of annotations and '
                    f'{len(annotated_axis)} sets of annotated axii.'
                )

        # Now, let's sort things out so that things are ordered correctly
        permutation = [
            x[0] for x in sorted(enumerate(annotated_axis), key=lambda x: x[1])
        ]
        annotations = [
            annotations[x] for x in permutation
        ]
        annotated_axis = [
            annotated_axis[x] for x in permutation
        ]

        for annotation_idx in annotated_axis:
            if annotation_idx < 0 or annotation_idx >= len(tensor.shape):
                raise ValueError(
                    f"Annotation axis {annotation_idx} is out of range for "
                    f"tensor with shape {tensor.shape}."
                )

        # Finally make it so that all dimensions are provied with annotations (empty)
        # for those dimensions whose annotations we were not provided
        if annotated_axis == []:
            annotations = [[] for _ in tensor.shape]
        else:
            annotations = [[] for _ in range(annotated_axis[0])] + annotations
            annotations =  annotations + [
                [] for _ in range(annotated_axis[-1] + 1, len(tensor.shape))
            ]
        return annotations

    @staticmethod
    def _check_annotations(
        tensor: torch.Tensor,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
    ) -> Tuple[List[List[str]], List[int]]:

        # First standarize the arguments
        annotations = AnnotatedTensor._standarize_arguments(
            tensor=tensor,
            annotations=annotations,
            annotated_axis=annotated_axis,
        )
        new_annotations = [
            [] for _ in tensor.shape
        ]
        # At this point we know we have as many sets of annotations as
        # provided indices
        for annotation_idx, annotation_set in enumerate(annotations):
            if annotation_set is None:
                current_annotations = [
                    f"dim_{annotated_axis}_{i}"
                    for i in range(tensor.shape[annotation_idx])
                ]
            elif annotation_set == []:
                current_annotations = None
            elif (len(annotation_set) != tensor.shape[annotation_idx]):
                raise ValueError(
                    f'For dimension at axis {annotation_idx} we were given an '
                    f'annotation set with {len(annotation_set)} entries. '
                    f'However, we expected an annotation set with '
                    f'{tensor.shape[annotation_idx]} elements as the tensor to '
                    f'be annotated has shape {tensor.shape}.'
                )
            else:
                # Copy the list so that we can do manipulation without affecting
                # previous pointers to this array
                current_annotations = annotations[annotation_idx][:]
            new_annotations[annotation_idx] = current_annotations
        return new_annotations

    def __str__(self):
        """
        Returns a string representation of the AnnotatedTensor.
        """
        return (
            f"AnnotatedTensor of shape {self.shape}, dtype {self.dtype}, and "
            f"annotations {self.annotations} for each dimension."
        )

    @classmethod
    def tensor(
        cls,
        tensor: torch.Tensor,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
    ) -> 'AnnotatedTensor':
        """
        Create a AnnotatedTensor from a torch.Tensor.

        Attributes:
            tensor: Input tensor.
            annotations: Names of dimensions.
            annotated_axis: dimension of tensor which indexes concepts.
        Returns:
            AnnotatedTensor: AnnotatedTensor instance.
        """
        # Ensure the tensor has the correct shape
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor.")
        if len(tensor.shape) < 2:
            raise ValueError(
                "AnnotatedTensor must have at least two dimensions: batch size "
                "and number of concepts."
            )

        # Convert the existing tensor to AnnotatedTensor
        instance = tensor.as_subclass(cls)
        instance.annotations = cls._check_annotations(
            tensor=tensor,
            annotations=annotations,
            annotated_axis=annotated_axis,
        )
        return instance

    def assign_annotations(
        self,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
    ):
        """
        Assign new concept names to the AnnotatedTensor.

        Attributes:
            annotations: Dictionary of concept names.
            annotated_axis: dimension of tensor which indexes concepts.
        """
        self.annotations = self._check_annotations(
            tensor=self,
            annotations=annotations,
            annotated_axis=annotated_axis,
        )

    def update_annotations(
        self,
        new_annotations: List[List[str]],
        annotated_axis: int,
    ):
        """
        Update the concept names for specified dimensions.

        Attributes:
            new_annotations: Dictionary with dimension indices as keys and
                lists of new concept names as values.
        """
        if len(new_annotations) != self.shape[annotated_axis]:
            raise ValueError(
                f"When updating the annotations of tensor with "
                f"shape {self.shape} and annotation axis {annotated_axis}, "
                f"we expected the new names to "
                f"have {self.shape[annotated_axis]} elements in it. "
                f"Instead, the list has {len(new_annotations)} entries in it."
            )
        self.annotations[annotated_axis] = new_annotations[:]

    def annotated_axis(self) -> List[int]:
        return [
            idx for idx, annotations in enumerate(self.annotations)
            if (annotations is not None) and len(annotations)
        ]

    def extract_by_annotations(
        self,
        target_annotations: List[Union[int, str]],
        target_axis: int = None,
    ) -> 'AnnotatedTensor':
        """
        Extract a subset of concepts from the AnnotatedTensor.

        Attributes:
            target_annotations: List of concept names or indices to extract.

        Returns:
            AnnotatedTensor: Extracted AnnotatedTensor.
        """
        if self.annotations is None:
            raise ValueError(
                "Annotations names are not set for this AnnotatedTensor."
            )
        if target_axis is not None:
            # Then we take this to be the last annotated axis
            annotated_dims = self.annotated_axis()
            if len(annotated_dims) == 0:
                raise ValueError(
                    f'We cannot access any axis through annotations for '
                    f'AnnotatedTensor without any dimensions annotated.'
                )

            target_axis = annotated_dims[-1]

        indices = []
        for annotation_name in target_annotations:
            if isinstance(annotation_name, str):
                if annotation_name not in self.annotations[target_axis]:
                    raise ValueError(
                        f"Annotation {annotation_name} was not found amongst "
                        f"annotations {self.annotations[target_axis]} of "
                        f"axis {target_axis} in AnnotatedTensor."
                    )
                indices.append(self.annotations[target_axis].index(annotation_name))
            else:
                # Else this is a numerical index
                indices.append(annotation_name)

        extracted_data = self.index_select(
            dim=target_axis,
            index=torch.tensor(indices, device=self.device),
        )
        new_annotations = copy.deepcopy(self.annotations)
        new_annotations[target_axis] = [
            self.annotations[target_axis][i] for i in indices
        ]
        # replace None with empty list
        new_annotations = [
            annotation for annotation in new_annotations if annotation is not None
        ]

        return AnnotatedTensor(
            extracted_data,
            annotations=new_annotations,
            annotated_axis=self.annotated_axis(),
        )

    def new_empty(self, *shape):
        """
        Create a new empty AnnotatedTensor with the same concept names,
        shape, and concept axis.

        Attributes:
            shape: Shape of the new tensor.

        Returns:
            AnnotatedTensor: A new empty AnnotatedTensor.
        """
        # Create a new empty tensor with the specified shape
        new_tensor = super().new_empty(*shape, device=self.device)

        new_annotations = [
            annotation for annotation in self.annotations if annotation is not None
        ]
        return AnnotatedTensor(
            new_tensor,
            annotations=new_annotations,
            annotated_axis=self.annotated_axis()
        )

    def to_standard_tensor(self) -> torch.Tensor:
        """
        Convert the AnnotatedTensor to a standard torch.Tensor while preserving
        gradients.

        Returns:
            torch.Tensor: Standard tensor with gradients.
        """
        return self.as_subclass(torch.Tensor)

    def view(
        self,
        *shape,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
    ):
        """
        View the tensor with a new shape and update concept names accordingly.
        """
        new_tensor = super().view(*shape)
        new_tensor = new_tensor.as_subclass(AnnotatedTensor)
        new_tensor.assign_annotations(
            annotations=annotations,
            annotated_axis=annotated_axis,
        )
        return new_tensor

    def reshape(
        self,
        *shape,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
    ):
        """
        Reshape the tensor to the specified shape and update concept names
        accordingly.
        """
        new_tensor = super().reshape(*shape)
        new_tensor = new_tensor.as_subclass(AnnotatedTensor)
        new_tensor.assign_annotations(
            annotations=annotations,
            annotated_axis=annotated_axis,
        )
        return new_tensor

    def transpose(self, dim0, dim1):
        """
        Transpose two dimensions of the tensor and update concept names
        accordingly.
        """
        new_tensor = super().transpose(dim0, dim1)
        return AnnotatedTensor(
            new_tensor,
            annotations=list(np.transpose(
                np.array(self.annotations),
                (dim0, dim1),
            )),
        )

    def permute(self, *dims):
        """
        Permute the dimensions of the tensor and update concept names
        accordingly.
        """
        new_tensor = super().permute(*dims)
        return AnnotatedTensor(
            new_tensor,
            annotations=list(np.transpose(
                np.array(self.annotations),
                dims,
            )),
        )

    def squeeze(self, dim=None):
        """
        Squeeze the tensor and update concept names accordingly.
        """
        if dim is not None:
            new_tensor = super().squeeze(dim)
        else:
            new_tensor = super().squeeze()

        new_tensor = new_tensor.as_subclass(AnnotatedTensor)
        if hasattr(self, 'annotations'):
            new_tensor.annotations = (
                self.annotations[:dim] + self.annotations[dim+1:]
            )
        return new_tensor

    def unsqueeze(self, dim):
        """
        Unsqueeze the tensor and update concept names accordingly.
        """
        new_tensor = super().unsqueeze(dim)
        new_tensor = new_tensor.as_subclass(AnnotatedTensor)
        if hasattr(self, 'annotations'):
            new_tensor.annotations = (
                self.annotations[:dim] + [None] + self.annotations[dim:]
            )
        return new_tensor

    def __getitem__(self, key):
        sliced_tensor = super().__getitem__(key)
        if isinstance(sliced_tensor, torch.Tensor) and (
            not isinstance(sliced_tensor, AnnotatedTensor)
        ):
            sliced_tensor = sliced_tensor.as_subclass(AnnotatedTensor)

        if not isinstance(key, (list, tuple)):
            key = [key]

        sliced_tensor.annotations = []
        for axis, idx in enumerate(range(len(self.annotations))):
            if idx < len(key) and self.annotations[axis] is not None and len(self.annotations[axis]):
                sliced_tensor.annotations.append(self.annotations[axis].__getitem__(key[idx]))
            else:
                sliced_tensor.annotations.append(None)

        return sliced_tensor

    def ravel(self):
        new_tensor = super().ravel()
        return new_tensor.as_subclass(torch.Tensor)
