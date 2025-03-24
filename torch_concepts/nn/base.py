import numpy as np
import torch

from torch_concepts.base import AnnotatedTensor
from typing import List, Union

def _standardize_annotations(
    annotations: Union[List[Union[List[str], int]], List[str], int]
) -> List[Union[List[str], int]]:
    """
    Helper function to standardize the annotations arguments so that we can
    support singleton arguments (e.g., a single axis is being annotated), as
    well as axis-specific annotations.
    """
    if annotations is None:
        return None

    if isinstance(annotations, int):
        # Then this is a singleton annotation. We will wrap it up to
        # standardize on always using lists
        annotations = [annotations]
    elif isinstance(annotations, list) and len(annotations) and (
            isinstance(annotations[0], str)
    ):
        # Then this is a singleton annotation with named dimensions. We will
        # wrap it up to standardize on always using lists
        annotations = [annotations]
    return annotations


class Annotate(torch.nn.Module):
    """
    Annotate is a class for annotation layers.
    The output objects are annotated tensors with the exact shape of the input
    tensors.
    """

    def __init__(
            self,
            annotations: Union[List[Union[List[str], int]], List[str], int] = None,
            annotated_axis: Union[List[int], int] = None,
    ):
        super().__init__()
        annotations = _standardize_annotations(annotations)
        self.annotated_axis = annotated_axis
        self.annotations = annotations

    def forward(
            self,
            x: torch.Tensor,
    ) -> AnnotatedTensor:
        return AnnotatedTensor.tensor(
            tensor=x,
            annotations=self.annotations,
            annotated_axis=self.annotated_axis,
        )


class LinearConceptLayer(torch.nn.Module):
    """
    LinearConceptLayer is a class which first applies a linear
        transformation to the input tensor, then it reshapes and
        annotates the output tensor.
    """

    def __init__(
            self,
            in_features: int,
            out_annotations: Union[List[Union[List[str], int]], List[str], int],
            *args,
            **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        out_annotations = _standardize_annotations(out_annotations)

        self.annotations = []
        shape = []
        for dim, annotation in enumerate(out_annotations):
            if isinstance(annotation, int):
                self.annotations.append([])
                shape.append(annotation)
            else:
                self.annotations.append(annotation)
                shape.append(len(annotation))

        self.annotated_axes = []
        for dim, annotation in enumerate(out_annotations):
            self.annotated_axes.append(-len(shape) + dim)
        self._shape = shape
        self.output_size = np.prod(self.shape())

        self.transform = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.output_size,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self.shape()),
            Annotate(self.annotations, self.annotated_axes)
        )

    def shape(self):
        return self._shape

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs,
    ) -> AnnotatedTensor:
        """
        Forward pass of a LinearConceptLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            AnnotatedTensor: Transformed AnnotatedTensor.
        """
        return self.transform(x, *args, **kwargs)
