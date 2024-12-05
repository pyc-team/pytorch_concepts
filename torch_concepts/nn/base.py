from abc import ABC, abstractmethod

import torch
import numpy as np

from typing import List, Union, Dict, Tuple

from torch_concepts.base import AnnotatedTensor


class Annotate(torch.nn.Module):
    """
    Annotate is a class for annotation layers.
    The output objects are annotated tensors with the same shape of the input tensors.
    """
    def __init__(
        self,
        annotations: Union[List[List[str]], List[str]] = None,
        annotated_axis: Union[List[int], int] = None,
    ):
        super().__init__()
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
        tranformation to the input tensor, then it reshapes and
        annotates the output tensor.
    """
    def __init__(
        self,
        in_features: int,
        annotations: List[Union[List[str], int]],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features

        self.annotations = []
        shape = []
        self.annotated_axes = []
        for dim, annotation in enumerate(annotations):
            if isinstance(annotation, int):
                self.annotations.append([])
                shape.append(annotation)
            else:
                self.annotations.append(annotation)
                shape.append(len(annotation))
            self.annotated_axes.append(dim+1)

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
