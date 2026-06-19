from abc import ABC, abstractmethod
from typing import Any, Callable

from torch.utils.data import Dataset

from torch_concepts import AxisAnnotation

Prompt = str | Callable[..., Any]
LLM = Callable[..., Any]
Parser = Callable[[Any], list[Any]]
Postprocessor = Callable[[list[Any]], list[Any]]


class ConceptGenerator(ABC):
    """Base class for automatic concept generators.

    Dataset -> concepts
    A generator maps dataset-level information to a concept-axis annotation.
    """

    @abstractmethod
    def generate(
        self,
        dataset: Dataset | None = None,
        class_names: list[str] | None = None,
        **kwargs: Any,
    ) -> AxisAnnotation:
        pass
