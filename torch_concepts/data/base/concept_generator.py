from abc import ABC, abstractmethod
from typing import Any, Callable, List
from torch.utils.data import Dataset


Prompt = str | Callable[..., Any]
LLM = Callable[..., Any]
Parser = Callable[[Any], List[str]]
Postprocessor = Callable[[List[str]], List[str]]


class ConceptGenerator(ABC):
    """Abstract base class for automatic concept generators.
    
    The generator maps:
        dataset -> concepts
    """

    @abstractmethod
    def generate(
        self,
        dataset: Dataset | None = None,
        class_names: List[str] | None = None,
        **kwargs,
    ) -> List[str]:
        pass