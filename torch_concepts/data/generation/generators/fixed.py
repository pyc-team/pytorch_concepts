from typing import Any, List, Optional

from torch_concepts import Annotations
from torch_concepts.data.generation.base.generator import Generator


class FixedConceptGenerator(Generator):
    """Return a predefined concept vocabulary, ignoring the dataset.

    Use it when the vocabulary comes from domain expertise or a published
    paper rather than from a model.

    Parameters
    ----------
    labels : list of str
        Concept names.
    states : list of list of str, optional
        State names per concept. None assume binary concepts.
    cardinalities : list of int, optional
        Cardinality per concept. None assume binary concepts.
    types : list of str, optional
        ``'binary'``, ``'categorical'``, or ``'continuous'`` per concept.
        None assume binary concepts.

    Examples
    --------
    .. code-block:: python

        from torch_concepts.data.generation.generators import (
            FixedConceptGenerator,
        )

        generator = FixedConceptGenerator(
            labels=["striped", "colour"],
            types=["binary", "categorical"],
            cardinalities=[1, 3],
        )
        print(generator.generate().labels)
    """

    def __init__(
        self,
        labels: List[str],
        states: Optional[List[List[str]]] = None,
        cardinalities: Optional[List[int]] = None,
        types: Optional[List[str]] = None,
    ):
        if not labels:
            raise ValueError("FixedConceptGenerator must define at least one concept.")
        self.annotations = Annotations(
            labels=list(labels),
            states=states,
            cardinalities=cardinalities,
            types=types,
        )

    def generate(self, **kwargs: Any) -> Annotations:
        """Return the predefined vocabulary, ignoring every argument."""
        return self.annotations
