import json
import re
from typing import List, Any
from torch.utils.data import Dataset

from torch_concepts.data.base.concept_generator import LLM, ConceptGenerator, Parser, Postprocessor, Prompt


class LLMConceptGenerator(ConceptGenerator):
    """Generate concepts using an arbitrary LLM/VLM and arbitrary prompt.
    
    Parameters
    ----------
    llm : LLM
        A callable that takes a prompt and returns the LLM's output.
        It should accept the prompt as a string and return the output as a string.
    parser : Parser, optional
        A callable that takes the LLM's output and returns a list of concept strings.
        If not provided, a default parser is used that handles common output formats.
    postprocessor : Postprocessor, optional
        A callable that takes a list of concept strings and returns a cleaned and deduplicated list.
        If not provided, a default postprocessor is used that strips whitespace, removes duplicates, and normalizes the concepts.
    llm_kwargs : dict, optional
        Additional keyword arguments to pass to the LLM callable.
    """

    def __init__(
        self,
        llm: LLM,
        parser: Parser | None = None,
        postprocessor: Postprocessor | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ):
        self.llm = llm
        self.parser = parser or default_concept_parser
        self.postprocessor = postprocessor or default_concept_postprocessor
        self.llm_kwargs = llm_kwargs or {}

    def generate(
        self,
        dataset: Dataset | None = None,
        class_names: List[str] | None = None,
        prompt: Prompt | None = None,
        **kwargs,
    ) -> List[str]:
        """Generate concepts using the LLM and prompt.
        
        Parameters
        ----------
        dataset : Dataset, optional
            The dataset to provide context for generating concepts. Can be None if not needed.
        class_names : List[str], optional
            A list of class names to provide context for generating concepts. Can be None if not needed.
        prompt : Prompt, optional
            A string or callable that generates the prompt for the LLM.
            If a string, it can contain placeholders for formatting.
            If a callable, it should accept the dataset and class_names as arguments and return a string.
        **kwargs : Any
            Additional keyword arguments to pass to the prompt renderer.
            
        Returns
        -------
        List[str]
            A list of generated concept strings.
        """

        if prompt is None:
            raise ValueError(
                "No prompt provided. Pass a prompt either in the constructor "
                "or in generate(prompt=...)."
            )

        prompt_payload = self._render_prompt(
            prompt=prompt,
            dataset=dataset,
            class_names=class_names,
            **kwargs,
        )

        raw_output = self.llm(prompt_payload, **self.llm_kwargs)

        concepts = self.parser(raw_output)
        concepts = self.postprocessor(concepts)

        return concepts

    def _render_prompt(
        self,
        prompt: Prompt,
        dataset: Dataset | None = None,
        class_names: List[str] | None = None,
        **kwargs,
    ) -> Any:
        """Render the prompt, calling it if it's a function or formatting it if it's a string.
        
        The prompt can be either a string with format placeholders or a callable that generates the prompt dynamically.
        
        Example usage:
            # String prompt with placeholders
            prompt = "Generate concepts for the following classes: {class_names}."
            # Callable prompt
            def prompt_fn(class_names, **kwargs):
                return f"Generate concepts for these classes: {', '.join(class_names)}. Additional info: {kwargs.get('info', 'none')}."
                
        Args:
            prompt: The prompt to render, either a string or a callable.
            dataset: Optional dataset to provide context for rendering.
            class_names: Optional list of class names to provide context for rendering.
            **kwargs: Additional keyword arguments to pass to the prompt renderer.
            
        Returns:
            The rendered prompt, ready to be passed to the LLM.
        """
        if callable(prompt):
            return prompt(
                dataset=dataset,
                class_names=class_names,
                **kwargs,
            )

        if not isinstance(prompt, str):
            raise TypeError("prompt must be either a string or a callable.")

        format_vars = {
            "class_names": ", ".join(class_names) if class_names else None,
            "num_classes": len(class_names) if class_names else None,
            **kwargs,
        }

        return prompt.format(**format_vars)
            
            
def default_concept_parser(text: str) -> List[str]:
    """Parse LLM outputs into a list of concept strings.
    
    This parser is designed to handle a variety of common output formats, including:
    - JSON lists of strings or objects with "name"/"concept" fields
    - Line-separated lists with optional bullets or numbering
    - Comma-separated lists
    
    It returns a list of cleaned concept strings extracted from the LLM output.
    """

    text = text.strip()
    
    try:
        data = json.loads(text)

        if isinstance(data, list):
            concepts = []

            for item in data:
                if isinstance(item, str):
                    concepts.append(item)
                elif isinstance(item, dict):
                    if "name" in item:
                        concepts.append(str(item["name"]))
                    elif "concept" in item:
                        concepts.append(str(item["concept"]))

            if concepts:
                return concepts

    except json.JSONDecodeError:
        pass

    # Otherwise parse line by line
    concepts = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue
        
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)\-:]\s*", "", line)
        line = line.strip("\"'` ")

        if line:
            concepts.append(line)

    # comma-separated output
    if len(concepts) <= 1 and "," in text:
        concepts = [
            c.strip("\"'` ")
            for c in text.split(",")
            if c.strip()
        ]

    return concepts


def default_concept_postprocessor(concepts: List[str]) -> List[str]:
    """Clean and deduplicate generated concepts."""

    processed = []
    seen = set()

    for concept in concepts:
        c = concept.strip()
        c = c.rstrip(".")
        c = re.sub(r"\s+", " ", c)

        if not c:
            continue

        key = c.lower()

        if key in seen:
            continue

        seen.add(key)
        processed.append(c)

    return processed