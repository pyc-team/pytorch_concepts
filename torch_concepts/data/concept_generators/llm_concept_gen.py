import json
import re
from typing import Any

from torch.utils.data import Dataset

from torch_concepts import AxisAnnotation
from torch_concepts.data.base.concept_generator import (
    LLM,
    ConceptGenerator,
    Parser,
    Postprocessor,
    Prompt,
)


ConceptSpec = dict[str, Any]


class LLMConceptGenerator(ConceptGenerator):
    """Generate concept-axis annotations using an arbitrary LLM or VLM.

    Parameters
    ----------
    llm : LLM
        Callable that accepts a rendered prompt and returns the model output.
    prompt : Prompt, optional
        Default string or callable prompt used by :meth:`generate`.
    parser : Parser, optional
        Converts raw model output into intermediate concept specifications.
        The default parser accepts JSON, line-separated lists, bullets,
        numbering, and comma-separated output.
    postprocessor : Postprocessor, optional
        Cleans and deduplicates concept specifications by concept name.
    llm_kwargs : dict, optional
        Additional keyword arguments passed to the LLM callable.
    """

    def __init__(
        self,
        llm: LLM,
        prompt: Prompt | None = None,
        parser: Parser | None = None,
        postprocessor: Postprocessor | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ):
        self.llm = llm
        self.prompt = prompt
        self.parser = parser or default_concept_parser
        self.postprocessor = postprocessor or default_concept_postprocessor
        self.llm_kwargs = llm_kwargs or {}

    def generate(
        self,
        dataset: Dataset | None = None,
        class_names: list[str] | None = None,
        prompt: Prompt | None = None,
        **kwargs: Any,
    ) -> AxisAnnotation:
        """Generate an :class:`AxisAnnotation` from an LLM response.

        Parameters
        ----------
        dataset : Dataset, optional
            Dataset-level context supplied to callable prompts.
        class_names : list[str], optional
            Class names supplied to the prompt renderer.
        prompt : Prompt, optional
            Per-call prompt override. If omitted, the constructor prompt is
            used.
        **kwargs : Any
            Additional values supplied to the prompt renderer.

        Returns
        -------
        AxisAnnotation
            Generated binary and/or categorical concept definitions.
        """
        prompt = prompt if prompt is not None else self.prompt
        if prompt is None:
            raise ValueError(
                "No prompt provided. Pass a prompt to the constructor "
                "or to generate(prompt=...)."
            )

        prompt_payload = self._render_prompt(
            prompt=prompt,
            dataset=dataset,
            class_names=class_names,
            **kwargs,
        )
        raw_output = self.llm(prompt_payload, **self.llm_kwargs)
        specs = self.postprocessor(self.parser(raw_output))
        normalized = default_concept_postprocessor(specs)
        return concept_specs_to_annotation(normalized)

    def _render_prompt(
        self,
        prompt: Prompt,
        dataset: Dataset | None = None,
        class_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a string or callable prompt.

        String prompts are formatted with ``class_names``, ``num_classes``,
        and any additional keyword arguments. Callable prompts receive the
        dataset, class names, and keyword arguments directly.

        Returns
        -------
        Any
            Rendered payload passed to the configured LLM callable.
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


def default_concept_parser(text: str) -> list[ConceptSpec]:
    """Parse common LLM output formats into concept specifications.

    Supported formats include JSON lists of strings, JSON objects containing
    ``name``/``concept`` and optional ``states`` fields, fenced JSON, plain
    line-separated lists, bullets, numbering, and comma-separated output.

    Returns
    -------
    list[ConceptSpec]
        Intermediate binary or categorical concept specifications.
    """

    if not isinstance(text, str):
        raise TypeError("LLM output must be a string.")

    text = _strip_code_fence(text.strip())
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        if isinstance(data.get("concepts"), list):
            data = data["concepts"]
        else:
            data = [data]

    if isinstance(data, list):
        parsed = [_item_to_spec(item) for item in data]
        parsed = [spec for spec in parsed if spec is not None]
        if parsed:
            return parsed

    concepts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)\-:]\s*", "", line)
        line = line.strip("\"'` ")
        if line:
            concepts.append(line)

    if len(concepts) <= 1 and "," in text:
        concepts = [
            concept.strip("\"'` ")
            for concept in text.split(",")
            if concept.strip("\"'` ")
        ]

    return [{"name": concept, "type": "binary"} for concept in concepts]


def default_concept_postprocessor(concepts: list[Any]) -> list[ConceptSpec]:
    """Normalize and deduplicate concept specifications by concept name."""

    processed: list[ConceptSpec] = []
    seen: set[str] = set()

    for concept in concepts:
        spec = _item_to_spec(concept)
        if spec is None:
            continue

        name = _clean_text(str(spec["name"]))
        if not name:
            continue

        key = name.casefold()
        if key in seen:
            continue

        states = spec.get("states")
        if isinstance(states, str):
            states = [state.strip() for state in states.split(",")]
        if states is not None:
            state_names = []
            state_keys = set()
            for state in states:
                cleaned = _clean_text(str(state))
                state_key = cleaned.casefold()
                if cleaned and state_key not in state_keys:
                    state_keys.add(state_key)
                    state_names.append(cleaned)
            states = state_names or None

        seen.add(key)
        normalized: ConceptSpec = {
            "name": name,
            "type": "categorical" if states else "binary",
        }
        if states:
            normalized["states"] = states
        processed.append(normalized)

    return processed


def concept_specs_to_annotation(concepts: list[Any]) -> AxisAnnotation:
    specs = default_concept_postprocessor(concepts)
    labels = [spec["name"] for spec in specs]
    states = [
        list(spec["states"]) if spec.get("states") else ["0"]
        for spec in specs
    ]
    cardinalities = [len(state_names) for state_names in states]
    metadata = {
        label: {"type": "discrete"}
        for label in labels
    }
    return AxisAnnotation(
        labels=labels,
        states=states,
        cardinalities=cardinalities,
        metadata=metadata,
    )


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.I)
    return match.group(1).strip() if match else text


def _item_to_spec(item: Any) -> ConceptSpec | None:
    if isinstance(item, str):
        return {"name": item, "type": "binary"}
    if not isinstance(item, dict):
        return None

    name = item.get("name", item.get("concept"))
    if name is None:
        return None

    spec: ConceptSpec = {
        "name": str(name),
        "type": item.get("type", "categorical" if item.get("states") else "binary"),
    }
    if "states" in item:
        spec["states"] = item["states"]
    return spec


def _clean_text(value: str) -> str:
    value = value.strip().rstrip(".")
    return re.sub(r"\s+", " ", value)
