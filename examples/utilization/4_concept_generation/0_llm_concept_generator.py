"""
Example: LLMConceptGenerator with Gemini

This example demonstrates how to generate candidate visual concepts for a
concept bottleneck model using the LLMConceptGenerator utility and Gemini API.

Before running, create a free Gemini API key in Google AI Studio and export it:

    export GEMINI_API_KEY="your_google_ai_studio_key"

Run from the repository root with:

    python examples/utilization/4_concept_generation/0_llm_concept_generator.py
"""

import json
import os
import urllib.error
import urllib.request

from torch_concepts.data.concept_generators.llm_concept_gen import LLMConceptGenerator


PROMPT = """I am building a concept bottleneck model for image classification.
The classes are: {class_names}.

Generate {num_concepts} short, visual, human-understandable concepts that could
help distinguish these classes.

Rules:
- concepts must be visually observable
- avoid class names themselves
- use short noun phrases
- return one concept per line
"""


def gemini_llm(prompt, **kwargs) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a free key in Google AI Studio "
            "and export it before running this example."
        )

    model = kwargs.pop("model", "gemini-2.5-flash")
    temperature = kwargs.pop("temperature", None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported Gemini request options: {unsupported}")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
    }
    if temperature is not None:
        payload["generationConfig"] = {"temperature": temperature}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8")
        raise RuntimeError(f"Gemini API request failed: {detail}") from error

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected Gemini API response: {data}") from error

    return "\n".join(part.get("text", "") for part in parts).strip()


def main():
    generator = LLMConceptGenerator(
        llm=gemini_llm,
        prompt=PROMPT,
        llm_kwargs={
            "temperature": 0.2,
        },
    )

    concepts = generator.generate(
        class_names=["cat", "dog", "horse"],
        num_concepts=10,
    )

    print(f"Generated {len(concepts)} concepts:\n")
    for i, concept in enumerate(concepts, start=1):
        print(f"{i}. {concept}")


if __name__ == "__main__":
    main()
