"""Shared pairwise prompts, response voting and invalid-answer retries.

Used by direct LLM discovery and reciprocal-edge refinement. Backends are
callables accepting a prompt and completion options, including repeats.
Only exact token lines A->B, B->A and none contribute to voting; provider
exceptions propagate instead of being treated as invalid model answers.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

_EDGE_TOKENS = ("A->B", "B->A", "none")

DIRECTED_PROMPT_TEMPLATE = (
    "You are a causal-inference expert {domain_clause}.\n"
    "Assess the direct causal relationship between:\n"
    "A: {concept_1_details}\n"
    "B: {concept_2_details}\n"
    "Choose one answer, accounting for confounding, indirect effects, and "
    "mere association:\n"
    "A->B: A directly causes B\n"
    "B->A: B directly causes A\n"
    "none: no direct causal relationship\n\n"
    "Reason internally. Output exactly A->B, B->A, or none. "
    "No other response is allowed."
)


def _query_pair(
    llm_backend: Callable[..., str],
    concept_a: str,
    concept_a_description: str,
    concept_b: str,
    concept_b_description: str,
    *,
    domain: str = "",
    repeats: int = 1,
    max_attempts: int = 3,
    completion_kwargs: dict[str, Any] | None = None,
) -> str | None:
    """Query and vote on the direct causal direction for one concept pair.

    Parameters
    ----------
    llm_backend : callable
        Receives the formatted prompt and completion keywords, including repeats.
    concept_a, concept_b : str
        Names substituted as A and B in the prompt.
    concept_a_description, concept_b_description : str
        Optional descriptions appended to those names.
    domain : str, default ""
        Optional causal-inference domain for the prompt.
    repeats : int, default 1
        Number of completions requested from the backend for token voting.
    max_attempts : int, default 3
        Maximum calls after answers containing no valid token. Each retry adds
        corrective instructions; deterministic backends may get a higher temperature.
    completion_kwargs : dict, optional
        Additional options forwarded to the backend. repeats is controlled by
        the argument above; retry temperature may override an option on retries.

    Returns
    -------
    str or None
        Majority of exact token lines A->B, B->A or none; tied valid-token votes
        resolve to none. After all invalid-answer attempts, warn and return None.
        Direct discovery treats None as no edge; refinement retains the old pair.

    Notes
    -----
    Provider exceptions propagate. These attempts handle malformed answers,
    not network errors or rate limits. Public source/refinement factories
    validate repeats before using this helper.
    """
    tokens = _EDGE_TOKENS
    template = DIRECTED_PROMPT_TEMPLATE
    domain_clause = f"in the domain of {domain}" if domain else ""
    concept_a_details = _concept_details(concept_a, concept_a_description)
    concept_b_details = _concept_details(concept_b, concept_b_description)

    base_prompt = template.format(
        domain_clause=domain_clause,
        concept_1_details=concept_a_details,
        concept_2_details=concept_b_details,
    )
    retry_suffix = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + retry_suffix
        call_kwargs = {**(completion_kwargs or {}), "repeats": repeats}
        temperature = _retry_temperature(llm_backend, attempt)
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        response = llm_backend(prompt, **call_kwargs)
        try:
            return _most_frequent_token(response, tokens)
        except ValueError:
            if attempt == max_attempts:
                warnings.warn(
                    "LLM did not return a valid edge token after "
                    f"{max_attempts} attempts for pair "
                    f"{concept_a!r}, {concept_b!r}. Last response: "
                    f"{response!r}. Falling back to None.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            warnings.warn(
                "LLM returned no valid edge token for pair "
                f"{concept_a!r}, {concept_b!r}; retrying "
                f"({attempt}/{max_attempts}).",
                UserWarning,
                stacklevel=2,
            )
            retry_suffix = (
                "\n\nYour previous response was invalid: "
                f"{response!r}. Return only one of these exact tokens: "
                f"{', '.join(tokens)}."
            )


def _retry_temperature(llm_backend: Callable[..., str], attempt: int) -> float | None:
    """Increase retry temperature only for a backend configured with temperature=0.

    Return None on the first attempt or when the backend's completion_kwargs
    are absent/nonzero. Otherwise return min(0.1 * (attempt - 1), 1.0).
    """
    if attempt == 1:
        return None
    temperature = getattr(llm_backend, "completion_kwargs", {}).get("temperature")
    if temperature != 0:
        return None
    return min(0.1 * (attempt - 1), 1.0)


def _concept_details(name: str, description: str) -> str:
    """Combine a concept name and optional description for prompt text."""
    return f"{name} - {description}" if description else name


def _most_frequent_token(response: Any, tokens: tuple[str, ...] = _EDGE_TOKENS) -> str:
    """Vote over exact valid-token lines, with none on ties.

    Whitespace around lines is ignored; prose and embedded tokens do not count.
    Raise ValueError if no line matches a valid token. All valid lines contribute
    one vote, allowing a backend to concatenate repeated completion outputs.
    """
    valid_tokens = [
        line.strip()
        for line in str(response).splitlines()
        if line.strip() in tokens
    ]
    if not valid_tokens:
        raise ValueError("LLM response contains no valid edge token.")
    counts = {token: 0 for token in tokens}
    for token in valid_tokens:
        counts[token] += 1

    highest_count = max(counts.values())
    winners = [token for token, count in counts.items() if count == highest_count]
    return winners[0] if len(winners) == 1 else "none"
