def compose_refinements(
    *refinements: Callable[[ConceptGraph], ConceptGraph],
) -> Callable[[ConceptGraph], ConceptGraph]:
    """Compose graph refinements left-to-right.

    Parameters
    ----------
    refinements : callable
        Functions that each accept and return a :class:`ConceptGraph`.

    Returns
    -------
    callable
        A single refinement that applies the inputs in order.
    """
    if not refinements:
        raise ValueError("At least one refinement is required.")
    if not all(callable(refinement) for refinement in refinements):
        raise TypeError("All refinements must be callable.")

    def composed(graph: ConceptGraph) -> ConceptGraph:
        for refinement in refinements:
            graph = refinement(graph)
        return graph

    composed._refinements = tuple(refinements)
    return composed




DEFAULT_REFINEMENT_MODEL = "groq/openai/gpt-oss-20b"

_EDGE_TOKENS = ("A->B", "B->A", "none")

_PROMPT_TEMPLATE = (
    "You are a causal-inference expert {domain_clause}.\n"
    "Assess the direct causal relationship between:\n"
    "A: {concept_1_details}\n"
    "B: {concept_2_details}\n"
    "{context_section}"
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
) -> str | None:
    """Ask an LLM to orient one unordered concept pair.

    The backend must return one of the allowed edge tokens. Invalid answers are
    retried a few times with a stricter suffix; if all attempts fail, ``None``
    is returned so the caller can leave the pair unchanged.
    """
    domain_clause = f"in the domain of {domain}" if domain else ""
    concept_a_details = _concept_details(concept_a, concept_a_description)
    concept_b_details = _concept_details(concept_b, concept_b_description)

    context = ""

    base_prompt = _PROMPT_TEMPLATE.format(
        domain_clause=domain_clause,
        concept_1_details=concept_a_details,
        concept_2_details=concept_b_details,
        context_section=(
            f"\nRelevant context:\n{context}\n" if context else ""
        ),
    )
    retry_suffix = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + retry_suffix
        completion_kwargs = {"repeats": repeats}
        temperature = _retry_temperature(llm_backend, attempt)
        if temperature is not None:
            completion_kwargs["temperature"] = temperature
        response = llm_backend(prompt, **completion_kwargs)
        try:
            return _most_frequent_token(response)
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
                "A->B, B->A, none."
            )


def _retry_temperature(llm_backend: Callable[..., str], attempt: int) -> float | None:
    """Return a small retry temperature when a deterministic backend failed."""
    if attempt == 1:
        return None
    temperature = getattr(llm_backend, "completion_kwargs", {}).get("temperature")
    if temperature != 0:
        return None
    return min(0.1 * (attempt - 1), 1.0)


def _concept_details(name: str, description: str) -> str:
    """Combine a concept name and optional description for prompt text."""
    return f"{name} - {description}" if description else name


def _most_frequent_token(response: Any) -> str:
    """Return the majority valid token, using ``none`` for valid-token ties."""
    valid_tokens = [
        line.strip()
        for line in str(response).splitlines()
        if line.strip() in _EDGE_TOKENS
    ]
    if not valid_tokens:
        raise ValueError("LLM response contains no valid edge token.")
    counts = {token: 0 for token in _EDGE_TOKENS}
    for token in valid_tokens:
        counts[token] += 1

    highest_count = max(counts.values())
    winners = [token for token, count in counts.items() if count == highest_count]
    return winners[0] if len(winners) == 1 else "none"
