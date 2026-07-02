from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import time
from typing import Any


Message = dict[str, Any]
PromptJob = dict[str, Any]


class LiteLLMBackend:
    """Callable LLM backend built on LiteLLM's unified completion API.

    The backend is intentionally provider-agnostic: model names use LiteLLM's
    provider-prefixed format, for example ``"openai/gpt-4o"`` or
    ``"anthropic/claude-3-5-sonnet-20241022"``.

    Parameters
    ----------
    model : str
        LiteLLM model identifier.
    system_prompt : str, optional
        Optional system message prepended to string prompts.
    retry_on_rate_limit : bool, optional
        Whether to wait and retry once when a provider returns a short
        rate-limit countdown.
    max_rate_limit_wait : float, optional
        Maximum countdown, in seconds, that the backend is allowed to wait.
    **completion_kwargs : Any
        Default keyword arguments forwarded to ``litellm.completion``.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        retry_on_rate_limit: bool = False,
        max_rate_limit_wait: float = 60.0,
        **completion_kwargs: Any,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.retry_on_rate_limit = retry_on_rate_limit
        self.max_rate_limit_wait = max_rate_limit_wait
        self.completion_kwargs = {
            key: value
            for key, value in completion_kwargs.items()
            if value is not None
        }

    def __call__(self, prompt: Any, **kwargs: Any) -> str:
        """Run one or more prompt payloads and return concatenated text."""
        completion_kwargs = {
            key: value
            for key, value in {**self.completion_kwargs, **kwargs}.items()
            if value is not None
        }
        repeats = completion_kwargs.pop("repeats", 1)
        prompt_payloads = self._as_prompt_payloads(prompt)
        outputs = []
        for payload in prompt_payloads:
            for _ in range(repeats):
                outputs.append(self._complete(payload, completion_kwargs))
        return "\n".join(output for output in outputs if output).strip()

    def _complete(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
        completion_kwargs: dict[str, Any],
    ) -> str:
        try:
            from litellm import completion
        except ImportError as error:
            raise ImportError(
                "LiteLLMBackend requires litellm. Install it with: "
                "pip install litellm"
            ) from error

        request = self._completion_request(prompt, completion_kwargs)
        response = self._completion_with_rate_limit_retry(completion, request)
        return self._extract_content(response)

    def _completion_request(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
        completion_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": self._to_messages(prompt),
            **completion_kwargs,
        }

    def _completion_with_rate_limit_retry(
        self,
        completion: Any,
        request: dict[str, Any],
    ) -> Any:
        try:
            return completion(**request)
        except Exception as error:
            if not self.retry_on_rate_limit:
                raise

            wait_seconds = self._rate_limit_wait_seconds(error)
            if wait_seconds is None or wait_seconds > self.max_rate_limit_wait:
                raise

            print(
                "LiteLLM rate limit reached. Waiting "
                f"{wait_seconds:.1f}s before retrying..."
            )
            time.sleep(wait_seconds)
            return completion(**request)

    def _to_messages(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
    ) -> list[Message]:
        if isinstance(prompt, str):
            messages: list[Message] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            return messages

        messages = [dict(message) for message in prompt]
        if self.system_prompt and not any(
            message.get("role") == "system"
            for message in messages
        ):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    @staticmethod
    def _rate_limit_wait_seconds(error: Exception) -> float | None:
        error_text = str(error)
        if (
            "RateLimitError" not in error.__class__.__name__
            and "429" not in error_text
        ):
            return None

        patterns = [
            r"retryDelay\"\s*:\s*\"(?P<seconds>\d+(?:\.\d+)?)s\"",
            r"Please retry in (?P<seconds>\d+(?:\.\d+)?)s",
            r"retry(?:-|_)?after[^\d]*(?P<seconds>\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_text, flags=re.IGNORECASE)
            if match:
                return float(match.group("seconds")) + 1.0
        return None

    @classmethod
    def _as_prompt_payloads(cls, prompt: Any) -> list[Any]:
        if isinstance(prompt, str):
            return [prompt]

        if isinstance(prompt, Mapping):
            if "messages" in prompt:
                return [prompt["messages"]]
            if "prompt" in prompt:
                return [prompt["prompt"]]
            raise TypeError(
                "Prompt mappings must contain either 'prompt' or 'messages'."
            )

        if cls._is_message_sequence(prompt):
            return [prompt]

        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
            payloads = []
            for item in prompt:
                if isinstance(item, Mapping):
                    if "messages" in item:
                        payloads.append(item["messages"])
                    elif "prompt" in item:
                        payloads.append(item["prompt"])
                    else:
                        raise TypeError(
                            "Prompt job mappings must contain either "
                            "'prompt' or 'messages'."
                        )
                elif isinstance(item, str):
                    payloads.append(item)
                else:
                    raise TypeError(
                        "Prompt sequences must contain strings or mappings."
                    )
            return payloads

        raise TypeError(
            "prompt must be a string, a messages sequence, a prompt mapping, "
            "or a sequence of prompt jobs."
        )

    @staticmethod
    def _is_message_sequence(prompt: Any) -> bool:
        if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
            return False
        return all(
            isinstance(item, Mapping)
            and "role" in item
            and "content" in item
            for item in prompt
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except AttributeError:
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise RuntimeError(
                    f"Unexpected LiteLLM response format: {response!r}"
                ) from error
        except (IndexError, TypeError) as error:
            raise RuntimeError(
                f"Unexpected LiteLLM response format: {response!r}"
            ) from error

        return "" if content is None else str(content)
