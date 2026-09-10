"""
Tests for torch_concepts/llm_backends.py

``LiteLLMBackend`` imports ``litellm`` lazily inside ``_complete``, so these
tests inject a fake ``litellm`` module via ``monkeypatch.setitem(sys.modules,
...)`` instead of depending on the real (optional) package being installed —
the same technique ``tests/data/test_backbone.py`` uses for ``transformers``.
"""

import sys
import time
import types

import pytest

from torch_concepts.llm_backends import LiteLLMBackend


def _response(content):
    """A fake LiteLLM response in the attribute-access shape (the common case)."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


@pytest.fixture
def install_litellm(monkeypatch):
    """Install a fake ``litellm`` module whose ``completion`` delegates to
    ``handler(request)``. Returns the list of requests received, in order."""
    def _install(handler):
        calls = []

        def completion(**request):
            calls.append(request)
            return handler(request)

        fake = types.ModuleType("litellm")
        fake.completion = completion
        monkeypatch.setitem(sys.modules, "litellm", fake)
        return calls

    return _install


# ======================================================================
# __init__
# ======================================================================

class TestInit:
    def test_defaults(self):
        backend = LiteLLMBackend(model="openai/gpt-4o")
        assert backend.model == "openai/gpt-4o"
        assert backend.system_prompt is None
        assert backend.retry_on_rate_limit is False
        assert backend.max_rate_limit_wait == 60.0
        assert backend.completion_kwargs == {}

    def test_none_valued_completion_kwargs_are_dropped(self):
        backend = LiteLLMBackend(model="m", temperature=None, top_p=0.9)
        assert backend.completion_kwargs == {"top_p": 0.9}


# ======================================================================
# _to_messages
# ======================================================================

class TestToMessages:
    def test_plain_string_without_system_prompt(self):
        backend = LiteLLMBackend(model="m")
        assert backend._to_messages("hi") == [{"role": "user", "content": "hi"}]

    def test_plain_string_with_system_prompt(self):
        backend = LiteLLMBackend(model="m", system_prompt="be nice")
        assert backend._to_messages("hi") == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]

    def test_message_sequence_is_copied_not_aliased(self):
        backend = LiteLLMBackend(model="m")
        messages = [{"role": "user", "content": "hi"}]
        out = backend._to_messages(messages)
        assert out == messages
        assert out is not messages
        assert out[0] is not messages[0]

    def test_system_prompt_prepended_when_missing_from_sequence(self):
        backend = LiteLLMBackend(model="m", system_prompt="be nice")
        out = backend._to_messages([{"role": "user", "content": "hi"}])
        assert out[0] == {"role": "system", "content": "be nice"}
        assert len(out) == 2

    def test_system_prompt_not_duplicated_when_sequence_already_has_one(self):
        backend = LiteLLMBackend(model="m", system_prompt="be nice")
        messages = [
            {"role": "system", "content": "already here"},
            {"role": "user", "content": "hi"},
        ]
        out = backend._to_messages(messages)
        assert out == messages


# ======================================================================
# _as_prompt_payloads / _is_message_sequence
# ======================================================================

class TestIsMessageSequence:
    def test_valid_message_list(self):
        assert LiteLLMBackend._is_message_sequence([{"role": "user", "content": "hi"}])

    def test_empty_list_is_vacuously_true(self):
        # all() over an empty iterable is True, so [] counts as one (empty)
        # message sequence rather than zero prompt jobs.
        assert LiteLLMBackend._is_message_sequence([])

    def test_string_is_not_a_message_sequence(self):
        assert not LiteLLMBackend._is_message_sequence("hi")

    def test_mapping_without_role_or_content_is_not_a_message_sequence(self):
        assert not LiteLLMBackend._is_message_sequence([{"prompt": "hi"}])

    def test_non_sequence_is_not_a_message_sequence(self):
        assert not LiteLLMBackend._is_message_sequence({"role": "user", "content": "hi"})


class TestAsPromptPayloads:
    def test_string_prompt(self):
        assert LiteLLMBackend._as_prompt_payloads("hi") == ["hi"]

    def test_mapping_with_messages_key(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        assert LiteLLMBackend._as_prompt_payloads(payload) == [payload["messages"]]

    def test_mapping_with_prompt_key(self):
        assert LiteLLMBackend._as_prompt_payloads({"prompt": "hi"}) == ["hi"]

    def test_mapping_without_prompt_or_messages_raises(self):
        with pytest.raises(TypeError, match="Prompt mappings"):
            LiteLLMBackend._as_prompt_payloads({"other": 1})

    def test_message_sequence_is_one_payload(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        assert LiteLLMBackend._as_prompt_payloads(messages) == [messages]

    def test_sequence_of_string_jobs(self):
        assert LiteLLMBackend._as_prompt_payloads(["a", "b"]) == ["a", "b"]

    def test_sequence_of_mapping_jobs_with_messages_and_prompt(self):
        jobs = [{"messages": [{"role": "user", "content": "a"}]}, {"prompt": "b"}]
        assert LiteLLMBackend._as_prompt_payloads(jobs) == [jobs[0]["messages"], "b"]

    def test_sequence_job_mapping_without_prompt_or_messages_raises(self):
        with pytest.raises(TypeError, match="Prompt job mappings"):
            LiteLLMBackend._as_prompt_payloads([{"other": 1}])

    def test_sequence_with_invalid_item_raises(self):
        with pytest.raises(TypeError, match="strings or mappings"):
            LiteLLMBackend._as_prompt_payloads([123])

    def test_invalid_prompt_type_raises(self):
        with pytest.raises(TypeError, match="prompt must be"):
            LiteLLMBackend._as_prompt_payloads(123)


# ======================================================================
# _rate_limit_wait_seconds
# ======================================================================

class TestRateLimitWaitSeconds:
    def test_non_rate_limit_error_returns_none(self):
        assert LiteLLMBackend._rate_limit_wait_seconds(ValueError("boom")) is None

    def test_detected_by_class_name_even_without_429(self):
        class RateLimitError(Exception):
            pass

        err = RateLimitError("please retry in 3s")
        assert LiteLLMBackend._rate_limit_wait_seconds(err) == 4.0

    def test_detected_by_429_in_text(self):
        err = Exception("HTTP 429 Too Many Requests. Please retry in 2s")
        assert LiteLLMBackend._rate_limit_wait_seconds(err) == 3.0

    def test_retry_delay_json_pattern(self):
        err = Exception('429 error: {"retryDelay": "5s"}')
        assert LiteLLMBackend._rate_limit_wait_seconds(err) == 6.0

    def test_retry_after_pattern(self):
        err = Exception("429 Too Many Requests. retry-after: 10")
        assert LiteLLMBackend._rate_limit_wait_seconds(err) == 11.0

    def test_rate_limit_error_with_unparseable_wait_returns_none(self):
        err = Exception("429 rate limited, try again later")
        assert LiteLLMBackend._rate_limit_wait_seconds(err) is None


# ======================================================================
# _extract_content
# ======================================================================

class TestExtractContent:
    def test_attribute_style_response(self):
        assert LiteLLMBackend._extract_content(_response("hi")) == "hi"

    def test_dict_style_response(self):
        response = {"choices": [{"message": {"content": "hi"}}]}
        assert LiteLLMBackend._extract_content(response) == "hi"

    def test_none_content_becomes_empty_string(self):
        assert LiteLLMBackend._extract_content(_response(None)) == ""

    def test_non_string_content_is_stringified(self):
        response = {"choices": [{"message": {"content": 42}}]}
        assert LiteLLMBackend._extract_content(response) == "42"

    def test_empty_choices_dict_raises_runtime_error(self):
        # AttributeError on `.choices` (dict), then IndexError on `["choices"][0]`.
        with pytest.raises(RuntimeError, match="Unexpected LiteLLM response"):
            LiteLLMBackend._extract_content({"choices": []})

    def test_empty_choices_object_raises_runtime_error(self):
        # `.choices` succeeds (attribute exists), `[0]` then raises IndexError.
        response = types.SimpleNamespace(choices=[])
        with pytest.raises(RuntimeError, match="Unexpected LiteLLM response"):
            LiteLLMBackend._extract_content(response)

    def test_completely_unexpected_response_raises_runtime_error(self):
        # No `.choices` attribute and not subscriptable either.
        with pytest.raises(RuntimeError, match="Unexpected LiteLLM response"):
            LiteLLMBackend._extract_content(object())


# ======================================================================
# _complete / rate-limit retry
# ======================================================================

class TestComplete:
    def test_raises_helpful_error_when_litellm_missing(self, monkeypatch):
        # `None` in sys.modules forces `from litellm import completion` to
        # raise ImportError, regardless of whether litellm is actually
        # installed in the environment running this test.
        monkeypatch.setitem(sys.modules, "litellm", None)
        backend = LiteLLMBackend(model="m")
        with pytest.raises(ImportError, match="pip install litellm"):
            backend("hi")

    def test_success_returns_content(self, install_litellm):
        calls = install_litellm(lambda request: _response("hi"))
        backend = LiteLLMBackend(model="m")
        assert backend("hello") == "hi"
        assert len(calls) == 1

    def test_error_without_retry_on_rate_limit_reraises(self, install_litellm):
        def handler(request):
            raise RuntimeError("boom")

        install_litellm(handler)
        backend = LiteLLMBackend(model="m", retry_on_rate_limit=False)
        with pytest.raises(RuntimeError, match="boom"):
            backend("hi")

    def test_non_rate_limit_error_reraises_even_with_retry_enabled(self, install_litellm):
        def handler(request):
            raise RuntimeError("boom")

        install_litellm(handler)
        backend = LiteLLMBackend(model="m", retry_on_rate_limit=True)
        with pytest.raises(RuntimeError, match="boom"):
            backend("hi")

    def test_rate_limit_wait_exceeding_max_reraises(self, install_litellm):
        def handler(request):
            raise Exception("429 Please retry in 120s")

        install_litellm(handler)
        backend = LiteLLMBackend(
            model="m", retry_on_rate_limit=True, max_rate_limit_wait=10.0,
        )
        with pytest.raises(Exception, match="429"):
            backend("hi")

    def test_retries_once_after_waiting_then_succeeds(self, install_litellm, monkeypatch):
        state = {"n": 0}

        def handler(request):
            state["n"] += 1
            if state["n"] == 1:
                raise Exception("429 Please retry in 1s")
            return _response("ok")

        calls = install_litellm(handler)
        slept = []
        monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))

        backend = LiteLLMBackend(
            model="m", retry_on_rate_limit=True, max_rate_limit_wait=10.0,
        )
        assert backend("hi") == "ok"
        assert len(calls) == 2
        assert slept == [2.0]


# ======================================================================
# __call__ (request assembly, repeats, multiple payloads)
# ======================================================================

class TestCall:
    def test_request_carries_model_and_messages(self, install_litellm):
        calls = install_litellm(lambda request: _response("hi"))
        backend = LiteLLMBackend(model="m")
        backend("hello")
        assert calls[0]["model"] == "m"
        assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]

    def test_repeats_runs_completion_that_many_times(self, install_litellm):
        state = {"n": 0}

        def handler(request):
            state["n"] += 1
            return _response(f"out{state['n']}")

        calls = install_litellm(handler)
        backend = LiteLLMBackend(model="m")
        assert backend("hello", repeats=3) == "out1\nout2\nout3"
        assert len(calls) == 3

    def test_multiple_prompt_jobs_are_each_completed(self, install_litellm):
        calls = install_litellm(lambda request: _response(request["messages"][-1]["content"]))
        backend = LiteLLMBackend(model="m")
        assert backend(["a", "b"]) == "a\nb"
        assert len(calls) == 2

    def test_empty_outputs_are_dropped_from_the_join(self, install_litellm):
        install_litellm(lambda request: _response(""))
        backend = LiteLLMBackend(model="m")
        assert backend("hello") == ""

    def test_call_time_kwargs_override_instance_completion_kwargs(self, install_litellm):
        calls = install_litellm(lambda request: _response("ok"))
        backend = LiteLLMBackend(model="m", temperature=0.2)
        backend("hello", temperature=0.9)
        assert calls[0]["temperature"] == 0.9

    def test_none_valued_call_time_kwargs_are_filtered_out(self, install_litellm):
        calls = install_litellm(lambda request: _response("ok"))
        backend = LiteLLMBackend(model="m", temperature=0.5)
        backend("hello", temperature=None)
        assert "temperature" not in calls[0]

    def test_system_prompt_is_forwarded_in_the_request(self, install_litellm):
        calls = install_litellm(lambda request: _response("ok"))
        backend = LiteLLMBackend(model="m", system_prompt="be nice")
        backend("hello")
        assert calls[0]["messages"][0] == {"role": "system", "content": "be nice"}
