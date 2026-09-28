"""Tests for the backend interface."""

import pytest
from src.backends import ModelResponse, LMStudioBackend, AnthropicBackend, create_backend, _split_reasoning


class TestModelResponse:
    def test_basic_properties(self):
        r = ModelResponse(text="hello world", model="test", backend="test")
        assert r.text == "hello world"
        assert not r.is_empty

    def test_empty_response(self):
        r = ModelResponse(text="", model="test", backend="test")
        assert r.is_empty

    def test_whitespace_is_empty(self):
        r = ModelResponse(text="   \n  ", model="test", backend="test")
        assert r.is_empty

    def test_token_count_estimate(self):
        r = ModelResponse(text="one two three four", model="test", backend="test")
        # 4 words * 4/3 ≈ 5
        assert r.token_count_estimate > 0

    def test_metadata_default(self):
        r = ModelResponse(text="hi", model="test", backend="test")
        assert r.metadata == {}


class TestCreateBackend:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("nonexistent")

    def test_connection_failure_raises_runtime_error(self, monkeypatch):
        import requests
        def offline(*args, **kwargs):
            raise requests.ConnectionError("offline")
        monkeypatch.setattr(requests, "get", offline)
        with pytest.raises(RuntimeError, match="offline"):
            LMStudioBackend()


class TestSplitReasoning:
    def test_no_think_block(self):
        text, reasoning = _split_reasoning("Just an answer.")
        assert text == "Just an answer."
        assert reasoning == ""

    def test_think_block_extracted(self):
        text, reasoning = _split_reasoning("<think>pondering the void</think>The answer.")
        assert text == "The answer."
        assert reasoning == "pondering the void"

    def test_multiple_think_blocks(self):
        text, reasoning = _split_reasoning("<think>one</think>A.<think>two</think>B.")
        assert text == "A.B."
        assert "one" in reasoning and "two" in reasoning


class TestDockerBackend:
    def setup_backend(self, monkeypatch, completion=None):
        import requests
        from unittest.mock import Mock
        from src.backends import DockerModelRunnerBackend
        model = {"id": "docker.io/ai/gpt-oss:20B", "dmr": {"quantization": "Q4"}}
        get = Mock(return_value=Mock(json=lambda: {"data": [model]}))
        post = Mock(return_value=Mock(json=lambda: completion or {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}))
        monkeypatch.setattr(requests, "get", get)
        monkeypatch.setattr(requests, "post", post)
        backend = DockerModelRunnerBackend(model="ai/gpt-oss:20B", base_url="http://localhost:12434/engines/v1", max_tokens=123)
        return backend, get, post

    def test_routes_alias_and_full_metadata(self, monkeypatch):
        backend, get, post = self.setup_backend(monkeypatch)
        response = backend.query("Question", "System", 0.2)
        assert backend.model == "docker.io/ai/gpt-oss:20B"
        assert get.call_args.args[0].endswith("/engines/v1/models")
        assert post.call_args.args[0].endswith("/engines/v1/chat/completions")
        request = post.call_args.kwargs["json"]
        assert request["max_tokens"] == 123
        assert request["temperature"] == 0.2
        assert request["messages"][0]["content"] == "System"
        assert response.backend == "docker"
        assert response.metadata["raw_response"]["choices"]
        assert response.metadata["model_info"]["dmr"]["quantization"] == "Q4"

    def test_unclosed_reasoning_is_not_visible_answer(self, monkeypatch):
        backend, _, _ = self.setup_backend(monkeypatch, {"choices": [{"message": {"content": "<think>unfinished"}, "finish_reason": "length"}]})
        response = backend.query("q")
        assert response.text == ""
        assert response.metadata["reasoning"] == "unfinished"
        assert response.metadata["raw_response"]["choices"][0]["message"]["content"] == "<think>unfinished"

    def test_missing_choices_is_transport_error(self, monkeypatch):
        backend, _, _ = self.setup_backend(monkeypatch, {"error": "load failed"})
        with pytest.raises(RuntimeError, match="missing choices"):
            backend.query("q")

    def test_timeout_is_actionable(self, monkeypatch):
        import requests
        backend, _, post = self.setup_backend(monkeypatch)
        post.side_effect = requests.Timeout("timed out")
        with pytest.raises(RuntimeError, match="Docker"):
            backend.query("q")

    def test_reasoning_fields_not_lost(self, monkeypatch):
        backend, _, _ = self.setup_backend(monkeypatch, {"choices": [{"message": {"content": "<think>inline</think>answer", "reasoning": "trace one", "reasoning_content": "trace two"}}]})
        response = backend.query("q")
        assert response.text == "answer"
        assert set(response.metadata["reasoning_sources"].values()) == {"inline", "trace one", "trace two"}
