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

    def test_anthropic_without_package(self):
        """Anthropic backend should fail gracefully if package not installed or no key."""
        # This test just verifies the factory doesn't crash before trying to init
        # The actual init may fail due to missing API key, which is expected
        try:
            backend = create_backend("anthropic")
        except (RuntimeError, Exception):
            pass  # Expected — no API key or package

    def test_lmstudio_requires_server(self):
        """LMStudioBackend should give a clear error if server isn't reachable."""
        try:
            # Try connecting to a port that's (likely) not running LM Studio
            backend = LMStudioBackend(base_url="http://localhost:99999/v1")
            pytest.fail("Should have raised RuntimeError")
        except (RuntimeError, Exception):
            pass  # Expected


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
