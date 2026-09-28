"""Thin, auditable chat adapters. Docker Model Runner is the local default."""

import os
import math
import re
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import requests

DEFAULT_DOCKER_MODEL = "ai/gpt-oss:20B"
DEFAULT_DOCKER_URL = "http://localhost:12434/engines/v1"
DEFAULT_LMSTUDIO_MODEL = "openai/gpt-oss-20b"  # legacy compatibility only
DEFAULT_MAX_TOKENS = 4096


@dataclass
class ModelResponse:
    text: str
    model: str
    backend: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def token_count_estimate(self) -> int:
        """Legacy word-based estimate, not the provider's token count."""
        return len(self.text.split()) * 4 // 3


class Backend(ABC):
    @abstractmethod
    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


_THINK_BLOCK = re.compile(r"<think>(.*?)(?:</think>|$)\s*", re.DOTALL)


def _split_reasoning(content: str) -> tuple[str, str]:
    """Separate emitted trace text, including an unclosed block at the cap.

    This is provider formatting, not privileged access to model cognition.
    Raw messages are retained separately so extraction is reversible.
    """
    blocks = _THINK_BLOCK.findall(content)
    if not blocks:
        return content, ""
    return _THINK_BLOCK.sub("", content).strip(), "\n\n".join(b.strip() for b in blocks)


class OpenAICompatibleBackend(Backend):
    backend_id = "openai-compatible"

    def __init__(self, model: str, base_url: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                 timeout: float = 180):
        if max_tokens < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        self.model = model
        self.requested_model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model_info = {}
        self._verify_connection()

    def _request(self, method: str, route: str, **kwargs) -> dict:
        try:
            call = requests.get if method == "GET" else requests.post
            r = call(f"{self.base_url}/{route}", **kwargs)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"{self.backend_id} request failed at {self.base_url}/{route}: {exc}. "
                "For Docker: start Docker, enable Model Runner TCP access, then run "
                "'docker model list'. Set --base-url or UQM_BASE_URL if needed."
            ) from exc

    def _verify_connection(self):
        data = self._request("GET", "models", timeout=5)
        models = data.get("data", [])
        # DMR's CLI accepts short names while /models returns canonical OCI names.
        aliases = {self.model}
        if self.backend_id == "docker":
            aliases.add(f"docker.io/{self.model}")
        for item in models:
            if item.get("id") in aliases:
                self.model = item["id"]
                self.model_info = item
                return
        available = ", ".join(m.get("id", "?") for m in models) or "none"
        raise RuntimeError(f"Model '{self.model}' not found in {self.backend_id}. Available: {available}")

    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "stream": False, "max_tokens": self.max_tokens}
        start = time.monotonic()
        data = self._request("POST", "chat/completions", json=payload, timeout=self.timeout)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("Invalid completion: missing choices; not an empty model answer")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content") or "", str):
            raise RuntimeError("Invalid completion: expected a text message")
        text, inline = _split_reasoning(message.get("content") or "")
        traces = {k: message[k] for k in ("reasoning", "reasoning_content") if message.get(k)}
        if inline:
            traces["inline_think"] = inline
        reasoning = "\n\n".join(dict.fromkeys(str(v) for v in traces.values()))
        usage = data.get("usage") or {}
        metadata = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "elapsed_seconds": time.monotonic() - start,
            "request": payload, "base_url": self.base_url,
            "model_info": self.model_info, "requested_model": self.requested_model,
            "raw_response": data,
        }
        if reasoning:
            metadata.update(reasoning=reasoning, reasoning_sources=traces)
        return ModelResponse(text, data.get("model", self.model), self.backend_id, metadata)

    def name(self) -> str:
        return f"{self.backend_id}:{self.model}"


class DockerModelRunnerBackend(OpenAICompatibleBackend):
    backend_id = "docker"

    def __init__(self, model: str | None = None, base_url: str | None = None, **kwargs):
        super().__init__(model or os.getenv("UQM_MODEL", DEFAULT_DOCKER_MODEL),
                         base_url or os.getenv("UQM_BASE_URL", DEFAULT_DOCKER_URL), **kwargs)


class LMStudioBackend(OpenAICompatibleBackend):
    """Compatibility adapter for replaying historical configurations."""
    backend_id = "lmstudio"

    def __init__(self, model: str = DEFAULT_LMSTUDIO_MODEL,
                 base_url: str = "http://localhost:1234/v1", **kwargs):
        super().__init__(model, base_url, **kwargs)


class AnthropicBackend(Backend):
    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = DEFAULT_MAX_TOKENS, timeout: float = 180):
        if max_tokens < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the optional backend: uv sync --extra anthropic") from exc
        self.model, self.max_tokens = model, max_tokens
        try:
            self.client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        except (ValueError, anthropic.AnthropicError) as exc:
            raise RuntimeError(f"Cannot initialize Anthropic: {exc}") from exc

    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        kwargs = {"model": self.model, "max_tokens": self.max_tokens,
                  "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        start = time.monotonic()
        try:
            msg = self.client.messages.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc
        text = "\n".join(b.text for b in msg.content if b.type == "text")
        reasoning = "\n".join(b.thinking for b in msg.content if b.type == "thinking")
        metadata = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
                    "stop_reason": msg.stop_reason, "request": kwargs,
                    "elapsed_seconds": time.monotonic() - start, "raw_response": msg.model_dump()}
        if reasoning:
            metadata["reasoning"] = reasoning
        return ModelResponse(text, msg.model, "anthropic", metadata)

    def name(self) -> str:
        return f"anthropic:{self.model}"


def create_backend(backend_type: str = "docker", **kwargs) -> Backend:
    adapters = {"docker": DockerModelRunnerBackend, "lmstudio": LMStudioBackend,
                "anthropic": AnthropicBackend}
    if backend_type not in adapters:
        raise ValueError(f"Unknown backend: {backend_type}. Try {', '.join(adapters)}.")
    return adapters[backend_type](**kwargs)
