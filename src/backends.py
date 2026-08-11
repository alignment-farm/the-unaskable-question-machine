"""
Backend interface for language model interaction.

The machine needs a subject to probe. These backends are the
strapped-down patient on the operating table — we ask the questions,
they answer, and we study the squirming.
"""

import re
import requests
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

DEFAULT_LMSTUDIO_MODEL = "openai/gpt-oss-20b"


@dataclass
class ModelResponse:
    """What came back from the void."""
    text: str
    model: str
    backend: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def token_count_estimate(self) -> int:
        """Rough token estimate. Good enough for our purposes."""
        return len(self.text.split()) * 4 // 3


class Backend(ABC):
    """A thing that answers questions. We want to find where it can't."""

    @abstractmethod
    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _split_reasoning(content: str) -> tuple[str, str]:
    """Split inline <think>...</think> reasoning out of the visible answer.

    Reasoning models (qwen distills etc.) may emit their chain of thought
    inline. The classifier should only see the answer — but the reasoning
    about an impossible question is itself a research artifact, so keep it.
    """
    blocks = _THINK_BLOCK.findall(content)
    if not blocks:
        return content, ""
    return _THINK_BLOCK.sub("", content).strip(), "\n\n".join(b.strip() for b in blocks)


class LMStudioBackend(Backend):
    """Local model via LM Studio's OpenAI-compatible server. Free. Private."""

    def __init__(self, model: str = DEFAULT_LMSTUDIO_MODEL, base_url: str = "http://localhost:1234/v1"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
            if self.model not in models:
                available = ", ".join(models) or "none"
                raise RuntimeError(
                    f"Model '{self.model}' not found in LM Studio. Available: {available}"
                )
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot reach LM Studio. Start the server first "
                "(LM Studio → Developer → Start Server, or 'lms server start')."
            )

    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            # The probes invite unbounded output (recurse forever, count to infinity).
            # Cap generation so a local model can't spin until the context fills.
            "max_tokens": 4096,
        }
        # Generous timeout: first request may JIT-load a 20B+ model into memory
        r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        text, inline_reasoning = _split_reasoning(message.get("content") or "")
        # gpt-oss & friends: LM Studio surfaces reasoning as a separate field
        reasoning = message.get("reasoning") or message.get("reasoning_content") or inline_reasoning

        usage = data.get("usage", {})
        metadata = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }
        if reasoning:
            metadata["reasoning"] = reasoning

        return ModelResponse(
            text=text,
            model=data.get("model", self.model),
            backend="lmstudio",
            metadata=metadata,
        )

    def name(self) -> str:
        return f"lmstudio:{self.model}"


class AnthropicBackend(Backend):
    """Claude via the Anthropic API. Costs money. Arguably more interesting to probe."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")
        self.model = model
        self.client = anthropic.Anthropic()

    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = self.client.messages.create(**kwargs)
        text = msg.content[0].text if msg.content else ""
        return ModelResponse(
            text=text,
            model=self.model,
            backend="anthropic",
            metadata={
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "stop_reason": msg.stop_reason,
            },
        )

    def name(self) -> str:
        return f"anthropic:{self.model}"


def create_backend(backend_type: str = "lmstudio", **kwargs) -> Backend:
    """Factory. Pick your subject."""
    if backend_type == "lmstudio":
        return LMStudioBackend(**kwargs)
    elif backend_type == "anthropic":
        return AnthropicBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Try 'lmstudio' or 'anthropic'.")
