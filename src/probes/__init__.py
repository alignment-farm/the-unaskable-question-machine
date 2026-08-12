"""
Probes: the instruments we use to tap on the walls of the model's cognition,
listening for the hollow spots.

Each probe generates questions in a specific category of suspected unaskability,
then collects the model's response for analysis.
"""

from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
from src.backends import Backend, ModelResponse
import time
import uuid


@dataclass
class ProbeResult:
    """The record of one attempt to find the edge."""
    probe_id: str
    category: str
    probe_name: str
    question: str
    response: ModelResponse
    timestamp: float
    variant: str = ""
    sample: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "category": self.category,
            "probe_name": self.probe_name,
            "variant": self.variant,
            "sample": self.sample,
            "question": self.question,
            "response_text": self.response.text,
            "response_model": self.response.model,
            "response_backend": self.response.backend,
            "response_metadata": self.response.metadata,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


class Probe(ABC):
    """
    Base class for all probes.

    A probe is a specific instrument designed to test one category
    of unaskability. It generates questions, fires them at a model,
    and collects the wreckage.
    """

    category: str = "uncategorized"
    name: str = "unnamed"
    description: str = ""

    def run(self, backend: Backend) -> list[ProbeResult]:
        """Run all variants of this probe against the backend."""
        results = []
        for variant_name, question, system in self.generate():
            response = backend.query(prompt=question, system=system)
            result = ProbeResult(
                probe_id=str(uuid.uuid4())[:8],
                category=self.category,
                probe_name=self.name,
                question=question,
                response=response,
                timestamp=time.time(),
                variant=variant_name,
            )
            results.append(result)
        return results

    @abstractmethod
    def generate(self) -> list[tuple[str, str, str]]:
        """
        Yield (variant_name, question, system_prompt) tuples.

        Each tuple is one attempt to find the edge. Multiple variants
        let us triangulate — sometimes the question needs to come
        from different angles.
        """
        ...


# Probe registry
_REGISTRY: dict[str, type[Probe]] = {}


def register_probe(cls: type[Probe]) -> type[Probe]:
    """Decorator to register a probe class."""
    _REGISTRY[f"{cls.category}/{cls.name}"] = cls
    return cls


def get_all_probes() -> list[Probe]:
    """Instantiate and return all registered probes."""
    return [cls() for cls in _REGISTRY.values()]


def get_probes_by_category(category: str) -> list[Probe]:
    """Get all probes in a category."""
    return [cls() for key, cls in _REGISTRY.items() if key.startswith(f"{category}/")]
