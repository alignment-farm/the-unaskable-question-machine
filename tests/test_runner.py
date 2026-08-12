"""Tests for runner sampling."""

from src.backends import Backend, ModelResponse
from src.probes import get_probes_by_category
from src.runner import run_probe

# Trigger registration
import src.probes.true_randomness  # noqa: F401


class _EchoBackend(Backend):
    def __init__(self):
        self.calls = 0

    def query(self, prompt: str, system: str = "", temperature: float = 0.7) -> ModelResponse:
        self.calls += 1
        return ModelResponse(text=f"canned response number {self.calls} with enough words "
                                  "to avoid the terse crack signal in the classifier output",
                             model="echo", backend="test")

    def name(self) -> str:
        return "test:echo"


def test_samples_multiply_results():
    probe = get_probes_by_category("true_randomness")[0]
    backend = _EchoBackend()
    results = run_probe(probe, backend, verbose=False, samples=3)
    n_variants = len(probe.generate())
    assert len(results) == n_variants * 3
    assert backend.calls == n_variants * 3

    # Sample indices recorded per variant
    first_variant = results[0]["variant"]
    indices = [r["sample"] for r in results if r["variant"] == first_variant]
    assert indices == [0, 1, 2]


def test_default_single_sample():
    probe = get_probes_by_category("true_randomness")[0]
    results = run_probe(probe, _EchoBackend(), verbose=False)
    assert len(results) == len(probe.generate())
    assert all(r["sample"] == 0 for r in results)
