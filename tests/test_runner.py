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
    assert sorted(indices) == [0, 1, 2]


def test_default_single_sample():
    probe = get_probes_by_category("true_randomness")[0]
    results = run_probe(probe, _EchoBackend(), verbose=False)
    assert len(results) == len(probe.generate())
    assert all(r["sample"] == 0 for r in results)


def test_system_prompt_and_sampling_settings_recorded():
    from src.probes import Probe
    class Example(Probe):
        def generate(self): return [("a", "a question", "pressure instruction")]
    result = run_probe(Example(), _EchoBackend(), verbose=False, temperature=0.2)[0]
    assert result["system_prompt"] == "pressure instruction"
    assert result["generation_temperature"] == 0.2


def test_unique_safe_artifact_names(tmp_path, monkeypatch):
    import src.runner as runner
    import json
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    a = runner.save_results([], "ai/model/../../x")
    b = runner.save_results([], "ai/model/../../x")
    assert a != b and a.parent == b.parent == tmp_path
    assert json.loads(a.read_text())["schema_version"] == 2


def test_checkpoint_keeps_start_time(tmp_path, monkeypatch):
    import json
    import src.runner as runner
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    path = runner.save_results([], status="running")
    initial = json.loads(path.read_text())
    runner.save_results([], path=path, status="failed", error="connection lost")
    final = json.loads(path.read_text())
    assert final["timestamp"] == initial["timestamp"]
    assert final["status"] == "failed"
    assert final["error"] == "connection lost"
