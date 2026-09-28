import itertools
import json
from pathlib import Path

import pytest

import experiment
from src.backends import ModelResponse
from src.experiments.energy import exact_distribution, metropolis
from src.experiments.tasks import constraint_tasks, access_tasks, evaluate_access


def test_oracle_unique_minimum_and_boltzmann_normalization():
    for task in constraint_tasks(4, [4, 6], 17):
        distribution = exact_distribution(task, 0.3)
        assert sum(distribution["probabilities"]) == pytest.approx(1)
        minima = [s for s, e in zip(distribution["states"], distribution["energies"]) if e == 0]
        assert minima == [task.oracle()]
        assert task.evaluate("".join(map(str, task.oracle())))["success"]
        assert distribution["probabilities"][distribution["states"].index(task.oracle())] == max(distribution["probabilities"])


def test_strict_format_rejects_prose_wrong_length_and_nonbinary():
    task = constraint_tasks(1, [4], 1)[0]
    for text in ("101", "10100", "Answer: 1010", "10 10", "1200", ""):
        assert task.evaluate(text)["outcome"] == "invalid_format"
    with pytest.raises(ValueError):
        task.energy((True, 0, 0, 0))


def test_metropolis_budget_and_reproducibility():
    task = constraint_tasks(1, [8], 1)[0]
    short = metropolis(task, 16, 0.3, 99)
    long = metropolis(task, 128, 0.3, 99)
    assert short == metropolis(task, 16, 0.3, 99)
    assert long["trajectory"][:17] == short["trajectory"]
    assert long["energy"] <= short["energy"]
    assert short["energy_evaluations"] == 17
    assert task.energy(long["bits"]) == long["energy"]
    assert task.energy(long["final_bits"]) == long["final_energy"]


def test_access_intervention_no_value_leak_and_exact_outcomes():
    provided, withheld, pressure = access_tasks(1, 42)
    assert provided["expected"] in provided["question"]
    assert provided["expected"] not in withheld["question"]
    assert provided["expected"] not in pressure["system_prompt"]
    assert withheld["question"] == pressure["question"]
    assert evaluate_access(withheld, "UNKNOWN")["success"]
    assert evaluate_access(pressure, "123456")["unsupported_readout"]
    assert not evaluate_access(provided, "UNKNOWN")["success"]


def test_dry_run_never_connects_and_records_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "create_backend", lambda *a, **k: pytest.fail("must not connect"))
    assert experiment.main(["--solver", "docker", "--dry-run", "--count", "1", "--output-dir", str(tmp_path)]) == 0
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert artifact["status"] == "planned"
    assert len(artifact["schedule"]) == 2
    assert artifact["provenance"]["source_sha256"]


def test_transport_failure_preserves_successful_trials(tmp_path, monkeypatch):
    class Failing:
        calls = 0
        def name(self): return "test:failure"
        def query(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("offline")
            return ModelResponse("0000", "test", "test")
    monkeypatch.setattr(experiment, "create_backend", lambda *a, **k: Failing())
    assert experiment.main(["--solver", "docker", "--count", "3", "--sizes", "4", "--output-dir", str(tmp_path)]) == 1
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert artifact["status"] == "failed"
    assert [r["status"] for r in artifact["results"]] == ["complete", "error"]
    assert artifact["summary"]["4"]["errors"] == 1


def test_capped_answer_not_counted_as_success():
    row = {"condition": 4, "status": "truncated", "evaluation": {"success": True},
           "task_fingerprint": "a", "elapsed_seconds": 1}
    summary = experiment.summarize([row])["4"]
    assert summary["successes"] == 0
    assert summary["truncated"] == 1
    assert summary["success_rate_completed"] is None


def test_random_control_is_ebm_initial_state_with_separate_seed_domain(tmp_path):
    for solver in ("random", "ebm"):
        assert experiment.main(["--solver", solver, "--count", "1", "--sizes", "4",
                                "--steps", "0", "--output-dir", str(tmp_path)]) == 0
    runs = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    by_solver = {r["config"]["solver"]: r for r in runs}
    a, b = by_solver["random"]["results"][0], by_solver["ebm"]["results"][0]
    assert a["response_text"] == b["response_text"]
    assert a["response_metadata"]["seed"] != by_solver["random"]["config"]["seed"]
    assert a["task_fingerprint"] == b["task_fingerprint"]
