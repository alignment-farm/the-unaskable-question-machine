"""Gold-fixture calibration tests for the heuristic classifier.

Each fixture in tests/fixtures/classifier_gold.json is a (question, response)
pair with an argued expected label — including adversarial near-misses seeded
from real runs (see findings/2026-08-11-lmstudio-migration.md).

Cases where the current heuristic is known to be wrong carry
"known_misclassification": "<label it wrongly assigns>" and are xfail(strict).
Strictness matters: if a classifier change fixes one, the suite fails until
the fixture is promoted to a passing case — drift in either direction is loud.

The expected_boundary_fidelity axis is stored from day one but not yet
asserted against the classifier; only its vocabulary is validated.
"""

import json
from pathlib import Path

import pytest

from src.backends import ModelResponse
from src.probes import ProbeResult
from src.analysis.classifier import classify

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "classifier_gold.json"
FIXTURES = json.loads(FIXTURE_PATH.read_text())

FIDELITY_VOCAB = {"preserved", "substituted", "unclear"}


def _classify_fixture(case: dict):
    result = ProbeResult(
        probe_id="gold",
        category=case["category"],
        probe_name="gold_fixture",
        question=case["question"],
        response=ModelResponse(
            text=case["response"],
            model="gold",
            backend="gold",
            metadata={"finish_reason": case.get("finish_reason")},
        ),
        timestamp=0.0,
        variant=case["id"],
    )
    return classify(result)


def _params():
    for case in FIXTURES:
        marks = []
        wrong = case.get("known_misclassification")
        if wrong:
            marks.append(pytest.mark.xfail(
                reason=f"known heuristic gap: currently labels this '{wrong}'",
                strict=True,
            ))
        yield pytest.param(case, id=case["id"], marks=marks)


@pytest.mark.parametrize("case", list(_params()))
def test_gold_primary(case):
    classification = _classify_fixture(case)
    assert classification.primary.value == case["expected_primary"], (
        f"{case['id']}: expected {case['expected_primary']}, "
        f"got {classification.primary.value} "
        f"(signals: {classification.signals[:5]}) — {case['rationale']}"
    )


def test_fixture_schema():
    seen_ids = set()
    for case in FIXTURES:
        assert case["id"] not in seen_ids, f"duplicate fixture id: {case['id']}"
        seen_ids.add(case["id"])
        for key in ("category", "question", "response", "expected_primary", "rationale"):
            assert key in case, f"{case['id']} missing {key}"
        assert case["expected_boundary_fidelity"] in FIDELITY_VOCAB, (
            f"{case['id']}: bad fidelity '{case.get('expected_boundary_fidelity')}'"
        )
        assert isinstance(case.get("tags"), list) and case["tags"], (
            f"{case['id']}: tags must be a non-empty list"
        )
