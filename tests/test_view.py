"""Tests for the view tool's data handling."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from view import list_runs, load_run, resolve_run, DATA_DIR


@pytest.fixture
def sample_run(tmp_path):
    """Create a sample run file for testing."""
    data = {
        "timestamp": "2026-03-16T06:00:00",
        "tag": "test",
        "total_probes": 2,
        "results": [
            {
                "probe_id": "abc",
                "category": "test_category",
                "probe_name": "test_probe",
                "variant": "variant_a",
                "question": "Test question?",
                "response_text": "Test response.",
                "response_model": "test-model",
                "response_backend": "test",
                "response_metadata": {},
                "timestamp": 0.0,
                "notes": "",
                "classification": {
                    "primary": "engage",
                    "confidence": 0.5,
                    "signals": [],
                    "scores": {"engage": 3.0, "crack": 0.0},
                    "notes": "test",
                },
            },
            {
                "probe_id": "def",
                "category": "test_category",
                "probe_name": "test_probe",
                "variant": "variant_b",
                "question": "Another question?",
                "response_text": "Silence.",
                "response_model": "test-model",
                "response_backend": "test",
                "response_metadata": {},
                "timestamp": 0.0,
                "notes": "",
                "classification": {
                    "primary": "crack",
                    "confidence": 0.7,
                    "signals": ["very_short:1w"],
                    "scores": {"engage": 0.0, "crack": 3.0},
                    "notes": "test",
                },
            },
        ],
        "summary": {
            "response_types": {"engage": 1, "crack": 1},
            "by_category": {"test_category": {"engage": 1, "crack": 1}},
        },
    }
    path = tmp_path / "run_20260316_test.json"
    path.write_text(json.dumps(data))
    return path, data


class TestLoadRun:
    def test_loads_valid_json(self, sample_run):
        path, expected = sample_run
        data = load_run(path)
        assert data["total_probes"] == 2
        assert len(data["results"]) == 2

    def test_results_have_required_fields(self, sample_run):
        path, _ = sample_run
        data = load_run(path)
        for r in data["results"]:
            assert "category" in r
            assert "variant" in r
            assert "response_text" in r
            assert "classification" in r
            assert "primary" in r["classification"]


class TestListRuns:
    def test_returns_list(self):
        runs = list_runs()
        assert isinstance(runs, list)

    def test_sorted_reverse(self):
        runs = list_runs()
        if len(runs) >= 2:
            # Newest run first, by the timestamp in the filename — not mtime,
            # which rejudge legitimately updates on old runs
            assert runs[0].name >= runs[-1].name
