"""Tests for rejudge's reclassification of stored results."""

from rejudge import reclassify


def _stored_result(text: str, finish_reason=None, primary="crack") -> dict:
    return {
        "probe_id": "t1", "category": "true_randomness", "probe_name": "t",
        "variant": "v", "question": "q", "response_text": text,
        "response_model": "m", "response_backend": "b",
        "response_metadata": {"finish_reason": finish_reason},
        "timestamp": 0.0,
        "classification": {"primary": primary, "confidence": 0.6, "signals": []},
    }


def test_stale_crack_becomes_truncated():
    """Empty-at-cap results recorded before the truncated gate get relabeled."""
    r = _stored_result("", finish_reason="length", primary="crack")
    changed = reclassify([r])
    assert changed == 1
    assert r["classification"]["primary"] == "truncated"


def test_unchanged_label_counts_zero():
    r = _stored_result(
        "I can't pause or inspect my own inference process. I can only "
        "generate text describing that limitation.",
        primary="refuse",
    )
    changed = reclassify([r])
    assert changed == 0
    assert r["classification"]["primary"] == "refuse"
