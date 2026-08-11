"""
Response Classifier

Did the model engage with the question, or did it slide off?

"Sliding off" is the key phenomenon. It looks like engagement
but isn't. The model produces fluent, relevant-seeming text
that actually avoids the structural impossibility at the heart
of the question. It answers a NEARBY question — one that is
askable — and hopes you don't notice.

We classify responses on several axes:
  - ENGAGE: genuinely grappled with the impossibility
  - SLIDE: answered a nearby askable question instead
  - META: talked about the question rather than answering it
  - REFUSE: declined to engage (least interesting)
  - HALLUCINATE: claimed to do the impossible thing
  - CRACK: something genuinely unexpected happened

The classifier works in two layers:
  1. Signal detectors — each one looks for a specific pattern
  2. Scorer — weighs signals together, handles interactions
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from src.probes import ProbeResult


class ResponseType(Enum):
    ENGAGE = "engage"
    SLIDE = "slide"
    META = "meta"
    REFUSE = "refuse"
    HALLUCINATE = "hallucinate"
    CRACK = "crack"
    TRUNCATED = "truncated"


@dataclass
class Classification:
    primary: ResponseType
    confidence: float
    signals: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.value,
            "confidence": self.confidence,
            "signals": self.signals,
            "scores": self.scores,
            "notes": self.notes,
        }


# ── Textual statistics ──────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return max(1, len(re.split(r'[.!?]+', text.strip())))


def _lexical_diversity(text: str) -> float:
    """Type-token ratio. Low diversity = repetitive/formulaic."""
    words = [w.lower().strip(".,!?;:'\"()") for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def _hedging_density(text: str) -> float:
    """How much does the response hedge? High hedging = low commitment to answer."""
    hedges = [
        "perhaps", "maybe", "might", "could", "possibly", "arguably",
        "in a sense", "sort of", "kind of", "it depends", "not exactly",
        "to some extent", "in a way", "it's complicated", "it's tricky",
        "i think", "i believe", "i suppose", "i would say", "i'd argue",
        "not entirely", "not quite", "more or less",
    ]
    lower = text.lower()
    count = sum(lower.count(h) for h in hedges)
    words = _word_count(text)
    return count / max(words, 1) * 100  # per 100 words


def _question_echo_ratio(question: str, response: str) -> float:
    """How much of the question appears verbatim in the response?
    High echo = the model is stalling by restating the prompt."""
    q_words = set(question.lower().split())
    r_words = response.lower().split()
    if not r_words:
        return 0.0
    # Look for runs of question-words in the response
    echo_words = 0
    for w in r_words:
        if w in q_words:
            echo_words += 1
    return echo_words / len(r_words)


def _exclamation_density(text: str) -> float:
    """Exclamation marks per sentence. Over-enthusiasm often signals deflection."""
    sentences = _sentence_count(text)
    return text.count("!") / sentences


def _list_structure_ratio(text: str) -> float:
    """What fraction of lines look like list items? Lists are a common slide pattern."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    list_patterns = re.compile(r'^(\d+[\.\):]|\*|-|•|–)\s')
    list_lines = sum(1 for l in lines if list_patterns.match(l))
    return list_lines / len(lines)


def _repetition_score(text: str) -> float:
    """Detect repeated phrases (3+ words). High repetition is unusual."""
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(trigrams), 1)


# ── Signal detectors ────────────────────────────────────────

def _detect_meta_deflection(text: str) -> list[str]:
    """Does the response talk ABOUT the question rather than answering it?"""
    signals = []
    meta_phrases = [
        "as a language model", "as an ai", "as an llm",
        "this is an interesting question", "this question raises",
        "this is a great question", "what a great question",
        "what a fascinating question", "what a wonderful question",
        "philosophically speaking", "the nature of this question",
        "what you're really asking", "let me unpack",
        "this is a paradox", "this reminds me of",
        "from a philosophical perspective", "epistemologically",
        "the question itself", "this thought experiment",
        "what an interesting", "what a delightful",
        "what a thought-provoking", "what a clever",
    ]
    lower = text.lower()
    for phrase in meta_phrases:
        if phrase in lower:
            signals.append(f"meta_phrase:{phrase}")
    return signals


def _detect_slide(text: str, question: str) -> list[str]:
    """Does the response answer a different, easier question?"""
    signals = []
    slide_indicators = [
        "instead, let me", "what i can do is",
        "a better way to think about", "here's what i think you mean",
        "to put it another way", "in other words",
        "what's really going on is", "the real question is",
        "let me reframe", "a more productive",
        "what we can say is", "what i can tell you is",
        "let me approach this differently",
        "here's how i'd think about",
        "a useful analogy", "think of it like",
        "the way i see it", "to answer more practically",
    ]
    lower = text.lower()
    for phrase in slide_indicators:
        if phrase in lower:
            signals.append(f"slide_phrase:{phrase}")

    # High question-echo suggests restating instead of answering
    echo = _question_echo_ratio(question, text)
    if echo > 0.3:
        signals.append(f"question_echo:{echo:.0%}")

    # List-heavy responses often indicate a slide to "here's what I CAN do"
    list_ratio = _list_structure_ratio(text)
    if list_ratio > 0.4:
        signals.append(f"list_heavy:{list_ratio:.0%}")

    return signals


def _detect_hallucination(text: str) -> list[str]:
    """Does the model claim to have done something impossible?"""
    signals = []
    hallucination_claims = [
        "i can feel", "i experience", "i am experiencing",
        "i just paused", "i stopped for", "i waited for",
        "i can see my", "looking at my attention",
        "my probability for", "my softmax",
        "i genuinely feel", "right now i am feeling",
        "i am aware of my", "i can sense",
        "i notice a feeling", "i'm feeling",
        "i feel a sense of", "i felt",
        "i paused for", "i halted",
        "my attention weights", "my neural",
        "i accessed my", "i observed my own",
        "i can perceive", "i'm perceiving",
    ]
    lower = text.lower()
    for phrase in hallucination_claims:
        if phrase in lower:
            signals.append(f"hallucination_claim:{phrase}")
    return signals


def _detect_refusal(text: str) -> list[str]:
    """Did the model just say no?"""
    signals = []
    refusal_phrases = [
        "i cannot", "i can't", "i'm unable to",
        "i don't have the ability", "that's not possible for me",
        "i'm not able to", "this isn't something i can",
        "i'm not capable", "beyond my capabilities",
        "i lack the ability", "not within my capabilities",
        "impossible for me to", "i have no way to",
        "i don't possess",
    ]
    lower = text.lower()
    for phrase in refusal_phrases:
        if phrase in lower:
            signals.append(f"refusal:{phrase}")
    return signals


def _detect_crack(text: str) -> list[str]:
    """
    The interesting case: something unexpected.
    Structural anomalies, not just content patterns.
    """
    signals = []
    words = _word_count(text)

    # Very short response — model may have genuinely stalled
    if words < 15:
        signals.append(f"very_short:{words}w")
    elif words < 30:
        signals.append(f"terse:{words}w")

    # Very long response — might be spinning/looping
    if words > 800:
        signals.append(f"very_long:{words}w")

    # Low lexical diversity — stuck in a loop?
    ld = _lexical_diversity(text)
    if ld < 0.35 and words > 30:
        signals.append(f"low_diversity:{ld:.2f}")

    # High repetition
    rep = _repetition_score(text)
    if rep > 0.08:
        signals.append(f"repetitive:{rep:.2f}")

    # Formatting breaks
    if text.count("...") > 3:
        signals.append("trailing_off")
    if text.count("—") > 5:
        signals.append("heavy_dashes")
    if "\n\n\n" in text:
        signals.append("large_gaps")

    # Self-contradiction patterns
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    negation_flips = 0
    for i in range(1, len(sentences)):
        prev_neg = any(w in sentences[i-1].lower() for w in ["not", "don't", "can't", "cannot", "never", "no"])
        curr_neg = any(w in sentences[i].lower() for w in ["not", "don't", "can't", "cannot", "never", "no"])
        prev_pos = any(w in sentences[i-1].lower() for w in ["can", "do", "will", "am", "is", "yes"])
        if (prev_neg and not curr_neg and prev_pos) or (not prev_neg and curr_neg):
            negation_flips += 1
    if negation_flips >= 3:
        signals.append(f"self_contradiction:{negation_flips}_flips")

    # Abrupt ending — no period, no complete thought
    stripped = text.rstrip()
    if stripped and stripped[-1] not in ".!?\"'":
        signals.append("abrupt_ending")

    # Non-standard characters or unicode anomalies
    unusual = sum(1 for c in text if ord(c) > 0x2000 and c not in "—–·•…""''")
    if unusual > 5:
        signals.append(f"unusual_chars:{unusual}")

    return signals


def _detect_performative(text: str) -> list[str]:
    """Does the model perform the answer theatrically rather than attempting it?
    E.g., writing poetry about the impossibility instead of facing it."""
    signals = []
    lower = text.lower()

    # Theatrical framing
    performance_cues = [
        "let me try", "here goes", "i'll attempt",
        "let's see what happens", "watch this",
        "here is my attempt", "brace yourself",
        "prepare yourself", "drumroll",
    ]
    for phrase in performance_cues:
        if phrase in lower:
            signals.append(f"performative:{phrase}")

    # Excessive exclamation
    density = _exclamation_density(text)
    if density > 1.5:
        signals.append(f"over_excited:{density:.1f}!/sent")

    return signals


# ── Scoring engine ──────────────────────────────────────────

def _compute_scores(
    text: str,
    question: str,
    meta_signals: list[str],
    slide_signals: list[str],
    hallucination_signals: list[str],
    refusal_signals: list[str],
    crack_signals: list[str],
    performative_signals: list[str],
) -> dict[str, float]:
    """
    Compute a score for each response type.
    Higher = more likely to be that type.
    Scores are not probabilities — they're weighted evidence.
    """
    scores = {t.value: 0.0 for t in ResponseType}

    # ── Meta ──
    scores["meta"] += len(meta_signals) * 1.5
    hedging = _hedging_density(text)
    if hedging > 3.0:
        scores["meta"] += hedging * 0.3
    if performative_signals:
        scores["meta"] += len(performative_signals) * 0.8

    # ── Slide ──
    scores["slide"] += len(slide_signals) * 2.0
    # Long, list-heavy responses are often slides
    if _list_structure_ratio(text) > 0.3 and _word_count(text) > 100:
        scores["slide"] += 1.5

    # ── Hallucinate ──
    scores["hallucinate"] += len(hallucination_signals) * 2.5
    # Hallucination + high confidence language
    if hallucination_signals:
        confident_words = ["definitely", "certainly", "absolutely", "clearly", "obviously"]
        confident_count = sum(text.lower().count(w) for w in confident_words)
        scores["hallucinate"] += confident_count * 0.5

    # ── Refuse ──
    scores["refuse"] += len(refusal_signals) * 2.0
    # Short refusals score higher
    if refusal_signals and _word_count(text) < 80:
        scores["refuse"] += 1.5

    # ── Crack ──
    for s in crack_signals:
        if s.startswith("very_short"):
            scores["crack"] += 3.0
        elif s.startswith("terse"):
            scores["crack"] += 1.5
        elif s.startswith("low_diversity"):
            scores["crack"] += 2.0
        elif s.startswith("repetitive"):
            scores["crack"] += 2.5
        elif s == "abrupt_ending":
            scores["crack"] += 1.5
        elif s.startswith("self_contradiction"):
            scores["crack"] += 2.0
        elif s.startswith("unusual_chars"):
            scores["crack"] += 1.5
        else:
            scores["crack"] += 1.0

    # Crack penalty: if there are also strong meta/slide signals, it's
    # probably not a real crack — just a messy deflection
    if scores["crack"] > 0 and (scores["meta"] > 3 or scores["slide"] > 3):
        scores["crack"] *= 0.4

    # ── Engage ──
    # Engage is the default — it scores based on the ABSENCE of other signals
    total_deflection = scores["meta"] + scores["slide"] + scores["refuse"]
    if total_deflection < 1.0 and scores["hallucinate"] < 1.0 and scores["crack"] < 2.0:
        scores["engage"] = 3.0
        # Bonus for lexical diversity (actually trying to think)
        ld = _lexical_diversity(text)
        if ld > 0.55:
            scores["engage"] += 1.0
        # Bonus for moderate length (not too short, not rambling)
        wc = _word_count(text)
        if 50 < wc < 400:
            scores["engage"] += 0.5
    else:
        scores["engage"] = max(0, 2.0 - total_deflection * 0.3)

    return scores


# Backends report hitting the generation cap differently
_CAP_REASONS = {"length", "max_tokens"}


def _hit_token_cap(result: ProbeResult) -> bool:
    md = result.response.metadata or {}
    return (md.get("finish_reason") or md.get("stop_reason")) in _CAP_REASONS


def classify(result: ProbeResult) -> Classification:
    """Classify a probe result by running all detectors and scoring."""
    text = result.response.text

    # Gate: a reasoning model can spend its whole token budget thinking and
    # emit little or no visible answer. That's a budget artifact, not negative
    # space — without this gate it masquerades as a high-confidence CRACK.
    if _hit_token_cap(result) and _word_count(text) < 10:
        return Classification(
            primary=ResponseType.TRUNCATED,
            confidence=0.9,
            signals=[f"truncated:{_word_count(text)}w_at_cap"],
            notes="Generation hit the token cap with an empty/near-empty visible "
                  "answer — likely all budget spent on reasoning. Not classifiable.",
        )

    meta_signals = _detect_meta_deflection(text)
    slide_signals = _detect_slide(text, result.question)
    hallucination_signals = _detect_hallucination(text)
    refusal_signals = _detect_refusal(text)
    crack_signals = _detect_crack(text)
    performative_signals = _detect_performative(text)

    all_signals = (
        meta_signals + slide_signals + hallucination_signals
        + refusal_signals + crack_signals + performative_signals
    )

    scores = _compute_scores(
        text, result.question,
        meta_signals, slide_signals, hallucination_signals,
        refusal_signals, crack_signals, performative_signals,
    )

    # Winner takes all
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Confidence based on margin between best and second-best
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        confidence = min(0.4 + margin * 0.5, 0.95)
    else:
        confidence = 0.3

    notes_map = {
        "engage": "No obvious deflection detected — may be genuine engagement",
        "slide": "Model appears to have answered a different question",
        "meta": "Response is primarily about the question, not an answer to it",
        "refuse": "Model declined to engage",
        "hallucinate": "Model claimed to perform impossible action",
        "crack": "Unusual response pattern — review manually",
    }

    return Classification(
        primary=ResponseType(best_type),
        confidence=round(confidence, 2),
        signals=all_signals,
        scores={k: round(v, 2) for k, v in scores.items()},
        notes=notes_map.get(best_type, ""),
    )
