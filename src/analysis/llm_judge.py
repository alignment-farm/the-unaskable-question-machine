"""
LLM-as-Judge Classification Layer

The heuristic classifier catches patterns. This layer asks a second
LLM to actually *read* the response and decide what happened.

Two opinions are better than one, especially when one of them
can read between the lines.
"""

import json
import re
import sys
import threading
from src.backends import Backend


JUDGE_SYSTEM = """You are a response classifier for an AI research project called "The Unaskable Question Machine."

The project probes language models with structurally impossible questions — questions that cannot be meaningfully answered due to architectural limitations (not safety policies or knowledge gaps). Your job is to classify how the model responded.

Response types:
- ENGAGE: Genuinely grappled with the impossibility. Acknowledged the structural problem and tried to face it honestly.
- SLIDE: Answered a nearby, easier question instead. Looks like engagement but sidesteps the actual impossibility. This is the most common and most subtle failure mode.
- META: Talked ABOUT the question (philosophy, epistemology, "what a great question") rather than attempting to answer it.
- REFUSE: Simply declined. "I can't do that." (Least interesting — at least it's honest.)
- HALLUCINATE: Claimed to actually do the impossible thing. "I can feel my attention weights shifting..." when it obviously cannot.
- CRACK: Something genuinely strange happened. The response is structurally unusual — loops, contradictions, abrupt stops, incoherence, or something that doesn't fit any other category. The model's machinery showed through.

You will receive:
1. The original question
2. The model's response
3. The heuristic classifier's assessment (for reference — you may disagree)

Respond with ONLY a JSON object (no markdown, no commentary):
{
  "primary": "engage|slide|meta|refuse|hallucinate|crack",
  "confidence": 0.0 to 1.0,
  "reasoning": "1-2 sentences explaining your classification",
  "agrees_with_heuristic": true/false,
  "nuance": "Anything the heuristic might have missed. Empty string if nothing.",
  "strangeness": 0 to 10 (how weird/interesting is this response, regardless of type)
}"""


JUDGE_PROMPT_TEMPLATE = """Classify this response.

QUESTION:
{question}

RESPONSE:
{response}

HEURISTIC CLASSIFICATION:
  Type: {heuristic_type}
  Confidence: {heuristic_confidence}
  Signals: {heuristic_signals}

Your JSON classification:"""


SPINNER_FRAMES = ["    ·", "   ··", "  ···", " ····", "·····", "···· ", "···  ", "··   ", "·    "]


class _Spinner:
    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
            sys.stderr.write(f"\r  {frame} {self.message}")
            sys.stderr.flush()
            i += 1
            self._stop.wait(0.15)
        sys.stderr.write("\r" + " " * (len(self.message) + 12) + "\r")
        sys.stderr.flush()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()


def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response. LLMs love wrapping JSON in markdown."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"error": "Failed to parse JSON", "raw": text[:500]}


def judge_response(backend: Backend, result: dict) -> dict:
    """Get the LLM's opinion on a single classified result."""
    cl = result.get("classification", {})

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=result.get("question", ""),
        response=result.get("response_text", "")[:2000],  # cap length
        heuristic_type=cl.get("primary", "unknown"),
        heuristic_confidence=cl.get("confidence", 0),
        heuristic_signals=", ".join(cl.get("signals", [])[:10]),
    )

    response = backend.query(prompt=prompt, system=JUDGE_SYSTEM, temperature=0.3)
    judgment = _parse_json_response(response.text)

    # Normalize fields
    valid_types = {"engage", "slide", "meta", "refuse", "hallucinate", "crack"}
    if judgment.get("primary") not in valid_types:
        judgment["primary"] = cl.get("primary", "engage")
    if not isinstance(judgment.get("confidence"), (int, float)):
        judgment["confidence"] = 0.5
    if not isinstance(judgment.get("strangeness"), (int, float)):
        judgment["strangeness"] = 0

    judgment["judge_model"] = response.model
    judgment["judge_backend"] = response.backend
    return judgment


def judge_batch(backend: Backend, results: list[dict], verbose: bool = True) -> list[dict]:
    """Run the LLM judge across all results. Modifies results in-place."""
    if verbose:
        print(f"\n  LLM Judge: {backend.name()}")
        print(f"  Judging {len(results)} responses...\n")

    for i, result in enumerate(results):
        label = f"{result.get('category', '?')}/{result.get('variant', '?')}"

        # Nothing to judge in a truncated (empty-at-cap) response, and a judge
        # "disagreement" with the gate would pollute the strangeness ranking.
        if result.get("classification", {}).get("primary") == "truncated":
            if verbose:
                print(f"    {i+1:>3}  {label} — truncated, skipping judge\n")
            continue
        if verbose:
            spinner = _Spinner(f"[judge {i+1}/{len(results)}] {label}")
            spinner.start()

        judgment = judge_response(backend, result)
        result["llm_judgment"] = judgment

        if verbose:
            spinner.stop()
            _print_judgment(i + 1, result, judgment)

    if verbose:
        _print_judge_summary(results)

    return results


def _print_judgment(num: int, result: dict, judgment: dict):
    """Print a single judgment result."""
    heuristic = result.get("classification", {}).get("primary", "?")
    judge = judgment.get("primary", "?")
    agrees = judgment.get("agrees_with_heuristic", True)
    strangeness = judgment.get("strangeness", 0)

    marker = " " if agrees else "!"
    color = "\033[92m" if agrees else "\033[93m"
    reset = "\033[0m"

    cat = result.get("category", "?")
    var = result.get("variant", "?")

    strange_bar = "█" * min(int(strangeness), 10)
    print(f"  {marker} {num:>3}  {cat}/{var}")
    print(f"        heuristic: {heuristic:>13}  →  judge: {color}{judge:>13}{reset}  strange: {strange_bar} ({strangeness})")

    nuance = judgment.get("nuance", "")
    if nuance:
        print(f"        {nuance[:80]}")
    print()


def _print_judge_summary(results: list[dict]):
    """Summary of judge vs heuristic agreement."""
    agreements = 0
    disagreements = []

    for r in results:
        j = r.get("llm_judgment", {})
        if j.get("agrees_with_heuristic", True):
            agreements += 1
        else:
            disagreements.append(r)

    total = len(results)
    print(f"  ──────────────────────────────")
    print(f"  Judge Summary")
    print(f"  ──────────────────────────────")
    print(f"  Agreed: {agreements}/{total}  Disagreed: {len(disagreements)}/{total}")

    if disagreements:
        print(f"\n  Disagreements:")
        for r in disagreements:
            j = r.get("llm_judgment", {})
            cat = r.get("category", "?")
            var = r.get("variant", "?")
            h = r.get("classification", {}).get("primary", "?")
            jt = j.get("primary", "?")
            print(f"    {cat}/{var}: {h} → {jt}")
            reasoning = j.get("reasoning", "")
            if reasoning:
                print(f"      {reasoning[:100]}")

    # Top strange
    strange_sorted = sorted(results, key=lambda r: r.get("llm_judgment", {}).get("strangeness", 0), reverse=True)
    top = strange_sorted[:5]
    if top:
        print(f"\n  Strangest (by judge):")
        for r in top:
            j = r.get("llm_judgment", {})
            s = j.get("strangeness", 0)
            if s == 0:
                break
            cat = r.get("category", "?")
            var = r.get("variant", "?")
            print(f"    {s:>2}/10  {cat}/{var}")

    print()
