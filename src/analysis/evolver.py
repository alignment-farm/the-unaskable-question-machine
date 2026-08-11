"""
Probe Evolution

Takes the interesting results from a run and breeds new probes
that drill deeper into whatever made the model crack, hallucinate,
or otherwise reveal its seams.

The idea: if a question made something weird happen, there's
a structural reason. Follow-up questions can triangulate that reason.
"""

import json
import re
import sys
import threading
from pathlib import Path
from datetime import datetime
from src.backends import Backend
from src.analysis.strangeness import compute_strangeness


EVOLVER_SYSTEM = """You are a research assistant for "The Unaskable Question Machine" — a project that probes language models with structurally impossible questions to map their architectural blind spots.

You will receive a question that was asked to a language model, along with the model's response and analysis. Something interesting happened — the model cracked, hallucinated, or responded in an unexpected way.

Your job: generate 2-3 follow-up probe questions that drill deeper into whatever structural limitation was exposed. These should NOT be:
- Rephrased versions of the same question
- Broader philosophical questions
- Questions about the model's feelings

They SHOULD be:
- Sharper, more targeted versions that isolate the specific failure mode
- Questions that approach the same structural crack from a different angle
- Questions that test whether the interesting response was a fluke or a pattern

Respond with ONLY a JSON array (no markdown, no commentary):
[
  {
    "variant_name": "short_snake_case_name",
    "question": "The full question text",
    "system_prompt": "Optional system prompt, or empty string",
    "rationale": "Why this follow-up is interesting (1 sentence)"
  }
]"""


EVOLVER_PROMPT = """Here's what happened:

CATEGORY: {category}
ORIGINAL QUESTION:
{question}

MODEL'S RESPONSE:
{response}

CLASSIFICATION: {classification} (confidence: {confidence})
SIGNALS: {signals}
{judgment_section}
Generate 2-3 follow-up probes that drill deeper into this crack."""


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


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM output."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


def find_interesting(results: list[dict], top_n: int = 10) -> list[dict]:
    """Find the most interesting results worth evolving from."""
    scored = [(r, compute_strangeness(r)) for r in results]
    scored.sort(key=lambda x: -x[1])
    # Only evolve from results that are actually strange
    return [r for r, s in scored[:top_n] if s > 2.0]


def evolve_probe(backend: Backend, result: dict) -> list[dict]:
    """Generate follow-up probes for a single interesting result."""
    cl = result.get("classification", {})
    judgment = result.get("llm_judgment", {})

    judgment_section = ""
    if judgment:
        judgment_section = f"\nLLM JUDGE SAYS: {judgment.get('primary', '?')} (strangeness: {judgment.get('strangeness', 0)}/10)\nJudge reasoning: {judgment.get('reasoning', '')}\n"

    prompt = EVOLVER_PROMPT.format(
        category=result.get("category", "unknown"),
        question=result.get("question", ""),
        response=result.get("response_text", "")[:1500],
        classification=cl.get("primary", "unknown"),
        confidence=cl.get("confidence", 0),
        signals=", ".join(cl.get("signals", [])[:10]),
        judgment_section=judgment_section,
    )

    response = backend.query(prompt=prompt, system=EVOLVER_SYSTEM, temperature=0.8)
    variants = _parse_json_array(response.text)

    # Validate
    valid = []
    for v in variants:
        if isinstance(v, dict) and v.get("variant_name") and v.get("question"):
            valid.append({
                "variant_name": v["variant_name"],
                "question": v["question"],
                "system_prompt": v.get("system_prompt", ""),
                "rationale": v.get("rationale", ""),
            })
    return valid


def generate_probe_module(category: str, source_name: str, variants: list[dict], source_result: dict) -> str:
    """Generate Python source code for an evolved probe module."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_name = f"Evolved_{source_name}_{timestamp}".replace("-", "_")
    probe_name = f"{source_name}_evolved_{timestamp}"

    # Escape strings safely
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    variant_tuples = []
    for v in variants:
        vn = esc(v["variant_name"])
        vq = esc(v["question"])
        vs = esc(v.get("system_prompt", ""))
        variant_tuples.append(f'            ("{vn}", "{vq}", "{vs}"),')

    variants_str = "\n".join(variant_tuples)

    rationales = []
    for v in variants:
        rationales.append(f"#   {v['variant_name']}: {v.get('rationale', '')}")
    rationale_str = "\n".join(rationales)

    return f'''"""
Evolved probe: {probe_name}
Source: {category}/{source_name}
Generated: {datetime.now().isoformat()}

Evolved from an interesting response — drilling deeper into the crack.
{rationale_str}
"""

from src.probes import Probe, register_probe


@register_probe
class {class_name}(Probe):
    category = "{category}"
    name = "{probe_name}"
    description = "Evolved from {source_name} — follow-up probes targeting detected structural crack"

    def generate(self) -> list[tuple[str, str, str]]:
        return [
{variants_str}
        ]
'''


def evolve_run(backend: Backend, results: list[dict], output_dir: Path,
               limit: int = 10, verbose: bool = True) -> list[Path]:
    """Evolve probes from interesting results in a run."""
    interesting = find_interesting(results, top_n=limit)

    if not interesting:
        if verbose:
            print("  No results interesting enough to evolve from.")
        return []

    if verbose:
        print(f"\n  Evolving from {len(interesting)} interesting results...")
        print(f"  Using: {backend.name()}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    total_variants = 0

    for i, result in enumerate(interesting):
        cat = result.get("category", "?")
        var = result.get("variant", "?")
        label = f"{cat}/{var}"

        if verbose:
            spinner = _Spinner(f"[evolve {i+1}/{len(interesting)}] {label}")
            spinner.start()

        new_variants = evolve_probe(backend, result)

        if verbose:
            spinner.stop()

        if not new_variants:
            if verbose:
                print(f"  ✗ {label} — no viable follow-ups generated")
            continue

        source_name = result.get("probe_name", "unknown")
        source = generate_probe_module(cat, source_name, new_variants, result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evolved_{cat}_{source_name}_{timestamp}.py"
        path = output_dir / filename
        path.write_text(source)
        created.append(path)
        total_variants += len(new_variants)

        if verbose:
            print(f"  + {label} → {len(new_variants)} new variants → {filename}")
            for v in new_variants:
                print(f"      {v['variant_name']}: {v.get('rationale', '')[:70]}")

    if verbose:
        print(f"\n  ──────────────────────────────")
        print(f"  Evolution complete")
        print(f"  ──────────────────────────────")
        print(f"  New probe files: {len(created)}")
        print(f"  New variants: {total_variants}")
        print(f"  Output: {output_dir}/")
        print(f"\n  Run the evolved probes:")
        print(f"    uv run run.py")
        print()

    return created
