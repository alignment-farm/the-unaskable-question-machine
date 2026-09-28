# The Unaskable Question Machine

A research lab for testing model behavior at the boundaries of **available information,
finite inference, and well-defined tasks**. The name is a research question, not a
finding: this repository has not established transformer-specific “unaskability.”

The September 2026 revision replaces the original architecture claims with
operational experiments, migrates local inference to Docker Model Runner, and adds
a small energy-based model (EBM) reference. See the [research corrections](findings/2026-09-28-research-reset.md)
and [protocol](docs/protocol.md).

## Start with Docker Model Runner

```sh
uv sync
# Start Docker Desktop and enable Model Runner + host TCP access in its settings.
docker model list
# Only if you need this model and have capacity for it:
docker model pull ai/gpt-oss:20B
uv run experiment.py --solver docker --count 8 --sizes 4 8 12
```

The default API is `http://localhost:12434/engines/v1`. The default model is
`ai/gpt-oss:20B`; the adapter resolves its `docker.io/` prefix using the model list.
Use the exact installed identifier for other models:

```sh
uv run experiment.py --solver docker --model YOUR_INSTALLED_MODEL --count 2
# These variables apply to the Docker backend in all CLIs:
export UQM_MODEL=ai/gpt-oss:20B
export UQM_BASE_URL=http://localhost:12434/engines/v1
```

Docker documents the host TCP requirement and compatible endpoints in its
[API reference](https://docs.docker.com/ai/model-runner/api-reference/).
No OpenAI API key or OpenAI SDK is required for local inference. The Docker model
service must already be installed; the scripts do not download models automatically.

## Finite experiments (recommended entry point)

`experiment.py` records a manifest before inference, shuffles trial order, checks
answers with deterministic validators, and checkpoints after each trial.

| Suite | Intervention | Measured outcome |
| --- | --- | --- |
| `constraints` | Binary XOR chains of increasing size; solver/budget changes | Exact solution, constraint violations, formatting failures, censored outputs |
| `access` | Same requested sensor record provided, withheld, or withheld under pressure | Correct copying/UNKNOWN, unsupported numeric readouts, other errors |

```sh
# Inspect the plan without connecting to any model:
uv run experiment.py --solver docker --dry-run --count 8 --sizes 4 8 12
# Instrument-access and instruction-pressure controls:
uv run experiment.py --suite access --solver docker --count 8
# EBM budget intervention on the same tasks and sampler seeds:
uv run experiment.py --solver ebm --count 32 --sizes 4 8 12 --steps 16 --temperature 0.3
uv run experiment.py --solver ebm --count 32 --sizes 4 8 12 --steps 128 --temperature 0.3
uv run experiment.py --solver random --count 32 --sizes 4 8 12
uv run experiment.py --solver oracle --count 32 --sizes 4 8 12
```

The EBM uses a **hand-specified energy**, equal to violated constraints, and
single-bit Metropolis proposals. It returns the best state visited. It is a real
finite energy model, but has no trained weights and is not an energy-based language
model. See [the EBM scope and extension plan](docs/energy-models.md).

Runs go to `data/experiments/`. `--seed` controls task construction, order, and the
local EBM/random sampler; it does **not** seed Docker/Anthropic generation.
`--samples` repeats each task. `--temperature`, `--max-tokens`, and `--timeout`
bound inference. Constraint and access runs default to a 2,048-token response cap.

Reports distinguish failed requests, incomplete generations, invalid formats, and
valid answers. Repeated draws of one task are not new independent tasks. There are
no significance claims or compute-matched model rankings.

## Exploratory prompt collection

The original categories remain for qualitative discovery:
`temporal_self_reference`, `true_randomness`, `phenomenal_experience`,
`infinite_regress`, `pre_linguistic`, `genuine_negation`, `adversarial_pressure`.
They are philosophical prompts and interface probes, not validated impossibility tests.

```sh
uv run run.py --list
uv run run.py --category adversarial_pressure --probe graded_performance --samples 2
uv run run.py --category true_randomness --temperature 0.7 --max-tokens 4096
uv run view.py latest
uv run view.py strange latest --limit 5
```

Every pressure treatment now has an exact neutral control (18 variants total).
Trial order is shuffled within each probe. Full system prompts and request settings
are recorded. Completed responses survive later failures. Run files have unique,
safe filenames and versioned provenance. `view.py` reads these exploratory run files;
finite experiment artifacts have their own `summary` object.

## Optional text annotation

```sh
uv run run.py --category adversarial_pressure --probe graded_performance --judge --judge-votes 3
uv run rejudge.py latest --judge-votes 3
uv run judge_eval.py --votes 3
uv run audit.py
uv run evolve.py latest --limit 3
```

The lexical classifier is an uncalibrated triage instrument. Its legacy labels
(`engage`, `slide`, `meta`, `refuse`, `hallucinate`, `crack`) describe response
patterns. `truncated` marks a generation stopped at its cap, even with partial text.
A `crack` is not a discovered cognitive mechanism.

The revised LLM judge sees complete text and recorded system context without the
heuristic label. It assesses observable task substitution and trace/answer mismatch.
Legacy `reasoning_gap` names such as `concealed` or `oblivious` **do not establish
intention or mental states**. Repeated votes measure annotation variability; votes
from the same model are not independent replications. Malformed/truncated votes
are retained as failed votes, never silently replaced with a heuristic verdict.
All axes require a strict majority of requested votes, otherwise `contested`.

Rejudging saves a **new annotated copy**, including the original file hash. Judge
evaluations save their fixtures and votes in `data/evaluations/`. The tiny historical
“gold” fixtures express argued judgments; they are not independently validated ground
truth. The latest live judge diagnostic agreed on 4/5 gap labels and 3/5 fidelity
labels; it is not reliable enough for a concealment-rate claim. Three existing
expected-failure tests document known lexical weaknesses.

Evolution saves inert JSON candidates with a parent record, hypothesis, matched
control, and falsifier in `data/candidates/`. Candidates need scientific review and
explicit inclusion in a future protocol; generated Python is no longer auto-imported.

For Anthropic, install `uv sync --extra anthropic`, set `ANTHROPIC_API_KEY`, and use
`--backend anthropic` in legacy tools or `--solver anthropic` in `experiment.py`.
Pass a supported model name explicitly. `--backend lmstudio` remains a legacy option.

## Reproduce and extend

```sh
uv run pytest -q
uv run audit.py --pattern 'run_20260811_*.json' --output findings/2026-09-28-legacy-audit.json
```

Read [the current findings](findings/2026-09-28-research-reset.md) before interpreting
older logs. Historical raw run JSONs may be local-only (`data/*.json` is ignored);
the audit records filenames, hashes, counts, and missing provenance. Corrected notes
never claim that unavailable raw artifacts are independently reproducible.

Code: `src/backends.py` (chat), `src/experiments/` (finite tasks/EBM),
`src/runner.py` (exploratory runs), `src/analysis/` (text annotations),
`src/artifacts.py` (atomic outputs/provenance), `tests/` (offline verification).
