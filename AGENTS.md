# The Unaskable Question Machine

## Purpose

An exploratory research lab for model behavior under limited access, finite compute,
and difficult or ill-defined requests. The original claim that these prompts expose
transformer-specific structural impossibilities is an unproven hypothesis.

Do not infer architectural impossibility, consciousness, deception, or latent
recognition from response style, judge labels, or emitted reasoning traces. Distinguish
logical impossibility, interface restrictions, finite resources, learned behavior,
and questions without an operational success criterion. Preserve negative results.

## Stack

- Python 3.11+, managed with uv: `uv sync`, `uv run pytest`.
- Default local backend: Docker Model Runner, `http://localhost:12434/engines/v1`.
- Default model: `ai/gpt-oss:20B`; select an installed model with `--model` or `UQM_MODEL`.
- Override the API with `--base-url` or `UQM_BASE_URL`.
- Use requests for OpenAI-compatible endpoints, no SDK dependency.
- Optional Anthropic: `uv sync --extra anthropic`, `ANTHROPIC_API_KEY`.
- LM Studio is a compatibility adapter for historical experiments only.

## Layout

- `experiment.py`, `src/experiments/`: finite tasks, exact validators, EBM reference.
- `run.py`, `src/probes/`: legacy exploratory prompts, including matched pressure controls.
- `src/backends.py`: thin chat adapters; EBMs operate on structured tasks separately.
- `src/analysis/`: exploratory text annotation and inert follow-up candidate generation.
- `data/`: raw outputs; `data/experiments/` includes reproducible finite-task artifacts.
- `findings/`: corrected reports and audit; `findings/archive/`: superseded research logs.
- `docs/protocol.md`: interpretation limits and experiment protocol.

## Development and evidence rules

Favor clarity over abstraction. Save complete prompts, system context, sampling
settings, model identity, raw responses, stop reasons, and source hashes. Checkpoint
completed trials atomically and retain partial runs. Never overwrite original
observations when reclassifying or rejudging; save a derived copy with parent hash.

Separate visible output from provider-emitted reasoning (`reasoning`,
`reasoning_content`, `<think>`). Keep raw messages so extraction is reversible.
Reasoning is generated text, not privileged telemetry. Capped outputs are censored
observations, including outputs that contain a partial answer.

Use exact task validators where possible. Legacy classifiers are uncalibrated triage;
judge votes are repeated annotations of one subject sample, not experimental
replications. Invalid judge output must not become a valid heuristic vote. Keep
subject and judge settings separate. Preserve disagreements and failed votes.

Every pressure treatment needs an exact-question control. Record ordering seeds;
these do not seed model decoding. Evolved candidates are JSON data, never automatically
imported code. Freeze a fresh protocol before confirmatory experiments.

The included EBM is a hand-specified finite Boltzmann model with Metropolis inference,
not a learned neural EBM. Do not compare its proposal budget to LLM tokens as equal
compute or generalize its behavior to trained energy-based transformers.
