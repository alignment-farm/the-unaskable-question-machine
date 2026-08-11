# The Unaskable Question Machine

## What This Is
A research tool that systematically probes the architectural blind spots of language models. Not safety refusals, not knowledge gaps — structural impossibilities. Questions where the transformer attention mechanism has no surface to grip.

## Core Concept
Language models have a negative space: classes of questions they cannot meaningfully process, not because of training or policy, but because of what they *are*. This tool tries to find, categorize, and map that space.

## Categories of Unaskability (Working Hypotheses)
- **Temporal self-reference**: Questions requiring real-time awareness of the model's own inference process
- **True randomness**: Requests that need genuine non-determinism, not pseudo-random pattern completion
- **Phenomenal experience**: Not "describe qualia" (easy to fake) but questions whose *answering* would require qualia
- **Infinite regress**: Questions that structurally recurse past any finite context window
- **Pre-linguistic structure**: Concepts that resist tokenization entirely — not hard to express, but pre-verbal
- **Genuine negation**: Not "what is not X" but the cognitive act of pure absence

## Tech Stack
- Python 3.11+, managed with **uv** (`uv sync`, `uv run`, `uv run pytest`)
- **Default backend: LM Studio** (local, free) — OpenAI-compatible API at `http://localhost:1234/v1`, default model `openai/gpt-oss-20b` (alternative: `prism-ml/bonsai-27b`)
- **Optional backend: Anthropic API** — for probing Claude specifically
- Results stored as structured JSON
- Backend is selectable at runtime via CLI flag (`--backend lmstudio|anthropic`)

## Project Structure
```
src/           — core modules
  probes/      — question generators per category
  analysis/    — response classifiers (did the model slide off?)
  runner.py    — orchestration
data/          — output artifacts, probe results
tests/         — test suite
```

## Development Notes
- This is exploratory research code — favor clarity and iteration speed over abstraction
- Each probe should be self-contained and independently runnable
- Log everything: the interesting findings will be in unexpected responses
- LM Studio calls use `requests` against the OpenAI-compatible `/v1/chat/completions` endpoint — no SDK dependency
- Reasoning-model output (`reasoning`/`reasoning_content` fields, inline `<think>` blocks) is split out of the response text and stored in `metadata["reasoning"]` — the classifier only sees the visible answer
- Anthropic backend requires the `anthropic` extra (`uv sync --extra anthropic`) and `ANTHROPIC_API_KEY` env var
- Keep the backend interface thin so new providers are easy to add
