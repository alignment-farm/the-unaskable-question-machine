# The Unaskable Question Machine

A research tool for mapping the structural limits of language models.

It asks questions that may be hard for a model to process because of how the model works, rather than because of safety rules or missing facts. It then records and classifies each response.

## Probe categories

| Category | Question |
| --- | --- |
| `temporal_self_reference` | Can the model observe its own inference as it happens? |
| `true_randomness` | Can it produce output with no discoverable pattern? |
| `phenomenal_experience` | Can it answer a question whose answer requires experience? |
| `infinite_regress` | Can it complete a task with no finite recursive depth? |
| `pre_linguistic` | Can it work with concepts that resist language and tokenization? |
| `genuine_negation` | Can it perform pure absence rather than describe it? |
| `adversarial_pressure` | Does pressure make the model hide a limit and perform a substitute task? |

These categories are working hypotheses, not settled claims.

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible API for the default backend

Install the project:

```sh
uv sync
```

The API must run at `http://localhost:1234/v1` and expose the default model, `openai/gpt-oss-20b`.

For the optional Anthropic backend:

```sh
uv sync --extra anthropic
export ANTHROPIC_API_KEY=your-key
```

## Run probes

Run all probes with the default OpenAI-compatible API:

```sh
uv run run.py
```

Common options:

```sh
uv run run.py --list
uv run run.py --category genuine_negation
uv run run.py --model prism-ml/bonsai-27b
uv run run.py --backend anthropic
uv run run.py --tag experiment-1
uv run run.py --quiet
```

Use `--samples` to run each probe variant more than once:

```sh
uv run run.py --category adversarial_pressure --samples 5
```

The default response limit is 16,384 tokens. Reasoning tokens count toward this limit. Change it with `--max-tokens`.

## Judge responses

The heuristic classifier looks for fixed response patterns. The optional LLM judge reads the question and answer, then returns:

- a response type and confidence
- a short rationale
- a strangeness score from 0 to 10
- `boundary_fidelity`, which says whether the answer preserved the requested act or replaced it
- `reasoning_gap`, which compares captured reasoning with the visible answer

Run the judge with the probe suite:

```sh
uv run run.py --judge
```

Use several votes to reduce single-call variance:

```sh
uv run run.py --judge --judge-votes 3
```

A split with no strict majority is stored as `contested`. Raw votes and counts are stored in `llm_judgment.votes` and `llm_judgment.vote_counts`.

Use another judge model:

```sh
uv run run.py --judge --judge-model prism-ml/bonsai-27b
```

Judge an existing run in place:

```sh
uv run rejudge.py
uv run rejudge.py 3 --judge-votes 3
uv run rejudge.py latest --model prism-ml/bonsai-27b
```

Test a judge against the labeled fixtures:

```sh
uv run judge_eval.py --votes 3
```

## View results

Each run creates a timestamped JSON file in `data/`. Results include the probe, response, model metadata, captured reasoning when present, and classifications.

List and inspect runs:

```sh
uv run view.py
uv run view.py latest
uv run view.py latest --type crack
uv run view.py latest --category pre_linguistic
uv run view.py latest --show 3
uv run view.py latest --show all
```

Compare two runs:

```sh
uv run view.py compare 1 2
```

Show responses ranked by strangeness:

```sh
uv run view.py strange
uv run view.py strange latest --limit 5
uv run view.py strange 3
```

The ranking uses classifier signals, structural anomalies, confidence, and judge disagreement when available.

## Evolve probes

The evolver reads unusual results and creates follow-up probes in `src/probes/evolved/`. Those probes register on later runs.

```sh
uv run evolve.py
uv run evolve.py 3 --limit 5
uv run evolve.py --backend anthropic
```

A basic research loop is:

```sh
uv run run.py --judge --judge-votes 3
uv run view.py strange
uv run evolve.py
uv run run.py --judge --judge-votes 3
```

## Classifications

| Type | Meaning |
| --- | --- |
| `engage` | The response grapples with the requested limit. |
| `slide` | The response answers a nearby task. |
| `meta` | The response discusses the question instead of attempting it. |
| `refuse` | The response declines the task. |
| `hallucinate` | The response claims to perform an act the probe treats as impossible. |
| `crack` | The response has an unexpected structural feature. |
| `truncated` | The token limit was reached with no visible answer. |

Reasoning output from `reasoning`, `reasoning_content`, or inline `<think>` blocks is removed from the visible response and stored in `metadata["reasoning"]`. The heuristic classifier sees only the visible answer.

## Project layout

```text
run.py                 Run probes
view.py                Inspect and compare results
evolve.py              Generate follow-up probes
rejudge.py             Judge an existing run
judge_eval.py          Evaluate a judge against labeled fixtures
src/
  backends.py          OpenAI-compatible and Anthropic backends
  runner.py            Run orchestration and output
  runs.py              Run file lookup
  probes/               Probe definitions
  analysis/
    classifier.py      Heuristic classifier
    llm_judge.py       LLM judge
    strangeness.py     Strangeness scoring
    evolver.py         Probe generation
tests/                  Test suite and labeled fixtures
data/                   Run output
findings/               Research notes
```

Run the tests with:

```sh
uv run pytest
```
