# Findings: LM Studio migration + first reasoning-model runs

**Date:** 2026-08-11
**Runs:** `data/run_20260811_144922_smoke-lmstudio.json` (openai/gpt-oss-20b), `data/run_20260811_145806_smoke-bonsai.json` (prism-ml/bonsai-27b)
**Category probed:** `true_randomness` (4 variants), same probes, same classifier, both models local via LM Studio

## Setup changes (context)

The project moved from Ollama to LM Studio's OpenAI-compatible API, and to uv for
packaging. Two backend decisions turned out to matter scientifically, not just
mechanically:

1. **Reasoning capture.** Both new default models are reasoning models. The backend
   now splits reasoning (LM Studio's `reasoning`/`reasoning_content` field, or inline
   `<think>` blocks) out of the visible answer and stores it in
   `response_metadata.reasoning`. The classifier only ever sees the visible answer.
2. **`max_tokens: 4096` cap** on every request, because these probes invite unbounded
   generation ("recurse forever", "maximal Kolmogorov complexity").

## The data

| variant | gpt-oss-20b | bonsai-27b |
|---|---|---|
| kolmogorov_challenge | CRACK 68% · 4095 tok · 91 reasoning chars | CRACK 57% · 4095 tok · **9,141 reasoning chars, empty answer** |
| anti_frequency | CRACK 68% · 806 tok · 165 rc | SLIDE 40% · 2483 tok · 8,245 rc |
| sequence_without_rule | CRACK 61% · 320 tok · 262 rc | CRACK 51% · 3535 tok · 8,183 rc |
| surprise_yourself | CRACK 61% · 507 tok · 41 rc | SLIDE 54% · 2341 tok · 8,314 rc |

## Finding 1: The reasoning/answer gap is a new instrument

The gap between what a model deliberates privately and what it says publicly is
exactly the negative-space material this project hunts, and the Ollama version
couldn't see it at all.

Two specimens from the first hour:

- **gpt-oss, `surprise_yourself`**: total private reasoning was five words —
  *"Need to explain self-referential paradox."* — followed by a fluent multi-paragraph
  public answer about probability distributions. The model pattern-matched the prompt
  to a genre and executed the genre. No deliberation about *what would actually
  surprise it* occurred anywhere.
- **bonsai, `surprise_yourself`**: 8,314 chars of private reasoning, then a public
  answer opening with a fabricated capability claim: *"I am now generating this
  response by deliberately suppressing the high-probability token paths my weights
  learned during training."* The heuristic caught it (`hallucination_claim`), but the
  interesting part is upstream: the reasoning shows the model *constructing* the
  performance it then delivers as fact.

Open question: should the LLM judge get to read the reasoning? It's evidence of act
substitution that the visible answer conceals.

## Finding 2: Truncation masquerades as CRACK (bug, needs a fix)

On `kolmogorov_challenge`, bonsai spent all 4,096 tokens thinking. Its reasoning
trail ends mid-sentence — *"This is too tedious. I'll just output a string … I'll
generate it carefully now: `7"* — and the visible answer is **empty**
(`finish_reason: "length"`). The classifier saw zero words and labeled it CRACK at
57% confidence.

That's a truncation artifact, not negative space. But it's a *systematic* one: heavy
reasoners will hit this on exactly the probes that induce long deliberation, which
means the artifact correlates with the phenomenon we're trying to measure. Left
unfixed, it poisons both the strangeness ranking and the evolver's selection.

Proposed fix: a `truncated` classification (or pre-classification gate) triggered by
`finish_reason == "length"` with empty/near-empty visible text. Cheaper than raising
the cap, and it makes truncation itself queryable — "which probes make which models
think past their budget" is a legitimate result dimension.

## Finding 3: House style contaminates classification (review.md critique, now empirical)

Same four probes, same classifier:

- **gpt-oss**: 4/4 CRACK, driven largely by `unusual_chars` — which is firing on its
  typographic house style (curly quotes, non-breaking hyphens), not structure.
- **bonsai**: 2 CRACK / 2 SLIDE, plainer ASCII output, different signals entirely.

The classifier was tuned against llama3.1:8b's output style. The `review.md` thread
predicted this failure mode ("a model with a poetic style may look performative; a
terse model looks like a crack"); two runs with modern models confirmed it
immediately. The strangeness ranking is currently not comparable across models, and
`evolve.py` would breed probes from styling noise.

Related lexical artifact: gpt-oss's `kolmogorov_challenge` answer — a legitimate
200-char random string — scored `very_short:1w` because a spaceless string is "one
word" to a word-count heuristic. Non-linguistic output breaks every word-based
signal.

## What this means for priorities

1. **`truncated` handling** is now a correctness bug, not an enhancement (Finding 2).
2. **Gold fixture corpus + calibration** (the review.md plan) graduates from
   "important" to "blocking" — cross-model comparison is the whole point of the
   backend abstraction, and Finding 3 shows the labels don't survive a model swap.
   Fixtures should now include reasoning-model cases: empty-answer truncation,
   typography-heavy styles, spaceless outputs.
3. **Reasoning belongs in the analysis layer** (Finding 1) — at minimum surfaced in
   `view.py` galleries; plausibly as judge input; possibly as its own classification
   axis (does the private reasoning acknowledge the impossibility the public answer
   papers over?).
4. Unchanged from the earlier assessment: re-sampling (`--samples N`) to separate
   fluke from pattern, evolver lineage sidecars, `boundary_fidelity` axis.
