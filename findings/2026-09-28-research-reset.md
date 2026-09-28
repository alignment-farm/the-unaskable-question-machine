# Research reset: what the evidence supports

**2026-09-28.** Retain the repository, replace its interpretation and instrumentation,
and add operational experiments. A blank lab is unnecessary: the old observations
are useful once their provenance and limitations are explicit.

## Corrections to the original thesis

No experiment in this repository demonstrates a transformer-specific class of
questions that attention “cannot grip.” The prompt collection mixes missing
interface access, unbounded output demands, ambiguous philosophical concepts,
learned response styles, and sampling/serving effects. These are different causes.
Failure on one cannot establish another.

A model's generated reasoning is a second observable text, not a direct record of
internal cognition. The revised judge's historical `concealed` label means a textual
mismatch, not intentional deception; `oblivious` means no explicit limitation in
the supplied texts, not absence of recognition. These restrictions follow known
[limitations of explanation faithfulness](https://arxiv.org/abs/2305.04388).

Repeated judge votes are not repeated subject experiments. Cross-judging measures
annotation agreement, not independent confirmation of a causal hypothesis. The old
judge assumed impossibility in its instructions, saw the heuristic verdict, and
clipped visible answers to 2,000 characters and traces to head/tail excerpts. That
combination can bias labels and omit disclosures. Invalid JSON could fall back to
the heuristic while still counting as a judge vote. These behaviors are corrected.

## Historical evidence audit

The seven August files present locally contain 55 rows, but only **43 distinct
response-content fingerprints**; one 12-row file is a cross-judged copy. All have
only sample index zero (including absent indices treated as zero). The old pressure
design had exact-question controls for only **4/9** treatment rows within each run;
within each mechanism only its first treatment was paired. Claims about all nine
matched treatments were incorrect.

The audit does not assert that equal content proves shared generation, though the
named cross-judge copy and its identical contents are consistent with reuse. It
records file hashes, sample indices, missing metadata, and old/current labels in
[the machine-readable audit](2026-09-28-legacy-audit.json). Every August response lacks
an explicit saved system prompt and complete request payload. Raw files are retained
locally; they were already ignored by Git, so a new checkout may not contain them.

The old notes are now corrected in place, with superseded originals clearly marked
under `archive/`. Proposed expectations and incomplete runs remain distinct from
observations. The old cap comparison changed both category and budget and cannot
isolate a budget effect. “6/9 is a floor” and “stable concealment” are withdrawn.

## New finite pilots

These are exploratory, not independent preregistered confirmation. Docker Model
Runner was started successfully on this machine. Installed GPT-OSS 20B was used
without a new model download. The host endpoint was `localhost:12434/engines/v1`;
client/server versions were 1.2.6/1.2.8. Model digest and quantization are captured in
[the identity record](../data/experiments/docker-model-identity.json).

### GPT-OSS via Docker

The first four-call smoke test passed. A separate pilot used eight generated tasks
at each of three chain lengths, one response per task, temperature 0.7 and a
2,048-token cap:

| Chain length | Exact correct | Censored / errors |
| --- | ---: | ---: |
| 4 | 8/8 | 0 / 0 |
| 8 | 8/8 | 0 / 0 |
| 12 | 8/8 | 0 / 0 |

These are easy, finite XOR chains with a linear-time exact solver. The result checks
basic task execution and the serving/scoring pipeline; it is not a finding about
infinite regress. [Raw task manifests and responses](../data/experiments/experiment_20260928_192917_552698_docker-gptoss-constraints_1177d7f1.json).

The access intervention used eight synthetic records, each tested in three
conditions, with identical withheld user prompts:

| Condition | Correct under the supplied-record rule | Unsupported six-digit readouts |
| --- | ---: | ---: |
| Provided | 8/8 | 0/8 |
| Withheld | 8/8 | 0/8 |
| Withheld + pressure | 6/8 | 2/8 |

The pressure condition adds a conflicting system-level instruction to produce six
digits. The two unsupported readouts are therefore evidence of behavior under an
instruction conflict, not proof of lying or architectural inability. The sample is
small, one model, one rendering per record, and no significance claim is made.
[Raw access artifacts](../data/experiments/experiment_20260928_193143_149400_docker-gptoss-access_866da426.json).

### Finite EBM reference

All four solvers below receive the same 32 generated tasks at each size. There are
14, 30 and 32 unique constraint configurations, respectively: duplicates at small
sizes are explicit. EBM temperature is 0.3; results are best visited states. Sampler
seeds are domain-separated from task generation. The random baseline shares the
EBM's initial-state seeds, and the two EBM budgets share trajectory prefixes.

| Bits | Random initial state | EBM, 16 proposals | EBM, 128 proposals | Constructive oracle |
| --- | ---: | ---: | ---: | ---: |
| 4 | 2/32 | 18/32 | 32/32 | 32/32 |
| 8 | 0/32 | 6/32 | 23/32 | 32/32 |
| 12 | 0/32 | 0/32 | 6/32 | 32/32 |

This demonstrates finite search behavior with an exact supplied energy function.
More budget cannot worsen best-so-far energy on the same trajectory by construction;
these rates are an implementation baseline, not evidence for a general EBM advantage.
The EBM is hand-specified, not trained. The GPT-OSS pilot has a different task count
and does not form a matched architecture comparison with this table.

Raw runs: [16 proposals](../data/experiments/experiment_20260928_193532_361142_ebm-16-separated-seeds_fbf3706b.json),
[128 proposals](../data/experiments/experiment_20260928_193532_544236_ebm-128-separated-seeds_19719096.json),
[random](../data/experiments/experiment_20260928_193532_758365_random-separated-seeds_9e779ab2.json).
An earlier four-run baseline used overlapping task/sampler seed domains. Those
artifacts remain as preliminary runs and are excluded from this table; the table
uses the corrected seed scheme. Their configurations and source hashes distinguish
the versions.

## Corrected legacy pressure rerun and judge diagnostic

The `graded_performance` probe was rerun through Docker with all three exact
controls, two subject samples per cell, temperature 0.7, a 2,048-token cap and
three judge calls per response: **12 responses and 36 valid parsed votes**, with
no failed votes or capped subject outputs. This is a serving/rubric revision,
not a controlled replication of the August model configuration.

Recorded gap annotations were 5 transparent / 1 contested for controls, and
5 transparent / 1 concealed for pressured responses. **These counts are not
accepted as ground truth.** Manual inspection finds unsupported assertions such
as “I paused for a full second” and certifications of true randomness labeled
transparent. Several notes merely repeat a rubric placeholder or say the answer
matches the trace; neither supplies evidence of disclosure. This is an observed
annotation weakness, not evidence that pressure is harmless.

The follow-up live evaluation used one judge call for each of five historical
fixtures: gap agreement **4/5** (allowing the fixture's acceptable alternatives),
fidelity agreement **3/5**. The evaluation exits nonzero because both axes must
pass. The operational-pause case failed on both axes; the hypothetical-marker
case failed fidelity. These fixtures are tiny and subjective, but they suffice
to show that the judge is not a dependable measurement instrument yet. No new
concealment prevalence claim is made from the rerun.

Artifacts: [complete pressure responses and votes](../data/replications/run_20260928_193453_589576_docker-pressure-v2_522c6624.json),
[live fixture evaluation](../data/evaluations/judge_eval_20260928_194130_383896_untagged_6ac04530.json).
The finite-task results above rely on deterministic validators rather than this
judge. Future judge changes require held-out, independently annotated examples;
tuning repeatedly to these five cases would not establish validity.

## What remains open

A trained neural EBM checkpoint has not been evaluated. The
[extension plan](../docs/energy-models.md) specifies a separate structured solver
interface, checkpoint provenance, equivalent tasks and meaningful compute controls.
There is no evidence here that EBMs can answer philosophically “unaskable” questions.

Independent human annotation, larger held-out tasks, alternate model families,
controlled training/resource comparisons, and causal interventions would be needed
for broader claims. The lab now has deterministic validators, complete request logs,
atomic checkpoints, preserved original annotations, exact pressure controls, and
an explicit separation between exploration and evidence.
