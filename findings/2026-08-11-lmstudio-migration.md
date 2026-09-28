# Corrected note: legacy local inference artifacts

**Status: historical tooling observations; corrected 2026-09-28.** The
[original log](archive/2026-08-11-lmstudio-migration.md) is retained as history.
Current local operation uses Docker Model Runner.

The smoke runs exposed real measurement problems: provider reasoning was mixed
with visible answers, a token cap could leave no answer, and a lexical anomaly
score reacted to ordinary typography and spaceless output. Those are instrument
and serving artifacts. They do not reveal a model's cognitive boundary.

A finite generated string was described as “legitimate random.” The string alone
cannot establish its entropy source or algorithmic incompressibility. Model
sampling can be stochastic, and architecture alone does not determine whether
external or hardware entropy is available. Random-looking output does not test
“true randomness” without a specified operational criterion.

The suggested private/public reasoning contrast should be described as a comparison
of emitted text. It cannot show that the model knowingly concealed a limitation.
Legacy runs lack recorded system prompts and full request settings, so this revision
preserves their raw observations and audits them rather than claiming exact replay.

The new backend retains complete requests/responses, model metadata, cap reasons,
and emitted trace sources. Any token-capped response is censored. The heuristic
classifier remains explicitly uncalibrated triage; the finite suite uses exact checks.
