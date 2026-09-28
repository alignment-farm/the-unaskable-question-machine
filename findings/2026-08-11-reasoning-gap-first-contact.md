# Corrected note: emitted trace/answer annotations

**Status: exploratory annotation observations; corrected 2026-09-28.** The
[original log](archive/2026-08-11-reasoning-gap-first-contact.md) is retained as history.
See the [current audit](2026-09-28-research-reset.md).

Repeated judge calls sometimes assigned different `reasoning_gap` labels to the
same captured response. Majority votes reduce some observed disagreement but do
not prove the minority was noise, the majority was correct, or either model had a
particular mental state. The old conclusion that “every single-shot concealed was
judge noise, not model deception” was not established by this procedure.

A trace that does not state a limitation provides no evidence that recognition is
absent internally. A trace that does state it shows emitted wording, not necessarily
the cause of the final answer. Thus the claimed cross-model difference in “seeing
the wall” is withdrawn. At most, the available responses differed in explicit
limitation statements and in how judges annotated those statements.

The 7/7 answered run under a larger cap used a different probe category from the
earlier 2/4 truncated run. It does not isolate the effect of raising the cap, much
less show that the new cap eliminates truncation generally. A matched budget sweep
would be required. All capped outputs, including partial visible answers, are now
marked as censored in the revised instrumentation.

Expectations written while a run was already in flight are prospective notes, not
independently timestamped preregistration. The trace/answer axis remains available
for exploratory annotation, with revised definitions restricted to observable text.
