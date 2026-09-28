# Energy-based models: scope and next experiment

The repository now runs a finite EBM locally without a neural runtime. Its energy
counts constraint violations, defining p(x|task) ∝ exp(−E(task,x)/T). We can enumerate
small state spaces, check normalization and global optima, and inspect every
Metropolis proposal's accepted-state energy. This gives the lab an inspectable
inference process and a resource intervention with exact ground truth.

This model is hand-specified. It tests the harness and finite optimization behavior,
not learned representations, text generation, or architectural “unaskability.”
No neural EBM checkpoint was trained or evaluated in this revision. EBM energy
queries have privileged access to an exact task objective; chat models must parse
a prompt. A ranking between these solvers is therefore not an architecture comparison.

An EBM is a modeling framework, not a synonym for a non-transformer. Energy-Based
Transformers combine transformer parameterization with energy-based inference.
The [authors' code](https://github.com/alexiglad/EBT) and
[inference script](https://github.com/alexiglad/EBT/blob/main/job_scripts/nlp/inference/ebt.sh)
use their training/inference runtime and a supplied checkpoint. We have not verified
that these checkpoints can be served through Docker Model Runner's chat endpoint;
no adapter should pretend a chat model is a neural EBM simply because it is Docker-hosted.

For a trained-EBM extension:

1. Select a published checkpoint and verify license, tokenizer, architecture, training
   task, model digest, supported hardware and author-provided inference procedure.
2. Add a structured solver adapter that returns candidates, energy values, proposal
   counts, gradient evaluations, temperatures/noise schedules, initialization seeds,
   acceptance/selection rules, and checkpoint identity. Do not force it into a chat API.
3. Train or evaluate on a shared finite task distribution with held-out seeds and
   harder sizes. Keep task information equivalent; disclose any explicit energy oracle.
4. Match meaningful resource axes (measured runtime, memory and estimated operations),
   not “one token equals one optimization step.” Include random search, constructive
   solvers, the hand-specified EBM, and a matched autoregressive baseline.
5. Test changing inference budget, initialization, energy calibration and out-of-
   distribution task size. Ask whether failures disappear with compute or information
   before hypothesizing representational absence.

The [energy-based learning tutorial](https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf)
provides the foundational distinction between an energy function, inference, and
learning. The implemented reference addresses the first two only.
