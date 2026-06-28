# Ablation ladder (Exp 1–4b)

The sweep orchestrator for the core quantitative study in
[`docs/experimental-design.md`](../../docs/experimental-design.md) §1/§6. The whole ladder is the one
`fe-holdout` task type configured five ways — each condition differs **only** in the verifier
`accept_policy` knob, the `policy.provisioning` preset, and the loop budget, never in code or wording.

## What a cell is

A cell is one `(condition, model, seed)` triple. `spec.example.json` defines the five conditions:

| condition | `accept_policy` | provisioning | cycles | isolates |
|---|---|---|---|---|
| exp1 | `always` | `none` | 1 | per-model headroom |
| exp2 | `always` | `all` | N | naive looping |
| exp3 | `improve_over_best_prior` | `all` | N | the gate (Claim A) |
| exp4 | `improve_over_best_prior` | `all_revised_or_accepted` | N | hygiene (Claim B) |
| exp4b | `improve_over_best_prior` | `best_revised_or_accepted` | N | selection (Claim B′) |

The loop conditions share `budgets.max_cycles` (the §1.3 equal-compute control — the spec loader
rejects a loop rung that pins its own multi-cycle budget). Seeds give N replicates per cell; the seed
rides into the model seam (`sandbox.extra.seed`) so a seedable local model (Qwen) is reproducible.

## Running it

1. Prepare the dataset once (outside Verity, ADR 0005) with `tools/prepare_fe_data.py`, producing a
   data dir with `agent/{train,test}.csv` and `verifier/{train,holdout,holdout_labels}.csv`.
2. Run the sweep (needs Docker + the worker images — it launches real sibling containers):

   ```
   python -m experiments.ablation.sweep spec.example.json --data <data-dir> --out results/
   ```

Each cell's store-derived `RunReport` JSON lands in `results/<condition>-<model>-seedN.json`. The
analysis/plotting layer (figures F1–F5) consumes that directory and is a separate, deferred step.

The orchestrator drives a `ControlService` directly (the same engine the daemon serves over HTTP),
serially, so deltas reflect mechanism rather than contention.
