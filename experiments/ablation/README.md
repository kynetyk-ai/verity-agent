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

Each cell's store-derived `RunReport` JSON lands in `results/<condition>-<model>-seedN.json`, plus a
`manifest.json` (cell coordinates → report filename) the analysis layer reads.

The orchestrator drives a `ControlService` directly (the same engine the daemon serves over HTTP),
serially, so deltas reflect mechanism rather than contention.

## Analyzing results

Turn a results directory into the paper's figures + a machine-readable summary:

```
uv run --group analysis python -m experiments.ablation.analyze results/ --out figures/ --threshold 0.96
```

This writes `figures/summary.json` (per-`(condition, model)` metrics + the pre-registered deltas
**A (3−2)** / **B (4−3)** / **B′ (4b−4)** with a Cliff's-delta effect size) and the figures:

- **F1** — Exp 1 one-shot final-score distribution by model, with a headroom reference line.
- **F2** — best-so-far trajectory (mean ± IQR), faceted by model, one line per loop rung.
- **F3** — money plot: final score by condition × model, annotated with the ladder deltas.
- **F4** — provisioned-context bytes per cycle, one line per condition (bounded vs ballooning).
- **F5** — self-vs-independent divergence (fudge) by condition + the gate catch-rate bar.

The analysis *projection* (`analyze.py`) is pure stdlib; only the *figures* need the `analysis`
dependency group (matplotlib). A `summary.json`-only run works without it (the figures import is lazy).
`--threshold` adds a tokens-to-threshold readout; `--reference` overrides the F1 headroom line.

