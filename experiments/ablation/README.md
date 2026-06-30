# Ablation ladder (Exp 1–4b)

The sweep orchestrator for the core quantitative study in
[`docs/experimental-design.md`](../../docs/experimental-design.md) §1/§6. The whole ladder is the one
`fe-holdout` task type configured five ways — each condition differs **only** in the verifier
`accept_policy` knob, the `policy.provisioning` preset, and the loop budget, never in code or wording.

## What a cell is

A cell is one `(condition, model, seed)` triple. The five conditions:

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

## The bottom rung runs separately

The ladder is split into two specs so the **bottom rung (exp1) runs on its own, before the loop**:

- **`spec.exp1.json`** — exp1 only (one cycle, `provisioning: none`). The per-model **calibration /
  headroom** gate (§1, figure F1): it tells you each model's single-shot ceiling, so you can add or
  drop models and tune budgets *before* committing to the expensive loop runs.
- **`spec.loop.json`** — exp2–4b (the guarded/curated loop rungs), configured from what exp1 taught.
- **`spec.example.json`** — all five in one sweep, if you'd rather not split.

Each spec carries its own `models` / `budgets`, so the two phases can legitimately differ (e.g. more
models in exp1, an adjusted cycle budget for the loop).

## Running it

1. Prepare the dataset once (outside Verity, ADR 0005) with `tools/prepare_fe_data.py`, producing a
   data dir with `agent/{train,test}.csv` and `verifier/{train,holdout,holdout_labels}.csv`. The
   agent's `test.csv` is a **small unlabelled sample** (`--agent-test-rows`, default **200**) — a
   wiring/format check only; the agent estimates its score via stratified **CV on `train`**, while the
   gate scores on the verifier's **full** reserved `holdout`. Keep `--agent-test-rows` identical
   across phases so the agent's inputs never differ between conditions.
2. Run each phase **with `run_sweep_container.sh`** (needs Docker + the worker images built, and the
   `verity-net` network — i.e. the daemon stack has been brought up at least once, see
   `infra/compose.daemon.yml`):

   ```
   experiments/ablation/run_sweep_container.sh spec.exp1.json <data-dir> results/exp1/
   # inspect F1 / per-model headroom, adjust spec.loop.json if needed, then:
   experiments/ablation/run_sweep_container.sh spec.loop.json <data-dir> results/loop/
   ```

   Each cell's store-derived `RunReport` JSON lands in
   `results/<phase>/<condition>-<model>-seedN.json`, plus a `manifest.json` (cell coordinates →
   report filename) the analysis layer reads, plus the **durable store** at `results/<phase>/store/`
   (see the box below). The sweep also **auto-harvests** each cell from that store (no manual step):
   - `results/<phase>/transcripts/<cell>.json` + `<cell>.md` — the agent transcript (raw + rendered
     via `tools/render_transcript.py`), one per cycle (`-cycleN` suffix when a cell runs >1 cycle),
     present even for rejected / no-proposal / timed-out cells (F4);
   - `results/<phase>/submissions/<cell>/{submission.py,requirements.txt}` — the accepted submission
     objects (rejected cells have no *accepted* artifact, but their code is in the transcript).

> [!IMPORTANT]
> **Do NOT run `python -m experiments.ablation.sweep` directly on the host, and never without a
> `--store-root`.** Two traps, both of which silently waste a multi-hour run:
>
> 1. **Verifier reachability.** `build_service` reaches the §9.1 verifier sibling by its Docker-network
>    container name. A host process can't resolve that → *every cell aborts instantly* with
>    `verifier unreachable ... nodename nor servname provided`. The sweep must run **inside a container
>    on `verity-net`**, which is exactly what `run_sweep_container.sh` does (it mirrors
>    `infra/compose.daemon.yml`: docker socket + identical-path `/tmp/verity-staging` mount +
>    `VERITY_VERIFIER_NETWORK=verity-net` + model creds from `./.env`).
>
> 2. **Store persistence.** Without `--store-root`, `build_service(root=None)` opens an **in-memory
>    `SqliteStore`** that dies with the `--rm` container. The `RunReport` JSONs still land in `--out`
>    and carry inline scores / rationale / `agent_telemetry` (enough for `analyze.py` and *all*
>    figures) — **but the agent transcripts and the submission objects (`submission.py`,
>    `requirements.txt`) are LOST**: a report's `transcript_ref` is only a content hash, and the bytes
>    it points at lived in that dead store. `run_sweep_container.sh` passes
>    `--store-root "$OUT/store"` so the store persists at `results/<phase>/store/`
>    (`store.db` + `objects/`), root-owned on the host (`chown` if a non-root user needs it). Retrieve
>    objects via `service.read_object(task, hash)` or straight from `objects/`; render a transcript
>    with `python tools/render_transcript.py <bytes>`.

The orchestrator drives a `ControlService` directly (the same engine the daemon serves over HTTP),
serially, so deltas reflect mechanism rather than contention. The reason it must still run *inside* a
container is trap #1 above — it is the CP, so it needs the CP's network + mounts.

## Analyzing results

Turn the results into the paper's figures + a machine-readable summary. Pass **one or more** results
dirs — the analysis merges them, so the separately-run bottom rung and the loop combine (F1 from
exp1, the trajectory/context/fudge figures from the loop, the F3 money plot across all five rungs):

```
uv run --group analysis python -m experiments.ablation.analyze results/exp1/ results/loop/ --out figures/ --threshold 0.96
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

