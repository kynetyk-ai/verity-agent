# End-to-end `fe-kaggle` run — full data, Sakana **fugu** (2026-06-23)

The first run of the `fe-kaggle` task on a **new hosted provider reached over the OpenAI-compatible
seam** — Sakana's `fugu` model at `https://api.sakana.ai/v1` — on the same task, data, and
infrastructure as the earlier gpt-5.4 and sonnet runs. It is the Phase 6.2 payoff in practice: a
provider that is neither Anthropic nor OpenAI, driven with no code change (a `base_url` + an
`api_key_env` on the `ModelSpec`).

This run **aborted at cycle 8 of 10** — not on a Verity or model-quality fault, but because the
**Sakana account's prepaid credit balance was exhausted** mid-run (HTTP 429), which tripped the
3-consecutive-failure safeguard. It is recorded here as an honest, partial run: three scored full-data
cycles plus two distinct failure modes worth keeping.

| | |
|---|---|
| run_id | `4ec383633b624f07bb40803d946b56b7` |
| task_id | `34731e2a2dc74949a3a9fd30ad7611a5` |
| date | 2026-06-23 |
| model | `fugu` (Sakana, OpenAI-compatible at `api.sakana.ai/v1`; key via `OPENAI_API_KEY`) |
| competition | `playground-series-s6e6` (stellar classification, balanced accuracy) |
| data | **full** — train 577,347 rows · test 247,435 · agent-train 490,746 · hold-out 86,601 (15% stratified) |
| config | [`../../task.fugu.json`](../../task.fugu.json) (`max_cycles=10`, `stop_on_accept=false`, `sandbox_timeout_s=2400`, `code_timeout_s=3600`) |
| totals | 8 cycles run (of 10) · ~3 h 21 m wall · 2,537,997 tokens · 112 model steps · 132 tool calls |
| outcomes | 3 revised · 5 sandbox-failed (2 timeout · 3 credit-exhaustion) · **0 accepted / 0 Kaggle submissions** |

## Configuring fugu (the OpenAI-compatible seam)

fugu is a hosted, **keyed** OpenAI-compatible endpoint, so the `ModelSpec` needs all three of:
`model="fugu"`, `base_url="https://api.sakana.ai/v1"`, and — the non-obvious one —
**`api_key_env="OPENAI_API_KEY"`**. Without the last, a `base_url` spec resolves to the
`openai-compatible` provider, which has no default key env, so the resolver sends the `"EMPTY"`
placeholder and the endpoint 401s. The `verity create` CLI exposes `--model`/`--base-url` but **not**
`--api-key-env`, so the spec is supplied via a full `--request-file` ([`task.fugu.json`](../../task.fugu.json)).
A pre-flight against `api.sakana.ai/v1` confirmed both auth (HTTP 200) and **structured `tool_calls`**
(what the propose seam requires) before the run.

## Infrastructure exercised

- **A third model provider over one seam** — neither Anthropic nor OpenAI, no code change: a
  `base_url` + `api_key_env` on the `ModelSpec`, the daemon forwarding `OPENAI_API_KEY` by name into
  the agent worker. fugu drove the full `read → propose → gate → commit` loop and emitted structured
  tool calls throughout.
- **Full-data training** on all 490,746 agent-train rows, gated against the 86,601-row hold-out and
  the live 0.9714 top-10% leaderboard bar.
- **The competitive gate held every submission back** — no cycle's calibrated estimate reached the
  bar, so the harness spent **zero** Kaggle submissions (same discipline the gpt/sonnet runs showed).
- **Degrade-don't-crash on two boundaries** — a sandbox timeout (cycles 1, 3) and an external API
  error (cycles 5–7) were each recorded as a failed cycle and fed back, never crashing the daemon.
- **The 3-consecutive-failure safeguard fired correctly** — once the credit-exhaustion 429s stacked
  up three in a row, the run aborted with a recorded reason and still exported its `results.json`.

## What we observed

| cyc | outcome | proxy bal-acc | wall | tokens | submission |
|---:|---|---:|---:|---:|---|
| 0 | revised | **0.9664** | 21.6 m | 870,345 | 5-fold LightGBM (n_estimators=2000) + **OOF-tuned** per-class multipliers, balanced weights |
| 1 | timed out | — | 40.0 m | — | (no proposal — LightGBM run hit the 2400 s wall) |
| 2 | revised | **0.9663** | 34.4 m | 939,753 | "bounded" 3-model LightGBM — single validation pass, capped iterations (≤1250) |
| 3 | timed out | — | 40.0 m | — | (no proposal — bounded LightGBM still hit the wall) |
| 4 | revised | **0.9665** | 40.6 m | 727,899 | **dropped LightGBM** → 4-model sklearn `HistGradientBoosting`, hardcoded multipliers |
| 5 | sandbox-failed | — | 24.1 m | — | Sakana 429 *Prepaid credit balance is exhausted* (mid-cycle, after ~24 m of calls) |
| 6 | sandbox-failed | — | 4 s | — | Sakana 429 (instant) |
| 7 | sandbox-failed | — | 4 s | — | Sakana 429 (instant) |

Trajectory of the scored cycles:

```
0.9664 → ✗timeout → 0.9663 → ✗timeout → 0.9665 → ✗✗✗ credits-exhausted (abort)
```

Best hold-out balanced accuracy: **0.9665** (cycle 4). Competitive bar: **0.9714**. The scored cycles
were flat within ~0.0002 of each other and ~0.005 short of the bar throughout, so the gate spent zero
Kaggle submissions before the credit limit ended the run.

## The agent learned and adapted across cycles

The standout story is a **two-step diagnosis of its own timeout**, ending in the same conclusion the
gpt-5.4 run only hypothesized:

- **First it tried to make LightGBM cheaper.** Cycle 0's 5-fold, 2000-estimator LightGBM ensemble
  completed (21.6 m) but left little margin. After cycle 1 timed out, cycle 2 explicitly *"addresses
  the previous timeout by using a bounded LightGBM workflow"* — it dropped K-fold CV for a single
  validation pass and a 3-model ensemble with capped iterations. The score barely moved (0.9663) and
  it still ran 34 m.
- **When bounding didn't help, it changed model families.** Cycle 3 timed out again on the bounded
  LightGBM, and cycle 4 made the real move — *"avoids the LightGBM/OpenMP stall that caused the
  previous timeout"* — replacing LightGBM entirely with scikit-learn `HistGradientBoostingClassifier`
  and pinning the thread env (`OMP_NUM_THREADS`, BLAS=1). That cycle committed at the run's best
  score, 0.9665. The `requirements.txt` records the switch concretely: cycles 0/2 list `lightgbm==4.5.0`;
  cycle 4 drops it.

  This is the **same failure the gpt-5.4 run hit at its cycle 9** (a multi-library OpenMP/threading
  stall that froze at 0 % CPU), there left as an unconfirmed hypothesis. fugu independently reached
  that diagnosis *and* acted on it.

- **One regression worth noting.** The balanced-accuracy decision multipliers got *less* principled
  under time pressure, not more: cycle 0 **derived** them from out-of-fold predictions and cycle 2
  from a validation pass, but cycle 4 **froze them as hardcoded constants** (`{STAR: 1.195, …}`) to
  save time. That is the opposite of the gpt run's arc (guessed → OOF-tuned) and is the brittlest part
  of the best-scoring submission — the held-out gate measured 0.9665 against the script's self-reported
  ~0.967, the small gap consistent with multipliers tuned on the agent's own split.

## Failure modes observed (and how the harness handled each)

| cyc | failure | how the harness handled it |
|---:|---|---|
| 1, 3 | sandbox worker hit the 2400 s wall (LightGBM/OpenMP stall) with no proposal in the outbox | recorded as `sandbox_failed`, the reason fed back; **the run continued** (degrade-don't-crash) |
| 5 | Sakana **429 — prepaid credit balance exhausted** (mid-cycle, after ~24 m of model calls) | recorded as a failed cycle; the run continued |
| 6, 7 | Sakana 429 again, **instant** (no budget left for any call) | a third consecutive failure tripped `max_consecutive_sandbox_failures=3` → **run aborted** with a recorded reason; `results.json` still exported |

On the timeouts: the soft-deadline nudge (`DeadlineMiddleware`, fires at 80 % of the budget = 32 min)
is wired into this path, but it is injected only *between model steps* — when the agent is blocked
inside a long full-data training call as the deadline passes, the nudge can't land, and at 32 min there
is too little runway left for a final train-and-submit anyway. The hard 40-min kill was the backstop on
cycles 1 and 3.

On the abort: this was an **external account limit**, not a Verity or fugu fault. The harness behaved
exactly as designed — degraded each failed cycle, then stopped cleanly once failures became
consecutive rather than burning the rest of the budget on guaranteed-failing calls.

## The submissions

The three scripts fugu actually wrote, as the gate ran them (cycles 1, 3, 5–7 produced none):

| script | cycle | result |
|---|---:|---|
| `submissions/cycle0-submission.py` | 0 | revised · 0.9664 · 5-fold LightGBM + OOF multipliers |
| `submissions/cycle2-submission.py` | 2 | revised · 0.9663 · bounded 3-model LightGBM |
| `submissions/cycle4-submission.py` | 4 | revised · **0.9665 (best)** · sklearn HistGradientBoosting (LightGBM dropped) |

Each cycle's `requirements.txt` is alongside its script — the `lightgbm` line present in cycles 0/2
and absent in cycle 4 is the model-family switch, on the record. `results.json` is the full
machine-readable RunReport (every cycle's decisions, scores, timings, and telemetry).

## Reproduce

Per [`../../PROTOCOL.md`](../../PROTOCOL.md): bring up the daemon (with a funded `OPENAI_API_KEY` for
Sakana in `.env`), then `./prototyping_datasci_test/run.sh prototyping_datasci_test/task.fugu.json`
(full data, hosted fugu). To avoid the timeout losses seen here, raise `sandbox_timeout_s` (e.g. to
3600) in `task.fugu.json`. Runs are non-deterministic (hosted model + live leaderboard), and fugu's
default mode auto-routes across backend providers, so per-cycle latency varies.
