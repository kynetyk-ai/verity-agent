# The stellar `fe-kaggle` task — setup & run protocol

This folder is a **self-contained task package**: the data (gitignored), the task definition
(`task.json`), and the driver (`run.sh`). It runs the §12 model-building task aimed at the **top 10%
of the real Kaggle leaderboard** — Verity writes a single self-contained script that predicts (by any
means: engineered features, model choice, ensembling, calibration), and the goal-seeking gate keeps
it iterating until it believes the attempt is competitive, then submits to the live competition and
accepts only what **climbs our best public score**. Run it repeatedly and the accepted scores form a
documented climb (see *What you get*).

Competition: **`playground-series-s6e6`** (stellar classification, balanced accuracy). Local agent by
default (Ollama). See `overview.md` for the dataset.

## How the gate works (why this is the fun part)

A **goal-seeking ladder** (the gate re-runs your *script* on gold data — it never trusts the agent's
CSV):
1. **runs-clean (cheap, every cycle):** run the script on a labeled hold-out carved from `train.csv`;
   a crash / timeout / missing prediction is a reject.
2. **proxy-improves (cheap):** score balanced accuracy on the hold-out; reject if it doesn't beat our
   best **accepted** attempt by a noise margin — so we never spend a Kaggle submission on a
   locally-worse attempt.
3. **competitive (cheap, no budget spent):** read the **live leaderboard** and compute the score at
   the top-`target_percentile`% rank; compare it to our *calibrated* hold-out estimate. **Below the
   bar → revise** (the attempt rests `revised`, the agent is told the gap and edits its prior script
   next cycle — *no submission spent*); at/above the bar → proceed to submit.
4. **kaggle (hard, rate-limited — the competition's daily cap, read from its metadata, default 5):**
   regenerate on the full train + the real `test.csv`, submit, read the **public score**; accept iff
   it beats our best prior Kaggle-confirmed score, and record the `(proxy, public)` pair so the
   competitive estimate **calibrates itself** as real submissions accrue. When the daily cap is spent
   the gate **blocks** until it frees (the control plane is long-lived); a Kaggle API failure degrades
   the cycle (recorded, fed back), it doesn't crash the run.

The Kaggle submit happens on the **trusted verifier side** — credentials never reach a worker.

**Data prep is yours, done outside Verity (ADR 0005).** The control plane does *no* splitting — it
routes opaque, role-keyed blobs. `run.sh` runs the reference prep (`tools/prepare_fe_data.py`) for
you: by default it trains on the **full** `train.csv` (the data is the bottleneck for a top-10%
score) with a **large stratified hold-out** (~15% ≈ 86k rows — big enough that its balanced accuracy
tightly estimates the public score the competitive gate gates against). It writes two role bundles —
`agent/` (the training set with the hold-out removed + the real `test.csv`) and `verifier/` (the same
training set, the hold-out features + **answer key**, the full train, and `test.csv`). The answer key
lives **only** in the `verifier/` bundle, so it structurally cannot reach the agent.

## Setup (once)

1. **Data.** Download the competition's `train.csv` and `test.csv` into **this folder**
   (`feature-engineering-test/`). They're gitignored. (`sample_submission.csv` is optional — the gate
   derives the submission from `test.csv`.)

2. **Kaggle API token.** At <https://www.kaggle.com/settings> → *Create New Token* (downloads
   `kaggle.json`). Put its `username`/`key` in the repo `.env`:
   ```
   KAGGLE_USERNAME=...
   KAGGLE_KEY=...
   ```
   (`cp .env.example .env` first.) The daemon compose forwards these to the control-plane container
   only — never a worker.

3. **Accept the competition rules.** On the competition page on kaggle.com, click *Join* / accept the
   rules once. Without this the API returns **403**.

4. **Local model (default).** Have an OpenAI-compatible server running (e.g. Ollama on `:11434`)
   serving the model in `task.json` (`qwen3.6:27b-coding-mxfp8`). To use a hosted model instead, edit
   `task.json`'s `sandbox.model`/`base_url` (and set the matching key in `.env`).

## Run

```bash
# 1. bring up the standing daemon (builds images, mounts exchange + store + creds)
set -a; . ./.env; set +a          # load Kaggle creds into the environment compose reads
just cp-serve

# 2. run the task end to end (ingest -> create -> run -> poll -> results -> export)
./feature-engineering-test/run.sh

# 3. when you're done
just cp-down
```

`run.sh` prepares the role bundles (host-side, pure stdlib), stages them + `task.json` into the
daemon's exchange, ingests each role file, then drives the verity CLI. Runs
are asynchronous and may take a while (local model + the Kaggle daily cap); the run keeps going on the
daemon even if you Ctrl-C the poll. Re-attach with `just cp status <run_id>` / `just cp results <run_id>`.

## What you get

- An **accepted `Submission`** only if it climbed the **real public leaderboard** — provenance and
  the per-cycle proxy / competitive-estimate / public scores are in `verity results <run_id>`.
- The **documented climb.** Pipe one or more runs' results through the readout to see the trajectory
  of accepted scores toward the top-10% bar:
  ```bash
  docker exec verity-cp verity results <run_id> | python3 feature-engineering-test/trajectory.py
  # cross-run: collect each run's `verity results` JSON to files and pass them all
  python3 feature-engineering-test/trajectory.py run1.json run2.json ...
  ```
- The submitted script exported to `${VERITY_EXCHANGE_HOST:-/tmp/verity-exchange}/out/<run_id>/`.
- Your submissions appear on the competition's public leaderboard under your Kaggle account.

## Knobs (`task.json`)

- `sandbox.model` / `sandbox.base_url` — the agent's model (local by default). `sandbox.code_timeout_s`
  — the per-run script budget on the **verifier** side (1800s, sized for full-data training +
  calibration); `sandbox.sandbox_timeout_s` — the **agent's** own runtime budget (2400s).
- `policy.max_cycles` — cycles per run (raised for the goal-seeking climb). `policy.stop_on_accept` —
  stop at the first accept (keep `false` to keep climbing and document improvement).
- `verifier.knobs.competition` — the competition slug. Goal-seeking knobs: `target_percentile` (the
  bar, default 10), `proxy_margin` (hold-out noise margin), `min_calibration_points`, `pessimism`.
  Other optional: `budget_poll_interval_s`, `score_poll_interval_s`, `wait_deadline_s`,
  `submit_message`, `daily_submission_limit`.

**Prep knobs** (env vars to `run.sh`, *not* `task.json` — prep is user-side now): `PER_CLASS`
(rows/class — **unset = the full competitive train** (the default); set a number for a quick smaller
smoke) and `RESERVED_FRACTION` (the hold-out share, default 0.15). To prepare data yourself without
`run.sh`: `python3 -m tools.prepare_fe_data --train train.csv --test test.csv --out
feature-engineering-test` (add `--per-class N` for a smoke).

Mind the **daily submission cap** (the competition's own limit, read from its Kaggle metadata —
typically ~5/day, shared across all runs) — keep concurrent runs to a handful.
