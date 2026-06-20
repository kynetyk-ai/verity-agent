# The stellar `fe-kaggle` task — setup & run protocol

This folder is a **self-contained task package**: the data (gitignored), the task definition
(`task.json`), and the driver (`run.sh`). It runs the §12 model-building task with the **real
Kaggle leaderboard as the final-test gate** — Verity writes a single self-contained script that
predicts (by any means: engineered features, model choice, ensembling, calibration), regenerates it
on gold data, submits to the live competition, and accepts only what **beats our best public score**.

Competition: **`playground-series-s6e6`** (stellar classification, balanced accuracy). Local agent by
default (Ollama). See `overview.md` for the dataset.

## How the gate works (why this is the fun part)

Two-tier ladder (the gate re-runs your *script* on gold data — it never trusts the agent's CSV):
1. **cheap, every cycle:** run the script on a labeled hold-out carved from `train.csv`; reject if its
   balanced accuracy doesn't beat our best **accepted** attempt — so we never waste a Kaggle submission.
2. **hard, rate-limited (the competition's daily cap, read from its Kaggle metadata — default 5):**
   regenerate the submission on the full train + the real `test.csv`,
   submit to Kaggle, read the **public score**; accept iff it beats our best prior Kaggle-confirmed
   score. When the daily cap is spent the gate **blocks** until it frees (the control plane is
   long-lived); a Kaggle API failure degrades the cycle (recorded, fed back), it doesn't crash the run.

The Kaggle submit happens on the **trusted verifier side** — credentials never reach a worker.

**Data prep is yours, done outside Verity (ADR 0005).** The control plane does *no* splitting — it
routes opaque, role-keyed blobs. `run.sh` runs the reference prep (`tools/prepare_fe_data.py`) for
you: it splits `train.csv` into two role bundles — `agent/` (the training set with the hold-out
removed + the real `test.csv`) and `verifier/` (the same training set, the hold-out features +
**answer key**, the full train, and `test.csv`). The answer key lives **only** in the `verifier/`
bundle, so it structurally cannot reach the agent.

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

- An **accepted `Submission`** only if it improved the **real public leaderboard** — provenance and
  the public scores are in `verity results <run_id>`.
- The submitted script exported to `${VERITY_EXCHANGE_HOST:-/tmp/verity-exchange}/out/<run_id>/`.
- Your submissions appear on the competition's public leaderboard under your Kaggle account.

## Knobs (`task.json`)

- `sandbox.model` / `sandbox.base_url` — the agent's model (local by default).
- `policy.max_cycles` — refine cycles per run. `policy.stop_on_accept` — stop at the first accept.
- `verifier.knobs.competition` — the competition slug. Optional: `budget_poll_interval_s`,
  `score_poll_interval_s`, `wait_deadline_s`, `submit_message`, `daily_submission_limit`.

**Prep knobs** (env vars to `run.sh`, *not* `task.json` — prep is user-side now): `PER_CLASS`
(rows/class subsampled for speed — raise for a serious attempt; full data is slow on a local model)
and `RESERVED_FRACTION` (the cheap-proxy hold-out share). To prepare data yourself without `run.sh`:
`python3 -m tools.prepare_fe_data --train train.csv --test test.csv --out feature-engineering-test`.

Mind the **daily submission cap** (the competition's own limit, read from its Kaggle metadata —
typically ~5/day, shared across all runs) — keep concurrent runs to a handful.
