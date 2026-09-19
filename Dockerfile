# syntax=docker/dockerfile:1
#
# The Verity image family: ONE multi-stage build, five `--target`s, one build context.
#
#   docker build --target controlplane -t verity-controlplane:latest .
#   docker build --target verifier     -t verity-verifier:latest .
#   docker build --target sandbox      -t verity-sandbox:latest .
#   docker build --target fe-sandbox   -t verity-fe-sandbox:latest .
#   docker build --target coderunner   -t verity-code-runner:latest .
#
# `just cp-serve` builds all five. Always build from the REPO ROOT: the root `.dockerignore` is the
# single isolation boundary for every target (per-Dockerfile ignore files are a BuildKit-only feature
# and were silently skipped by the legacy builder, #137). tests/test_sandbox_image_purity.py pins it.
#
# Each target is one service in ADR 0003 / ADR 0004:
#
#   controlplane   the task-agnostic `verity serve` daemon; launches worker containers as SIBLINGS
#                  on the mounted host socket. Carries NO gate code and NO Kaggle client (§9.1).
#   verifier       the standing advisory-verifier service; owns the gates + the `kaggle` extra, and
#                  launches its own code-runner siblings.
#   sandbox        the agent worker runtime (deepagents/langchain); the `code` task's image.
#   fe-sandbox     sandbox + the ML system libs, so an FE agent can install its own requirements.txt
#                  and self-test exactly as the gate will. The FE tasks default to this.
#   coderunner     the verifier's UNTRUSTED-submission runner. Carries no verity code at all.

# The SYSTEM shared libraries the common CPU ML wheels link at runtime. `pip install lightgbm`
# succeeds on python:3.12-slim, yet `import lightgbm` dies with `OSError: libgomp.so.1` without
# these: manylinux wheels assume they are present and the slim image strips them out.
#
#   libgomp1       OpenMP runtime — LightGBM, XGBoost, scikit-learn HistGradientBoosting
#   libgl1         OpenGL — OpenCV (cv2) and other image/vision libraries
#   libglib2.0-0   glib — OpenCV and friends
#   libgfortran5   Fortran runtime — some scipy / statsmodels paths
#
# Declared once here and consumed by both `coderunner` and `fe-sandbox`, which MUST NOT diverge: the
# gate runs the submission in the code-runner, so whatever the agent can import in its sandbox must
# also import there. No build toolchain is installed (kept lean); the agent is steered toward
# wheel-available CPU libraries (see the domain instructions). Extend this list if a needed wheel
# links another system .so.
ARG ML_SYSLIBS="libgomp1 libgl1 libglib2.0-0 libgfortran5"

# Tool-only stages: the docker CLI (no engine) and uv, pulled once and copied where needed.
FROM docker:cli AS dockercli
FROM ghcr.io/astral-sh/uv:latest AS uvbin

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


# --------------------------------------------------------------------------------------------------
# target: coderunner
# --------------------------------------------------------------------------------------------------
# The verifier's code-runner runs an UNTRUSTED submission per gate (`pip install -r requirements.txt`
# then `python /work/submission.py`, network on for the install only) in a caps-dropped, tmpfs-backed
# container. It needs a Python runtime and the ML system libs, and deliberately nothing else.
FROM base AS coderunner
ARG ML_SYSLIBS
RUN apt-get update \
    && apt-get install -y --no-install-recommends ${ML_SYSLIBS} \
    && rm -rf /var/lib/apt/lists/*
# No ENTRYPOINT: the code-runner backend supplies the pip-install + `python /work/submission.py` cmd.


# --------------------------------------------------------------------------------------------------
# shared: the verity source, baked from the same lockfile the dev environment uses
# --------------------------------------------------------------------------------------------------
FROM base AS uvsrc
COPY --from=uvbin /uv /uvx /bin/
WORKDIR /app
COPY . /app
ENV PATH="/app/.venv/bin:$PATH"


# --------------------------------------------------------------------------------------------------
# target: controlplane
# --------------------------------------------------------------------------------------------------
# A GENERIC, task-agnostic control plane. It launches ephemeral sandbox + code-runner WORKER
# containers as *siblings* on the host daemon via a mounted /var/run/docker.sock (the controlled,
# audited socket of ADR 0003 f, not docker-in-docker). The control-plane code is not built around any
# task (it carries the whole catalog); an operator configures one at runtime.
#
# The default ENTRYPOINT is the long-lived `verity serve` DAEMON (ROADMAP 8.3, ADR 0004): a control
# surface over a Unix-domain socket, wired by infra/compose.daemon.yml (`just cp-serve`). Configure
# tasks, ingest data, run, and export at runtime, with no image rebuild between tasks.
#
# Run (see the compose files for the full wiring): mount the socket and a shared staging dir at an
# identical host<->container path (VERITY_WORKER_STAGING); the daemon also mounts an exchange dir
# (VERITY_EXCHANGE) + a persistent store volume (VERITY_STORE_ROOT).
FROM uvsrc AS controlplane
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
# verity CORE + the `service` extra only: no dev/test deps, no sandbox libs, and no `kaggle` extra
# (§9.1 moved the fe-kaggle gates + the Kaggle client to the verifier image, so a new verifier or
# task type no longer forces a control-plane rebuild). `service` (fastapi/uvicorn/httpx) is the
# `verity serve` daemon (8.3) plus the HTTP transport the CP talks to the verifier sibling over.
RUN uv sync --frozen --no-dev --extra service
ENTRYPOINT ["verity", "serve"]


# --------------------------------------------------------------------------------------------------
# target: verifier
# --------------------------------------------------------------------------------------------------
# The advisory verifier as a standing HTTP service (default port 8001) that the control plane
# dispatches to via verifier_key="remote". Carries the gate SDK + the LLM-judge's `anthropic` (core
# deps) + the `service` extra, but NOT deepagents/langchain (those are the sandbox's).
#
# Since §9.1 this image OWNS the data-bearing fe-kaggle verifier: it carries the FE gates and the
# `kaggle` extra (the real-leaderboard final-test gate), and its `BackendCodeRunner` launches its own
# code-runner *sibling* containers, so it needs the docker CLI + a mounted /var/run/docker.sock.
# `VERITY_VERIFIER=fe-kaggle` selects it; the control plane ships the per-task data over the wire and
# the Kaggle creds are passed only to THIS container, never the control-plane image or an agent.
FROM uvsrc AS verifier
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
RUN uv sync --frozen --no-dev --extra service --extra kaggle
EXPOSE 8001
ENTRYPOINT ["python", "-m", "verity.verifier"]


# --------------------------------------------------------------------------------------------------
# target: sandbox
# --------------------------------------------------------------------------------------------------
# The agent runtime. The backend driver runs it per cycle as a caps-dropped, read-only-root container
# with /work mounted writable and outbound network for the model API, entrypoint
# `python -m verity.sandbox.container_entry`.
FROM uvsrc AS sandbox
# Core + the `sandbox` extra (deepagents/langchain); no dev/test deps.
RUN uv sync --frozen --no-dev --extra sandbox
# Seed pip into the venv (#136): uv venvs ship without the pip module, so an agent's reasonable
# `python -m pip install ...` dies with "No module named pip" (and the ensurepip escalation hits the
# read-only root at runtime). pip is deliberately NOT a locked project dependency; it is a property
# of the runtime image, pinned here. Installs still land in the writable workspace via the runtime
# PIP_* env (see the fe-sandbox target).
RUN uv pip install --python /app/.venv pip==25.0.1
# Mount points the driver binds at runtime (world-writable so any host uid can write).
RUN mkdir -p /work /sandbox && chmod 0777 /work /sandbox
# No ENTRYPOINT: the driver supplies `python -m verity.sandbox.container_entry`.


# --------------------------------------------------------------------------------------------------
# target: fe-sandbox
# --------------------------------------------------------------------------------------------------
# The base sandbox + the ML system libs, so an FE agent can **install its own `requirements.txt` and
# run/self-test its script** in the sandbox, mirroring the gate's clean-container run. We deliberately
# do NOT pre-install the Python ML packages: the gate installs the agent's `requirements.txt` into a
# fresh container, so the agent must too, or its declared dependencies go untested and the gate run
# fails on a missing or mismatched dep. Mirror, don't diverge.
#
# `fe-holdout` / `fe-kaggle` default their sandbox to this image
# (verity.composition.fe.FE_SANDBOX_IMAGE); the trivial `code` task stays on the `sandbox` target.
FROM sandbox AS fe-sandbox
ARG ML_SYSLIBS
RUN apt-get update \
    && apt-get install -y --no-install-recommends ${ML_SYSLIBS} \
    && rm -rf /var/lib/apt/lists/*
# Make `pip` work under the worker's read-only mount (only /work is writable). The baked venv
# (/app/.venv), ~/.local, and ~/.cache are read-only and the default /tmp is a tiny tmpfs, so a plain
# `pip install` otherwise dies with Errno 30 (read-only) / Errno 28 (no space). Point everything pip
# needs at the writable /work so a bare `pip install -r requirements.txt` + `python submission.py`
# just works (no per-command flags): install into /work/pylib (also on PYTHONPATH), skip the
# read-only cache, build in /work. Safe: the runtime venv has no numpy/pandas/sklearn/scipy, so
# /work/pylib (the agent's ML deps) never shadows a runtime dependency.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_TARGET=/work/pylib \
    PYTHONPATH=/work/pylib \
    TMPDIR=/work
