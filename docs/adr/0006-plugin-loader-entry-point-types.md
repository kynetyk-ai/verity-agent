# ADR 0006 — Plugin loader: task & verifier types discovered via entry points

- **Status:** Accepted — shipped (ROADMAP 9.3; task & verifier types discovered from entry points at boot).
- **Date:** 2026-06-20
- **Affects spec:** §3.3–§3.4 (the deliberately-unintelligent control plane + its catalog), §3.6 (the
  opaque, user-selected verifier), §3.9 (deployment — adding a type without an image rebuild).
  **Resolves ADR 0004 (i)** (the placed-but-unbuilt registration seam). Builds on
  **[ADR 0004](0004-long-lived-configurable-control-plane.md)** (the catalog + `register` seam) and
  **[ADR 0005](0005-data-prep-out-of-the-control-plane.md)** (user-side prep — the prerequisite that
  left the CP with no domain *data* code).
- **Supersedes:** nothing. It *builds* the (i) seam ADR 0004 placed.

## Context

After Phase 9.1 (verifier → sibling container) and 9.2 (data prep → user side), the last compile-in
remains: **which types exist** is hardcoded. `default_catalog()` (`composition/catalog.py`) registered
the task types in code, and `verifier/__main__.py` held hardcoded `_IMPL_BUILDERS`/`_SETUP_BUILDERS`
dicts. So adding a task or verifier type meant editing `verity` source and rebuilding the image —
contradicting the founding concept (§3.3) that the control plane *dumbly* provisions and the litmus of
the whole phase: a new type ships **a container + a catalog entry, with no control-plane rebuild**.

The `register()` seam already existed (ADR 0004 (i)); this ADR wires **discovery** to it.

## Decision

**Task and verifier types are discovered at boot from Python entry-point groups, not compiled into a
hardcoded registry.** Two groups:

1. `verity.task_types` — read by the control-plane daemon (`composition/loader.py`).
2. `verity.verifier_types` — read by the verifier image (`verifier/registry.py`).

### (a) The entry-point contract

Each entry point resolves to a **`register(registry)` callable** — `register(catalog)` for task types,
`register_verifier(registry)` for verifier types. This mirrors the existing `TaskCatalog.register`
shape exactly, so a plugin may register one *or several* types and decides itself whether to pass a
describer (which `register` already makes optional). The entry-point *name* is diagnostic only; the
authoritative type name is whatever the plugin passes to `register`.

A third-party package advertises it in its own `pyproject.toml`:

```toml
[project.entry-points."verity.task_types"]
my-task = "my_pkg.verity_plugin:register"
```

### (b) Dogfood — the built-ins are entry points too

The built-ins (`code`, `fe-kaggle` tasks; `fake`, `fe-kaggle` verifiers) are declared as entry points
in the core `verity` distribution and flow through the **same** loader. `default_catalog()` is now
pure discovery (`load_task_plugins(TaskCatalog())`); `build_server_from_env()` is pure discovery into a
`VerifierRegistry`. One code path, exercised on every boot.

### (c) Built-ins keep priority; collisions are lenient (incumbent wins)

Discovery loads the core `verity` distribution's entry points **first** (deterministic order via
`ep.dist.name == "verity"`), then the rest. On a name collision the **incumbent wins** — the colliding
plugin type is logged (`*_type_collision`, `policy="incumbent_wins"`) and skipped — so an installed
package can never silently shadow a built-in, and two plugins claiming a name resolve to the first
discovered. (A strict/error mode is a possible future knob; the lenient default is degrade-don't-crash.)

### (d) Degrade-don't-crash + a loud built-ins guard

Every failure mode — `ep.load()` raising, a non-callable object, `register` raising — is logged
(structured) and **skipped** via a scratch-registry probe (a partial/failing `register` can never
mutate the real registry). The daemon still boots with whatever loaded. **But** because discovery
reads *installed dist metadata* (not live `pyproject.toml`), `default_catalog()` asserts the built-ins
are present and raises a **loud** error if they are missing (the install lacks its entry-point
metadata — run `uv sync`), rather than silently serving an empty catalog.

### (e) The loaders stay generic

`composition/loader.py` and `verifier/registry.py` import **no domain** (guarded by
`test_plugin_loaders_import_no_domain`). A domain enters only through a discovered plugin's *own*
package, lazily, on `ep.load()`.

### Trust model

An entry point is **install-time-trusted code that runs in the CP/verifier process** at boot —
equivalent to any installed dependency. This is acceptable under the ADR 0004 v1 trust boundary
("whoever controls the image/socket controls the CP"): the operator who installs a plugin package is
the same trusted party who controls the image. **Per-tenant / untrusted plugin isolation is explicitly
out of scope and deferred to the multi-tenancy engine (#58).**

## Consequences

- **New generic modules:** `composition/loader.py` (task discovery + `TaskCatalog.merge`) and
  `verifier/registry.py` (`VerifierRegistry` + verifier discovery). Both domain-free.
- **`pyproject.toml`** declares both entry-point groups with the built-ins. **Operational note:** entry
  points come from installed metadata — `uv sync` after editing them (CI already does).
- **Third-party task *or* verifier types add with no control-plane rebuild** — the Phase 9 exit.
- **`catalog.py` no longer imports any domain at module load** — the builders are pulled lazily when
  the loader resolves their entry points.
- **Deferred:** a strict collision mode; runtime hot-reload (boot-time discovery is enough); and
  authz over *who* may install plugins (with #58).

## Litmus test

An installed package advertising `verity.task_types` (or `verity.verifier_types`) makes its type
appear in `verity catalog` and dispatch via `create_task` (or be selectable by `VERITY_VERIFIER`),
with **no control-plane rebuild and no edit to `verity` source**; a broken plugin is logged and skipped
and the daemon still boots with the built-ins. Proven offline by `test_fixture_plugin_appears_without_core_import`
(a fixture type the core imports nowhere, reachable purely via discovery).
