# Archived ablation specs

Retired sweep specs, kept for provenance (not deleted) — several are still the *producing spec* of a
batch that lives in `../../../../verity-analysis/data/`. Moved out of the top-level catalog so
`experiments/ablation/` reflects only current, runnable specs.

| Spec | Retired because | Still the provenance of |
|---|---|---|
| `spec.exp1.rerun.json` | Superseded 2026-06-30 four-model exp1 mix (haiku / gpt-oss / sonnet / qwen); the current exp1 headroom specs are the per-model breadth files. | `verity-analysis/data/calibration_and_prototyping/exp1-model-breadth/exp1-rerun-120k/` |
| `spec.exp1.gemma4.json` | Byte-identical duplicate of `spec.exp1.gemma4-10seed.json` (which supersedes it in the catalog). | `verity-analysis/data/calibration_and_prototyping/exp1-model-breadth/exp1-gemma4-120k/` |

If a spec here needs to run again, copy it back up to `experiments/ablation/` rather than running from
`archive/`.
