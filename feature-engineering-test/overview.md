# Predicting Stellar Class — Kaggle Playground Series S6E6

**Competition:** Playground Series - Season 6, Episode 6
**Type:** Tabular multiclass classification (beginner-friendly)
**Timeline:** June 1 – June 30, 2026 (11:59 PM UTC final deadline)
**Evaluation metric:** Balanced accuracy
**Prize:** Kaggle merchandise (1st–3rd); no points or medals
**License:** CC BY 4.0

## Predictive Goal

Predict the **stellar class** of each astronomical observation — that is, classify each object as one of three categories:

- **GALAXY**
- **STAR**
- **QSO** (quasar / quasi-stellar object)

This is a single-target, three-class classification problem. The target column is `class`.

Because submissions are scored on **balanced accuracy** (the average of per-class recall), performance is measured equally across all three classes regardless of how common each one is. This matters because the classes are likely imbalanced, so a model that simply favors the majority class will score poorly.

## The Dataset

The data is **synthetically generated**, "inspired by" the well-known Stellar Classification Dataset (derived from the Sloan Digital Sky Survey, SDSS). Feature distributions are close to — but not identical to — the original real-world data. This is standard practice for the Playground Series: synthetic generation keeps named, interpretable features while ensuring test labels stay hidden.

**Total size:** ~144.8 MB across 3 CSV files (25 columns total across all files).

### Files

- **train.csv** — the training set (~100 MB, 12 columns), including the `class` target. Roughly 577,000 rows.
- **test.csv** — the test set (11 columns, no target); predict `class` for each row.
- **sample_submission.csv** — example submission in the correct format (~210,000 rows).

### Features (train.csv)

The training file contains an `id`, the `class` target, and the following predictive features:

- **alpha** — Right Ascension angle (sky coordinate, ~0–360°)
- **delta** — Declination angle (sky coordinate, ~ -18° to +79°)
- **u** — photometric magnitude in the ultraviolet filter
- **g** — photometric magnitude in the green filter
- **r** — photometric magnitude in the red filter
- **i** — photometric magnitude in the near-infrared filter
- **z** — photometric magnitude in the infrared filter
- **redshift** — measured redshift (a strong discriminator: quasars and galaxies show much higher redshift than stars)
- **spectral_type** — categorical stellar spectral classification (dominated by type M ~53%, A/F ~21%, with other types making up the remainder)
- **galaxy_population** — a categorical column present in **both** `train.csv` and `test.csv` (so it is usable at prediction time). Not part of the original SDSS feature set; explore the columns directly rather than relying on this overview to enumerate everything.

The five photometric bands (**u, g, r, i, z**) are the standard SDSS filter magnitudes; differences between them ("colors") are typically among the most informative engineered features for this kind of problem. Combined with **redshift**, they carry most of the signal for separating stars from galaxies and quasars.

## Submission Format

For each `id` in the test set, predict a class label. The CSV needs a header and two columns:

```
id,class
577347,STAR
577348,GALAXY
577349,STAR
```

## Notes for Modeling

- **Redshift** is usually the single most powerful feature for this task — near-zero for stars, larger for galaxies and quasars.
- Engineering **color indices** (e.g., u−g, g−r, r−i, i−z) from the photometric bands tends to boost accuracy.
- Encode the categorical **spectral_type** appropriately (e.g., target/ordinal/one-hot encoding).
- Optimize directly for **balanced accuracy** — handle class imbalance via class weights, resampling, or threshold/decision tuning rather than raw accuracy.
- The large row count (~577K train) makes gradient-boosted tree models (LightGBM, XGBoost, CatBoost) a strong, fast baseline.

---
*Source: https://www.kaggle.com/competitions/playground-series-s6e6 (Overview and Data tabs, retrieved June 8, 2026)*