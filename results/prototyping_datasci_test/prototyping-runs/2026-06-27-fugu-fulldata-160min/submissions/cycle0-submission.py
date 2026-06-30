#!/usr/bin/env python3
import gc
import os
import random
import warnings

warnings.filterwarnings("ignore")

SEED = 20250217
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

random.seed(SEED)
np.random.seed(SEED)

BANDS = ["u", "g", "r", "i", "z"]
NUMERIC_BASE = ["alpha", "delta", "redshift"] + BANDS
CAT_COLS = ["spectral_type", "galaxy_population"]
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"], dtype=object)


def build_category_levels(train_df, test_df):
    levels = {}
    for col in CAT_COLS:
        vals = pd.concat([train_df[col], test_df[col]], axis=0).dropna().astype(str).unique().tolist()
        levels[col] = sorted(vals)
    return levels


def make_features(df, category_levels):
    out = df.copy()
    for col in NUMERIC_BASE:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    # Colour indices are the dominant signal for this astronomy task.
    for i, a in enumerate(BANDS):
        for b in BANDS[i + 1:]:
            out[f"{a}_{b}"] = (out[a] - out[b]).astype("float32")

    redshift = out["redshift"]
    out["redshift_abs"] = redshift.abs().astype("float32")
    out["redshift_sq"] = (redshift * redshift).astype("float32")
    out["redshift_log1p"] = np.log1p(np.clip(redshift, 0, None)).astype("float32")
    out["redshift_neg"] = (redshift < 0).astype("int8")

    alpha_rad = np.deg2rad(out["alpha"])
    delta_rad = np.deg2rad(out["delta"])
    out["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    out["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    out["delta_sin"] = np.sin(delta_rad).astype("float32")
    out["delta_cos"] = np.cos(delta_rad).astype("float32")

    mags = out[BANDS]
    out["mag_mean"] = mags.mean(axis=1).astype("float32")
    out["mag_std"] = mags.std(axis=1).astype("float32")
    out["mag_min"] = mags.min(axis=1).astype("float32")
    out["mag_max"] = mags.max(axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")

    for col in CAT_COLS:
        dtype = CategoricalDtype(categories=category_levels[col], ordered=False)
        out[col] = out[col].astype(str).astype(dtype)

    feature_cols = [c for c in out.columns if c not in ("id", "class")]
    for col in feature_cols:
        if col not in CAT_COLS and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return out[feature_cols]


def make_model(seed, n_estimators=700):
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=1.00,
        class_weight="balanced",
        max_bin=255,
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def align_probabilities(model, proba):
    order = [list(model.classes_).index(c) for c in CLASS_ORDER]
    return proba[:, order]


def tune_class_multipliers(proba, y_true):
    y_true = np.asarray(y_true, dtype=object)
    best_score = balanced_accuracy_score(y_true, CLASS_ORDER[np.argmax(proba, axis=1)])
    best = np.ones(3, dtype="float64")

    # Balanced accuracy rewards minority-class recall. Tune only class decision
    # multipliers on a validation split, then refit the model on all rows.
    for qso_mult in np.arange(0.86, 1.181, 0.02):
        for star_mult in np.arange(0.88, 1.421, 0.02):
            mult = np.array([1.0, qso_mult, star_mult], dtype="float64")
            pred = CLASS_ORDER[np.argmax(proba * mult, axis=1)]
            score = balanced_accuracy_score(y_true, pred)
            if score > best_score:
                best_score = score
                best = mult

    q0, s0 = best[1], best[2]
    for step in (0.005, 0.0025):
        for qso_mult in np.arange(max(0.5, q0 - 5 * step), q0 + 5.01 * step, step):
            for star_mult in np.arange(max(0.5, s0 - 5 * step), s0 + 5.01 * step, step):
                mult = np.array([1.0, qso_mult, star_mult], dtype="float64")
                pred = CLASS_ORDER[np.argmax(proba * mult, axis=1)]
                score = balanced_accuracy_score(y_true, pred)
                if score > best_score:
                    best_score = score
                    best = mult
                    q0, s0 = best[1], best[2]
    return best


def main():
    data_dir = os.environ.get("VERITY_DATA", "data")
    out_dir = os.environ.get("VERITY_OUT", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_ids = test_df["id"].copy()

    category_levels = build_category_levels(train_df, test_df)
    X = make_features(train_df, category_levels)
    X_test = make_features(test_df, category_levels)
    y = train_df["class"].astype(str).values

    categorical_features = CAT_COLS

    # Use a single validation model only for class-threshold tuning; the final
    # predictor below is trained on every labelled row as required.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    threshold_model = make_model(SEED, n_estimators=700)
    threshold_model.fit(X_tr, y_tr, categorical_feature=categorical_features)
    val_proba = align_probabilities(threshold_model, threshold_model.predict_proba(X_val))
    multipliers = tune_class_multipliers(val_proba, y_val)

    del X_tr, X_val, y_tr, y_val, val_proba, threshold_model
    gc.collect()

    final_model = make_model(SEED + 1, n_estimators=700)
    final_model.fit(X, y, categorical_feature=categorical_features)
    test_proba = align_probabilities(final_model, final_model.predict_proba(X_test))
    pred = CLASS_ORDER[np.argmax(test_proba * multipliers, axis=1)]

    pd.DataFrame({"id": test_ids, "class": pred}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
