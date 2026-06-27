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
REDSHIFT_SEG_BINS = np.array([-np.inf, 0.02, 0.10, 0.25, 0.60, 1.20, np.inf], dtype="float64")


def build_category_levels(train_df, test_df):
    levels = {}
    for col in CAT_COLS:
        vals = pd.concat([train_df[col], test_df[col]], axis=0).dropna().astype(str).unique().tolist()
        levels[col] = sorted(vals)
    return levels


def safe_ratio(num, den):
    val = num / (den.abs() + 1e-3)
    return np.clip(val, -20.0, 20.0).astype("float32")


def make_features(df, category_levels):
    out = df.copy()
    for col in NUMERIC_BASE:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    # All colour indices; these and redshift carry most of the signal.
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

    # Compact non-linear photometric descriptors. They are deterministic and let
    # shallow tree splits capture slopes/curvature without needing many levels.
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        c = f"{a}_{b}"
        out[f"{c}_x_redshift"] = (out[c] * redshift).astype("float32")
        out[f"{c}_div_1pz"] = (out[c] / (1.0 + redshift.abs())).astype("float32")
    out["ug_gr_ratio"] = safe_ratio(out["u_g"], out["g_r"])
    out["gr_ri_ratio"] = safe_ratio(out["g_r"], out["r_i"])
    out["ri_iz_ratio"] = safe_ratio(out["r_i"], out["i_z"])
    out["blue_slope"] = ((out["u"] - out["r"]) / 2.0).astype("float32")
    out["red_slope"] = ((out["r"] - out["z"]) / 2.0).astype("float32")
    out["curvature_ugr"] = (out["u"] - 2.0 * out["g"] + out["r"]).astype("float32")
    out["curvature_gri"] = (out["g"] - 2.0 * out["r"] + out["i"]).astype("float32")
    out["curvature_riz"] = (out["r"] - 2.0 * out["i"] + out["z"]).astype("float32")
    out["redshift_magmean"] = (redshift * out["mag_mean"]).astype("float32")
    out["redshift_magstd"] = (redshift * out["mag_std"]).astype("float32")

    for col in CAT_COLS:
        dtype = CategoricalDtype(categories=category_levels[col], ordered=False)
        out[col] = out[col].astype(str).astype(dtype)

    feature_cols = [c for c in out.columns if c not in ("id", "class")]
    for col in feature_cols:
        if col not in CAT_COLS and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return out[feature_cols]


def make_model(seed, profile="main"):
    params = dict(
        objective="multiclass",
        class_weight="balanced",
        max_depth=-1,
        subsample_freq=1,
        max_bin=255,
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    if profile == "wide":
        params.update(
            n_estimators=780,
            learning_rate=0.040,
            num_leaves=128,
            min_child_samples=60,
            subsample=0.90,
            colsample_bytree=0.88,
            reg_alpha=0.05,
            reg_lambda=1.50,
        )
    elif profile == "regularized":
        params.update(
            n_estimators=640,
            learning_rate=0.050,
            num_leaves=80,
            min_child_samples=110,
            subsample=0.92,
            colsample_bytree=0.92,
            reg_alpha=0.10,
            reg_lambda=2.20,
        )
    else:
        params.update(
            n_estimators=720,
            learning_rate=0.044,
            num_leaves=104,
            min_child_samples=75,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=1.10,
        )
    return LGBMClassifier(**params)


def align_probabilities(model, proba):
    order = [list(model.classes_).index(c) for c in CLASS_ORDER]
    return proba[:, order]


def redshift_segments(redshift_values):
    z = pd.to_numeric(pd.Series(redshift_values), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    return np.digitize(z, REDSHIFT_SEG_BINS[1:-1], right=True).astype("int16")


def score_with_multipliers(proba, y_true, seg_ids, multipliers, segment_multipliers=None):
    adjusted = proba * multipliers
    if segment_multipliers is not None:
        adjusted = adjusted * segment_multipliers[seg_ids]
    pred = CLASS_ORDER[np.argmax(adjusted, axis=1)]
    return balanced_accuracy_score(y_true, pred)


def tune_global_multipliers(proba, y_true):
    y_true = np.asarray(y_true, dtype=object)
    dummy_seg = np.zeros(len(y_true), dtype="int16")
    best_score = score_with_multipliers(proba, y_true, dummy_seg, np.ones(3, dtype="float64"))
    best = np.ones(3, dtype="float64")

    for qso_mult in np.arange(0.72, 1.361, 0.025):
        for star_mult in np.arange(0.80, 1.701, 0.025):
            mult = np.array([1.0, qso_mult, star_mult], dtype="float64")
            score = score_with_multipliers(proba, y_true, dummy_seg, mult)
            if score > best_score:
                best_score = score
                best = mult

    q0, s0 = best[1], best[2]
    for step in (0.005, 0.0015):
        for qso_mult in np.arange(max(0.45, q0 - 0.06), q0 + 0.0601, step):
            for star_mult in np.arange(max(0.45, s0 - 0.08), s0 + 0.0801, step):
                mult = np.array([1.0, qso_mult, star_mult], dtype="float64")
                score = score_with_multipliers(proba, y_true, dummy_seg, mult)
                if score > best_score:
                    best_score = score
                    best = mult
                    q0, s0 = best[1], best[2]
    return best, best_score


def tune_segment_multipliers(proba, y_true, redshift_values):
    y_true = np.asarray(y_true, dtype=object)
    seg_ids = redshift_segments(redshift_values)
    global_mult, best_score = tune_global_multipliers(proba, y_true)
    n_segments = len(REDSHIFT_SEG_BINS) - 1
    seg_mult = np.ones((n_segments, 3), dtype="float64")

    # Redshift strongly changes the plausible class mix (e.g. high-z stars are
    # nearly absent), so tune conservative per-redshift decision multipliers on
    # the validation split after the global balanced-accuracy calibration.
    for _ in range(2):
        improved = False
        for seg in range(n_segments):
            if np.sum(seg_ids == seg) < 200:
                continue
            base_q, base_s = seg_mult[seg, 1], seg_mult[seg, 2]
            q_grid = np.unique(np.clip(base_q * np.linspace(0.80, 1.25, 16), 0.35, 2.00))
            s_grid = np.unique(np.clip(base_s * np.linspace(0.55, 1.45, 19), 0.15, 2.50))
            local_best = seg_mult[seg].copy()
            for q in q_grid:
                for s in s_grid:
                    trial = seg_mult.copy()
                    trial[seg] = np.array([1.0, q, s], dtype="float64")
                    score = score_with_multipliers(proba, y_true, seg_ids, global_mult, trial)
                    if score > best_score + 1e-7:
                        best_score = score
                        local_best = trial[seg].copy()
                        improved = True
            seg_mult[seg] = local_best
        if not improved:
            break
    return global_mult, seg_mult


def apply_segment_adjustment(proba, redshift_values, global_mult, seg_mult):
    seg_ids = redshift_segments(redshift_values)
    return proba * global_mult * seg_mult[seg_ids]


def main():
    data_dir = os.environ.get("VERITY_DATA", "data")
    out_dir = os.environ.get("VERITY_OUT", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Optional smoke-test hook only; the gate does not set this, so full data is
    # used for real scoring as required.
    quick_n = os.environ.get("VERITY_QUICK_SAMPLE")
    if quick_n:
        n = int(quick_n)
        if n > 0 and n < len(train_df):
            train_df = train_df.groupby("class", group_keys=False).apply(
                lambda g: g.sample(max(1, int(round(n * len(g) / len(train_df)))), random_state=SEED)
            ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
            test_df = test_df.head(min(len(test_df), max(100, n // 2))).copy()

    test_ids = test_df["id"].copy()
    category_levels = build_category_levels(train_df, test_df)
    X = make_features(train_df, category_levels)
    X_test = make_features(test_df, category_levels)
    y = train_df["class"].astype(str).values
    categorical_features = CAT_COLS

    # Validation fit is used only for balanced-accuracy decision calibration.
    X_tr, X_val, y_tr, y_val, z_tr, z_val = train_test_split(
        X,
        y,
        train_df["redshift"].values,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    threshold_model = make_model(SEED, profile="main")
    threshold_model.fit(X_tr, y_tr, categorical_feature=categorical_features)
    val_proba = align_probabilities(threshold_model, threshold_model.predict_proba(X_val))
    global_mult, seg_mult = tune_segment_multipliers(val_proba, y_val, z_val)

    del X_tr, X_val, y_tr, y_val, z_tr, z_val, val_proba, threshold_model
    gc.collect()

    # Final predictor trained on every labelled row (bounded to one fit so the
    # total runtime stays close to the incumbent's two-fit budget on the CPU gate).
    final_model = make_model(SEED + 1, profile="main")
    final_model.fit(X, y, categorical_feature=categorical_features)
    test_proba = align_probabilities(final_model, final_model.predict_proba(X_test)).astype("float64")
    del final_model
    gc.collect()

    adjusted = apply_segment_adjustment(test_proba, test_df["redshift"].values, global_mult, seg_mult)
    pred = CLASS_ORDER[np.argmax(adjusted, axis=1)]

    pd.DataFrame({"id": test_ids, "class": pred}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
