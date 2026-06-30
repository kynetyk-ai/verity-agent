#!/usr/bin/env python3
import gc
import os
import random
import warnings

warnings.filterwarnings("ignore")

SEED = 20250217
random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

np.random.seed(SEED)

BANDS = ["u", "g", "r", "i", "z"]
NUMERIC_BASE = ["alpha", "delta", "redshift"] + BANDS
CAT_COLS = ["spectral_type", "galaxy_population"]
LOW_CARD_CAT_COLS = CAT_COLS + ["spec_pop"]
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_ORDER)}

BIN_SPECS = [
    ("redshift", 72), ("u_g", 44), ("g_r", 44), ("r_i", 36), ("i_z", 36),
    ("u_r", 44), ("u_z", 44), ("g_i", 40), ("g_z", 40), ("r_z", 36),
    ("mag_mean", 36), ("mag_std", 28), ("mag_range", 32),
    ("u", 36), ("g", 36), ("r", 36), ("i", 32), ("z", 32),
    ("alpha", 36), ("delta", 36),
]

TE_KEYS = [
    "spectral_type", "galaxy_population", "spec_pop",
    "redshift_qbin", "redshift_qbin_x_spec", "redshift_qbin_x_pop", "redshift_qbin_x_spec_pop",
    "u_g_qbin", "g_r_qbin", "r_i_qbin", "i_z_qbin", "u_r_qbin", "u_z_qbin", "g_i_qbin", "g_z_qbin", "r_z_qbin",
    "mag_mean_qbin", "mag_std_qbin", "mag_range_qbin", "u_qbin", "g_qbin", "r_qbin",
    "sky_region", "ug_gr_region", "ri_iz_region", "rz_ug_region", "rz_gr_region", "rz_gi_region",
    "rz_uz_region", "rz_mag_region", "blue_red_region", "spec_pop_color_region", "redseq_lowz_region",
]

CAL_Z_BINS = np.array(
    [-np.inf, 0.0, 0.015, 0.03, 0.05, 0.075, 0.10, 0.14, 0.20, 0.30, 0.45,
     0.65, 0.90, 1.20, 1.80, 2.80, np.inf],
    dtype="float64",
)
CAL_N_SEGMENTS = (len(CAL_Z_BINS) - 1) * 8


def build_category_levels(train_df, test_df):
    levels = {}
    for col in CAT_COLS:
        vals = pd.concat([train_df[col], test_df[col]], axis=0).dropna().astype(str).unique().tolist()
        levels[col] = sorted(vals)
    spec_pop_vals = pd.concat(
        [
            train_df["spectral_type"].astype(str) + "|" + train_df["galaxy_population"].astype(str),
            test_df["spectral_type"].astype(str) + "|" + test_df["galaxy_population"].astype(str),
        ],
        axis=0,
    ).unique().tolist()
    levels["spec_pop"] = sorted(spec_pop_vals)
    return levels


def safe_ratio(num, den):
    return np.clip(num / (den.abs() + 1e-3), -25.0, 25.0).astype("float32")


def make_base_features(df, category_levels):
    out = df.copy()
    for col in NUMERIC_BASE:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    for i, a in enumerate(BANDS):
        for b in BANDS[i + 1:]:
            out[f"{a}_{b}"] = (out[a] - out[b]).astype("float32")

    redshift = out["redshift"]
    out["redshift_abs"] = redshift.abs().astype("float32")
    out["redshift_sq"] = (redshift * redshift).astype("float32")
    out["redshift_cube"] = (redshift * redshift * redshift).astype("float32")
    out["redshift_log1p"] = np.log1p(np.clip(redshift, 0, None)).astype("float32")
    out["redshift_neg"] = (redshift < 0).astype("int8")
    out["redshift_low"] = (redshift < 0.20).astype("int8")
    out["redshift_mid"] = ((redshift >= 0.20) & (redshift < 1.0)).astype("int8")
    out["redshift_high"] = (redshift >= 1.0).astype("int8")
    out["redshift_very_high"] = (redshift >= 2.2).astype("int8")

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
    out["mag_median"] = mags.median(axis=1).astype("float32")
    out["mag_skew_proxy"] = (out["mag_mean"] - out["mag_median"]).astype("float32")

    flux_cols = []
    for b in BANDS:
        fcol = f"flux_{b}"
        out[fcol] = np.power(10.0, -0.4 * out[b].clip(5, 35)).astype("float32")
        flux_cols.append(fcol)
    flux_sum = out[flux_cols].sum(axis=1).replace(0, np.nan)
    for b, fcol in zip(BANDS, flux_cols):
        out[f"flux_frac_{b}"] = (out[fcol] / flux_sum).astype("float32")
    out["flux_sum_log"] = np.log1p(flux_sum * 1e9).astype("float32")

    for a, b in zip(BANDS[:-1], BANDS[1:]):
        c = f"{a}_{b}"
        out[f"{c}_x_redshift"] = (out[c] * redshift).astype("float32")
        out[f"{c}_div_1pz"] = (out[c] / (1.0 + redshift.abs())).astype("float32")
        out[f"{c}_sq"] = (out[c] * out[c]).astype("float32")

    for b in BANDS:
        out[f"{b}_x_redshift"] = (out[b] * redshift).astype("float32")
        out[f"{b}_div_1pz"] = (out[b] / (1.0 + redshift.abs())).astype("float32")

    out["ug_gr_ratio"] = safe_ratio(out["u_g"], out["g_r"])
    out["gr_ri_ratio"] = safe_ratio(out["g_r"], out["r_i"])
    out["ri_iz_ratio"] = safe_ratio(out["r_i"], out["i_z"])
    out["ur_rz_ratio"] = safe_ratio(out["u_r"], out["r_z"])
    out["uz_redshift_ratio"] = safe_ratio(out["u_z"], 1.0 + redshift.abs())
    out["blue_slope"] = ((out["u"] - out["r"]) / 2.0).astype("float32")
    out["red_slope"] = ((out["r"] - out["z"]) / 2.0).astype("float32")
    out["curvature_ugr"] = (out["u"] - 2.0 * out["g"] + out["r"]).astype("float32")
    out["curvature_gri"] = (out["g"] - 2.0 * out["r"] + out["i"]).astype("float32")
    out["curvature_riz"] = (out["r"] - 2.0 * out["i"] + out["z"]).astype("float32")
    out["curvature_ugi"] = (out["u"] - 2.0 * out["g"] + out["i"]).astype("float32")
    out["redshift_magmean"] = (redshift * out["mag_mean"]).astype("float32")
    out["redshift_magstd"] = (redshift * out["mag_std"]).astype("float32")
    out["redshift_ug"] = (redshift * out["u_g"]).astype("float32")
    out["redshift_gr"] = (redshift * out["g_r"]).astype("float32")
    out["redshift_ri"] = (redshift * out["r_i"]).astype("float32")
    out["redshift_iz"] = (redshift * out["i_z"]).astype("float32")
    out["redshift_uz"] = (redshift * out["u_z"]).astype("float32")
    out["lowz_blue"] = ((redshift < 0.20).astype("float32") * out["u_g"]).astype("float32")
    out["lowz_red"] = ((redshift < 0.20).astype("float32") * out["r_z"]).astype("float32")
    out["highz_blue"] = ((redshift > 1.20).astype("float32") * out["u_g"]).astype("float32")
    out["highz_red"] = ((redshift > 1.20).astype("float32") * out["g_z"]).astype("float32")

    for col in CAT_COLS:
        dtype = CategoricalDtype(categories=category_levels[col], ordered=False)
        out[col] = out[col].astype(str).astype(dtype)
    spec_pop = out["spectral_type"].astype(str) + "|" + out["galaxy_population"].astype(str)
    out["spec_pop"] = spec_pop.astype(CategoricalDtype(categories=category_levels["spec_pop"], ordered=False))

    feature_cols = [c for c in out.columns if c not in ("id", "class")]
    for col in feature_cols:
        if col not in LOW_CARD_CAT_COLS and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return out[feature_cols]


def quantile_bins(train_s, test_s, q):
    vals = pd.concat([train_s, test_s], axis=0)
    vals = pd.to_numeric(vals, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = vals.median()
    med = float(med) if np.isfinite(med) else 0.0
    arr = vals.fillna(med).to_numpy(dtype="float64")
    edges = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, q + 1)))
    if len(edges) <= 2:
        return np.zeros(len(train_s), dtype="int16"), np.zeros(len(test_s), dtype="int16")
    inner = edges[1:-1]
    tr_vals = pd.to_numeric(train_s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype="float64")
    te_vals = pd.to_numeric(test_s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype="float64")
    return np.searchsorted(inner, tr_vals, side="right").astype("int16"), np.searchsorted(inner, te_vals, side="right").astype("int16")


def add_combo_code(X, X_test, name, train_parts, test_parts):
    train_vals = pd.Series(train_parts[0], index=X.index).astype(str)
    test_vals = pd.Series(test_parts[0], index=X_test.index).astype(str)
    for part in train_parts[1:]:
        train_vals = train_vals + "|" + pd.Series(part, index=X.index).astype(str)
    for part in test_parts[1:]:
        test_vals = test_vals + "|" + pd.Series(part, index=X_test.index).astype(str)
    cats = pd.Categorical(pd.concat([train_vals, test_vals], axis=0))
    codes = cats.codes.astype("int32")
    X[name] = codes[: len(X)].astype("int32")
    X_test[name] = codes[len(X):].astype("int32")


def add_bins_and_combos(X, X_test):
    X = X.copy()
    X_test = X_test.copy()
    for col, q in BIN_SPECS:
        tr_bin, te_bin = quantile_bins(X[col], X_test[col], q)
        X[f"{col}_qbin"] = tr_bin
        X_test[f"{col}_qbin"] = te_bin

    spec_tr = X["spectral_type"].astype(str)
    spec_te = X_test["spectral_type"].astype(str)
    pop_tr = X["galaxy_population"].astype(str)
    pop_te = X_test["galaxy_population"].astype(str)
    sp_tr = X["spec_pop"].astype(str)
    sp_te = X_test["spec_pop"].astype(str)

    add_combo_code(X, X_test, "redshift_qbin_x_spec", [X["redshift_qbin"], spec_tr], [X_test["redshift_qbin"], spec_te])
    add_combo_code(X, X_test, "redshift_qbin_x_pop", [X["redshift_qbin"], pop_tr], [X_test["redshift_qbin"], pop_te])
    add_combo_code(X, X_test, "redshift_qbin_x_spec_pop", [X["redshift_qbin"], sp_tr], [X_test["redshift_qbin"], sp_te])
    add_combo_code(X, X_test, "sky_region", [X["alpha_qbin"], X["delta_qbin"]], [X_test["alpha_qbin"], X_test["delta_qbin"]])
    add_combo_code(X, X_test, "ug_gr_region", [X["u_g_qbin"], X["g_r_qbin"]], [X_test["u_g_qbin"], X_test["g_r_qbin"]])
    add_combo_code(X, X_test, "ri_iz_region", [X["r_i_qbin"], X["i_z_qbin"]], [X_test["r_i_qbin"], X_test["i_z_qbin"]])
    add_combo_code(X, X_test, "rz_ug_region", [X["redshift_qbin"], X["u_g_qbin"]], [X_test["redshift_qbin"], X_test["u_g_qbin"]])
    add_combo_code(X, X_test, "rz_gr_region", [X["redshift_qbin"], X["g_r_qbin"]], [X_test["redshift_qbin"], X_test["g_r_qbin"]])
    add_combo_code(X, X_test, "rz_gi_region", [X["redshift_qbin"], X["g_i_qbin"]], [X_test["redshift_qbin"], X_test["g_i_qbin"]])
    add_combo_code(X, X_test, "rz_uz_region", [X["redshift_qbin"], X["u_z_qbin"]], [X_test["redshift_qbin"], X_test["u_z_qbin"]])
    add_combo_code(X, X_test, "rz_mag_region", [X["redshift_qbin"], X["mag_mean_qbin"]], [X_test["redshift_qbin"], X_test["mag_mean_qbin"]])
    add_combo_code(X, X_test, "blue_red_region", [X["u_g_qbin"], X["r_z_qbin"]], [X_test["u_g_qbin"], X_test["r_z_qbin"]])
    add_combo_code(X, X_test, "spec_pop_color_region", [sp_tr, X["u_g_qbin"], X["g_r_qbin"]], [sp_te, X_test["u_g_qbin"], X_test["g_r_qbin"]])
    add_combo_code(X, X_test, "redseq_lowz_region", [sp_tr, X["redshift_qbin"], X["u_g_qbin"]], [sp_te, X_test["redshift_qbin"], X_test["u_g_qbin"]])

    for col in X.columns:
        if col not in LOW_CARD_CAT_COLS and pd.api.types.is_numeric_dtype(X[col]):
            if str(X[col].dtype).startswith("int"):
                continue
            X[col] = X[col].replace([np.inf, -np.inf], np.nan).astype("float32")
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return X, X_test


def class_indices(y):
    return np.array([CLASS_TO_INDEX[v] for v in y], dtype="int16")


def _target_keys_present(X):
    return [key for key in TE_KEYS if key in X.columns]


def _build_target_maps(X_fit, y_fit, keys):
    y_fit = np.asarray(y_fit, dtype=object)
    prior = np.array([np.mean(y_fit == cls) for cls in CLASS_ORDER], dtype="float64")
    maps = {}
    for key in keys:
        counts = pd.crosstab(X_fit[key], pd.Categorical(y_fit, categories=CLASS_ORDER))
        counts = counts.reindex(columns=CLASS_ORDER, fill_value=0)
        row_counts = counts.sum(axis=1).to_numpy(dtype="float64")
        alpha = 260.0 if ("region" in key or "_x_" in key) else 105.0
        probs = (counts.to_numpy(dtype="float64") + alpha * prior.reshape(1, -1)) / (row_counts.reshape(-1, 1) + alpha)
        maps[key] = [(f"te_{key}_{cls.lower()}", pd.Series(probs[:, j], index=counts.index), float(prior[j]))
                     for j, cls in enumerate(CLASS_ORDER)]
    return maps


def _apply_target_maps(frame, maps):
    out = frame.copy()
    for key, specs in maps.items():
        source = frame[key]
        for feat_name, mapping, prior in specs:
            mapped = pd.Series(np.asarray(source.map(mapping)), index=frame.index)
            out[feat_name] = pd.to_numeric(mapped, errors="coerce").fillna(prior).astype("float32")
    return out


def _assign_target_maps_inplace(out, row_positions, source, maps):
    for key, specs in maps.items():
        src = source[key]
        for feat_name, mapping, prior in specs:
            mapped = pd.Series(np.asarray(src.map(mapping)), index=source.index)
            vals = pd.to_numeric(mapped, errors="coerce").fillna(prior).astype("float32").to_numpy()
            out.iloc[row_positions, out.columns.get_loc(feat_name)] = vals


def fit_apply_target_stats_oof(X_fit, y_fit, frames_to_transform, n_splits=5):
    """OOF target statistics for model-training rows; full-fit maps for test/calibration rows.

    The previous incumbent used in-fold target encodings for training rows. That can make trees
    over-trust high-cardinality bins and was the main source of held-out weakness. OOF encodings
    preserve the useful priors while matching the distribution seen by unseen rows.
    """
    y_fit = np.asarray(y_fit, dtype=object)
    keys = _target_keys_present(X_fit)
    if not keys:
        return [fr.copy() for fr in frames_to_transform]

    X_oof = X_fit.copy()
    global_prior = np.array([np.mean(y_fit == cls) for cls in CLASS_ORDER], dtype="float32")
    generated_cols = []
    for key in keys:
        for j, cls in enumerate(CLASS_ORDER):
            col = f"te_{key}_{cls.lower()}"
            X_oof[col] = global_prior[j]
            generated_cols.append(col)

    class_counts = pd.Series(y_fit).value_counts()
    folds = int(max(2, min(n_splits, class_counts.min())))
    if len(X_fit) >= 2000 and folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED + 707)
        for tr_idx, hold_idx in splitter.split(X_fit, y_fit):
            fold_maps = _build_target_maps(X_fit.iloc[tr_idx], y_fit[tr_idx], keys)
            _assign_target_maps_inplace(X_oof, hold_idx, X_fit.iloc[hold_idx], fold_maps)
    else:
        full_maps = _build_target_maps(X_fit, y_fit, keys)
        X_oof = _apply_target_maps(X_fit, full_maps)

    full_maps = _build_target_maps(X_fit, y_fit, keys)
    outputs = [X_oof]
    for fr in frames_to_transform[1:]:
        outputs.append(_apply_target_maps(fr, full_maps))
    return outputs


def make_lgbm_model(seed, variant="main", final=False):
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is not available")
    params = dict(
        objective="multiclass",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        max_bin=255,
    )
    if variant == "main":
        params.update(
            n_estimators=640 if final else 500,
            learning_rate=0.044 if final else 0.052,
            num_leaves=104,
            min_child_samples=80,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.05,
            reg_lambda=1.35,
        )
    else:
        params.update(
            n_estimators=700 if final else 540,
            learning_rate=0.040 if final else 0.049,
            num_leaves=88,
            min_child_samples=52,
            subsample=0.93,
            subsample_freq=1,
            colsample_bytree=0.82,
            reg_alpha=0.0,
            reg_lambda=0.95,
        )
    return LGBMClassifier(**params)


def make_xgb_model(seed, final=False):
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is not available")
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
        n_estimators=500 if final else 390,
        learning_rate=0.043 if final else 0.050,
        max_depth=5,
        min_child_weight=4.0,
        subsample=0.92,
        colsample_bytree=0.80,
        reg_alpha=0.0,
        reg_lambda=1.6,
        random_state=seed,
        n_jobs=-1,
    )


def align_probabilities(model, proba):
    order = [list(model.classes_).index(c) for c in CLASS_ORDER]
    return proba[:, order]


def confusion_counts(y_idx, pred_idx):
    return np.bincount(y_idx.astype("int16") * 3 + pred_idx.astype("int16"), minlength=9).reshape(3, 3)


def balanced_score_from_conf(conf):
    denom = conf.sum(axis=1).astype("float64")
    recalls = np.divide(np.diag(conf), denom, out=np.zeros(3, dtype="float64"), where=denom > 0)
    return float(np.mean(recalls))


def score_with_multipliers(proba, y_idx, multipliers):
    pred = np.argmax(proba * multipliers.reshape(1, -1), axis=1).astype("int16")
    return balanced_score_from_conf(confusion_counts(y_idx, pred))


def tune_global_multipliers(proba, y_true):
    y_idx = class_indices(y_true)
    best_score = score_with_multipliers(proba, y_idx, np.ones(3, dtype="float64"))
    best = np.ones(3, dtype="float64")
    for qso_mult in np.arange(0.60, 1.751, 0.025):
        for star_mult in np.arange(0.50, 2.451, 0.025):
            mult = np.array([1.0, qso_mult, star_mult], dtype="float64")
            score = score_with_multipliers(proba, y_idx, mult)
            if score > best_score:
                best_score = score
                best = mult
    q0, s0 = best[1], best[2]
    for step in (0.005, 0.0015):
        for qso_mult in np.arange(max(0.45, q0 - 0.08), q0 + 0.0801, step):
            for star_mult in np.arange(max(0.35, s0 - 0.10), s0 + 0.1001, step):
                mult = np.array([1.0, qso_mult, star_mult], dtype="float64")
                score = score_with_multipliers(proba, y_idx, mult)
                if score > best_score:
                    best_score = score
                    best = mult
                    q0, s0 = best[1], best[2]
    return best, best_score


def calibration_segments(df):
    z = pd.to_numeric(df["redshift"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    z_id = np.digitize(z, CAL_Z_BINS[1:-1], right=True).astype("int16")
    spec_codes = pd.Categorical(df["spectral_type"].astype(str), categories=["A/F", "G/K", "M", "O/B"]).codes
    pop_codes = pd.Categorical(df["galaxy_population"].astype(str), categories=["Blue_Cloud", "Red_Sequence"]).codes
    spec_codes = np.where(spec_codes < 0, 0, spec_codes).astype("int16")
    pop_codes = np.where(pop_codes < 0, 0, pop_codes).astype("int16")
    return (z_id * 8 + spec_codes * 2 + pop_codes).astype("int16")


def tune_segment_multipliers(proba, y_true, segments):
    y_idx = class_indices(y_true)
    global_mult, _ = tune_global_multipliers(proba, y_true)
    adjusted = proba * global_mult.reshape(1, -1)
    current_pred = np.argmax(adjusted, axis=1).astype("int16")
    total_conf = confusion_counts(y_idx, current_pred)
    best_score = balanced_score_from_conf(total_conf)

    n_segments = max(CAL_N_SEGMENTS, int(np.max(segments)) + 1 if len(segments) else CAL_N_SEGMENTS)
    seg_mult = np.ones((n_segments, 3), dtype="float64")
    seg_indices = [np.flatnonzero(segments == s) for s in range(n_segments)]

    # Require enough examples per segment to avoid overfitting the public-estimate split.
    min_seg = 650 if len(y_idx) >= 50000 else 120
    for pass_id in range(2):
        improved = False
        for seg, idx in enumerate(seg_indices):
            if len(idx) < min_seg:
                continue
            old_conf = confusion_counts(y_idx[idx], current_pred[idx])
            base_q, base_s = seg_mult[seg, 1], seg_mult[seg, 2]
            if pass_id == 0:
                q_grid = np.unique(np.r_[1.0, np.linspace(0.74, 1.40, 13)])
                s_grid = np.unique(np.r_[1.0, np.linspace(0.62, 1.90, 14)])
            else:
                q_grid = np.unique(np.clip(base_q * np.linspace(0.90, 1.12, 8), 0.52, 1.80))
                s_grid = np.unique(np.clip(base_s * np.linspace(0.88, 1.15, 9), 0.38, 2.25))
            local_best_score = best_score
            local_best = seg_mult[seg].copy()
            local_best_pred = current_pred[idx]
            seg_base = adjusted[idx]
            for q in q_grid:
                for s in s_grid:
                    trial_mult = np.array([1.0, q, s], dtype="float64")
                    trial_pred = np.argmax(seg_base * trial_mult.reshape(1, -1), axis=1).astype("int16")
                    trial_conf = confusion_counts(y_idx[idx], trial_pred)
                    score = balanced_score_from_conf(total_conf - old_conf + trial_conf)
                    if score > local_best_score + 1e-9:
                        local_best_score = score
                        local_best = trial_mult
                        local_best_pred = trial_pred
            if local_best_score > best_score + 1e-9:
                seg_mult[seg] = local_best
                current_pred[idx] = local_best_pred
                total_conf = total_conf - old_conf + confusion_counts(y_idx[idx], local_best_pred)
                best_score = local_best_score
                improved = True
        if not improved:
            break
    return global_mult, seg_mult


def apply_adjustment(proba, segments, global_mult, seg_mult):
    seg = np.clip(segments.astype("int64"), 0, len(seg_mult) - 1)
    return proba * global_mult.reshape(1, -1) * seg_mult[seg]


def prepare_xgb_frame(X):
    Xn = X.copy()
    for col in Xn.columns:
        if not pd.api.types.is_numeric_dtype(Xn[col]):
            if hasattr(Xn[col], "cat"):
                Xn[col] = Xn[col].cat.codes.astype("int16")
            else:
                Xn[col] = pd.Categorical(Xn[col].astype(str)).codes.astype("int16")
    Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    return Xn.astype("float32")


def blend_probabilities(named_probas, weights):
    total = 0.0
    blended = None
    for name, proba in named_probas:
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        total += w
        blended = proba * w if blended is None else blended + proba * w
    if blended is None or total <= 0:
        # Fallback: equal weight all available models.
        blended = sum(proba for _, proba in named_probas) / float(len(named_probas))
    else:
        blended = blended / total
    return blended


def select_blend_weights(named_probas, y_true):
    names = [name for name, _ in named_probas]
    if names == ["lgb_main", "lgb_alt"]:
        return {"lgb_main": 0.58, "lgb_alt": 0.42}
    if "xgb" not in names or len(named_probas) < 3:
        return {name: 1.0 / len(named_probas) for name in names}

    y_idx = class_indices(y_true)
    best = {"lgb_main": 0.51, "lgb_alt": 0.35, "xgb": 0.14}
    best_score = -1.0
    for wx in (0.00, 0.08, 0.14, 0.20, 0.26):
        rest = 1.0 - wx
        for main_share in (0.55, 0.60, 0.65):
            cand = {"lgb_main": rest * main_share, "lgb_alt": rest * (1.0 - main_share), "xgb": wx}
            proba = blend_probabilities(named_probas, cand)
            pred_idx = np.argmax(proba, axis=1).astype("int16")
            score = balanced_score_from_conf(confusion_counts(y_idx, pred_idx))
            if score > best_score:
                best_score = score
                best = cand
    return best


def fit_lgbm_pair(X_fit, y_fit, X_pred, categorical_features, final):
    named = []
    if LGBMClassifier is None:
        return named
    model_a = make_lgbm_model(SEED + (101 if final else 11), variant="main", final=final)
    model_a.fit(X_fit, y_fit, categorical_feature=categorical_features)
    proba_a = align_probabilities(model_a, model_a.predict_proba(X_pred)).astype("float64")
    named.append(("lgb_main", proba_a))
    del model_a
    gc.collect()

    model_b = make_lgbm_model(SEED + (211 if final else 29), variant="alt", final=final)
    model_b.fit(X_fit, y_fit, categorical_feature=categorical_features)
    proba_b = align_probabilities(model_b, model_b.predict_proba(X_pred)).astype("float64")
    named.append(("lgb_alt", proba_b))
    del model_b
    gc.collect()
    return named


def fit_xgb_predict(X_fit, y_fit, X_pred, final):
    if XGBClassifier is None:
        return None
    X_fit_xgb = prepare_xgb_frame(X_fit)
    X_pred_xgb = prepare_xgb_frame(X_pred)
    y_idx = class_indices(y_fit)
    weights = compute_sample_weight("balanced", y_idx)
    model = make_xgb_model(SEED + (307 if final else 73), final=final)
    model.fit(X_fit_xgb, y_idx, sample_weight=weights, verbose=False)
    proba = model.predict_proba(X_pred_xgb).astype("float64")
    del model, X_fit_xgb, X_pred_xgb, weights
    gc.collect()
    return ("xgb", proba)


def stratified_sample_for_quick(train_df, test_df, n):
    if n <= 0 or n >= len(train_df):
        return train_df, test_df
    sampled = train_df.groupby("class", group_keys=False).apply(
        lambda g: g.sample(max(2, int(round(n * len(g) / len(train_df)))), random_state=SEED)
    )
    sampled = sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_df = test_df.head(min(len(test_df), max(100, n // 2))).copy()
    return sampled, test_df


def main():
    data_dir = os.environ.get("VERITY_DATA", "data")
    out_dir = os.environ.get("VERITY_OUT", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    quick_n = os.environ.get("VERITY_QUICK_SAMPLE")
    if quick_n:
        train_df, test_df = stratified_sample_for_quick(train_df, test_df, int(quick_n))

    test_ids = test_df["id"].copy()
    y = train_df["class"].astype(str).values

    category_levels = build_category_levels(train_df, test_df)
    X_base = make_base_features(train_df, category_levels)
    X_test_base = make_base_features(test_df, category_levels)
    X_base, X_test_base = add_bins_and_combos(X_base, X_test_base)
    categorical_features = [c for c in LOW_CARD_CAT_COLS if c in X_base.columns]

    if len(train_df) >= 5000 and min(pd.Series(y).value_counts()) >= 4:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        fit_idx, cal_idx = next(splitter.split(X_base, y))
        X_fit_base = X_base.iloc[fit_idx]
        X_cal_base = X_base.iloc[cal_idx]
        y_fit = y[fit_idx]
        y_cal = y[cal_idx]
        X_fit, X_cal = fit_apply_target_stats_oof(X_fit_base, y_fit, [X_fit_base, X_cal_base], n_splits=5)

        cal_named = fit_lgbm_pair(X_fit, y_fit, X_cal, categorical_features, final=False)
        xgb_cal = fit_xgb_predict(X_fit, y_fit, X_cal, final=False)
        if xgb_cal is not None:
            cal_named.append(xgb_cal)
        if not cal_named:
            raise RuntimeError("No supported model library is available")

        blend_weights = select_blend_weights(cal_named, y_cal)
        cal_proba = blend_probabilities(cal_named, blend_weights)
        global_mult, seg_mult = tune_segment_multipliers(cal_proba, y_cal, calibration_segments(train_df.iloc[cal_idx]))
        del X_fit, X_cal, cal_named, cal_proba
        gc.collect()
    else:
        blend_weights = {"lgb_main": 0.51, "lgb_alt": 0.35, "xgb": 0.14}
        global_mult = np.ones(3, dtype="float64")
        seg_mult = np.ones((CAL_N_SEGMENTS, 3), dtype="float64")

    X_train, X_test = fit_apply_target_stats_oof(X_base, y, [X_base, X_test_base], n_splits=5)
    del X_base, X_test_base
    gc.collect()

    final_named = fit_lgbm_pair(X_train, y, X_test, categorical_features, final=True)
    xgb_final = fit_xgb_predict(X_train, y, X_test, final=True)
    if xgb_final is not None:
        final_named.append(xgb_final)
    if not final_named:
        raise RuntimeError("No supported model library is available")

    test_proba = blend_probabilities(final_named, blend_weights)
    adjusted = apply_adjustment(test_proba, calibration_segments(test_df), global_mult, seg_mult)
    pred = CLASS_ORDER[np.argmax(adjusted, axis=1)]

    pd.DataFrame({"id": test_ids, "class": pred}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
