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
from sklearn.model_selection import StratifiedKFold

random.seed(SEED)
np.random.seed(SEED)

BANDS = ["u", "g", "r", "i", "z"]
NUMERIC_BASE = ["alpha", "delta", "redshift"] + BANDS
CAT_COLS = ["spectral_type", "galaxy_population"]
LOW_CARD_CAT_COLS = CAT_COLS + ["spec_pop"]
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_ORDER)}

# Fine enough to let the threshold optimizer learn the changing redshift prior,
# but still coarse enough that every calibration cell has many OOF examples.
CAL_Z_BINS = np.array(
    [-np.inf, 0.0, 0.02, 0.06, 0.10, 0.16, 0.25, 0.40, 0.60, 0.90, 1.20, 1.80, 2.80, np.inf],
    dtype="float64",
)

BIN_SPECS = [
    ("redshift", 64),
    ("u_g", 40),
    ("g_r", 40),
    ("r_i", 36),
    ("i_z", 36),
    ("u_r", 40),
    ("g_i", 36),
    ("g_z", 36),
    ("mag_mean", 36),
    ("mag_range", 32),
    ("alpha", 32),
    ("delta", 32),
]

TE_KEYS = [
    "spectral_type",
    "galaxy_population",
    "spec_pop",
    "redshift_qbin",
    "redshift_qbin_x_spec",
    "redshift_qbin_x_pop",
    "redshift_qbin_x_spec_pop",
    "u_g_qbin",
    "g_r_qbin",
    "r_i_qbin",
    "i_z_qbin",
    "u_r_qbin",
    "g_i_qbin",
    "g_z_qbin",
    "mag_mean_qbin",
    "mag_range_qbin",
    "alpha_qbin",
    "delta_qbin",
    "sky_region",
    "ug_gr_region",
    "ri_iz_region",
    "rz_ug_region",
    "rz_gr_region",
]


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
    val = num / (den.abs() + 1e-3)
    return np.clip(val, -20.0, 20.0).astype("float32")


def make_base_features(df, category_levels):
    out = df.copy()
    for col in NUMERIC_BASE:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    # All colour indices; these and redshift carry most of the signal.
    for i, a in enumerate(BANDS):
        for b in BANDS[i + 1 :]:
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

    # Relative flux features are another smooth representation of the same
    # photometry.  They help the ensemble on unusually bright/faint rows while
    # remaining bounded and deterministic.
    flux_cols = []
    for b in BANDS:
        fcol = f"flux_{b}"
        out[fcol] = np.power(10.0, -0.4 * out[b].clip(5, 35)).astype("float32")
        flux_cols.append(fcol)
    flux_sum = out[flux_cols].sum(axis=1).replace(0, np.nan)
    for b, fcol in zip(BANDS, flux_cols):
        out[f"flux_frac_{b}"] = (out[fcol] / flux_sum).astype("float32")
    out["flux_sum_log"] = np.log1p(flux_sum * 1e9).astype("float32")

    # Compact non-linear photometric descriptors.
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        c = f"{a}_{b}"
        out[f"{c}_x_redshift"] = (out[c] * redshift).astype("float32")
        out[f"{c}_div_1pz"] = (out[c] / (1.0 + redshift.abs())).astype("float32")
        out[f"{c}_sq"] = (out[c] * out[c]).astype("float32")
    out["ug_gr_ratio"] = safe_ratio(out["u_g"], out["g_r"])
    out["gr_ri_ratio"] = safe_ratio(out["g_r"], out["r_i"])
    out["ri_iz_ratio"] = safe_ratio(out["r_i"], out["i_z"])
    out["ur_rz_ratio"] = safe_ratio(out["u_r"], out["r_z"])
    out["blue_slope"] = ((out["u"] - out["r"]) / 2.0).astype("float32")
    out["red_slope"] = ((out["r"] - out["z"]) / 2.0).astype("float32")
    out["curvature_ugr"] = (out["u"] - 2.0 * out["g"] + out["r"]).astype("float32")
    out["curvature_gri"] = (out["g"] - 2.0 * out["r"] + out["i"]).astype("float32")
    out["curvature_riz"] = (out["r"] - 2.0 * out["i"] + out["z"]).astype("float32")
    out["redshift_magmean"] = (redshift * out["mag_mean"]).astype("float32")
    out["redshift_magstd"] = (redshift * out["mag_std"]).astype("float32")
    out["redshift_ug"] = (redshift * out["u_g"]).astype("float32")
    out["redshift_gr"] = (redshift * out["g_r"]).astype("float32")

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
    med = float(vals.median()) if np.isfinite(vals.median()) else 0.0
    arr = vals.fillna(med).to_numpy(dtype="float64")
    quantiles = np.linspace(0.0, 1.0, q + 1)
    edges = np.unique(np.quantile(arr, quantiles))
    if len(edges) <= 2:
        tr = np.zeros(len(train_s), dtype="int16")
        te = np.zeros(len(test_s), dtype="int16")
        return tr, te
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
    all_vals = pd.concat([train_vals, test_vals], axis=0)
    cats = pd.Categorical(all_vals)
    codes = cats.codes.astype("int32")
    X[name] = codes[: len(X)].astype("int32")
    X_test[name] = codes[len(X) :].astype("int32")


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
    spec_pop_tr = X["spec_pop"].astype(str)
    spec_pop_te = X_test["spec_pop"].astype(str)

    add_combo_code(X, X_test, "redshift_qbin_x_spec", [X["redshift_qbin"], spec_tr], [X_test["redshift_qbin"], spec_te])
    add_combo_code(X, X_test, "redshift_qbin_x_pop", [X["redshift_qbin"], pop_tr], [X_test["redshift_qbin"], pop_te])
    add_combo_code(
        X,
        X_test,
        "redshift_qbin_x_spec_pop",
        [X["redshift_qbin"], spec_pop_tr],
        [X_test["redshift_qbin"], spec_pop_te],
    )
    add_combo_code(X, X_test, "sky_region", [X["alpha_qbin"], X["delta_qbin"]], [X_test["alpha_qbin"], X_test["delta_qbin"]])
    add_combo_code(X, X_test, "ug_gr_region", [X["u_g_qbin"], X["g_r_qbin"]], [X_test["u_g_qbin"], X_test["g_r_qbin"]])
    add_combo_code(X, X_test, "ri_iz_region", [X["r_i_qbin"], X["i_z_qbin"]], [X_test["r_i_qbin"], X_test["i_z_qbin"]])
    add_combo_code(X, X_test, "rz_ug_region", [X["redshift_qbin"], X["u_g_qbin"]], [X_test["redshift_qbin"], X_test["u_g_qbin"]])
    add_combo_code(X, X_test, "rz_gr_region", [X["redshift_qbin"], X["g_r_qbin"]], [X_test["redshift_qbin"], X_test["g_r_qbin"]])

    for col in X.columns:
        if col not in LOW_CARD_CAT_COLS and pd.api.types.is_numeric_dtype(X[col]):
            if str(X[col].dtype).startswith("int"):
                continue
            X[col] = X[col].replace([np.inf, -np.inf], np.nan).astype("float32")
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return X, X_test


def fit_apply_target_stats(X_fit, y_fit, frames_to_transform):
    # OOF-safe smoothed class-prior features.  Each validation fold and test fold
    # sees priors estimated only from that fold's training rows.
    y_fit = np.asarray(y_fit, dtype=object)
    prior = np.array([np.mean(y_fit == cls) for cls in CLASS_ORDER], dtype="float64")
    out_frames = [fr.copy() for fr in frames_to_transform]

    for key in TE_KEYS:
        key_fit = X_fit[key]
        counts = pd.crosstab(key_fit, pd.Categorical(y_fit, categories=CLASS_ORDER))
        counts = counts.reindex(columns=CLASS_ORDER, fill_value=0)
        row_counts = counts.sum(axis=1).to_numpy(dtype="float64")
        alpha = 150.0 if ("region" in key or "_x_" in key) else 80.0
        probs = (counts.to_numpy(dtype="float64") + alpha * prior.reshape(1, -1)) / (row_counts.reshape(-1, 1) + alpha)
        for j, cls in enumerate(CLASS_ORDER):
            mapping = pd.Series(probs[:, j], index=counts.index)
            feat_name = f"te_{key}_{cls.lower()}"
            for fr in out_frames:
                mapped = pd.to_numeric(pd.Series(np.asarray(fr[key].map(mapping)), index=fr.index), errors="coerce")
                fr[feat_name] = mapped.fillna(prior[j]).astype("float32")
    return out_frames


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
            n_estimators=760,
            learning_rate=0.041,
            num_leaves=128,
            min_child_samples=65,
            subsample=0.90,
            colsample_bytree=0.86,
            reg_alpha=0.05,
            reg_lambda=1.60,
        )
    elif profile == "regularized":
        params.update(
            n_estimators=660,
            learning_rate=0.048,
            num_leaves=80,
            min_child_samples=115,
            subsample=0.92,
            colsample_bytree=0.92,
            reg_alpha=0.12,
            reg_lambda=2.40,
        )
    elif profile == "deep":
        params.update(
            n_estimators=700,
            learning_rate=0.042,
            num_leaves=160,
            min_child_samples=90,
            subsample=0.88,
            colsample_bytree=0.84,
            reg_alpha=0.08,
            reg_lambda=1.80,
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


def class_indices(y):
    return np.array([CLASS_TO_INDEX[v] for v in y], dtype="int8")


def confusion_counts(y_idx, pred_idx):
    return np.bincount(y_idx.astype("int16") * 3 + pred_idx.astype("int16"), minlength=9).reshape(3, 3)


def balanced_score_from_conf(conf):
    denom = conf.sum(axis=1).astype("float64")
    recalls = np.divide(np.diag(conf), denom, out=np.zeros(3, dtype="float64"), where=denom > 0)
    return float(np.mean(recalls))


def score_with_multipliers(proba, y_idx, multipliers):
    pred = np.argmax(proba * multipliers.reshape(1, -1), axis=1).astype("int8")
    return balanced_score_from_conf(confusion_counts(y_idx, pred))


def tune_global_multipliers(proba, y_true):
    y_idx = class_indices(y_true)
    best_score = score_with_multipliers(proba, y_idx, np.ones(3, dtype="float64"))
    best = np.ones(3, dtype="float64")

    for qso_mult in np.arange(0.68, 1.421, 0.025):
        for star_mult in np.arange(0.70, 1.901, 0.025):
            mult = np.array([1.0, qso_mult, star_mult], dtype="float64")
            score = score_with_multipliers(proba, y_idx, mult)
            if score > best_score:
                best_score = score
                best = mult

    q0, s0 = best[1], best[2]
    for step in (0.005, 0.0015):
        for qso_mult in np.arange(max(0.45, q0 - 0.07), q0 + 0.0701, step):
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
    global_mult, best_score = tune_global_multipliers(proba, y_true)
    adjusted = proba * global_mult.reshape(1, -1)
    current_pred = np.argmax(adjusted, axis=1).astype("int8")
    total_conf = confusion_counts(y_idx, current_pred)
    best_score = balanced_score_from_conf(total_conf)

    n_segments = int(np.max(segments)) + 1
    seg_mult = np.ones((n_segments, 3), dtype="float64")
    seg_indices = [np.flatnonzero(segments == s) for s in range(n_segments)]
    min_count = 900

    for pass_id in range(3):
        improved = False
        for seg, idx in enumerate(seg_indices):
            n = len(idx)
            if n < min_count:
                continue
            old_conf = confusion_counts(y_idx[idx], current_pred[idx])
            base_q, base_s = seg_mult[seg, 1], seg_mult[seg, 2]
            if pass_id == 0:
                q_grid = np.unique(np.r_[1.0, np.linspace(0.70, 1.38, 18), np.linspace(0.86, 1.18, 13)])
                s_grid = np.unique(np.r_[1.0, np.linspace(0.55, 1.80, 22), np.linspace(0.82, 1.25, 14)])
            else:
                q_grid = np.unique(np.clip(base_q * np.linspace(0.88, 1.14, 14), 0.55, 1.70))
                s_grid = np.unique(np.clip(base_s * np.linspace(0.82, 1.22, 17), 0.35, 2.20))
            local_best_score = best_score
            local_best = seg_mult[seg].copy()
            local_best_pred = current_pred[idx]
            seg_base = adjusted[idx]
            for q in q_grid:
                for s in s_grid:
                    trial_mult = np.array([1.0, q, s], dtype="float64")
                    trial_pred = np.argmax(seg_base * trial_mult.reshape(1, -1), axis=1).astype("int8")
                    trial_conf = confusion_counts(y_idx[idx], trial_pred)
                    score = balanced_score_from_conf(total_conf - old_conf + trial_conf)
                    if score > local_best_score + 1e-8:
                        local_best_score = score
                        local_best = trial_mult
                        local_best_pred = trial_pred
            if local_best_score > best_score + 1e-8:
                # Light shrinkage for smaller cells: preserves robust large-cell
                # corrections while avoiding brittle thresholds in tail segments.
                strength = min(1.0, max(0.35, np.sqrt(n / 7000.0)))
                shrunk = np.array([1.0, 1.0 + strength * (local_best[1] - 1.0), 1.0 + strength * (local_best[2] - 1.0)])
                shrunk_pred = np.argmax(seg_base * shrunk.reshape(1, -1), axis=1).astype("int8")
                shrunk_conf = confusion_counts(y_idx[idx], shrunk_pred)
                shrunk_score = balanced_score_from_conf(total_conf - old_conf + shrunk_conf)
                if shrunk_score >= best_score + 1e-8:
                    seg_mult[seg] = shrunk
                    current_pred[idx] = shrunk_pred
                    total_conf = total_conf - old_conf + shrunk_conf
                    best_score = shrunk_score
                else:
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


def stratified_sample_for_quick(train_df, test_df, n):
    if n <= 0 or n >= len(train_df):
        return train_df, test_df
    sampled = train_df.groupby("class", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(train_df)))), random_state=SEED)
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

    # Smoke-test hook only; the gate does not set this, so full data is used for
    # real scoring as required.
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
    n_splits = 4 if len(train_df) >= 100000 else 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    profiles = ["main", "wide", "regularized", "deep"]

    oof_proba = np.zeros((len(train_df), len(CLASS_ORDER)), dtype="float32")
    test_proba = np.zeros((len(test_df), len(CLASS_ORDER)), dtype="float64")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y)):
        X_tr_base = X_base.iloc[tr_idx]
        X_val_base = X_base.iloc[val_idx]
        y_tr = y[tr_idx]
        X_tr, X_val, X_test_fold = fit_apply_target_stats(X_tr_base, y_tr, [X_tr_base, X_val_base, X_test_base])

        model = make_model(SEED + 37 * fold, profiles[fold % len(profiles)])
        model.fit(X_tr, y_tr, categorical_feature=categorical_features)

        oof_proba[val_idx] = align_probabilities(model, model.predict_proba(X_val)).astype("float32")
        test_proba += align_probabilities(model, model.predict_proba(X_test_fold)).astype("float64") / n_splits

        del X_tr, X_val, X_test_fold, model
        gc.collect()

    train_segments = calibration_segments(train_df)
    test_segments = calibration_segments(test_df)
    global_mult, seg_mult = tune_segment_multipliers(oof_proba.astype("float64"), y, train_segments)
    adjusted = apply_adjustment(test_proba, test_segments, global_mult, seg_mult)
    pred = CLASS_ORDER[np.argmax(adjusted, axis=1)]

    pd.DataFrame({"id": test_ids, "class": pred}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
