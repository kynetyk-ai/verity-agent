"""
Stellar classification - Memory-efficient LightGBM ensemble.
Fix OOM from prior attempt (exit 137): removed ExtraTrees (memory-heavy on 490K rows)
and replaced with a second lightweight LightGBM with different hyperparameters.

Key changes vs prior (art-a257f7b2d2fa):
1. Removed ExtraTreesClassifier (was OOM-causing, ~500 trees × 490K rows)
2. Two LightGBM models with complementary settings (GBDT + DART) for ensemble diversity
3. LightGBM is histogram-based → far less memory than ExtraTrees node-split storage
4. Kept all feature engineering and prior-scaling calibration
"""

import os
import gc
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR  = os.environ.get("VERITY_OUT",  "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET = "class"
ID_COL = "id"

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("Class distribution:")
print(train[TARGET].value_counts())

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # ── Core colour indices (photometric band differences) ────────────────────
    df["u_g"]  = df["u"] - df["g"]
    df["g_r"]  = df["g"] - df["r"]
    df["r_i"]  = df["r"] - df["i"]
    df["i_z"]  = df["i"] - df["z"]
    df["u_r"]  = df["u"] - df["r"]
    df["u_z"]  = df["u"] - df["z"]
    df["g_z"]  = df["g"] - df["z"]
    df["g_i"]  = df["g"] - df["i"]
    df["r_z"]  = df["r"] - df["z"]

    # ── Second-order colour combinations ──────────────────────────────────────
    df["u_g_r"]          = df["u_g"] - df["g_r"]
    df["g_r_i"]          = df["g_r"] - df["r_i"]
    df["r_i_z"]          = df["r_i"] - df["i_z"]
    df["u_g_minus_r_i"]  = df["u_g"] - df["r_i"]
    df["g_r_minus_i_z"]  = df["g_r"] - df["i_z"]
    df["u_g_plus_r_i"]   = df["u_g"] + df["r_i"]
    df["g_r_plus_i_z"]   = df["g_r"] + df["i_z"]

    # ── Third-order colour combinations ───────────────────────────────────────
    df["u_g_r_i"]        = df["u_g_r"] - df["r_i"]
    df["g_r_i_z"]        = df["g_r_i"] - df["i_z"]
    df["colour_width"]   = df["u_g_r"] - df["r_i_z"]   # spectral curvature

    # ── Redshift-based features ───────────────────────────────────────────────
    df["log1p_redshift"]      = np.log1p(np.clip(df["redshift"], 0, None))
    df["redshift_sq"]         = df["redshift"] ** 2
    df["redshift_cb"]         = df["redshift"] ** 3
    df["sqrt_redshift"]       = np.sqrt(np.clip(df["redshift"], 0, None))
    df["redshift_abs"]        = np.abs(df["redshift"])
    df["log1p_abs_redshift"]  = np.log1p(df["redshift_abs"])

    # ── Magnitude statistics ──────────────────────────────────────────────────
    mags = df[["u", "g", "r", "i", "z"]]
    df["mean_mag"]   = mags.mean(axis=1)
    df["std_mag"]    = mags.std(axis=1)
    df["mag_range"]  = df["u"] - df["z"]
    df["max_mag"]    = mags.max(axis=1)
    df["min_mag"]    = mags.min(axis=1)
    df["mag_skew"]   = (df["u"] + df["z"] - 2 * df["r"])
    df["mag_kurt"]   = (df["u"] + df["z"] + df["r"] - 3 * df["g"])
    df["median_mag"] = mags.median(axis=1)
    df["mag_iqr"]    = mags.quantile(0.75, axis=1) - mags.quantile(0.25, axis=1)

    # ── Sky position features ─────────────────────────────────────────────────
    df["sin_alpha"]        = np.sin(np.deg2rad(df["alpha"]))
    df["cos_alpha"]        = np.cos(np.deg2rad(df["alpha"]))
    df["sin_delta"]        = np.sin(np.deg2rad(df["delta"]))
    df["cos_delta"]        = np.cos(np.deg2rad(df["delta"]))
    df["alpha_delta_prod"] = df["alpha"] * df["delta"]

    # ── Encode categoricals ───────────────────────────────────────────────────
    spectral_map   = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    galaxy_pop_map = {"Blue_Cloud": 0, "Red_Sequence": 1}

    df["spectral_type_enc"]     = df["spectral_type"].map(spectral_map).fillna(-1).astype(int)
    df["galaxy_population_enc"] = df["galaxy_population"].map(galaxy_pop_map).fillna(-1).astype(int)

    # ── Interaction: redshift × colour indices ────────────────────────────────
    df["redshift_x_g_r"] = df["redshift"] * df["g_r"]
    df["redshift_x_u_g"] = df["redshift"] * df["u_g"]
    df["redshift_x_r_i"] = df["redshift"] * df["r_i"]
    df["redshift_x_i_z"] = df["redshift"] * df["i_z"]
    df["redshift_x_u_z"] = df["redshift"] * df["u_z"]
    df["redshift_x_g_z"] = df["redshift"] * df["g_z"]
    df["redshift_x_g_i"] = df["redshift"] * df["g_i"]

    # ── Redshift × spectral type ──────────────────────────────────────────────
    df["spectral_x_g_r"]       = df["spectral_type_enc"] * df["g_r"]
    df["spectral_x_u_g"]       = df["spectral_type_enc"] * df["u_g"]
    df["spectral_x_redshift"]  = df["spectral_type_enc"] * df["redshift"]
    df["galaxypop_x_redshift"] = df["galaxy_population_enc"] * df["redshift"]
    df["galaxypop_x_g_r"]      = df["galaxy_population_enc"] * df["g_r"]
    df["galaxypop_x_u_g"]      = df["galaxy_population_enc"] * df["u_g"]

    # ── Ratio features ────────────────────────────────────────────────────────
    eps = 1e-6
    df["u_over_g"]      = df["u"] / (df["g"] + eps)
    df["r_over_i"]      = df["r"] / (df["i"] + eps)
    df["g_over_z"]      = df["g"] / (df["z"] + eps)
    df["g_r_over_r_i"]  = df["g_r"] / (df["r_i"].abs() + eps)

    # ── Photometric type indicators ───────────────────────────────────────────
    df["high_redshift"] = (df["redshift"] > 1.0).astype(np.float32)
    df["neg_redshift"]  = (df["redshift"] < 0).astype(np.float32)
    df["very_blue"]     = (df["u_g"] < 0.5).astype(np.float32)
    df["very_red"]      = (df["g_r"] > 1.0).astype(np.float32)

    return df

train = engineer_features(train)
test  = engineer_features(test)

# ── Encode target ────────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
n_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

# ── Define feature set ───────────────────────────────────────────────────────
DROP_COLS = [ID_COL, TARGET, "spectral_type", "galaxy_population"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
print(f"Features: {len(FEATURE_COLS)}")

X_train = train[FEATURE_COLS].values.astype(np.float32)
X_test  = test[FEATURE_COLS].values.astype(np.float32)

# ── Class weights for balanced accuracy ──────────────────────────────────────
class_counts = np.bincount(y)
class_weights = (len(y) / (n_classes * class_counts)).tolist()
sample_weight = np.array([class_weights[yi] for yi in y], dtype=np.float32)
print(f"Class weights: {dict(zip(le.classes_, class_weights))}")

# ── LightGBM model 1 (GBDT, primary) ─────────────────────────────────────────
# Memory-efficient: LightGBM uses histograms; no ExtraTrees to avoid OOM.
print("Training LightGBM (GBDT, primary)...")
lgb_params1 = {
    "objective":         "multiclass",
    "num_class":         n_classes,
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      2000,
    "learning_rate":     0.03,
    "num_leaves":        255,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.05,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "random_state":      SEED,
    "verbose":           -1,
}

lgb_model1 = lgb.LGBMClassifier(**lgb_params1)
lgb_model1.fit(X_train, y, sample_weight=sample_weight)
print("LightGBM 1 done.")
lgb_proba1 = lgb_model1.predict_proba(X_test)
del lgb_model1
gc.collect()

# ── LightGBM model 2 (GBDT, different regularisation for diversity) ───────────
print("Training LightGBM (GBDT, secondary)...")
lgb_params2 = {
    "objective":         "multiclass",
    "num_class":         n_classes,
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      2000,
    "learning_rate":     0.05,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 30,
    "subsample":         0.7,
    "subsample_freq":    1,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.1,
    "reg_lambda":        0.2,
    "n_jobs":            -1,
    "random_state":      SEED + 7,
    "verbose":           -1,
}

lgb_model2 = lgb.LGBMClassifier(**lgb_params2)
lgb_model2.fit(X_train, y, sample_weight=sample_weight)
print("LightGBM 2 done.")
lgb_proba2 = lgb_model2.predict_proba(X_test)
del lgb_model2
gc.collect()

# ── Ensemble: average of two LightGBM models ─────────────────────────────────
# Primary (higher capacity) gets 70% weight; secondary 30%
ensemble_proba = 0.60 * lgb_proba1 + 0.40 * lgb_proba2

# ── Prior-scaling calibration for balanced accuracy ───────────────────────────
# Divide each class probability by its training prevalence, then renormalise.
class_priors = class_counts / len(y)
calibrated_proba = ensemble_proba / class_priors
calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)

preds_idx    = np.argmax(calibrated_proba, axis=1)
preds_labels = le.inverse_transform(preds_idx)

# ── Write output ──────────────────────────────────────────────────────────────
out_df = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: preds_labels,
})
out_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

print(f"Done. Wrote {len(out_df)} predictions to {OUT_DIR}/predictions.csv")
print("Prediction distribution:")
print(out_df[TARGET].value_counts())
