"""
Stellar classification - Stronger model + ET ensemble.
Score on prior best: 0.9665 (art-a702aa65dfb5)

Key improvements vs prior:
1. Stronger LightGBM: num_leaves=255, n_estimators=2000, lr=0.03
   (Prior was limited to 127/1000 due to XGBoost memory competition; XGBoost removed now)
2. Add ExtraTreesClassifier ensemble (memory-efficient, diverse, fast with n_jobs=-1)
3. Richer feature engineering: higher-order colour terms, more interactions
4. Maintain simple+effective prior-scaling calibration for balanced accuracy
"""

import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score

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
    df["log1p_redshift"] = np.log1p(np.clip(df["redshift"], 0, None))
    df["redshift_sq"]    = df["redshift"] ** 2
    df["redshift_cb"]    = df["redshift"] ** 3
    df["sqrt_redshift"]  = np.sqrt(np.clip(df["redshift"], 0, None))
    df["redshift_abs"]   = np.abs(df["redshift"])
    df["log1p_abs_redshift"] = np.log1p(df["redshift_abs"])

    # ── Magnitude statistics ──────────────────────────────────────────────────
    mags = df[["u", "g", "r", "i", "z"]]
    df["mean_mag"]   = mags.mean(axis=1)
    df["std_mag"]    = mags.std(axis=1)
    df["mag_range"]  = df["u"] - df["z"]
    df["max_mag"]    = mags.max(axis=1)
    df["min_mag"]    = mags.min(axis=1)
    df["mag_skew"]   = (df["u"] + df["z"] - 2 * df["r"])
    df["mag_kurt"]   = (df["u"] + df["z"] + df["r"] - 3 * df["g"])  # brightness asymmetry
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
    df["u_over_g"]  = df["u"] / (df["g"] + eps)
    df["r_over_i"]  = df["r"] / (df["i"] + eps)
    df["g_over_z"]  = df["g"] / (df["z"] + eps)
    df["g_r_over_r_i"] = df["g_r"] / (df["r_i"].abs() + eps)

    # ── Photometric type indicators (hard thresholds that are class-separating) ─
    # QSOs tend to have large redshift and unusual colours
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

# ── LightGBM model — stronger than prior (num_leaves 127→255, trees 1000→2000) ─
# Memory is safe: prior OOM was caused by XGBoost co-training; that's removed now.
# Using all CPU cores for parallel training (n_jobs=-1).
print("Training LightGBM (dominant model)...")
lgb_params = {
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

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y, sample_weight=sample_weight)
print("LightGBM done.")

# ── ExtraTreesClassifier (diverse, fast, memory-efficient) ───────────────────
# ET is much faster than GBM — 500 trees on 490K rows ≈ 1-2 min with n_jobs=-1.
print("Training ExtraTrees (ensemble diversity)...")
et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED + 1,
)
et_model.fit(X_train, y)
print("ExtraTrees done.")

# ── Predictions ───────────────────────────────────────────────────────────────
print("Generating predictions...")
lgb_proba = lgb_model.predict_proba(X_test)
et_proba  = et_model.predict_proba(X_test)

# Ensemble: 80% LGB + 20% ET (LGB is dominant; ET adds orthogonal diversity)
BLEND_W = 0.80
ensemble_proba = BLEND_W * lgb_proba + (1 - BLEND_W) * et_proba

# ── Prior-scaling calibration for balanced accuracy ───────────────────────────
# Divide each class probability by its training prevalence, then renormalise.
# This rescales the decision boundary to treat all classes equally → improves balanced accuracy.
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
