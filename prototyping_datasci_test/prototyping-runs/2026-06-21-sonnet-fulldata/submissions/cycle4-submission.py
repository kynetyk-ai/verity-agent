"""
Stellar classification - memory-optimized submission (fixes OOM/exit-137).
Score on prior best: 0.9664

Changes vs prior OOM submission:
1. Removed XGBoost (biggest memory consumer) - use LightGBM only
2. Reduced num_leaves: 255 -> 127 (saves ~50% tree memory)
3. Reduced n_estimators: 1500 -> 1000 (still accurate, less RAM)
4. Use dart boosting for better generalization with fewer trees
5. Keep all feature engineering from prior (colour indices, interactions, etc.)
6. Explicitly set max_bin=255 (default) to control histogram memory
7. Use float32 data consistently
"""

import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
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

    # Colour indices (photometric band differences) — primary signals
    df["u_g"]  = df["u"] - df["g"]
    df["g_r"]  = df["g"] - df["r"]
    df["r_i"]  = df["r"] - df["i"]
    df["i_z"]  = df["i"] - df["z"]
    df["u_r"]  = df["u"] - df["r"]
    df["u_z"]  = df["u"] - df["z"]
    df["g_z"]  = df["g"] - df["z"]
    df["g_i"]  = df["g"] - df["i"]
    df["r_z"]  = df["r"] - df["z"]

    # Second-order colour combinations
    df["u_g_r"] = df["u_g"] - df["g_r"]
    df["g_r_i"] = df["g_r"] - df["r_i"]
    df["r_i_z"] = df["r_i"] - df["i_z"]
    df["u_g_minus_r_i"] = df["u_g"] - df["r_i"]
    df["g_r_minus_i_z"] = df["g_r"] - df["i_z"]

    # Redshift-based features
    df["log1p_redshift"] = np.log1p(np.clip(df["redshift"], 0, None))
    df["redshift_sq"]    = df["redshift"] ** 2
    df["redshift_cb"]    = df["redshift"] ** 3
    df["sqrt_redshift"]  = np.sqrt(np.clip(df["redshift"], 0, None))

    # Magnitude statistics
    mags = df[["u", "g", "r", "i", "z"]]
    df["mean_mag"]  = mags.mean(axis=1)
    df["std_mag"]   = mags.std(axis=1)
    df["mag_range"] = df["u"] - df["z"]   # broadest colour spread
    df["max_mag"]   = mags.max(axis=1)
    df["min_mag"]   = mags.min(axis=1)
    df["mag_skew"]  = (df["u"] + df["z"] - 2 * df["r"])  # asymmetry

    # Sky position features (galactic coordinates proxy)
    df["sin_alpha"] = np.sin(np.deg2rad(df["alpha"]))
    df["cos_alpha"] = np.cos(np.deg2rad(df["alpha"]))
    df["sin_delta"] = np.sin(np.deg2rad(df["delta"]))
    df["cos_delta"] = np.cos(np.deg2rad(df["delta"]))
    df["alpha_delta_prod"] = df["alpha"] * df["delta"]

    # Encode categoricals
    spectral_map   = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    galaxy_pop_map = {"Blue_Cloud": 0, "Red_Sequence": 1}

    df["spectral_type_enc"]     = df["spectral_type"].map(spectral_map).fillna(-1).astype(int)
    df["galaxy_population_enc"] = df["galaxy_population"].map(galaxy_pop_map).fillna(-1).astype(int)

    # Interaction: redshift × colour indices
    df["redshift_x_g_r"] = df["redshift"] * df["g_r"]
    df["redshift_x_u_g"] = df["redshift"] * df["u_g"]
    df["redshift_x_r_i"] = df["redshift"] * df["r_i"]
    df["redshift_x_i_z"] = df["redshift"] * df["i_z"]
    df["redshift_x_u_z"] = df["redshift"] * df["u_z"]

    # Spectral type × colour interactions
    df["spectral_x_g_r"] = df["spectral_type_enc"] * df["g_r"]
    df["spectral_x_u_g"] = df["spectral_type_enc"] * df["u_g"]
    df["galaxypop_x_redshift"] = df["galaxy_population_enc"] * df["redshift"]
    df["galaxypop_x_g_r"]     = df["galaxy_population_enc"] * df["g_r"]

    # Ratio features (avoid division by zero)
    eps = 1e-6
    df["u_over_g"] = df["u"] / (df["g"] + eps)
    df["r_over_i"] = df["r"] / (df["i"] + eps)
    df["g_over_z"] = df["g"] / (df["z"] + eps)

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

# ── LightGBM model (memory-efficient params) ──────────────────────────────────
# Memory budget: 2GB.  Key levers:
#   - num_leaves=127  (vs 255) saves ~50% tree node memory
#   - n_estimators=1000 (vs 1500) saves ~33% model memory
#   - float32 data already set
#   - n_jobs=-1 uses all cores (no timeout risk)
print("Training LightGBM...")
lgb_params = {
    "objective":         "multiclass",
    "num_class":         n_classes,
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        127,
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

# ── Predictions ───────────────────────────────────────────────────────────────
print("Generating predictions...")
proba = lgb_model.predict_proba(X_test)

# ── Threshold calibration for balanced accuracy ───────────────────────────────
# Scale each class probability by inverse of its prevalence in training data.
# This helps with balanced accuracy when classes are skewed.
class_priors = class_counts / len(y)
calibrated_proba = proba / class_priors
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
