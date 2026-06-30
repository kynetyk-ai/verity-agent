"""
Stellar classification - strong ensemble submission.
Target: balanced accuracy > 0.9714

Strategy:
1. LightGBM: 2000 trees, 255 leaves, lr=0.03, n_jobs=-1
2. XGBoost: 1000 trees, hist method, n_jobs=-1
3. Rich feature engineering: one-hot categoricals, polynomial color indices,
   composite diagnostic features (QSO/STAR/GALAXY scores)
4. Prior-calibrated ensemble (60% LGB + 40% XGB)
5. Sample weights for balanced accuracy
"""

import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR  = os.environ.get("VERITY_OUT",  "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET = "class"
ID_COL = "id"

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("Class distribution:")
print(train[TARGET].value_counts())

def engineer_features(df):
    df = df.copy()

    # Colour indices (photometric band differences)
    df["u_g"]  = df["u"] - df["g"]
    df["g_r"]  = df["g"] - df["r"]
    df["r_i"]  = df["r"] - df["i"]
    df["i_z"]  = df["i"] - df["z"]
    df["u_r"]  = df["u"] - df["r"]
    df["u_z"]  = df["u"] - df["z"]
    df["g_z"]  = df["g"] - df["z"]
    df["g_i"]  = df["g"] - df["i"]
    df["r_z"]  = df["r"] - df["z"]
    df["u_i"]  = df["u"] - df["i"]

    # Second-order colour combinations
    df["u_g_r"]           = df["u_g"] - df["g_r"]
    df["g_r_i"]           = df["g_r"] - df["r_i"]
    df["r_i_z"]           = df["r_i"] - df["i_z"]
    df["u_g_minus_r_i"]   = df["u_g"] - df["r_i"]
    df["g_r_minus_i_z"]   = df["g_r"] - df["i_z"]
    df["u_g_sq"]          = df["u_g"] ** 2
    df["g_r_sq"]          = df["g_r"] ** 2
    df["r_i_sq"]          = df["r_i"] ** 2
    df["i_z_sq"]          = df["i_z"] ** 2
    df["u_g_cb"]          = df["u_g"] ** 3
    df["g_r_cb"]          = df["g_r"] ** 3

    # Redshift features
    df["log1p_redshift"]  = np.log1p(np.clip(df["redshift"], 0, None))
    df["redshift_sq"]     = df["redshift"] ** 2
    df["redshift_cb"]     = df["redshift"] ** 3
    df["sqrt_redshift"]   = np.sqrt(np.clip(df["redshift"], 0, None))
    df["redshift_abs"]    = np.abs(df["redshift"])
    df["redshift_neg"]    = (df["redshift"] < 0).astype(np.int8)

    # Magnitude statistics
    mags = df[["u", "g", "r", "i", "z"]]
    df["mean_mag"]   = mags.mean(axis=1)
    df["std_mag"]    = mags.std(axis=1)
    df["mag_range"]  = df["u"] - df["z"]
    df["max_mag"]    = mags.max(axis=1)
    df["min_mag"]    = mags.min(axis=1)
    df["mag_skew"]   = (df["u"] + df["z"] - 2 * df["r"])
    df["mag_kurt"]   = (df["u"] + df["z"] - 2 * df["g"])

    # Sky position features
    df["sin_alpha"]  = np.sin(np.deg2rad(df["alpha"]))
    df["cos_alpha"]  = np.cos(np.deg2rad(df["alpha"]))
    df["sin_delta"]  = np.sin(np.deg2rad(df["delta"]))
    df["cos_delta"]  = np.cos(np.deg2rad(df["delta"]))
    df["alpha_delta_prod"] = df["alpha"] * df["delta"]
    df["sin_2alpha"] = np.sin(2 * np.deg2rad(df["alpha"]))
    df["cos_2alpha"] = np.cos(2 * np.deg2rad(df["alpha"]))
    df["sin_2delta"] = np.sin(2 * np.deg2rad(df["delta"]))

    # Encode categoricals (ordinal)
    spectral_map   = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    galaxy_pop_map = {"Blue_Cloud": 0, "Red_Sequence": 1}

    df["spectral_type_enc"]     = df["spectral_type"].map(spectral_map).fillna(-1).astype(int)
    df["galaxy_population_enc"] = df["galaxy_population"].map(galaxy_pop_map).fillna(-1).astype(int)

    # One-hot encode categoricals
    for stype in spectral_map.keys():
        df[f"spec_{stype.replace('/', '_')}"] = (df["spectral_type"] == stype).astype(np.int8)
    for gpop in galaxy_pop_map.keys():
        df[f"galpop_{gpop}"] = (df["galaxy_population"] == gpop).astype(np.int8)

    # Interaction: redshift x colour indices
    df["redshift_x_g_r"]     = df["redshift"] * df["g_r"]
    df["redshift_x_u_g"]     = df["redshift"] * df["u_g"]
    df["redshift_x_r_i"]     = df["redshift"] * df["r_i"]
    df["redshift_x_i_z"]     = df["redshift"] * df["i_z"]
    df["redshift_x_u_z"]     = df["redshift"] * df["u_z"]
    df["redshift_x_g_i"]     = df["redshift"] * df["g_i"]
    df["log_rs_x_g_r"]       = df["log1p_redshift"] * df["g_r"]
    df["log_rs_x_u_g"]       = df["log1p_redshift"] * df["u_g"]
    df["redshift_sq_x_g_r"]  = df["redshift_sq"] * df["g_r"]

    # Spectral type x colour interactions
    df["spectral_x_g_r"]        = df["spectral_type_enc"] * df["g_r"]
    df["spectral_x_u_g"]        = df["spectral_type_enc"] * df["u_g"]
    df["galaxypop_x_redshift"]  = df["galaxy_population_enc"] * df["redshift"]
    df["galaxypop_x_g_r"]       = df["galaxy_population_enc"] * df["g_r"]
    df["galaxypop_x_u_g"]       = df["galaxy_population_enc"] * df["u_g"]

    # Ratio features
    eps = 1e-6
    df["u_over_g"]      = df["u"] / (df["g"] + eps)
    df["r_over_i"]      = df["r"] / (df["i"] + eps)
    df["g_over_z"]      = df["g"] / (df["z"] + eps)
    df["g_r_over_u_g"]  = df["g_r"] / (df["u_g"].abs() + eps)

    # Color-redshift composite diagnostics
    df["qso_score"]    = -df["u_g"] - df["g_r"] + df["log1p_redshift"]
    df["star_score"]   = -df["redshift_abs"] + df["g_r"]
    df["galaxy_score"] = df["redshift"] * (df["u_g"] + df["g_r"])

    return df


train = engineer_features(train)
test  = engineer_features(test)

# Encode target
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
n_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

# Define feature set
DROP_COLS = [ID_COL, TARGET, "spectral_type", "galaxy_population"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
print(f"Features: {len(FEATURE_COLS)}")

X_train = train[FEATURE_COLS].values.astype(np.float32)
X_test  = test[FEATURE_COLS].values.astype(np.float32)

# Class weights for balanced accuracy
class_counts = np.bincount(y)
class_weights = (len(y) / (n_classes * class_counts)).tolist()
sample_weight = np.array([class_weights[yi] for yi in y], dtype=np.float32)

print(f"Class weights: {dict(zip(le.classes_, [round(w, 4) for w in class_weights]))}")

# LightGBM model (main workhorse)
print("Training LightGBM on full data...")
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

# XGBoost model (complementary)
print("Training XGBoost on full data...")
xgb_params = {
    "objective":        "multi:softprob",
    "num_class":        n_classes,
    "eval_metric":      "mlogloss",
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.05,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "n_jobs":           -1,
    "random_state":     SEED,
    "verbosity":        0,
    "max_bin":          256,
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y, sample_weight=sample_weight)
print("XGBoost done.")

# Final predictions
print("Generating predictions...")
lgb_proba = lgb_model.predict_proba(X_test)
xgb_proba = xgb_model.predict_proba(X_test)

# Ensemble: 60% LGB + 40% XGB (LGB typically stronger on tabular)
W_LGB = 0.6
W_XGB = 0.4
ensemble_proba = W_LGB * lgb_proba + W_XGB * xgb_proba

# Apply prior calibration for balanced accuracy
class_priors = class_counts / len(y)
calibrated_proba = ensemble_proba / class_priors
calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)

preds_idx    = np.argmax(calibrated_proba, axis=1)
preds_labels = le.inverse_transform(preds_idx)

# Write output
out_df = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: preds_labels,
})
out_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

print(f"Done. Wrote {len(out_df)} predictions to {OUT_DIR}/predictions.csv")
print("Prediction distribution:")
print(out_df[TARGET].value_counts())
