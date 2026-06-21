"""
Stellar classification - CatBoost + LightGBM ensemble with OOF target encoding.
Target: 0.9714+ balanced accuracy

Key improvements vs prior (0.9661):
1. CatBoost: natively handles categoricals, complementary to LightGBM
2. Stronger LightGBM: num_leaves=511, n_estimators=3000, lr=0.02
3. OOF target-mean encoding for spectral_type × galaxy_population
4. More features: color index ratios, extended interactions
5. Ensemble weight: 70% LGB + 30% CatBoost
6. Prior-scaling calibration for balanced accuracy
"""

import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

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

    # ── Combined categorical feature ──────────────────────────────────────────
    df["spectral_galaxy_combo"] = df["spectral_type_enc"] * 2 + df["galaxy_population_enc"]

    # ── Interaction: redshift × colour indices ────────────────────────────────
    df["redshift_x_g_r"] = df["redshift"] * df["g_r"]
    df["redshift_x_u_g"] = df["redshift"] * df["u_g"]
    df["redshift_x_r_i"] = df["redshift"] * df["r_i"]
    df["redshift_x_i_z"] = df["redshift"] * df["i_z"]
    df["redshift_x_u_z"] = df["redshift"] * df["u_z"]
    df["redshift_x_g_z"] = df["redshift"] * df["g_z"]
    df["redshift_x_g_i"] = df["redshift"] * df["g_i"]

    # ── Redshift × spectral/galaxy features ───────────────────────────────────
    df["spectral_x_g_r"]       = df["spectral_type_enc"] * df["g_r"]
    df["spectral_x_u_g"]       = df["spectral_type_enc"] * df["u_g"]
    df["spectral_x_redshift"]  = df["spectral_type_enc"] * df["redshift"]
    df["galaxypop_x_redshift"] = df["galaxy_population_enc"] * df["redshift"]
    df["galaxypop_x_g_r"]      = df["galaxy_population_enc"] * df["g_r"]
    df["galaxypop_x_u_g"]      = df["galaxy_population_enc"] * df["u_g"]

    # ── Ratio features ────────────────────────────────────────────────────────
    eps = 1e-6
    df["u_over_g"]       = df["u"] / (df["g"] + eps)
    df["r_over_i"]       = df["r"] / (df["i"] + eps)
    df["g_over_z"]       = df["g"] / (df["z"] + eps)
    df["g_r_over_r_i"]   = df["g_r"] / (df["r_i"].abs() + eps)

    # ── Photometric type indicators ───────────────────────────────────────────
    df["high_redshift"]  = (df["redshift"] > 1.0).astype(np.float32)
    df["neg_redshift"]   = (df["redshift"] < 0).astype(np.float32)
    df["very_blue"]      = (df["u_g"] < 0.5).astype(np.float32)
    df["very_red"]       = (df["g_r"] > 1.0).astype(np.float32)
    df["mid_redshift"]   = ((df["redshift"] >= 0.1) & (df["redshift"] <= 1.0)).astype(np.float32)
    df["low_redshift"]   = (df["redshift"] < 0.1).astype(np.float32)

    # ── Higher-order interactions with spectral type ─────────────────────────
    df["spectral_x_r_i"]        = df["spectral_type_enc"] * df["r_i"]
    df["spectral_x_i_z"]        = df["spectral_type_enc"] * df["i_z"]
    df["galaxypop_x_r_i"]       = df["galaxy_population_enc"] * df["r_i"]
    df["galaxypop_x_u_z"]       = df["galaxy_population_enc"] * df["u_z"]

    # ── Colour index ratios ────────────────────────────────────────────────────
    df["u_g_over_g_r"] = df["u_g"] / (df["g_r"].abs() + eps)
    df["g_r_over_i_z"] = df["g_r"] / (df["i_z"].abs() + eps)
    df["r_i_over_i_z"] = df["r_i"] / (df["i_z"].abs() + eps)

    return df


train = engineer_features(train)
test  = engineer_features(test)

# ── Encode target ────────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
n_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

# ── OOF Target Encoding for spectral_type and galaxy_population ───────────────
# Encode each category as mean probability of each class using out-of-fold
# to avoid leakage (avoids target leakage while providing smooth category encoding).
print("Computing OOF target encoding...")
N_OOF_FOLDS = 5
oof_skf = StratifiedKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=SEED)

cat_cols = ["spectral_type", "galaxy_population"]
te_train_dict = {}
te_test_dict  = {}

for cat_col in cat_cols:
    for cls_idx in range(n_classes):
        feat_name = f"te_{cat_col}_cls{cls_idx}"
        col_oof = np.zeros(len(train), dtype=np.float32)
        col_test_folds = np.zeros((len(test), N_OOF_FOLDS), dtype=np.float32)

        for fold_idx, (tr_idx, va_idx) in enumerate(oof_skf.split(train, y)):
            tr_df = train.iloc[tr_idx][[cat_col]].copy()
            tr_y  = (y[tr_idx] == cls_idx).astype(float)
            te_map = tr_df.assign(_t=tr_y).groupby(cat_col)["_t"].mean()
            global_mean = tr_y.mean()

            col_oof[va_idx] = train[cat_col].iloc[va_idx].map(te_map).fillna(global_mean).values
            col_test_folds[:, fold_idx] = test[cat_col].map(te_map).fillna(global_mean).values

        te_train_dict[feat_name] = col_oof
        te_test_dict[feat_name]  = col_test_folds.mean(axis=1)

for feat_name, vals in te_train_dict.items():
    train[feat_name] = vals
for feat_name, vals in te_test_dict.items():
    test[feat_name] = vals

print(f"Added {len(te_train_dict)} OOF target-encoded features")

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

# ── LightGBM model — stronger than prior ──────────────────────────────────────
print("Training LightGBM (dominant model)...")
lgb_params = {
    "objective":         "multiclass",
    "num_class":         n_classes,
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      3000,
    "learning_rate":     0.02,
    "num_leaves":        511,
    "max_depth":         -1,
    "min_child_samples": 15,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.05,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "random_state":      SEED,
    "verbose":           -1,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y, sample_weight=sample_weight)
print("LightGBM done.")

# ── CatBoost model — strong complementary signal ──────────────────────────────
# CatBoost handles categorical features natively, providing diverse predictions.
print("Training CatBoost (complementary model)...")
cat_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    subsample=0.8,
    colsample_bylevel=0.7,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    class_weights=class_weights,
    random_seed=SEED,
    thread_count=-1,
    verbose=False,
)
cat_model.fit(X_train, y)
print("CatBoost done.")

# ── Predictions ───────────────────────────────────────────────────────────────
print("Generating predictions...")
lgb_proba = lgb_model.predict_proba(X_test)
cat_proba = cat_model.predict_proba(X_test)

# Ensemble: 70% LGB + 30% CatBoost
# CatBoost adds diverse signal; this blend tends to outperform either alone.
BLEND_LGB = 0.70
BLEND_CAT = 0.30
ensemble_proba = BLEND_LGB * lgb_proba + BLEND_CAT * cat_proba

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
