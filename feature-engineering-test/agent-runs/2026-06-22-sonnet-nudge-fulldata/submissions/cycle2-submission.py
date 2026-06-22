"""
Stellar classification: GALAXY / QSO / STAR
Improved ensemble: diverse LightGBM configs + OOF threshold optimization.
Optimized for balanced accuracy.
"""
import os
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize

# Fix random seeds
random.seed(42)
np.random.seed(42)

DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR  = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

test_ids = test["id"].values


def engineer_features(df):
    df = df.copy()

    # Core color indices
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    # Non-adjacent color indices
    df["u_r"] = df["u"] - df["r"]
    df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    df["r_z"] = df["r"] - df["z"]

    # Magnitude statistics
    mags = df[["u", "g", "r", "i", "z"]]
    df["mag_range"]  = df["u"] - df["z"]
    df["mag_mean"]   = mags.mean(axis=1)
    df["mag_std"]    = mags.std(axis=1)
    df["mag_min"]    = mags.min(axis=1)
    df["mag_max"]    = mags.max(axis=1)
    df["mag_median"] = mags.median(axis=1)
    df["mag_skew"]   = mags.skew(axis=1)

    # Redshift transformations
    df["redshift_pos"]   = df["redshift"].clip(lower=0)
    df["redshift_neg"]   = (-df["redshift"]).clip(lower=0)
    df["log1p_redshift"] = np.log1p(np.abs(df["redshift"])) * np.sign(df["redshift"])
    df["redshift_sq"]    = df["redshift"] ** 2
    df["redshift_abs"]   = np.abs(df["redshift"])
    df["redshift_cube"]  = df["redshift"] ** 3
    df["redshift_sqrt"]  = np.sqrt(df["redshift"].clip(lower=0))

    # Redshift x color interactions
    df["redshift_u_g"]      = df["redshift"] * df["u_g"]
    df["redshift_g_r"]      = df["redshift"] * df["g_r"]
    df["redshift_r_i"]      = df["redshift"] * df["r_i"]
    df["redshift_i_z"]      = df["redshift"] * df["i_z"]
    df["redshift_g_z"]      = df["redshift"] * df["g_z"]
    df["redshift_u_z"]      = df["redshift"] * df["u_z"]
    df["redshift_mag_mean"] = df["redshift"] * df["mag_mean"]
    df["redshift_mag_std"]  = df["redshift"] * df["mag_std"]
    df["redshift_u_r"]      = df["redshift"] * df["u_r"]
    df["redshift_g_i"]      = df["redshift"] * df["g_i"]

    # Encode categoricals
    spectral_map = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    galaxy_map   = {"Blue_Cloud": 0, "Red_Sequence": 1}
    df["spectral_type_enc"]     = df["spectral_type"].map(spectral_map).fillna(-1).astype(int)
    df["galaxy_population_enc"] = df["galaxy_population"].map(galaxy_map).fillna(-1).astype(int)

    # Combined categorical features
    df["spectral_galaxy_interact"] = (df["spectral_type_enc"] + 1) * (df["galaxy_population_enc"] + 1)
    df["cat_combo"] = df["spectral_type_enc"] * 2 + df["galaxy_population_enc"]

    # Band ratios
    eps = 1e-6
    df["g_over_r"] = df["g"] / (df["r"].abs() + eps)
    df["u_over_g"] = df["u"] / (df["g"].abs() + eps)
    df["r_over_i"] = df["r"] / (df["i"].abs() + eps)
    df["i_over_z"] = df["i"] / (df["z"].abs() + eps)
    df["g_over_z"] = df["g"] / (df["z"].abs() + eps)
    df["u_over_r"] = df["u"] / (df["r"].abs() + eps)

    # Sky position as Cartesian coordinates
    alpha_rad = np.deg2rad(df["alpha"])
    delta_rad = np.deg2rad(df["delta"])
    df["x_sky"] = np.cos(delta_rad) * np.cos(alpha_rad)
    df["y_sky"] = np.cos(delta_rad) * np.sin(alpha_rad)
    df["z_sky"] = np.sin(delta_rad)

    # Color curvature
    df["color_curve"]  = df["g_r"] - df["r_i"]
    df["color_curve2"] = df["r_i"] - df["i_z"]
    df["color_curve3"] = df["u_g"] - df["g_r"]
    df["color_curve4"] = df["g_r"] - df["g_i"]

    # Lyman break proxy
    df["u_excess"]   = df["u"] - df["g"] - df["g_r"]
    df["lyman_jump"] = df["u_g"] - df["g_r"]

    # Redshift x categorical interactions
    df["redshift_spectral"]  = df["redshift"] * df["spectral_type_enc"]
    df["redshift_galaxy"]    = df["redshift"] * df["galaxy_population_enc"]
    df["redshift_cat_combo"] = df["redshift"] * df["cat_combo"]

    # Alpha mod for periodicity
    df["alpha_mod"] = df["alpha"] % 360

    # Redshift regime indicators (strong discriminators)
    df["is_low_redshift"]  = (df["redshift"].abs() < 0.01).astype(np.float32)
    df["is_high_redshift"] = (df["redshift"] > 1.0).astype(np.float32)
    df["is_med_redshift"]  = ((df["redshift"] >= 0.01) & (df["redshift"] <= 1.0)).astype(np.float32)

    # Polynomial color features
    df["u_g_sq"] = df["u_g"] ** 2
    df["g_r_sq"] = df["g_r"] ** 2
    df["r_i_sq"] = df["r_i"] ** 2
    df["u_g_gr"] = df["u_g"] * df["g_r"]
    df["gr_ri"]  = df["g_r"] * df["r_i"]
    df["ri_iz"]  = df["r_i"] * df["i_z"]

    # SED shape proxies
    df["blue_excess"] = df["u_g"] - df["g_r"]
    df["red_excess"]  = df["r_i"] - df["i_z"]
    df["color_slope"] = (df["z"] - df["u"]) / 4.0

    return df


train = engineer_features(train)
test  = engineer_features(test)

feature_cols = [
    "alpha", "delta",
    "u", "g", "r", "i", "z",
    "redshift",
    "spectral_type_enc", "galaxy_population_enc",
    "spectral_galaxy_interact", "cat_combo",
    "u_g", "g_r", "r_i", "i_z",
    "u_r", "u_i", "u_z", "g_i", "g_z", "r_z",
    "mag_range", "mag_mean", "mag_std", "mag_min", "mag_max", "mag_median", "mag_skew",
    "log1p_redshift", "redshift_sq", "redshift_abs", "redshift_cube", "redshift_sqrt",
    "redshift_pos", "redshift_neg",
    "redshift_u_g", "redshift_g_r", "redshift_r_i", "redshift_i_z",
    "redshift_g_z", "redshift_u_z", "redshift_mag_mean", "redshift_mag_std",
    "redshift_u_r", "redshift_g_i",
    "redshift_spectral", "redshift_galaxy", "redshift_cat_combo",
    "g_over_r", "u_over_g", "r_over_i", "i_over_z", "g_over_z", "u_over_r",
    "x_sky", "y_sky", "z_sky",
    "color_curve", "color_curve2", "color_curve3", "color_curve4",
    "u_excess", "lyman_jump",
    "alpha_mod",
    "is_low_redshift", "is_high_redshift", "is_med_redshift",
    "u_g_sq", "g_r_sq", "r_i_sq", "u_g_gr", "gr_ri", "ri_iz",
    "blue_excess", "red_excess", "color_slope",
]

X_train = train[feature_cols].values.astype(np.float32)
y_train = train["class"].values
X_test  = test[feature_cols].values.astype(np.float32)

le = LabelEncoder()
y_enc = le.fit_transform(y_train)
n_classes = len(le.classes_)
print(f"Classes: {le.classes_}, n_train={len(y_enc)}, n_test={len(X_test)}")
print(f"n_features={len(feature_cols)}")

sample_weights = compute_sample_weight("balanced", y_enc)


def make_lgbm(seed, n_estimators=2000, num_leaves=127, lr=0.03,
              min_child=20, subsample=0.8, colsample=0.8,
              reg_alpha=0.1, reg_lambda=0.1, max_depth=-1,
              extra_trees=False):
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        extra_trees=extra_trees,
    )


def make_xgb(seed, n_estimators=1500, max_depth=8, lr=0.05,
             subsample=0.8, colsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
             min_child_weight=3):
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
    )


# OOF threshold optimization
print("Computing OOF predictions for threshold calibration...")

SAMPLE_SIZE = 80000
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_train), size=min(SAMPLE_SIZE, len(X_train)), replace=False)
X_samp  = X_train[sample_idx]
y_samp  = y_enc[sample_idx]
sw_samp = sample_weights[sample_idx]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_probs = np.zeros((len(X_samp), n_classes), dtype=np.float64)

for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_samp, y_samp)):
    print(f"  OOF fold {fold_i+1}/3...")
    oof_model = make_lgbm(
        seed=fold_i * 100 + 42, n_estimators=1000, num_leaves=127, lr=0.05,
        min_child=20, subsample=0.8, colsample=0.8,
    )
    oof_model.fit(X_samp[tr_idx], y_samp[tr_idx], sample_weight=sw_samp[tr_idx])
    oof_probs[val_idx] = oof_model.predict_proba(X_samp[val_idx])
    del oof_model


def neg_ba_thresholds(thresholds, probs, y_true):
    log_probs = np.log(probs + 1e-15)
    adj_probs = log_probs + np.array(thresholds)
    preds = np.argmax(adj_probs, axis=1)
    return -balanced_accuracy_score(y_true, preds)


result = minimize(
    neg_ba_thresholds,
    x0=[0.0, 0.0, 0.0],
    args=(oof_probs, y_samp),
    method="Nelder-Mead",
    options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6},
)
optimal_thresholds = result.x
print(f"Optimal thresholds: {optimal_thresholds}")
print(f"OOF balanced accuracy (thresholded): {-result.fun:.4f}")
baseline_ba = balanced_accuracy_score(y_samp, np.argmax(oof_probs, axis=1))
print(f"OOF balanced accuracy (no threshold): {baseline_ba:.4f}")


# Ensemble: 5 LightGBM + 2 XGBoost
print("Training full ensemble...")

lgbm_configs = [
    dict(seed=42,  n_estimators=2500, num_leaves=255, lr=0.020,
         min_child=15, subsample=0.80, colsample=0.80, reg_alpha=0.05, reg_lambda=0.1),
    dict(seed=123, n_estimators=2500, num_leaves=511, lr=0.018,
         min_child=10, subsample=0.75, colsample=0.75, reg_alpha=0.1,  reg_lambda=0.2),
    dict(seed=777, n_estimators=2000, num_leaves=127, lr=0.035,
         min_child=30, subsample=0.85, colsample=0.90, reg_alpha=0.2,  reg_lambda=0.05),
    dict(seed=999, n_estimators=2000, num_leaves=255, lr=0.030,
         min_child=20, subsample=0.80, colsample=0.70, reg_alpha=0.15, reg_lambda=0.15,
         extra_trees=True),
    dict(seed=2024, n_estimators=2500, num_leaves=191, lr=0.022,
         min_child=40, subsample=0.70, colsample=0.75, reg_alpha=0.3, reg_lambda=0.3),
]

xgb_configs = [
    dict(seed=42,  n_estimators=1800, max_depth=8,  lr=0.04,
         subsample=0.8,  colsample=0.80, reg_alpha=0.1,  reg_lambda=1.0, min_child_weight=3),
    dict(seed=123, n_estimators=1800, max_depth=10, lr=0.030,
         subsample=0.75, colsample=0.75, reg_alpha=0.05, reg_lambda=0.5, min_child_weight=5),
]

all_probs = []

for i, cfg in enumerate(lgbm_configs):
    print(f"  LightGBM {i+1}/{len(lgbm_configs)}...")
    model = make_lgbm(**cfg)
    model.fit(X_train, y_enc, sample_weight=sample_weights)
    all_probs.append((model.predict_proba(X_test), 1.0))
    del model

for i, cfg in enumerate(xgb_configs):
    print(f"  XGBoost {i+1}/{len(xgb_configs)}...")
    model = make_xgb(**cfg)
    model.fit(X_train, y_enc, sample_weight=sample_weights)
    all_probs.append((model.predict_proba(X_test), 0.75))
    del model

# Weighted soft voting
total_weight = sum(w for _, w in all_probs)
avg_probs = sum(p * w for p, w in all_probs) / total_weight

# Apply optimized thresholds in log space
log_avg = np.log(avg_probs + 1e-15)
adj_log = log_avg + np.array(optimal_thresholds)
preds_enc = np.argmax(adj_log, axis=1)
preds = le.inverse_transform(preds_enc)

# Write output
out_df = pd.DataFrame({"id": test_ids, "class": preds})
out_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
print(f"Wrote {len(out_df)} predictions.")
print(f"Prediction distribution:\n{out_df['class'].value_counts()}")
