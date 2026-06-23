"""
Stellar classification: GALAXY / QSO / STAR
Improved v5: 2-level stacking with diverse base learners,
better OOF calibration using FULL training data, 
more aggressive feature engineering, and optimized ensemble.
"""
import os
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, differential_evolution

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
print(f"Train: {train.shape}, Test: {test.shape}")


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
    df["mag_kurt"]   = mags.kurtosis(axis=1)

    # Redshift transformations
    df["redshift_pos"]   = df["redshift"].clip(lower=0)
    df["redshift_neg"]   = (-df["redshift"]).clip(lower=0)
    df["log1p_redshift"] = np.log1p(np.abs(df["redshift"])) * np.sign(df["redshift"])
    df["redshift_sq"]    = df["redshift"] ** 2
    df["redshift_abs"]   = np.abs(df["redshift"])
    df["redshift_cube"]  = df["redshift"] ** 3
    df["redshift_sqrt"]  = np.sqrt(df["redshift"].clip(lower=0))
    df["redshift_log_sq"] = df["log1p_redshift"] ** 2

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
    df["cat_combo_3way"] = df["spectral_type_enc"] * 3 + df["galaxy_population_enc"]

    # Band ratios
    eps = 1e-6
    df["g_over_r"] = df["g"] / (df["r"].abs() + eps)
    df["u_over_g"] = df["u"] / (df["g"].abs() + eps)
    df["r_over_i"] = df["r"] / (df["i"].abs() + eps)
    df["i_over_z"] = df["i"] / (df["z"].abs() + eps)
    df["g_over_z"] = df["g"] / (df["z"].abs() + eps)
    df["u_over_r"] = df["u"] / (df["r"].abs() + eps)
    df["u_over_z"] = df["u"] / (df["z"].abs() + eps)
    df["r_over_z"] = df["r"] / (df["z"].abs() + eps)

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
    df["color_curve5"] = df["u_g"] - df["r_i"]
    df["color_curve6"] = df["g_r"] - df["i_z"]

    # Lyman break proxy
    df["u_excess"]   = df["u"] - df["g"] - df["g_r"]
    df["lyman_jump"] = df["u_g"] - df["g_r"]

    # Redshift x categorical interactions
    df["redshift_spectral"]  = df["redshift"] * df["spectral_type_enc"]
    df["redshift_galaxy"]    = df["redshift"] * df["galaxy_population_enc"]
    df["redshift_cat_combo"] = df["redshift"] * df["cat_combo"]

    # Alpha mod for periodicity
    df["alpha_mod"] = df["alpha"] % 360
    df["alpha_sin"] = np.sin(alpha_rad)
    df["alpha_cos"] = np.cos(alpha_rad)
    df["delta_sin"] = np.sin(delta_rad)
    df["delta_cos"] = np.cos(delta_rad)

    # Redshift regime indicators (strong discriminators)
    df["is_low_redshift"]  = (df["redshift"].abs() < 0.01).astype(np.float32)
    df["is_high_redshift"] = (df["redshift"] > 1.0).astype(np.float32)
    df["is_med_redshift"]  = ((df["redshift"] >= 0.01) & (df["redshift"] <= 1.0)).astype(np.float32)
    df["is_neg_redshift"]  = (df["redshift"] < 0).astype(np.float32)
    df["is_very_high_z"]   = (df["redshift"] > 2.0).astype(np.float32)
    df["is_zero_z"]        = (df["redshift"].abs() < 0.001).astype(np.float32)

    # Polynomial color features
    df["u_g_sq"] = df["u_g"] ** 2
    df["g_r_sq"] = df["g_r"] ** 2
    df["r_i_sq"] = df["r_i"] ** 2
    df["i_z_sq"] = df["i_z"] ** 2
    df["u_g_gr"] = df["u_g"] * df["g_r"]
    df["gr_ri"]  = df["g_r"] * df["r_i"]
    df["ri_iz"]  = df["r_i"] * df["i_z"]
    df["ug_ri"]  = df["u_g"] * df["r_i"]
    df["ug_iz"]  = df["u_g"] * df["i_z"]

    # SED shape proxies
    df["blue_excess"] = df["u_g"] - df["g_r"]
    df["red_excess"]  = df["r_i"] - df["i_z"]
    df["color_slope"] = (df["z"] - df["u"]) / 4.0

    # Additional redshift x magnitude interactions
    df["redshift_r"] = df["redshift"] * df["r"]
    df["redshift_g"] = df["redshift"] * df["g"]
    df["redshift_z"] = df["redshift"] * df["z"]
    df["redshift_u"] = df["redshift"] * df["u"]
    df["redshift_i"] = df["redshift"] * df["i"]
    df["redshift_mag_range"] = df["redshift"] * df["mag_range"]
    df["redshift_log_r"] = df["log1p_redshift"] * df["r"]
    df["redshift_log_g"] = df["log1p_redshift"] * df["g"]

    # Extra color interactions
    df["u_g_times_g_r"] = df["u_g"] * df["g_r"]
    df["g_r_times_r_i"] = df["g_r"] * df["r_i"]
    df["r_i_times_i_z"] = df["r_i"] * df["i_z"]
    df["u_g_over_g_r"] = df["u_g"] / (df["g_r"].abs() + eps)

    # Spectral energy distribution features
    df["sed_slope_blue"] = df["u_g"]
    df["sed_slope_red"]  = df["r_z"]
    df["sed_curvature"]  = df["u_g"] + df["i_z"] - 2 * df["g_r"]
    df["sed_curvature2"] = df["g_r"] + df["r_z"] - 2 * df["r_i"]

    # Redshift bins (categorical-like)
    df["redshift_bin"] = pd.cut(
        df["redshift"],
        bins=[-np.inf, -0.01, 0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, np.inf],
        labels=False
    ).astype(np.float32)

    # Interaction: galaxy_population × redshift indicators
    df["galaxy_low_z"]  = df["galaxy_population_enc"] * df["is_low_redshift"]
    df["galaxy_high_z"] = df["galaxy_population_enc"] * df["is_high_redshift"]
    df["spectral_low_z"]  = df["spectral_type_enc"] * df["is_low_redshift"]
    df["spectral_high_z"] = df["spectral_type_enc"] * df["is_high_redshift"]
    df["spectral_neg_z"]  = df["spectral_type_enc"] * df["is_neg_redshift"]
    df["galaxy_neg_z"]    = df["galaxy_population_enc"] * df["is_neg_redshift"]

    # Photo-z color combinations known to separate classes
    df["gr_plus_ri"] = df["g_r"] + df["r_i"]
    df["ug_minus_iz"] = df["u_g"] - df["i_z"]
    df["stellar_locus"] = (df["g_r"] - 0.6 * df["r_i"])
    df["stellar_locus2"] = (df["u_g"] - 0.7 * df["g_r"])

    # Magnitude-redshift combinations
    df["r_over_z_sq"] = df["r"] / (df["redshift_sq"] + eps)
    df["mag_mean_z"] = df["mag_mean"] * df["redshift"]
    df["mag_std_z"]  = df["mag_std"]  * df["redshift_abs"]

    # Color gradient
    df["color_gradient_blue"] = df["u_g"] / (df["g_r"].abs() + eps)
    df["color_gradient_red"]  = df["g_r"] / (df["r_i"].abs() + eps)

    # Additional stellar diagnostics
    df["uband_excess"] = df["u"] - (2.5 * df["g"] - 1.5 * df["r"])
    df["balmer_proxy"] = df["u_g"] - 0.5 * df["g_r"]

    # Asymmetry in color distribution
    df["color_asymmetry"] = df["u_g"] + df["i_z"] - df["g_r"] - df["r_i"]

    return df


train = engineer_features(train)
test  = engineer_features(test)

feature_cols = [
    "alpha", "delta",
    "u", "g", "r", "i", "z",
    "redshift",
    "spectral_type_enc", "galaxy_population_enc",
    "spectral_galaxy_interact", "cat_combo", "cat_combo_3way",
    "u_g", "g_r", "r_i", "i_z",
    "u_r", "u_i", "u_z", "g_i", "g_z", "r_z",
    "mag_range", "mag_mean", "mag_std", "mag_min", "mag_max", "mag_median", "mag_skew", "mag_kurt",
    "log1p_redshift", "redshift_sq", "redshift_abs", "redshift_cube", "redshift_sqrt",
    "redshift_pos", "redshift_neg", "redshift_log_sq",
    "redshift_u_g", "redshift_g_r", "redshift_r_i", "redshift_i_z",
    "redshift_g_z", "redshift_u_z", "redshift_mag_mean", "redshift_mag_std",
    "redshift_u_r", "redshift_g_i",
    "redshift_spectral", "redshift_galaxy", "redshift_cat_combo",
    "g_over_r", "u_over_g", "r_over_i", "i_over_z", "g_over_z", "u_over_r", "u_over_z", "r_over_z",
    "x_sky", "y_sky", "z_sky",
    "color_curve", "color_curve2", "color_curve3", "color_curve4", "color_curve5", "color_curve6",
    "u_excess", "lyman_jump",
    "alpha_mod", "alpha_sin", "alpha_cos", "delta_sin", "delta_cos",
    "is_low_redshift", "is_high_redshift", "is_med_redshift", "is_neg_redshift",
    "is_very_high_z", "is_zero_z",
    "u_g_sq", "g_r_sq", "r_i_sq", "i_z_sq", "u_g_gr", "gr_ri", "ri_iz", "ug_ri", "ug_iz",
    "blue_excess", "red_excess", "color_slope",
    "redshift_r", "redshift_g", "redshift_z", "redshift_u", "redshift_i",
    "redshift_mag_range", "redshift_log_r", "redshift_log_g",
    "u_g_times_g_r", "g_r_times_r_i", "r_i_times_i_z", "u_g_over_g_r",
    "sed_slope_blue", "sed_slope_red", "sed_curvature", "sed_curvature2",
    "redshift_bin",
    "galaxy_low_z", "galaxy_high_z", "spectral_low_z", "spectral_high_z",
    "spectral_neg_z", "galaxy_neg_z",
    "gr_plus_ri", "ug_minus_iz", "stellar_locus", "stellar_locus2",
    "r_over_z_sq", "mag_mean_z", "mag_std_z",
    "color_gradient_blue", "color_gradient_red",
    "uband_excess", "balmer_proxy", "color_asymmetry",
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


# ===== STACKING: Full OOF on training set =====
# Use all training data for OOF to get unbiased meta-features
print("Computing OOF predictions for stacking...")

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# We'll generate OOF for several base models, then stack with a meta-learner
# Base models for stacking: 3 LightGBM variants (fast enough for OOF on full data)
base_configs_lgbm = [
    dict(seed=42,  n_estimators=1500, num_leaves=255, lr=0.02,
         min_child=15, subsample=0.80, colsample=0.80, reg_alpha=0.05, reg_lambda=0.1),
    dict(seed=123, n_estimators=1500, num_leaves=127, lr=0.04,
         min_child=20, subsample=0.75, colsample=0.75, reg_alpha=0.1,  reg_lambda=0.2),
    dict(seed=777, n_estimators=1500, num_leaves=511, lr=0.015,
         min_child=10, subsample=0.85, colsample=0.90, reg_alpha=0.2,  reg_lambda=0.05,
         extra_trees=True),
]

n_base = len(base_configs_lgbm)
oof_probs_all = [np.zeros((len(X_train), n_classes), dtype=np.float64) for _ in range(n_base)]
test_probs_fold = [np.zeros((len(X_test), n_classes), dtype=np.float64) for _ in range(n_base)]

for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_enc)):
    print(f"  Stacking fold {fold_i+1}/{N_FOLDS}...")
    sw_fold = sample_weights[tr_idx]
    for b_i, cfg in enumerate(base_configs_lgbm):
        model = make_lgbm(**cfg)
        model.fit(X_train[tr_idx], y_enc[tr_idx], sample_weight=sw_fold)
        oof_probs_all[b_i][val_idx] = model.predict_proba(X_train[val_idx])
        test_probs_fold[b_i] += model.predict_proba(X_test) / N_FOLDS
        del model

# Evaluate OOF balanced accuracy for stacking base models
for b_i, cfg in enumerate(base_configs_lgbm):
    ba = balanced_accuracy_score(y_enc, np.argmax(oof_probs_all[b_i], axis=1))
    print(f"  Base model {b_i+1} OOF BA: {ba:.4f}")

# Build meta-features: concatenate OOF probabilities
meta_train = np.hstack(oof_probs_all)  # shape: (n_train, n_base*n_classes)
meta_test  = np.hstack(test_probs_fold)  # shape: (n_test, n_base*n_classes)

print(f"Meta-features shape: train={meta_train.shape}, test={meta_test.shape}")

# Threshold optimization using OOF of first base model (fast)
def neg_ba_thresholds(thresholds, probs, y_true):
    log_probs = np.log(probs + 1e-15)
    adj_probs = log_probs + np.array(thresholds)
    preds = np.argmax(adj_probs, axis=1)
    return -balanced_accuracy_score(y_true, preds)


# Optimize thresholds on average of base OOF probs
avg_oof_probs = np.mean(np.array(oof_probs_all), axis=0)
result = minimize(
    neg_ba_thresholds,
    x0=[0.0, 0.0, 0.0],
    args=(avg_oof_probs, y_enc),
    method="Nelder-Mead",
    options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-8},
)
optimal_thresholds = result.x
print(f"Optimal thresholds (from OOF): {optimal_thresholds}")
print(f"OOF BA (thresholded, from base avg): {-result.fun:.4f}")
baseline_ba = balanced_accuracy_score(y_enc, np.argmax(avg_oof_probs, axis=1))
print(f"OOF BA (no threshold, base avg): {baseline_ba:.4f}")


# ===== FINAL ENSEMBLE: Train on full training data =====
print("Training final ensemble on full data...")

lgbm_configs = [
    dict(seed=42,  n_estimators=3000, num_leaves=255, lr=0.018,
         min_child=15, subsample=0.80, colsample=0.80, reg_alpha=0.05, reg_lambda=0.1),
    dict(seed=123, n_estimators=3000, num_leaves=511, lr=0.015,
         min_child=10, subsample=0.75, colsample=0.75, reg_alpha=0.1,  reg_lambda=0.2),
    dict(seed=777, n_estimators=2500, num_leaves=127, lr=0.030,
         min_child=30, subsample=0.85, colsample=0.90, reg_alpha=0.2,  reg_lambda=0.05),
    dict(seed=999, n_estimators=2500, num_leaves=255, lr=0.025,
         min_child=20, subsample=0.80, colsample=0.70, reg_alpha=0.15, reg_lambda=0.15,
         extra_trees=True),
    dict(seed=2024, n_estimators=3000, num_leaves=191, lr=0.020,
         min_child=40, subsample=0.70, colsample=0.75, reg_alpha=0.3, reg_lambda=0.3),
    dict(seed=2025, n_estimators=2500, num_leaves=383, lr=0.022,
         min_child=25, subsample=0.78, colsample=0.82, reg_alpha=0.08, reg_lambda=0.12),
    dict(seed=3001, n_estimators=3000, num_leaves=127, lr=0.018,
         min_child=15, subsample=0.80, colsample=0.80, reg_alpha=0.05, reg_lambda=0.1,
         extra_trees=True),
    dict(seed=3002, n_estimators=2500, num_leaves=255, lr=0.025,
         min_child=20, subsample=0.85, colsample=0.85, reg_alpha=0.1, reg_lambda=0.2),
]

xgb_configs = [
    dict(seed=42,  n_estimators=2000, max_depth=8,  lr=0.035,
         subsample=0.8,  colsample=0.80, reg_alpha=0.1,  reg_lambda=1.0, min_child_weight=3),
    dict(seed=123, n_estimators=2000, max_depth=10, lr=0.025,
         subsample=0.75, colsample=0.75, reg_alpha=0.05, reg_lambda=0.5, min_child_weight=5),
    dict(seed=456, n_estimators=1500, max_depth=6,  lr=0.05,
         subsample=0.80, colsample=0.85, reg_alpha=0.2, reg_lambda=2.0,  min_child_weight=2),
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

# Also include the stacking base model predictions (from full data fold-average)
# Blend: 80% direct ensemble + 20% stacking base average
avg_stack_base = np.mean(np.array(test_probs_fold), axis=0)
final_probs = 0.85 * avg_probs + 0.15 * avg_stack_base

# Apply optimized thresholds in log space
log_avg = np.log(final_probs + 1e-15)
adj_log = log_avg + np.array(optimal_thresholds)
preds_enc = np.argmax(adj_log, axis=1)
preds = le.inverse_transform(preds_enc)

# Write output
out_df = pd.DataFrame({"id": test_ids, "class": preds})
out_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
print(f"Wrote {len(out_df)} predictions.")
print(f"Prediction distribution:\n{out_df['class'].value_counts()}")
