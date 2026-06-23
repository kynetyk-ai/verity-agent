"""
Stellar classification: GALAXY / QSO / STAR
Optimized for speed + balanced accuracy:
- 3 LightGBM + 1 XGBoost ensemble (reduced from 6+2)
- Moderate estimators (1200-1500 vs 2500-3000)
- Small OOF sample (50K, 3-fold) for threshold calibration
- All n_jobs=-1 for full CPU utilization
- Rich feature engineering from photometric bands and redshift
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

    # Additional features: more redshift x magnitude interactions
    df["redshift_r"] = df["redshift"] * df["r"]
    df["redshift_g"] = df["redshift"] * df["g"]
    df["redshift_z"] = df["redshift"] * df["z"]
    df["redshift_mag_range"] = df["redshift"] * df["mag_range"]
    df["redshift_log_r"] = df["log1p_redshift"] * df["r"]

    # Extra color interactions
    df["u_g_times_g_r"] = df["u_g"] * df["g_r"]
    df["g_r_times_r_i"] = df["g_r"] * df["r_i"]
    df["r_i_times_i_z"] = df["r_i"] * df["i_z"]
    df["u_g_over_g_r"]  = df["u_g"] / (df["g_r"].abs() + eps)

    # Spectral energy distribution features
    df["sed_slope_blue"] = df["u_g"] / 1.0
    df["sed_slope_red"]  = df["r_z"] / 1.0
    df["sed_curvature"]  = df["u_g"] + df["i_z"] - 2 * df["g_r"]

    # Redshift bins (categorical-like)
    df["redshift_bin"] = pd.cut(
        df["redshift"],
        bins=[-np.inf, 0, 0.05, 0.2, 0.5, 1.0, 2.0, np.inf],
        labels=False
    ).astype(np.float32)

    # Interaction: galaxy_population × redshift indicators
    df["galaxy_low_z"]    = df["galaxy_population_enc"] * df["is_low_redshift"]
    df["galaxy_high_z"]   = df["galaxy_population_enc"] * df["is_high_redshift"]
    df["spectral_low_z"]  = df["spectral_type_enc"] * df["is_low_redshift"]
    df["spectral_high_z"] = df["spectral_type_enc"] * df["is_high_redshift"]

    # Photo-z color combinations
    df["gr_plus_ri"]   = df["g_r"] + df["r_i"]
    df["ug_minus_iz"]  = df["u_g"] - df["i_z"]
    df["stellar_locus"] = (df["g_r"] - 0.6 * df["r_i"])

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
    "redshift_r", "redshift_g", "redshift_z",
    "redshift_mag_range", "redshift_log_r",
    "u_g_times_g_r", "g_r_times_r_i", "r_i_times_i_z", "u_g_over_g_r",
    "sed_slope_blue", "sed_slope_red", "sed_curvature",
    "redshift_bin",
    "galaxy_low_z", "galaxy_high_z", "spectral_low_z", "spectral_high_z",
    "gr_plus_ri", "ug_minus_iz", "stellar_locus",
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

# ---------------------------------------------------------------
# OOF threshold optimization on a SMALL sample (50K, 3-fold)
# to stay within time budget
# ---------------------------------------------------------------
print("Computing OOF predictions for threshold calibration...")

SAMPLE_SIZE = 50000
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_train), size=min(SAMPLE_SIZE, len(X_train)), replace=False)
X_samp  = X_train[sample_idx]
y_samp  = y_enc[sample_idx]
sw_samp = sample_weights[sample_idx]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_probs = np.zeros((len(X_samp), n_classes), dtype=np.float64)

for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_samp, y_samp)):
    print(f"  OOF fold {fold_i+1}/3...")
    oof_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=fold_i * 100 + 42,
        n_jobs=-1,
        verbose=-1,
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


# ---------------------------------------------------------------
# Full ensemble: 3 LightGBM + 1 XGBoost (lean, within time budget)
# ---------------------------------------------------------------
print("Training full ensemble...")

all_probs = []

# LightGBM model 1: moderate depth, balanced
print("  LightGBM 1/3...")
lgbm1 = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=255,
    max_depth=-1,
    min_child_samples=15,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.80,
    reg_alpha=0.05,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgbm1.fit(X_train, y_enc, sample_weight=sample_weights)
all_probs.append((lgbm1.predict_proba(X_test), 1.0))
del lgbm1

# LightGBM model 2: larger leaves, different seed
print("  LightGBM 2/3...")
lgbm2 = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.025,
    num_leaves=255,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.1,
    reg_lambda=0.2,
    class_weight="balanced",
    random_state=123,
    n_jobs=-1,
    verbose=-1,
)
lgbm2.fit(X_train, y_enc, sample_weight=sample_weights)
all_probs.append((lgbm2.predict_proba(X_test), 1.0))
del lgbm2

# LightGBM model 3: extra_trees for diversity
print("  LightGBM 3/3...")
lgbm3 = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=255,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.70,
    reg_alpha=0.15,
    reg_lambda=0.15,
    class_weight="balanced",
    random_state=777,
    n_jobs=-1,
    verbose=-1,
    extra_trees=True,
)
lgbm3.fit(X_train, y_enc, sample_weight=sample_weights)
all_probs.append((lgbm3.predict_proba(X_test), 1.0))
del lgbm3

# XGBoost model: diversity via different algorithm
print("  XGBoost 1/1...")
xgb1 = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.04,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.80,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    tree_method="hist",
)
xgb1.fit(X_train, y_enc, sample_weight=sample_weights)
all_probs.append((xgb1.predict_proba(X_test), 0.75))
del xgb1

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
