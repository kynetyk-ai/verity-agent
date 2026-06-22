"""
Stellar classification: GALAXY / QSO / STAR
Ensemble of LightGBM + XGBoost with extensive feature engineering.
Optimized for balanced accuracy via soft-voting.
"""
import os
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Fix random seeds
random.seed(42)
np.random.seed(42)

DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

test_ids = test["id"].values

# ── Feature engineering ────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Core color indices (photometric differences between adjacent bands)
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

    # Redshift x color interactions (critical for QSO separation)
    df["redshift_u_g"]      = df["redshift"] * df["u_g"]
    df["redshift_g_r"]      = df["redshift"] * df["g_r"]
    df["redshift_r_i"]      = df["redshift"] * df["r_i"]
    df["redshift_i_z"]      = df["redshift"] * df["i_z"]
    df["redshift_g_z"]      = df["redshift"] * df["g_z"]
    df["redshift_u_z"]      = df["redshift"] * df["u_z"]
    df["redshift_mag_mean"] = df["redshift"] * df["mag_mean"]

    # Encode categoricals
    spectral_map = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    galaxy_map   = {"Blue_Cloud": 0, "Red_Sequence": 1}
    df["spectral_type_enc"]     = df["spectral_type"].map(spectral_map).fillna(-1).astype(int)
    df["galaxy_population_enc"] = df["galaxy_population"].map(galaxy_map).fillna(-1).astype(int)

    # Interaction: spectral_type x galaxy_population
    df["spectral_galaxy_interact"] = (df["spectral_type_enc"] + 1) * (df["galaxy_population_enc"] + 1)

    # Band ratios
    eps = 1e-6
    df["g_over_r"] = df["g"] / (df["r"].abs() + eps)
    df["u_over_g"] = df["u"] / (df["g"].abs() + eps)
    df["r_over_i"] = df["r"] / (df["i"].abs() + eps)
    df["i_over_z"] = df["i"] / (df["z"].abs() + eps)
    df["g_over_z"] = df["g"] / (df["z"].abs() + eps)

    # Sky position as Cartesian coordinates
    alpha_rad = np.deg2rad(df["alpha"])
    delta_rad = np.deg2rad(df["delta"])
    df["x_sky"] = np.cos(delta_rad) * np.cos(alpha_rad)
    df["y_sky"] = np.cos(delta_rad) * np.sin(alpha_rad)
    df["z_sky"] = np.sin(delta_rad)

    # Color curvature (second differences in photometric sequence)
    df["color_curve"]  = df["g_r"] - df["r_i"]
    df["color_curve2"] = df["r_i"] - df["i_z"]
    df["color_curve3"] = df["u_g"] - df["g_r"]

    # Lyman break proxy (large u-band dropout -> high-z QSO)
    df["u_excess"]   = df["u"] - df["g"] - df["g_r"]
    df["lyman_jump"] = df["u_g"] - df["g_r"]

    # Redshift x categorical interactions
    df["redshift_spectral"] = df["redshift"] * df["spectral_type_enc"]
    df["redshift_galaxy"]   = df["redshift"] * df["galaxy_population_enc"]

    # Alpha mod for periodicity
    df["alpha_mod"] = df["alpha"] % 360

    return df


train = engineer_features(train)
test  = engineer_features(test)

# ── Features & target ──────────────────────────────────────────────────────
feature_cols = [
    "alpha", "delta",
    "u", "g", "r", "i", "z",
    "redshift",
    "spectral_type_enc", "galaxy_population_enc",
    "spectral_galaxy_interact",
    # color indices
    "u_g", "g_r", "r_i", "i_z",
    "u_r", "u_i", "u_z", "g_i", "g_z", "r_z",
    # magnitude stats
    "mag_range", "mag_mean", "mag_std", "mag_min", "mag_max", "mag_median", "mag_skew",
    # redshift transforms
    "log1p_redshift", "redshift_sq", "redshift_abs", "redshift_cube",
    "redshift_pos", "redshift_neg",
    # redshift x color
    "redshift_u_g", "redshift_g_r", "redshift_r_i", "redshift_i_z",
    "redshift_g_z", "redshift_u_z", "redshift_mag_mean",
    # redshift x categorical
    "redshift_spectral", "redshift_galaxy",
    # band ratios
    "g_over_r", "u_over_g", "r_over_i", "i_over_z", "g_over_z",
    # sky cartesian
    "x_sky", "y_sky", "z_sky",
    # color curvature
    "color_curve", "color_curve2", "color_curve3",
    "u_excess", "lyman_jump",
    # position
    "alpha_mod",
]

X_train = train[feature_cols].values.astype(np.float32)
y_train = train["class"].values
X_test  = test[feature_cols].values.astype(np.float32)

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y_train)
n_classes = len(le.classes_)
print(f"Classes: {le.classes_}, n_train={len(y_enc)}, n_test={len(X_test)}")

# Sample weights for balanced accuracy
sample_weights = compute_sample_weight("balanced", y_enc)

# ── Model definitions ──────────────────────────────────────────────────────
def make_lgbm(seed, n_estimators=2000, num_leaves=127, lr=0.03,
              min_child=20, subsample=0.8, colsample=0.8,
              reg_alpha=0.1, reg_lambda=0.1, max_depth=-1):
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
    )

def make_xgb(seed, n_estimators=1500, max_depth=8, lr=0.05,
             subsample=0.8, colsample=0.8, reg_alpha=0.1, reg_lambda=1.0):
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
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


# ── Ensemble: 4 LightGBM + 1 XGBoost (soft-voting) ────────────────────────
print("Training ensemble...")

lgbm_models = [
    make_lgbm(seed=42,  n_estimators=2500, num_leaves=127, lr=0.025,
              min_child=20,  subsample=0.80, colsample=0.80, reg_alpha=0.1,  reg_lambda=0.1),
    make_lgbm(seed=123, n_estimators=2500, num_leaves=255, lr=0.020,
              min_child=15,  subsample=0.75, colsample=0.75, reg_alpha=0.05, reg_lambda=0.2),
    make_lgbm(seed=777, n_estimators=2000, num_leaves=63,  lr=0.040,
              min_child=30,  subsample=0.85, colsample=0.90, reg_alpha=0.2,  reg_lambda=0.05),
    make_lgbm(seed=999, n_estimators=2000, num_leaves=191, lr=0.030,
              min_child=25,  subsample=0.80, colsample=0.70, reg_alpha=0.15, reg_lambda=0.15),
]

xgb_model = make_xgb(
    seed=42, n_estimators=1500, max_depth=8, lr=0.05,
    subsample=0.8, colsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
)

all_probs = []
lgbm_weight = 1.0
xgb_weight  = 0.75

# Train LightGBM models
for i, model in enumerate(lgbm_models):
    print(f"  LightGBM model {i+1}/{len(lgbm_models)}...")
    model.fit(X_train, y_enc, sample_weight=sample_weights)
    probs = model.predict_proba(X_test)
    all_probs.append((probs, lgbm_weight))

# Train XGBoost
print("  XGBoost model...")
xgb_model.fit(X_train, y_enc, sample_weight=sample_weights)
xgb_probs = xgb_model.predict_proba(X_test)
all_probs.append((xgb_probs, xgb_weight))

# Weighted soft voting
total_weight = sum(w for _, w in all_probs)
avg_probs = sum(p * w for p, w in all_probs) / total_weight

preds_enc = np.argmax(avg_probs, axis=1)
preds = le.inverse_transform(preds_enc)

# ── Write output ───────────────────────────────────────────────────────────
out_df = pd.DataFrame({"id": test_ids, "class": preds})
out_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
print(f"Wrote {len(out_df)} predictions.")
print(f"Prediction distribution:\n{out_df['class'].value_counts()}")
