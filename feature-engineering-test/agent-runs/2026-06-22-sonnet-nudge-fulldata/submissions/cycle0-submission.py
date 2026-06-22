"""
Stellar classification: GALAXY / QSO / STAR
Optimized for balanced accuracy with LightGBM + feature engineering.
"""
import os
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

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
    
    # Color indices (photometric differences between adjacent bands)
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    df["u_z"] = df["u"] - df["z"]
    df["r_z"] = df["r"] - df["z"]
    
    # Magnitude spread (proxy for spectral shape)
    df["mag_range"] = df["u"] - df["z"]
    
    # Mean and std of all magnitudes
    mags = df[["u", "g", "r", "i", "z"]]
    df["mag_mean"] = mags.mean(axis=1)
    df["mag_std"]  = mags.std(axis=1)
    
    # Encode categoricals
    spectral_map = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    galaxy_map   = {"Blue_Cloud": 0, "Red_Sequence": 1}
    df["spectral_type_enc"]   = df["spectral_type"].map(spectral_map).fillna(-1).astype(int)
    df["galaxy_population_enc"] = df["galaxy_population"].map(galaxy_map).fillna(-1).astype(int)
    
    return df

train = engineer_features(train)
test  = engineer_features(test)

# ── Features & target ──────────────────────────────────────────────────────
feature_cols = [
    "alpha", "delta",
    "u", "g", "r", "i", "z",
    "redshift",
    "spectral_type_enc", "galaxy_population_enc",
    # engineered
    "u_g", "g_r", "r_i", "i_z",
    "u_r", "g_i", "g_z", "u_z", "r_z",
    "mag_range", "mag_mean", "mag_std",
]

X_train = train[feature_cols]
y_train = train["class"]
X_test  = test[feature_cols]

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y_train)

# ── Class weights for balanced accuracy ────────────────────────────────────
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight("balanced", y_enc)

# ── LightGBM model ─────────────────────────────────────────────────────────
model = LGBMClassifier(
    n_estimators=1500,
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
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(X_train, y_enc, sample_weight=sample_weights)

# ── Predict ────────────────────────────────────────────────────────────────
preds_enc = model.predict(X_test)
preds = le.inverse_transform(preds_enc)

# ── Write output ───────────────────────────────────────────────────────────
out_df = pd.DataFrame({"id": test_ids, "class": preds})
out_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
print(f"Wrote {len(out_df)} predictions.")
print(f"Prediction distribution:\n{out_df['class'].value_counts()}")
