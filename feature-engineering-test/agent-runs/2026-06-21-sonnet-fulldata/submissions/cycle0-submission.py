"""
Stellar classification submission.
Uses LightGBM with engineered colour-index features, handles class imbalance via
class_weight='balanced', trains on full data with all CPU cores.
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

    # Redshift-based features
    df["log1p_redshift"] = np.log1p(np.clip(df["redshift"], 0, None))
    df["redshift_sq"]    = df["redshift"] ** 2

    # Magnitude ratios / sums
    df["mean_mag"] = (df["u"] + df["g"] + df["r"] + df["i"] + df["z"]) / 5.0
    df["mag_range"] = df["u"] - df["z"]   # broadest colour spread

    # Encode categoricals
    spectral_map = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    galaxy_pop_map = {"Blue_Cloud": 0, "Red_Sequence": 1}

    df["spectral_type_enc"]    = df["spectral_type"].map(spectral_map).fillna(-1).astype(int)
    df["galaxy_population_enc"] = df["galaxy_population"].map(galaxy_pop_map).fillna(-1).astype(int)

    # Interaction: redshift × colour
    df["redshift_x_g_r"] = df["redshift"] * df["g_r"]
    df["redshift_x_u_g"] = df["redshift"] * df["u_g"]
    df["redshift_x_r_i"] = df["redshift"] * df["r_i"]

    return df

train = engineer_features(train)
test  = engineer_features(test)

# ── Encode target ────────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(train[TARGET])

# ── Define feature set ───────────────────────────────────────────────────────
DROP_COLS = [ID_COL, TARGET, "spectral_type", "galaxy_population"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]

X_train = train[FEATURE_COLS].values.astype(np.float32)
X_test  = test[FEATURE_COLS].values.astype(np.float32)

# ── Class weights for balanced accuracy ──────────────────────────────────────
n_classes = len(le.classes_)
class_counts = np.bincount(y)
class_weights = (len(y) / (n_classes * class_counts)).tolist()

sample_weight = np.array([class_weights[yi] for yi in y], dtype=np.float32)

# ── LightGBM model ────────────────────────────────────────────────────────────
params = {
    "objective":        "multiclass",
    "num_class":        n_classes,
    "metric":           "multi_logloss",
    "boosting_type":    "gbdt",
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 20,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "random_state":     SEED,
    "verbose":          -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y,
    sample_weight=sample_weight,
)

# ── Predict ───────────────────────────────────────────────────────────────────
preds_proba = model.predict_proba(X_test)
preds_idx   = np.argmax(preds_proba, axis=1)
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
