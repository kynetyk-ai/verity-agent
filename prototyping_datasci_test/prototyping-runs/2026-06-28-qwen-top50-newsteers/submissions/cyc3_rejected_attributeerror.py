#!/usr/bin/env python3
"""Stellar classification — LightGBM + XGBoost ensemble, optimized for balanced accuracy."""

import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR = os.environ.get("VERITY_OUT", "out")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

target = train.pop("class")
le = LabelEncoder()
y = le.fit_transform(target)
n_classes = len(le.classes_)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["alpha"] = df["alpha"]
    feat["delta"] = df["delta"]
    feat["u"] = df["u"]
    feat["g"] = df["g"]
    feat["r"] = df["r"]
    feat["i"] = df["i"]
    feat["z"] = df["z"]
    feat["redshift"] = df["redshift"]

    # Color indices (key signal per domain hint)
    feat["u_g"] = df["u"] - df["g"]
    feat["g_r"] = df["g"] - df["r"]
    feat["r_i"] = df["r"] - df["i"]
    feat["i_z"] = df["i"] - df["z"]
    feat["u_r"] = df["u"] - df["r"]
    feat["u_i"] = df["u"] - df["i"]
    feat["g_z"] = df["g"] - df["z"]

    # redshift interactions with color indices
    feat["redshift_x_u_g"] = df["redshift"] * feat["u_g"]
    feat["redshift_x_g_r"] = df["redshift"] * feat["g_r"]
    feat["redshift_x_r_i"] = df["redshift"] * feat["r_i"]

    # Spectral type encoding
    spec_map = {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}
    feat["spectral_type"] = df["spectral_type"].map(spec_map).fillna(-1).astype(float)

    # Galaxy population encoding
    feat["galaxy_population"] = (df["galaxy_population"] == "Blue_Cloud").astype(float)

    return feat

X_train = build_features(train)
X_test = build_features(test)

# Convert to numpy for speed
X_train_arr = X_train.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
y_arr = y.values

# ---------------------------------------------------------------------------
# Quick CV on subset to verify approach (5-fold, 50K sample)
# ---------------------------------------------------------------------------
print("Quick 5-fold CV on 50K sample...")
sample_idx = np.random.choice(len(X_train_arr), 50000, replace=False)
X_sample = X_train_arr[sample_idx]
y_sample = y_arr[sample_idx]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_bacc = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
    X_tr, y_tr = X_sample[tr_idx], y_sample[tr_idx]
    X_val, y_val = X_sample[val_idx], y_sample[val_idx]

    # LightGBM
    m_lgb = lgb.LGBMClassifier(
        n_estimators=1500, max_depth=8, learning_rate=0.05,
        num_leaves=64, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    m_lgb.fit(X_tr, y_tr)
    p_lgb = m_lgb.predict_proba(X_val)

    # XGBoost
    m_xgb = XGBClassifier(
        n_estimators=1500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, min_child_weight=3,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    m_xgb.fit(X_tr, y_tr)
    p_xgb = m_xgb.predict_proba(X_val)

    p = (p_lgb + p_xgb) / 2.0
    preds = np.argmax(p, axis=1)
    classes = np.unique(y_val)
    bacc = sum((preds[y_val == c] == c).mean() for c in classes) / len(classes)
    cv_bacc.append(bacc)
    print(f"  Fold {fold+1}: BA = {bacc:.4f}")

print(f"  Mean CV balanced acc: {np.mean(cv_bacc):.4f}")

# ---------------------------------------------------------------------------
# Train final models on FULL data
# ---------------------------------------------------------------------------
print("Training final LightGBM on full data...")
lgb_final = lgb.LGBMClassifier(
    n_estimators=2500,
    max_depth=8,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=10,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
lgb_final.fit(X_train_arr, y_arr)

print("Training final XGBoost on full data...")
xgb_final = XGBClassifier(
    n_estimators=2500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_weight=3,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)
xgb_final.fit(X_train_arr, y_arr)

# Ensemble (equal weight)
print("Generating test predictions...")
p_lgb = lgb_final.predict_proba(X_test_arr)
p_xgb = xgb_final.predict_proba(X_test_arr)
ensemble_preds = (p_lgb + p_xgb) / 2.0
final_preds = np.argmax(ensemble_preds, axis=1)
pred_labels = le.inverse_transform(final_preds)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
out = pd.DataFrame({"id": test["id"].values, "class": pred_labels})
out.to_csv(f"{OUT_DIR}/predictions.csv", index=False)
print(f"Wrote {len(out)} predictions to {OUT_DIR}/predictions.csv")
assert len(out) == len(test), f"Prediction count {len(out)} != test count {len(test)}"
print("Done.")
