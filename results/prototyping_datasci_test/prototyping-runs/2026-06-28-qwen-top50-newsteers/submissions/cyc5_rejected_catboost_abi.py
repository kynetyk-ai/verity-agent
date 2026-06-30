#!/usr/bin/env python3
"""Stellar classification - Ensemble LightGBM + CatBoost with advanced features."""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import balanced_accuracy_score

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR = os.environ.get("VERITY_OUT", "out")

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# -------------------------------------------------------------------
# Encode categoricals
# -------------------------------------------------------------------
le_spec = LabelEncoder()
le_pop = LabelEncoder()
all_spec = pd.concat([train["spectral_type"], test["spectral_type"]])
all_pop = pd.concat([train["galaxy_population"], test["galaxy_population"]])
le_spec.fit(all_spec)
le_pop.fit(all_pop)

train["spectral_type_enc"] = le_spec.transform(train["spectral_type"])
train["galaxy_pop_enc"] = le_pop.transform(train["galaxy_population"])
test["spectral_type_enc"] = le_spec.transform(test["spectral_type"])
test["galaxy_pop_enc"] = le_pop.transform(test["galaxy_population"])

# Encode target
le_class = LabelEncoder()
le_class.fit(["GALAXY", "QSO", "STAR"])
y = le_class.transform(train["class"])
n_classes = len(le_class.classes_)
print("Classes:", le_class.classes_)
print("Class distribution:", np.bincount(y))

# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
def make_features(df):
    out = pd.DataFrame(index=df.index)

    # Raw features
    for col in ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]:
        out[col] = df[col].values

    # Encoded categoricals
    out["spectral_type_enc"] = df["spectral_type_enc"].values
    out["galaxy_pop_enc"] = df["galaxy_pop_enc"].values

    bands = ["u", "g", "r", "i", "z"]

    # All pair color indices
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            b1, b2 = bands[i], bands[j]
            out[f"{b1}_{b2}"] = df[b1] - df[b2]

    # Ratios
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            b1, b2 = bands[i], bands[j]
            out[f"rat_{b1}_{b2}"] = df[b1] / (df[b2] + 1e-9)

    # Log magnitudes
    for b in bands:
        out[f"log_{b}"] = np.log1p(df[b].clip(lower=0))

    # Redshift interactions
    for b in bands:
        out[f"z_{b}"] = df["redshift"] * df[b]
        out[f"z_{b}_sq"] = (df["redshift"] ** 2) * df[b]

    out["redshift_sq"] = df["redshift"] ** 2
    out["redshift_cu"] = df["redshift"] ** 3

    # Magnitude aggregates
    mag_values = df[bands].values
    out["mag_sum"] = mag_values.sum(axis=1)
    out["mag_mean"] = mag_values.mean(axis=1)
    out["mag_std"] = mag_values.std(axis=1)
    out["mag_max"] = mag_values.max(axis=1)
    out["mag_min"] = mag_values.min(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]

    # Color index ratios
    ug = df["u"] - df["g"]
    gr = df["g"] - df["r"]
    ri = df["r"] - df["i"]
    iz = df["i"] - df["z"]
    out["ug_gr_ratio"] = ug / (gr + 1e-9)
    out["gr_ri_ratio"] = gr / (ri + 1e-9)
    out["ri_iz_ratio"] = ri / (iz + 1e-9)

    # Absolute color indices
    out["abs_ug"] = np.abs(ug.values)
    out["abs_gr"] = np.abs(gr.values)
    out["abs_ri"] = np.abs(ri.values)
    out["abs_iz"] = np.abs(iz.values)

    # Redshift bins
    out["redshift_bin"] = pd.cut(df["redshift"], bins=[-1, 0.1, 0.5, 1.0, 3.0, 10.0], labels=False).astype(float)

    return out

train_f = make_features(train)
test_f = make_features(test)

# Get all feature columns
exclude = {"spectral_type", "galaxy_population", "class", "id"}
feat_cols = [c for c in train_f.columns if c not in exclude]
print(f"Number of features: {len(feat_cols)}")

X = train_f[feat_cols].values.astype(np.float64)
X_test = test_f[feat_cols].values.astype(np.float64)

# Handle any NaNs
X = np.nan_to_num(X, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

# -------------------------------------------------------------------
# Class weights
# -------------------------------------------------------------------
class_counts = np.bincount(y)
total = len(y)
weights = total / (n_classes * class_counts)
sample_weights = np.array([weights[yi] for yi in y], dtype=np.float64)

# -------------------------------------------------------------------
# Train LightGBM
# -------------------------------------------------------------------
print("\n--- Training LightGBM (5-fold CV) ---")

lgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "min_child_samples": 20,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
lgb_oof = np.zeros((len(y), n_classes), dtype=np.float64)
lgb_models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx],
          eval_set=[(X[val_idx], y[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_oof[val_idx] = m.predict_proba(X[val_idx])
    lgb_models.append(m)
    print(f"  LightGBM fold {fold} done")

lgb_ba = balanced_accuracy_score(y, np.argmax(lgb_oof, axis=1))
print(f"  LightGBM OOF balanced accuracy: {lgb_ba:.4f}")

# -------------------------------------------------------------------
# Train CatBoost
# -------------------------------------------------------------------
print("\n--- Training CatBoost (5-fold CV) ---")

cb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
    "random_state": SEED,
    "thread_count": -1,
    "verbose": 0,
    "loss_function": "MultiClass",
}

catboost_oof = np.zeros((len(y), n_classes), dtype=np.float64)
catboost_models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    m = cb.CatBoostClassifier(**cb_params)
    m.fit(X[tr_idx], y[tr_idx],
          sample_weight=sample_weights[tr_idx],
          eval_set=[(X[val_idx], y[val_idx])],
          early_stopping_rounds=100, verbose=False)
    catboost_oof[val_idx] = m.predict_proba(X[val_idx])
    catboost_models.append(m)
    print(f"  CatBoost fold {fold} done")

cb_ba = balanced_accuracy_score(y, np.argmax(catboost_oof, axis=1))
print(f"  CatBoost OOF balanced accuracy: {cb_ba:.4f}")

# -------------------------------------------------------------------
# Ensemble OOF
# -------------------------------------------------------------------
if lgb_ba >= cb_ba:
    ens_oof = 0.6 * lgb_oof + 0.4 * catboost_oof
else:
    ens_oof = 0.4 * lgb_oof + 0.6 * catboost_oof

ens_ba = balanced_accuracy_score(y, np.argmax(ens_oof, axis=1))
print(f"\nEnsemble OOF balanced accuracy: {ens_ba:.4f}")

# -------------------------------------------------------------------
# Retrain on full data for test predictions
# -------------------------------------------------------------------
print("\n--- Retraining on full data ---")

lgb_full = lgb.LGBMClassifier(**lgb_params)
lgb_full.fit(X, y, sample_weight=sample_weights)
lgb_test_pred = lgb_full.predict_proba(X_test)

cb_full = cb.CatBoostClassifier(**cb_params)
cb_full.fit(X, y, sample_weight=sample_weights)
cb_test_pred = cb_full.predict_proba(X_test)

if lgb_ba >= cb_ba:
    test_pred = 0.6 * lgb_test_pred + 0.4 * cb_test_pred
else:
    test_pred = 0.4 * lgb_test_pred + 0.6 * cb_test_pred

pred_classes = le_class.inverse_transform(np.argmax(test_pred, axis=1))

# -------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
preds = pd.DataFrame({
    "id": test["id"],
    "class": pred_classes,
})
preds.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
print(f"\nPredictions saved: {len(preds)} rows")
print(f"Predictions class distribution:\n{preds['class'].value_counts()}")
