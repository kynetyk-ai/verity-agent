#!/usr/bin/env python3
"""Stellar classification - LightGBM ensemble with enhanced color index features."""
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR = os.environ.get("VERITY_OUT", "out")

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Encode categoricals
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

def make_features(df):
    out = df.copy()
    # Basic color indices
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    # Extended color indices
    out["u_z"] = out["u"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["g_z"] = out["g"] - out["z"]
    out["r_z"] = out["r"] - out["z"]
    # Ratios
    out["u_g_rat"] = out["u"] / (out["g"] + 1e-9)
    out["g_r_rat"] = out["g"] / (out["r"] + 1e-9)
    out["r_i_rat"] = out["r"] / (out["i"] + 1e-9)
    out["i_z_rat"] = out["i"] / (out["z"] + 1e-9)
    # Redshift interactions
    out["redshift_times_u"] = out["redshift"] * out["u"]
    out["redshift_times_g"] = out["redshift"] * out["g"]
    out["redshift_sq"] = out["redshift"] ** 2
    # Additional interaction features
    out["redshift_times_r"] = out["redshift"] * out["r"]
    out["redshift_times_i"] = out["redshift"] * out["i"]
    out["redshift_times_z"] = out["redshift"] * out["z"]
    out["u_times_g"] = out["u"] * out["g"]
    out["r_times_i"] = out["r"] * out["i"]
    # Log features (handles potential scale issues)
    out["log_u"] = np.log1p(out["u"].clip(lower=0))
    out["log_g"] = np.log1p(out["g"].clip(lower=0))
    out["log_r"] = np.log1p(out["r"].clip(lower=0))
    out["log_i"] = np.log1p(out["i"].clip(lower=0))
    out["log_z"] = np.log1p(out["z"].clip(lower=0))
    # Color index ratios
    out["ug_gr_ratio"] = out["u_g"] / (out["g_r"] + 1e-9)
    out["ri_iz_ratio"] = out["r_i"] / (out["i_z"] + 1e-9)
    # Magnitude spreads
    out["mag_spread"] = out[["u", "g", "r", "i", "z"]].max(axis=1) - out[["u", "g", "r", "i", "z"]].min(axis=1)
    out["mag_mean"] = out[["u", "g", "r", "i", "z"]].mean(axis=1)
    out["mag_std"] = out[["u", "g", "r", "i", "z"]].std(axis=1)
    return out

train_f = make_features(train)
test_f = make_features(test)

feat_cols = [
    "alpha", "delta",
    "u", "g", "r", "i", "z",
    "redshift",
    "spectral_type_enc", "galaxy_pop_enc",
    "u_g", "g_r", "r_i", "i_z", "u_z", "u_r", "g_i", "g_z", "r_z",
    "u_g_rat", "g_r_rat", "r_i_rat", "i_z_rat",
    "redshift_times_u", "redshift_times_g", "redshift_sq",
    "redshift_times_r", "redshift_times_i", "redshift_times_z",
    "u_times_g", "r_times_i",
    "log_u", "log_g", "log_r", "log_i", "log_z",
    "ug_gr_ratio", "ri_iz_ratio",
    "mag_spread", "mag_mean", "mag_std",
]

X = np.asarray(train_f[feat_cols])
y_raw = np.asarray(train["class"])
X_test = np.asarray(test_f[feat_cols])

y_le = LabelEncoder()
y_enc = y_le.fit_transform(y_raw)
print("Classes:", y_le.classes_)
print("Class distribution:", np.bincount(y_enc))

n_classes = len(y_le.classes_)

# Compute class-balanced sample weights
class_counts = np.bincount(y_enc)
total = len(y_enc)
weights = total / (n_classes * class_counts)
sample_weights = np.array([weights[yi] for yi in y_enc])

def train_lgb_model(X_train, y_train, sw, X_val, y_val):
    params = {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 8,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "min_child_samples": 20,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    m = lgb.LGBMClassifier(**params)
    m.fit(X_train, y_train, sample_weight=sw,
          eval_set=[(X_val, y_val)])
    return m

print("\nTraining LightGBM with 5-fold CV...")
oof_preds = np.zeros((len(y_enc), n_classes))
for fold, (tr_idx, val_idx) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y_enc)
):
    m = train_lgb_model(X[tr_idx], y_enc[tr_idx], sample_weights[tr_idx],
                         X[val_idx], y_enc[val_idx])
    oof_preds[val_idx] = m.predict_proba(X[val_idx])
    fb = balanced_accuracy_score(y_enc[val_idx], np.argmax(oof_preds[val_idx], axis=1))
    print("  Fold {}: BA = {:.4f}".format(fold + 1, fb))

oof_ba = balanced_accuracy_score(y_enc, np.argmax(oof_preds, axis=1))
print("\nOOF Balanced Accuracy: {:.4f}".format(oof_ba))

# Second model with different params
print("\nTraining second LightGBM with different params (5-fold)...")
oof_preds2 = np.zeros((len(y_enc), n_classes))
for fold, (tr_idx, val_idx) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1).split(X, y_enc)
):
    params2 = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "num_leaves": 63,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.2,
        "reg_lambda": 0.2,
        "min_child_samples": 30,
        "random_state": SEED + 2,
        "n_jobs": -1,
        "verbose": -1,
    }
    m2 = lgb.LGBMClassifier(**params2)
    m2.fit(X[tr_idx], y_enc[tr_idx], sample_weight=sample_weights[tr_idx],
           eval_set=[(X[val_idx], y_enc[val_idx])])
    oof_preds2[val_idx] = m2.predict_proba(X[val_idx])
    fb2 = balanced_accuracy_score(y_enc[val_idx], np.argmax(oof_preds2[val_idx], axis=1))
    print("  Fold {}: BA = {:.4f}".format(fold + 1, fb2))

oof_ba2 = balanced_accuracy_score(y_enc, np.argmax(oof_preds2, axis=1))
print("\nOOF2 Balanced Accuracy: {:.4f}".format(oof_ba2))

# Ensemble OOF
ensemble_oof = 0.6 * oof_preds + 0.4 * oof_preds2
ensemble_ba = balanced_accuracy_score(y_enc, np.argmax(ensemble_oof, axis=1))
print("Ensemble OOF BA: {:.4f}".format(ensemble_ba))

# Train final models on full data
print("\nTraining final models on full data...")
final_model1 = train_lgb_model(X, y_enc, sample_weights, X[:100], y_enc[:100])

params2_final = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "num_leaves": 63,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "min_child_samples": 30,
    "random_state": SEED + 2,
    "n_jobs": -1,
    "verbose": -1,
}
final_model2 = lgb.LGBMClassifier(**params2_final)
final_model2.fit(X, y_enc, sample_weight=sample_weights)

# Ensemble predictions
proba1 = final_model1.predict_proba(X_test)
proba2 = final_model2.predict_proba(X_test)
proba = 0.6 * proba1 + 0.4 * proba2
preds = y_le.inverse_transform(np.argmax(proba, axis=1))

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "predictions.csv")
pd.DataFrame({"id": test["id"], "class": preds}).to_csv(out_path, index=False)
print("\nPredictions saved to", out_path)
print("Rows:", len(preds))
print("Prediction distribution:")
print(pd.Series(preds).value_counts())
