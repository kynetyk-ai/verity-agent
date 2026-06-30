#!/usr/bin/env python3
"""Stellar classification - Enhanced LightGBM + XGBoost ensemble with extended features."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR = os.environ.get("VERITY_OUT", "out")

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ── Encode categoricals ────────────────────────────────────────────────
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

# ── Feature engineering ─────────────────────────────────────────────────
def make_features(df):
    out = df.copy()
    # Basic color indices
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["g_z"] = out["g"] - out["z"]
    out["r_z"] = out["r"] - out["z"]

    # Color ratios
    out["u_g_rat"] = out["u"] / (out["g"] + 1e-9)
    out["g_r_rat"] = out["g"] / (out["r"] + 1e-9)
    out["r_i_rat"] = out["r"] / (out["i"] + 1e-9)
    out["i_z_rat"] = out["i"] / (out["z"] + 1e-9)

    # Redshift interactions
    out["redshift_times_u"] = out["redshift"] * out["u"]
    out["redshift_times_g"] = out["redshift"] * out["g"]
    out["redshift_times_r"] = out["redshift"] * out["r"]
    out["redshift_sq"] = out["redshift"] ** 2
    out["redshift_cube"] = out["redshift"] ** 3

    # Photometric sums
    out["sum_phot"] = out["u"] + out["g"] + out["r"] + out["i"] + out["z"]
    out["mean_phot"] = out["sum_phot"] / 5.0

    # Color index squared terms
    out["u_g_sq"] = out["u_g"] ** 2
    out["g_r_sq"] = out["g_r"] ** 2
    out["r_i_sq"] = out["r_i"] ** 2
    out["i_z_sq"] = out["i_z"] ** 2

    # Color index * redshift
    out["redshift_u_g"] = out["redshift"] * out["u_g"]
    out["redshift_g_r"] = out["redshift"] * out["g_r"]
    out["redshift_r_i"] = out["redshift"] * out["r_i"]
    out["redshift_i_z"] = out["redshift"] * out["i_z"]

    # Log transforms
    for col in ["u", "g", "r", "i", "z"]:
        out[f"log_{col}"] = np.log1p(out[col])

    # Abs differences
    out["abs_u_g"] = np.abs(out["u_g"])
    out["abs_g_r"] = np.abs(out["g_r"])

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
    "redshift_times_u", "redshift_times_g", "redshift_times_r",
    "redshift_sq", "redshift_cube",
    "sum_phot", "mean_phot",
    "u_g_sq", "g_r_sq", "r_i_sq", "i_z_sq",
    "redshift_u_g", "redshift_g_r", "redshift_r_i", "redshift_i_z",
    "log_u", "log_g", "log_r", "log_i", "log_z",
    "abs_u_g", "abs_g_r",
]

X = train_f[feat_cols].values
y = train["class"].values
X_test = test_f[feat_cols].values

y_le = LabelEncoder()
y_enc = y_le.fit_transform(y)
print("Classes:", y_le.classes_)
print("Class counts:", np.bincount(y_enc))

# ── Compute balanced sample weights ─────────────────────────────────────
class_counts = np.bincount(y_enc)
n_classes = len(class_counts)
total = len(y_enc)
weights = total / (n_classes * class_counts)
sample_weights = np.array([weights[yi] for yi in y_enc])

# ── Train LightGBM ─────────────────────────────────────────────────────
print("\nTraining LightGBM...")
lgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 95,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "min_child_samples": 20,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    "force_col_wise": True,
}

lgb_oof = np.zeros((len(y_enc), n_classes))
lgb_test_proba = np.zeros((len(X_test), n_classes))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_enc)):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X[tr_idx], y_enc[tr_idx], sample_weight=sample_weights[tr_idx],
          eval_set=[(X[val_idx], y_enc[val_idx])])
    lgb_oof[val_idx] = m.predict_proba(X[val_idx])
    lgb_test_proba += m.predict_proba(X_test) / 5
    fb = balanced_accuracy_score(y_enc[val_idx], np.argmax(lgb_oof[val_idx], axis=1))
    print(f"  [LGB] Fold {fold + 1}: BA = {fb:.4f}")

lgb_ba = balanced_accuracy_score(y_enc, np.argmax(lgb_oof, axis=1))
print(f"[LGB] OOF BA: {lgb_ba:.4f}")

# ── Train XGBoost ─────────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "min_child_weight": 5,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
}

xgb_oof = np.zeros((len(y_enc), n_classes))
xgb_test_proba = np.zeros((len(X_test), n_classes))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_enc)):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X[tr_idx], y_enc[tr_idx], sample_weight=sample_weights[tr_idx],
          eval_set=[(X[val_idx], y_enc[val_idx])])
    xgb_oof[val_idx] = m.predict_proba(X[val_idx])
    xgb_test_proba += m.predict_proba(X_test) / 5
    fb = balanced_accuracy_score(y_enc[val_idx], np.argmax(xgb_oof[val_idx], axis=1))
    print(f"  [XGB] Fold {fold + 1}: BA = {fb:.4f}")

xgb_ba = balanced_accuracy_score(y_enc, np.argmax(xgb_oof, axis=1))
print(f"[XGB] OOF BA: {xgb_ba:.4f}")

# ── Ensemble ───────────────────────────────────────────────────────────
total_ba = lgb_ba + xgb_ba
w_lgb = lgb_ba / total_ba
w_xgb = xgb_ba / total_ba
print(f"\nEnsemble weights: LGB={w_lgb:.3f}, XGB={w_xgb:.3f}")

final_proba = w_lgb * lgb_test_proba + w_xgb * xgb_test_proba
preds_enc = np.argmax(final_proba, axis=1)
preds = y_le.inverse_transform(preds_enc)

# ── Save predictions ───────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "predictions.csv")
pd.DataFrame({"id": test["id"], "class": preds}).to_csv(out_path, index=False)

print(f"\nPredictions saved to {out_path}")
print(f"Rows: {len(preds)}")
print("Prediction distribution:")
print(pd.Series(preds).value_counts())
