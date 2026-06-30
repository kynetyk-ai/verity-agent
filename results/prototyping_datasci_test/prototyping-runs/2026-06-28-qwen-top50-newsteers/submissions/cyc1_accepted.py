#!/usr/bin/env python3
"""Stellar classification - LightGBM with color index features."""
import os
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
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["g_z"] = out["g"] - out["z"]
    out["r_z"] = out["r"] - out["z"]
    out["u_g_rat"] = out["u"] / (out["g"] + 1e-9)
    out["g_r_rat"] = out["g"] / (out["r"] + 1e-9)
    out["r_i_rat"] = out["r"] / (out["i"] + 1e-9)
    out["i_z_rat"] = out["i"] / (out["z"] + 1e-9)
    out["redshift_times_u"] = out["redshift"] * out["u"]
    out["redshift_times_g"] = out["redshift"] * out["g"]
    out["redshift_sq"] = out["redshift"] ** 2
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
]

X = train_f[feat_cols].values
y = train["class"].values
X_test = test_f[feat_cols].values

y_le = LabelEncoder()
y_enc = y_le.fit_transform(y)
print("Classes:", y_le.classes_)
print("Class distribution:", np.bincount(y_enc))

class_counts = np.bincount(y_enc)
n_classes = len(class_counts)
total = len(y_enc)
weights = total / (n_classes * class_counts)
sample_weights = np.array([weights[yi] for yi in y_enc])

lgb_params = {
      "n_estimators": 1000,
      "learning_rate": 0.05,
      "max_depth": 7,
      "num_leaves": 63,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "reg_alpha": 0.1,
      "reg_lambda": 0.1,
      "random_state": SEED,
      "n_jobs": -1,
      "verbose": -1,
}

print("\nTraining LightGBM with 5-fold CV...")
oof_preds = np.zeros((len(y_enc), n_classes))
for fold, (tr_idx, val_idx) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y_enc)
):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X[tr_idx], y_enc[tr_idx], sample_weight=sample_weights[tr_idx],
          eval_set=[(X[val_idx], y_enc[val_idx])])
    oof_preds[val_idx] = m.predict_proba(X[val_idx])
    fb = balanced_accuracy_score(y_enc[val_idx], np.argmax(oof_preds[val_idx], axis=1))
    print("  Fold {}: BA = {:.4f}".format(fold + 1, fb))

oof_ba = balanced_accuracy_score(y_enc, np.argmax(oof_preds, axis=1))
print("\nOOF Balanced Accuracy: {:.4f}".format(oof_ba))

print("\nTraining final model on full data...")
final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(X, y_enc, sample_weight=sample_weights)

proba = final_model.predict_proba(X_test)
preds = y_le.inverse_transform(np.argmax(proba, axis=1))

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "predictions.csv")
pd.DataFrame({"id": test["id"], "class": preds}).to_csv(out_path, index=False)
print("\nPredictions saved to", out_path)
print("Rows:", len(preds))
print("Prediction distribution:")
print(pd.Series(preds).value_counts())
