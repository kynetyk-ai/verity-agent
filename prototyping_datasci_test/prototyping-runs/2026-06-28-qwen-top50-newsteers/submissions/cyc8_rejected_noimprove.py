#!/usr/bin/env python3
"""Stellar classification - LightGBM ensemble with enhanced features."""
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
    """Enhanced feature engineering with color indices, gradients, ratios, and interactions."""
    out = df.copy()

    # Basic color indices (differences between photometric bands)
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["g_z"] = out["g"] - out["z"]
    out["r_z"] = out["r"] - out["z"]

    # Color gradients (second-order differences)
    out["cg1"] = (out["u"] - out["g"]) - (out["g"] - out["r"])   # u - 2g + r
    out["cg2"] = (out["g"] - out["r"]) - (out["r"] - out["i"])   # g - 2r + i
    out["cg3"] = (out["r"] - out["i"]) - (out["i"] - out["z"])   # r - 2i + z

    # Ratios
    out["u_g_rat"] = out["u"] / (out["g"] + 1e-9)
    out["g_r_rat"] = out["g"] / (out["r"] + 1e-9)
    out["r_i_rat"] = out["r"] / (out["i"] + 1e-9)
    out["i_z_rat"] = out["i"] / (out["z"] + 1e-9)

    # Log ratios
    out["log_ug"] = np.log1p(out["u"] / (out["g"] + 1e-9))
    out["log_gr"] = np.log1p(out["g"] / (out["r"] + 1e-9))

    # Redshift interactions
    out["redshift_times_u"] = out["redshift"] * out["u"]
    out["redshift_times_g"] = out["redshift"] * out["g"]
    out["redshift_times_r"] = out["redshift"] * out["r"]
    out["redshift_sq"] = out["redshift"] ** 2
    out["redshift_x_ug"] = out["redshift"] * out["u_g"]
    out["redshift_x_gr"] = out["redshift"] * out["g_r"]
    out["redshift_x_ri"] = out["redshift"] * out["r_i"]

    # Cross-band products
    out["u_x_g"] = out["u"] * out["g"]
    out["g_x_r"] = out["g"] * out["r"]
    out["r_x_i"] = out["r"] * out["i"]
    out["i_x_z"] = out["i"] * out["z"]

    # Sum features
    out["u_plus_g"] = out["u"] + out["g"]
    out["r_plus_i"] = out["r"] + out["i"]
    out["total_flux"] = out["u"] + out["g"] + out["r"] + out["i"] + out["z"]

    # Coordinate features (sin/cos to handle circular nature of azimutal angles)
    out["alpha_sin"] = np.sin(np.radians(out["alpha"]))
    out["alpha_cos"] = np.cos(np.radians(out["alpha"]))
    out["delta_sin"] = np.sin(np.radians(out["delta"]))
    out["delta_cos"] = np.cos(np.radians(out["delta"]))

    # Magnitude statistics
    mag_mean = (out["u"] + out["g"] + out["r"] + out["i"] + out["z"]) / 5
    out["mag_mean"] = mag_mean
    out["mag_std"] = np.sqrt(((out["u"] - mag_mean)**2 + (out["g"] - mag_mean)**2 +
                                (out["r"] - mag_mean)**2 + (out["i"] - mag_mean)**2 +
                                (out["z"] - mag_mean)**2) / 5)

    # Blue vs Red indicator (same as g_r but explicit)
    out["blue_red"] = out["g"] - out["r"]

    # Polynomial features for top signal bands
    out["u_sq"] = out["u"] ** 2
    out["g_sq"] = out["g"] ** 2
    out["r_sq"] = out["r"] ** 2

    # Spectral_type x redshift (one-hot style)
    for st in le_spec.classes_:
        out[f"spec_{st}_x_rz"] = (train["spectral_type"] == st).astype(int) * out["redshift"]

    # Interaction: spectral_type_enc * color indices
    out["spec_x_ug"] = out["spectral_type_enc"] * out["u_g"]
    out["spec_x_gr"] = out["spectral_type_enc"] * out["g_r"]
    out["pop_x_ug"] = out["galaxy_pop_enc"] * out["u_g"]
    out["pop_x_gr"] = out["galaxy_pop_enc"] * out["g_r"]

    return out


print("Building features...")
train_f = make_features(train)
test_f = make_features(test)

feat_cols = [
    "alpha", "delta",
    "u", "g", "r", "i", "z",
    "redshift",
    "spectral_type_enc", "galaxy_pop_enc",
    "u_g", "g_r", "r_i", "i_z", "u_z", "u_r", "g_i", "g_z", "r_z",
    "cg1", "cg2", "cg3",
    "u_g_rat", "g_r_rat", "r_i_rat", "i_z_rat",
    "log_ug", "log_gr",
    "redshift_times_u", "redshift_times_g", "redshift_times_r", "redshift_sq",
    "redshift_x_ug", "redshift_x_gr", "redshift_x_ri",
    "u_x_g", "g_x_r", "r_x_i", "i_x_z",
    "u_plus_g", "r_plus_i", "total_flux",
    "alpha_sin", "alpha_cos", "delta_sin", "delta_cos",
    "mag_mean", "mag_std", "blue_red",
    "u_sq", "g_sq", "r_sq",
    "spec_x_ug", "spec_x_gr", "pop_x_ug", "pop_x_gr",
]

# Add spectral_type x redshift columns
for feat in [c for c in train_f.columns if c.startswith("spec_") and c.endswith("_x_rz")]:
    feat_cols.append(feat)

print(f"Number of features: {len(feat_cols)}")

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

# Multiple model configurations for ensemble
configs = [
    {
        "name": "deep_wide",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 8,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    },
    {
        "name": "deeper",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 9,
        "num_leaves": 255,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.01,
        "reg_lambda": 0.05,
        "min_child_samples": 10,
        "random_state": SEED + 1,
        "n_jobs": -1,
        "verbose": -1,
    },
    {
        "name": "shallow_fast",
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 0.2,
        "min_child_samples": 30,
        "random_state": SEED + 2,
        "n_jobs": -1,
        "verbose": -1,
    },
    {
        "name": "balanced_aggressive",
        "n_estimators": 2000,
        "learning_rate": 0.04,
        "max_depth": 7,
        "num_leaves": 100,
        "subsample": 0.78,
        "colsample_bytree": 0.82,
        "reg_alpha": 0.08,
        "reg_lambda": 0.12,
        "min_child_samples": 15,
        "random_state": SEED + 3,
        "n_jobs": -1,
        "verbose": -1,
    },
]

# Train OOF predictions for validation
all_oof = []
for cfg in configs:
    name = cfg.pop("name")
    print(f"\nTraining {name}...")
    oof_preds = np.zeros((len(y_enc), n_classes))
    for fold, (tr_idx, val_idx) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y_enc)):
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X[tr_idx], y_enc[tr_idx], sample_weight=sample_weights[tr_idx],
              eval_set=[(X[val_idx], y_enc[val_idx])])
        oof_preds[val_idx] = m.predict_proba(X[val_idx])
        fb = balanced_accuracy_score(y_enc[val_idx], np.argmax(oof_preds[val_idx], axis=1))
        print(f"  Fold {fold+1}: BA = {fb:.4f}")
    oof_ba = balanced_accuracy_score(y_enc, np.argmax(oof_preds, axis=1))
    print(f"  {name} OOF BA: {oof_ba:.4f}")
    all_oof.append((name, oof_preds.copy(), cfg.copy()))

# Average OOF ensemble
avg_oof = np.mean([oof for _, oof, _ in all_oof], axis=0)
ensemble_ba = balanced_accuracy_score(y_enc, np.argmax(avg_oof, axis=1))
print(f"\n{'='*50}")
print(f"Ensemble OOF Balanced Accuracy: {ensemble_ba:.4f}")
print(f"{'='*50}")

# Train final models on full data and predict
print("\nTraining final ensemble on full data...")
test_preds = np.zeros((len(X_test), n_classes))
for name, _, cfg in all_oof:
    print(f"  Final {name}...")
    final_model = lgb.LGBMClassifier(**cfg)
    final_model.fit(X, y_enc, sample_weight=sample_weights)
    test_preds += final_model.predict_proba(X_test)

test_preds /= len(all_oof)
preds = y_le.inverse_transform(np.argmax(test_preds, axis=1))

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "predictions.csv")
pd.DataFrame({"id": test["id"], "class": preds}).to_csv(out_path, index=False)
print(f"\nPredictions saved to {out_path}")
print(f"Rows: {len(preds)}")
print("Prediction distribution:")
print(pd.Series(preds).value_counts())
