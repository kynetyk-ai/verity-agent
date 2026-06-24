"""Stellar object classification (GALAXY / QSO / STAR).

Pipeline:
  - Physics-informed feature engineering: photometric colour indices (all band
    pairs), colour curvature, redshift transforms and colour x redshift
    interactions, magnitude aggregates, and angular sky-position encodings.
  - Native categorical handling for `spectral_type` and `galaxy_population`.
  - A stratified K-fold LightGBM ensemble trained with balanced class weights
    (scoring metric is BALANCED accuracy on an imbalanced 3-class problem).
  - Out-of-fold probabilities drive a per-class probability-multiplier search
    that directly optimises balanced accuracy; multipliers are then applied to
    the averaged test probabilities.

Deterministic and self-contained. Trains on the FULL training set.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 20240607
np.random.seed(SEED)

DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]
N_SPLITS = 5


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # All pairwise colour indices (u-g, u-r, ..., i-z).
    for idx, a in enumerate(BANDS):
        for b in BANDS[idx + 1:]:
            df[f"{a}_{b}"] = df[a] - df[b]

    # Adjacent colours: redshift interactions (most discriminative signal).
    adjacent = [f"{a}_{b}" for a, b in zip(BANDS[:-1], BANDS[1:])]
    for c in adjacent:
        df[f"{c}_x_redshift"] = df[c] * df["redshift"]
        df[f"{c}_div_redshift1p"] = df[c] / (1.0 + np.abs(df["redshift"]))

    # Colour curvature (second differences of the spectral energy distribution).
    df["ug_gr_diff"] = df["u_g"] - df["g_r"]
    df["gr_ri_diff"] = df["g_r"] - df["r_i"]
    df["ri_iz_diff"] = df["r_i"] - df["i_z"]

    # Redshift transforms.
    z = df["redshift"]
    df["redshift_log1p"] = np.log1p(np.clip(z, 0, None))
    df["redshift_sq"] = z ** 2
    df["redshift_sqrt_abs"] = np.sqrt(np.abs(z))

    # Magnitude aggregates.
    vals = df[BANDS]
    df["mag_mean"] = vals.mean(axis=1)
    df["mag_std"] = vals.std(axis=1)
    df["mag_min"] = vals.min(axis=1)
    df["mag_max"] = vals.max(axis=1)
    df["mag_range"] = df["mag_max"] - df["mag_min"]
    for b in BANDS:
        df[f"{b}_minus_mean"] = df[b] - df["mag_mean"]

    # Angular sky-position encodings (cyclic).
    ar = np.deg2rad(df["alpha"])
    dr = np.deg2rad(df["delta"])
    df["alpha_sin"] = np.sin(ar)
    df["alpha_cos"] = np.cos(ar)
    df["delta_sin"] = np.sin(dr)
    df["delta_cos"] = np.cos(dr)

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    return df


def tune_multipliers(proba: np.ndarray, y_true: np.ndarray, n_classes: int):
    """Search per-class probability multipliers maximising balanced accuracy."""
    best = np.ones(n_classes)
    best_score = balanced_accuracy_score(y_true, np.argmax(proba, axis=1))

    # Coarse grid (GALAXY fixed at 1.0; search the two minority classes).
    grid = np.arange(0.4, 2.21, 0.1)
    for b in grid:
        for c in grid:
            mult = np.array([1.0, b, c])
            pred = np.argmax(proba * mult, axis=1)
            s = balanced_accuracy_score(y_true, pred)
            if s > best_score:
                best_score = s
                best = mult

    # Coordinate refinement over all classes.
    for step in [0.05, 0.02, 0.01, 0.005]:
        improved = True
        while improved:
            improved = False
            for j in range(n_classes):
                for delta in (-step, step):
                    mult = best.copy()
                    mult[j] = max(0.2, mult[j] + delta)
                    pred = np.argmax(proba * mult, axis=1)
                    s = balanced_accuracy_score(y_true, pred)
                    if s > best_score + 1e-9:
                        best_score = s
                        best = mult
                        improved = True
    return best, best_score


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    classes = np.array(sorted(train["class"].unique()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = train["class"].map(class_to_idx).values.astype(int)
    n_classes = len(classes)

    X = feature_engineer(train.drop(columns=["class"]))
    X_test = feature_engineer(test)

    feature_cols = [c for c in X.columns if c != "id"]
    X = X[feature_cols]
    X_test = X_test[feature_cols]
    cat_idx = [feature_cols.index(c) for c in CAT_COLS]

    params = dict(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=2000,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=45,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        max_depth=-1,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
        deterministic=True,
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), n_classes))
    test_proba = np.zeros((len(X_test), n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Balanced class weights for this fold.
        counts = np.bincount(y_tr, minlength=n_classes)
        weights = len(y_tr) / (n_classes * counts)
        sw = weights[y_tr]

        model = lgb.LGBMClassifier(random_state=SEED + fold, **params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sw,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            categorical_feature=cat_idx,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        oof[va_idx] = model.predict_proba(X_va)
        test_proba += model.predict_proba(X_test) / N_SPLITS

    mult, oof_score = tune_multipliers(oof, y, n_classes)
    print(f"OOF balanced accuracy (tuned): {oof_score:.5f}  multipliers={mult}")

    pred_idx = np.argmax(test_proba * mult, axis=1)
    pred_labels = classes[pred_idx]

    out = pd.DataFrame({"id": test["id"].values, "class": pred_labels})
    out.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    print(f"Wrote {len(out)} predictions to {os.path.join(OUT_DIR, 'predictions.csv')}")


if __name__ == "__main__":
    main()
