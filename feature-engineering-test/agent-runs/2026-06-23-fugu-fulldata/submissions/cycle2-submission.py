"""Fast competitive stellar object classifier.

This revision addresses the previous timeout by using a bounded LightGBM workflow:
one stratified validation pass to select iteration counts and tune balanced-accuracy
class multipliers, followed by a small three-model LightGBM probability ensemble
trained on the full labelled data. It keeps the high-signal astronomy features
(colour indices, redshift transforms/interactions, spectral/population categorical
signals) and uses class-balanced sample weights for the balanced-accuracy metric.
"""

import os
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 20240623
np.random.seed(SEED)

DATA_DIR = os.environ.get("VERITY_DATA", "data")
OUT_DIR = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ["u", "g", "r", "i", "z"]
BASE_NUMERIC = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BASE_CATS = ["spectral_type", "galaxy_population"]
CAT_COLS = ["spectral_type", "galaxy_population", "spectral_population"]
VALID_SIZE = 0.16
MAX_ESTIMATORS = 1250

MODEL_SPECS = [
    (
        "base63",
        101,
        dict(
            learning_rate=0.045,
            num_leaves=63,
            min_child_samples=70,
            subsample=0.90,
            colsample_bytree=0.88,
            reg_alpha=0.04,
            reg_lambda=1.25,
            max_bin=255,
        ),
    ),
    (
        "conservative47",
        202,
        dict(
            learning_rate=0.050,
            num_leaves=47,
            min_child_samples=85,
            subsample=0.94,
            colsample_bytree=0.92,
            reg_alpha=0.05,
            reg_lambda=1.50,
            max_bin=255,
        ),
    ),
    (
        "wide95",
        303,
        dict(
            learning_rate=0.040,
            num_leaves=95,
            min_child_samples=55,
            subsample=0.92,
            colsample_bytree=0.90,
            reg_alpha=0.03,
            reg_lambda=1.00,
            max_bin=255,
        ),
    ),
]


def read_inputs():
    dtype = {"id": "int64"}
    dtype.update({c: "float32" for c in BASE_NUMERIC})
    dtype.update({c: "string" for c in BASE_CATS})
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), dtype=dtype)
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), dtype=dtype)
    return train, test


def align_categories(train: pd.DataFrame, test: pd.DataFrame):
    for c in BASE_CATS:
        cats = sorted(set(train[c].astype(str).unique()).union(test[c].astype(str).unique()))
        cat_type = CategoricalDtype(categories=cats)
        train[c] = train[c].astype(str).astype(cat_type)
        test[c] = test[c].astype(str).astype(cat_type)

    train["spectral_population"] = (
        train["spectral_type"].astype(str) + "_" + train["galaxy_population"].astype(str)
    )
    test["spectral_population"] = (
        test["spectral_type"].astype(str) + "_" + test["galaxy_population"].astype(str)
    )
    combo_cats = sorted(
        set(train["spectral_population"].unique()).union(test["spectral_population"].unique())
    )
    combo_type = CategoricalDtype(categories=combo_cats)
    train["spectral_population"] = train["spectral_population"].astype(combo_type)
    test["spectral_population"] = test["spectral_population"].astype(combo_type)


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = np.float32(1e-3)

    # Pairwise colour indices; these are the strongest photometric signals.
    for i, a in enumerate(BANDS):
        av = out[a].astype("float32")
        for b in BANDS[i + 1 :]:
            out[f"{a}_{b}"] = (av - out[b].astype("float32")).astype("float32")

    adjacent = ["u_g", "g_r", "r_i", "i_z"]
    z = out["redshift"].astype("float32")
    abs_z = np.abs(z).astype("float32")
    z1p = (1.0 + abs_z).astype("float32")

    # Redshift transforms and thresholds: near-zero redshift is highly star-like,
    # while high redshift separates many QSOs from galaxies.
    out["redshift_abs"] = abs_z
    out["redshift_sq"] = (z * z).astype("float32")
    out["redshift_log1p_pos"] = np.log1p(np.clip(z, 0, None)).astype("float32")
    out["redshift_sqrt_abs"] = np.sqrt(abs_z).astype("float32")
    out["redshift_neg"] = (z < 0).astype("int8")
    out["redshift_lt_0p12"] = (abs_z < 0.12).astype("int8")
    out["redshift_gt_0p8"] = (z > 0.8).astype("int8")
    out["redshift_gt_1p6"] = (z > 1.6).astype("int8")
    out["redshift_gt_2p5"] = (z > 2.5).astype("int8")

    for c in adjacent:
        cv = out[c].astype("float32")
        out[f"{c}_x_redshift"] = (cv * z).astype("float32")
        out[f"{c}_div_1p_abs_redshift"] = (cv / z1p).astype("float32")

    # Curvature / ratios of the observed spectral-energy distribution.
    out["ug_gr_diff"] = (out["u_g"] - out["g_r"]).astype("float32")
    out["gr_ri_diff"] = (out["g_r"] - out["r_i"]).astype("float32")
    out["ri_iz_diff"] = (out["r_i"] - out["i_z"]).astype("float32")
    out["ug_gr_ratio"] = (out["u_g"] / (np.abs(out["g_r"]) + eps)).astype("float32")
    out["gr_ri_ratio"] = (out["g_r"] / (np.abs(out["r_i"]) + eps)).astype("float32")
    out["ri_iz_ratio"] = (out["r_i"] / (np.abs(out["i_z"]) + eps)).astype("float32")

    # Magnitude aggregates and centred bands.
    vals = out[BANDS].astype("float32")
    out["mag_mean"] = vals.mean(axis=1).astype("float32")
    out["mag_std"] = vals.std(axis=1).astype("float32")
    out["mag_min"] = vals.min(axis=1).astype("float32")
    out["mag_max"] = vals.max(axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")
    for b in BANDS:
        out[f"{b}_minus_mag_mean"] = (out[b] - out["mag_mean"]).astype("float32")
        out[f"{b}_x_redshift"] = (out[b].astype("float32") * z).astype("float32")

    # Cyclic sky-coordinate encodings capture weak survey-position systematics.
    ar = np.deg2rad(out["alpha"].astype("float32"))
    dr = np.deg2rad(out["delta"].astype("float32"))
    out["alpha_sin"] = np.sin(ar).astype("float32")
    out["alpha_cos"] = np.cos(ar).astype("float32")
    out["delta_sin"] = np.sin(dr).astype("float32")
    out["delta_cos"] = np.cos(dr).astype("float32")

    for c in CAT_COLS:
        out[c] = out[c].astype("category")
    return out


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype("float64")
    weights = len(y) / (n_classes * np.maximum(counts, 1.0))
    return weights[y].astype("float32")


def tune_multipliers(proba: np.ndarray, y_true: np.ndarray, n_classes: int):
    best = np.ones(n_classes, dtype="float64")
    best_score = balanced_accuracy_score(y_true, np.argmax(proba, axis=1))

    # Anchor class 0; common scaling is invariant. The sorted class order for this
    # data is GALAXY, QSO, STAR, so this tunes minority-class recall trade-offs.
    if n_classes == 3:
        grid = np.arange(0.65, 1.551, 0.025)
        for m1 in grid:
            scaled_1 = proba[:, 1] * m1
            for m2 in grid:
                pred = np.argmax(
                    np.column_stack((proba[:, 0], scaled_1, proba[:, 2] * m2)), axis=1
                )
                score = balanced_accuracy_score(y_true, pred)
                if score > best_score:
                    best_score = score
                    best = np.array([1.0, m1, m2], dtype="float64")

    # Light coordinate refinement around the grid optimum.
    for step in (0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for j in range(1, n_classes):
                for direction in (-1.0, 1.0):
                    trial = best.copy()
                    trial[j] = max(0.25, trial[j] + direction * step)
                    score = balanced_accuracy_score(y_true, np.argmax(proba * trial, axis=1))
                    if score > best_score + 1e-12:
                        best_score = score
                        best = trial
                        improved = True
    return best.astype("float32"), best_score


def make_model(params: dict, n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=int(n_estimators),
        max_depth=-1,
        subsample_freq=1,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
        force_col_wise=True,
        deterministic=True,
        **params,
    )


def main():
    train, test = read_inputs()
    test_ids = test["id"].to_numpy(copy=True)

    classes = np.array(sorted(train["class"].unique()))
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y = train["class"].map(class_to_idx).to_numpy(dtype="int32")
    n_classes = len(classes)

    align_categories(train, test)
    X = feature_engineer(train.drop(columns=["class"]))
    X_test = feature_engineer(test)
    feature_cols = [c for c in X.columns if c != "id"]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

    tr_idx, va_idx = train_test_split(
        np.arange(len(y)), test_size=VALID_SIZE, random_state=SEED, stratify=y
    )
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    sw_tr = balanced_weights(y_tr, n_classes)
    sw_va = balanced_weights(y_va, n_classes)

    val_proba_sum = np.zeros((len(va_idx), n_classes), dtype="float64")
    final_iters = []

    # Validation models estimate both per-spec iteration counts and ensemble
    # decision multipliers. This costs only three bounded fits, not K-fold CV.
    for name, seed_offset, params in MODEL_SPECS:
        model = make_model(params, MAX_ESTIMATORS, SEED + seed_offset)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[sw_va],
            eval_metric="multi_logloss",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None or best_iter <= 0:
            best_iter = MAX_ESTIMATORS
        # Refit on more data; a small iteration uplift compensates for the larger
        # full training set while keeping runtime bounded.
        final_iter = int(min(MAX_ESTIMATORS, max(360, round(best_iter * 1.08))))
        final_iters.append(final_iter)
        val_proba_sum += model.predict_proba(X_va, num_iteration=best_iter)
        print(f"validation {name}: best_iter={best_iter}, final_iter={final_iter}")

    val_proba = val_proba_sum / len(MODEL_SPECS)
    multipliers, val_score = tune_multipliers(val_proba, y_va, n_classes)
    print(
        f"ensemble validation balanced_accuracy={val_score:.6f}; "
        f"multipliers={multipliers.tolist()}"
    )

    # Final ensemble trains every model on all labelled rows, as required.
    sw_all = balanced_weights(y, n_classes)
    test_proba = np.zeros((len(X_test), n_classes), dtype="float64")
    for (name, seed_offset, params), n_estimators in zip(MODEL_SPECS, final_iters):
        model = make_model(params, n_estimators, SEED + 1000 + seed_offset)
        model.fit(X, y, sample_weight=sw_all, categorical_feature=CAT_COLS)
        test_proba += model.predict_proba(X_test)
        print(f"final {name}: n_estimators={n_estimators}")
    test_proba /= len(MODEL_SPECS)

    pred_idx = np.argmax(test_proba * multipliers, axis=1)
    pred_labels = classes[pred_idx]

    out = pd.DataFrame({"id": test_ids, "class": pred_labels})
    pred_path = os.path.join(OUT_DIR, "predictions.csv")
    out.to_csv(pred_path, index=False)
    print(f"wrote {len(out)} predictions to {pred_path}")


if __name__ == "__main__":
    main()
