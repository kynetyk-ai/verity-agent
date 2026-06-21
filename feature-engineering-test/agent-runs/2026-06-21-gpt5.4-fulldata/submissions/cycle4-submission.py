import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
TARGET = "class"
ID_COL = "id"
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
MAG_COLS = ["u", "g", "r", "i", "z"]

random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")


class AveragingEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = np.array(weights if weights is not None else [1.0] * len(models), dtype=float)
        self.weights = self.weights / self.weights.sum()

    def fit(self, X, y, sample_weight=None):
        for model in self.models:
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            model.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for model in self.models]
        return np.average(np.stack(probs, axis=0), axis=0, weights=self.weights)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for a, b in zip(MAG_COLS[:-1], MAG_COLS[1:]):
        out[f"{a}_{b}_diff"] = out[a] - out[b]

    for a, b in [("u", "r"), ("u", "i"), ("u", "z"), ("g", "i"), ("g", "z"), ("r", "z")]:
        out[f"{a}_{b}_color"] = out[a] - out[b]

    eps = 1e-3
    out["ug_gr_ratio"] = (out["u"] - out["g"]) / (np.abs(out["g"] - out["r"]) + eps)
    out["gr_ri_ratio"] = (out["g"] - out["r"]) / (np.abs(out["r"] - out["i"]) + eps)
    out["ri_iz_ratio"] = (out["r"] - out["i"]) / (np.abs(out["i"] - out["z"]) + eps)
    out["uz_redness"] = out["u"] - out["z"]
    out["gr_sum"] = out["g"] + out["r"]

    out["mag_mean"] = out[MAG_COLS].mean(axis=1)
    out["mag_std"] = out[MAG_COLS].std(axis=1)
    out["mag_min"] = out[MAG_COLS].min(axis=1)
    out["mag_max"] = out[MAG_COLS].max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["brightest_band"] = out[MAG_COLS].idxmin(axis=1)
    out["faintest_band"] = out[MAG_COLS].idxmax(axis=1)

    redshift = out["redshift"].astype(float)
    clipped_redshift = np.clip(redshift, a_min=-0.999999, a_max=None)
    out["redshift_abs"] = np.abs(redshift)
    out["redshift_sq"] = redshift ** 2
    out["redshift_cube"] = redshift ** 3
    out["redshift_sqrt_abs"] = np.sqrt(np.abs(redshift))
    out["redshift_log1p"] = np.log1p(clipped_redshift)
    redshift_bucket = pd.cut(
        redshift,
        bins=[-np.inf, 0.02, 0.15, 0.6, 1.5, 3.0, np.inf],
        labels=["very_low", "low", "mid", "high", "very_high", "extreme"],
    )
    out["redshift_bucket"] = redshift_bucket.astype(object).fillna("missing")
    for band in MAG_COLS:
        out[f"redshift_{band}"] = redshift * out[band]
    out["redshift_mag_mean"] = redshift * out["mag_mean"]
    out["redshift_color_ur"] = redshift * out["u_r_color"]
    out["redshift_color_gz"] = redshift * out["g_z_color"]

    alpha_rad = np.deg2rad(out["alpha"].astype(float))
    delta_rad = np.deg2rad(out["delta"].astype(float))
    out["alpha_sin"] = np.sin(alpha_rad)
    out["alpha_cos"] = np.cos(alpha_rad)
    out["delta_sin"] = np.sin(delta_rad)
    out["delta_cos"] = np.cos(delta_rad)
    out["alpha_delta_interaction"] = out["alpha_sin"] * out["delta_cos"]
    out["sky_quadrant"] = (
        (out["alpha"] // 90).astype(int).astype(str)
        + "_"
        + np.floor((out["delta"] + 90.0) / 45.0).astype(int).astype(str)
    )

    out["spectral_population"] = out["spectral_type"].astype(str) + "__" + out["galaxy_population"].astype(str)
    out["spectral_brightest"] = out["spectral_type"].astype(str) + "__" + out["brightest_band"].astype(str)
    out["population_redshift_bucket"] = out["galaxy_population"].astype(str) + "__" + out["redshift_bucket"].astype(str)
    return out


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_model() -> AveragingEnsemble:
    model_a = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=1.5,
        gamma=0.0,
        max_bin=256,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model_b = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=900,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.95,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=2.0,
        gamma=0.0,
        max_bin=256,
        tree_method="hist",
        random_state=SEED + 1,
        n_jobs=-1,
    )
    model_c = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=700,
        max_depth=10,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=2.5,
        gamma=0.1,
        max_bin=256,
        tree_method="hist",
        random_state=SEED + 2,
        n_jobs=-1,
    )
    return AveragingEnsemble([model_a, model_b, model_c], weights=[0.4, 0.35, 0.25])


def main() -> None:
    data_dir = os.environ.get("VERITY_DATA", "data")
    out_dir = os.environ.get("VERITY_OUT", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(str).to_numpy()
    test_ids = test_df[ID_COL].to_numpy()

    y_idx = np.searchsorted(CLASS_ORDER, y_train)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    train_features = add_features(X_train)
    test_features = add_features(test_df)

    preprocessor = build_preprocessor(train_features)
    X_train_processed = preprocessor.fit_transform(train_features)
    X_test_processed = preprocessor.transform(test_features)

    model = build_model()
    model.fit(X_train_processed, y_idx, sample_weight=sample_weight)

    pred_idx = model.predict(X_test_processed).astype(int)
    preds = CLASS_ORDER[pred_idx]

    pd.DataFrame({ID_COL: test_ids, TARGET: preds}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
