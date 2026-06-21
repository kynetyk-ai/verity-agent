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
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

TARGET = "class"
ID_COL = "id"
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mags = ["u", "g", "r", "i", "z"]

    for a, b in zip(mags[:-1], mags[1:]):
        out[f"{a}_{b}_diff"] = out[a] - out[b]

    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["r_z"] = out["r"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    out["ug_gr_ratio"] = (out["u"] - out["g"]) / (np.abs(out["g"] - out["r"]) + 1e-3)
    out["gr_ri_ratio"] = (out["g"] - out["r"]) / (np.abs(out["r"] - out["i"]) + 1e-3)
    out["ri_iz_ratio"] = (out["r"] - out["i"]) / (np.abs(out["i"] - out["z"]) + 1e-3)

    out["mag_mean"] = out[mags].mean(axis=1)
    out["mag_std"] = out[mags].std(axis=1)
    out["mag_range"] = out[mags].max(axis=1) - out[mags].min(axis=1)
    out["brightest_band"] = out[mags].idxmin(axis=1)
    out["faintest_band"] = out[mags].idxmax(axis=1)

    out["redshift_abs"] = np.abs(out["redshift"])
    out["redshift_sq"] = out["redshift"] ** 2
    out["redshift_cube"] = out["redshift"] ** 3
    out["redshift_log1p"] = np.log1p(np.clip(out["redshift"], a_min=-0.999999, a_max=None))
    out["redshift_u"] = out["redshift"] * out["u"]
    out["redshift_g"] = out["redshift"] * out["g"]
    out["redshift_r"] = out["redshift"] * out["r"]
    out["redshift_i"] = out["redshift"] * out["i"]
    out["redshift_z"] = out["redshift"] * out["z"]

    out["alpha_rad"] = np.deg2rad(out["alpha"])
    out["delta_rad"] = np.deg2rad(out["delta"])
    out["alpha_sin"] = np.sin(out["alpha_rad"])
    out["alpha_cos"] = np.cos(out["alpha_rad"])
    out["delta_sin"] = np.sin(out["delta_rad"])
    out["delta_cos"] = np.cos(out["delta_rad"])
    out["sky_quadrant"] = (
        (out["alpha"] // 90).astype(int).astype(str)
        + "_"
        + (np.floor((out["delta"] + 90) / 45).astype(int).astype(str))
    )

    out["spectral_population"] = out["spectral_type"].astype(str) + "__" + out["galaxy_population"].astype(str)
    return out


def build_pipeline(feature_df: pd.DataFrame) -> Pipeline:
    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=900,
        max_depth=8,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=1.5,
        gamma=0.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )

    return Pipeline([
        ("features", DataFrameFunctionTransformer(add_features)),
        ("preprocessor", preprocessor),
        ("model", model),
    ])


class DataFrameFunctionTransformer:
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)


def main() -> None:
    data_dir = os.environ.get("VERITY_DATA", "data")
    out_dir = os.environ.get("VERITY_OUT", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(str).values
    test_ids = test_df[ID_COL].values

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    y_idx = np.searchsorted(CLASS_ORDER, y_train)

    pipeline = build_pipeline(add_features(X_train))
    pipeline.fit(X_train, y_idx, model__sample_weight=sample_weight)

    pred_idx = pipeline.predict(test_df)
    preds = CLASS_ORDER[pred_idx.astype(int)]

    pd.DataFrame({ID_COL: test_ids, TARGET: preds}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
