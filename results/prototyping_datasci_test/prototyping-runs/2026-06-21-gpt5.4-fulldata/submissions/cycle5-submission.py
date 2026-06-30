import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
TARGET = "class"
ID_COL = "id"
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_ORDER)}
MAG_COLS = ["u", "g", "r", "i", "z"]
COLOR_COLS = [
    "u_g_diff",
    "g_r_diff",
    "r_i_diff",
    "i_z_diff",
    "u_r_color",
    "u_i_color",
    "u_z_color",
    "g_i_color",
    "g_z_color",
    "r_z_color",
]

random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")


class AveragingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        self.fitted_models_ = []
        for model in self.models:
            fitted = clone(model)
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            fitted.fit(X, y, **fit_kwargs)
            self.fitted_models_.append(fitted)
        self.weights_ = np.array(self.weights if self.weights is not None else [1.0] * len(self.models), dtype=float)
        self.weights_ = self.weights_ / self.weights_.sum()
        return self

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for model in self.fitted_models_]
        return np.average(np.stack(probs, axis=0), axis=0, weights=self.weights_)

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
    out["color_sum"] = out[COLOR_COLS].sum(axis=1)
    out["color_std"] = out[COLOR_COLS].std(axis=1)
    out["color_min"] = out[COLOR_COLS].min(axis=1)
    out["color_max"] = out[COLOR_COLS].max(axis=1)
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
    out["redshift_bucket"] = pd.cut(
        redshift,
        bins=[-np.inf, 0.0, 0.02, 0.08, 0.2, 0.6, 1.2, 2.0, 3.0, np.inf],
        labels=["neg", "vlow", "low", "mlow", "mid", "high", "vhigh", "extreme", "ultra"],
    ).astype(object).fillna("missing")
    for band in MAG_COLS:
        out[f"redshift_{band}"] = redshift * out[band]
        out[f"{band}_over_redshift1p"] = out[band] / (1.0 + np.abs(redshift))
    out["redshift_mag_mean"] = redshift * out["mag_mean"]
    out["redshift_color_ur"] = redshift * out["u_r_color"]
    out["redshift_color_gz"] = redshift * out["g_z_color"]
    out["redshift_color_sum"] = redshift * out["color_sum"]

    alpha_rad = np.deg2rad(out["alpha"].astype(float))
    delta_rad = np.deg2rad(out["delta"].astype(float))
    out["alpha_sin"] = np.sin(alpha_rad)
    out["alpha_cos"] = np.cos(alpha_rad)
    out["delta_sin"] = np.sin(delta_rad)
    out["delta_cos"] = np.cos(delta_rad)
    out["alpha_delta_interaction"] = out["alpha_sin"] * out["delta_cos"]
    out["sky_quadrant"] = (
        (out["alpha"] // 60).astype(int).astype(str)
        + "_"
        + np.floor((out["delta"] + 90.0) / 30.0).astype(int).astype(str)
    )

    spectral = out["spectral_type"].astype(str)
    population = out["galaxy_population"].astype(str)
    brightest = out["brightest_band"].astype(str)
    red_bucket = out["redshift_bucket"].astype(str)
    out["spectral_population"] = spectral + "__" + population
    out["spectral_brightest"] = spectral + "__" + brightest
    out["population_redshift_bucket"] = population + "__" + red_bucket
    out["spectral_redshift_bucket"] = spectral + "__" + red_bucket
    out["spectral_sky"] = spectral + "__" + out["sky_quadrant"].astype(str)
    out["population_brightest"] = population + "__" + brightest
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
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_model() -> AveragingEnsemble:
    common = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
    )
    model_a = XGBClassifier(
        **common,
        n_estimators=1800,
        max_depth=0,
        grow_policy="lossguide",
        max_leaves=256,
        learning_rate=0.028,
        subsample=0.9,
        colsample_bytree=0.86,
        colsample_bynode=0.8,
        min_child_weight=2,
        reg_alpha=0.02,
        reg_lambda=1.8,
        gamma=0.0,
        max_bin=256,
        random_state=SEED,
    )
    model_b = XGBClassifier(
        **common,
        n_estimators=1400,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.88,
        colsample_bytree=0.84,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=2.0,
        gamma=0.0,
        max_bin=256,
        random_state=SEED + 1,
    )
    model_c = XGBClassifier(
        **common,
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.95,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.5,
        gamma=0.0,
        max_bin=512,
        random_state=SEED + 2,
    )
    return AveragingEnsemble([model_a, model_b, model_c], weights=[0.45, 0.3, 0.25])


def main() -> None:
    data_dir = os.environ.get("VERITY_DATA", "data")
    out_dir = os.environ.get("VERITY_OUT", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(str).map(CLASS_TO_INDEX).to_numpy()
    test_ids = test_df[ID_COL].to_numpy()

    sample_weight = compute_sample_weight(
        class_weight={CLASS_TO_INDEX[label]: weight for label, weight in zip(CLASS_ORDER, [1.0, 1.25, 1.1])},
        y=y_train,
    )

    train_features = add_features(X_train)
    test_features = add_features(test_df)

    preprocessor = build_preprocessor(train_features)
    X_train_processed = preprocessor.fit_transform(train_features)
    X_test_processed = preprocessor.transform(test_features)

    model = build_model()
    model.fit(X_train_processed, y_train, sample_weight=sample_weight)

    pred_idx = model.predict(X_test_processed).astype(int)
    preds = CLASS_ORDER[pred_idx]

    pd.DataFrame({ID_COL: test_ids, TARGET: preds}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
