import os
import random
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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

random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mags = ["u", "g", "r", "i", "z"]

    for a, b in zip(mags[:-1], mags[1:]):
        out[f"{a}_{b}_diff"] = out[a] - out[b]

    for a, b in [("u", "r"), ("u", "z"), ("g", "i"), ("r", "z"), ("g", "z"), ("u", "i")]:
        out[f"{a}_{b}_color"] = out[a] - out[b]

    out["ug_gr_ratio"] = (out["u"] - out["g"]) / (np.abs(out["g"] - out["r"]) + 1e-3)
    out["gr_ri_ratio"] = (out["g"] - out["r"]) / (np.abs(out["r"] - out["i"]) + 1e-3)
    out["ri_iz_ratio"] = (out["r"] - out["i"]) / (np.abs(out["i"] - out["z"]) + 1e-3)
    out["ug_plus_gr"] = (out["u"] - out["g"]) + (out["g"] - out["r"])
    out["gr_plus_ri"] = (out["g"] - out["r"]) + (out["r"] - out["i"])
    out["ri_plus_iz"] = (out["r"] - out["i"]) + (out["i"] - out["z"])

    out["mag_mean"] = out[mags].mean(axis=1)
    out["mag_std"] = out[mags].std(axis=1)
    out["mag_min"] = out[mags].min(axis=1)
    out["mag_max"] = out[mags].max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["brightest_band"] = out[mags].idxmin(axis=1)
    out["faintest_band"] = out[mags].idxmax(axis=1)

    redshift = out["redshift"].astype(float)
    out["redshift_abs"] = np.abs(redshift)
    out["redshift_sq"] = redshift ** 2
    out["redshift_cube"] = redshift ** 3
    out["redshift_sqrt_abs"] = np.sqrt(np.abs(redshift))
    out["redshift_log1p"] = np.log1p(np.clip(redshift, a_min=-0.999999, a_max=None))
    out["redshift_sign"] = np.sign(redshift)
    for band in mags:
        out[f"redshift_{band}"] = redshift * out[band]
    out["redshift_mag_mean"] = redshift * out["mag_mean"]
    out["redshift_color_ur"] = redshift * out["u_r_color"]

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
    out["alpha_bin"] = pd.cut(out["alpha"], bins=24, labels=False, include_lowest=True).astype(str)
    out["delta_bin"] = pd.cut(out["delta"], bins=18, labels=False, include_lowest=True).astype(str)

    out["spectral_population"] = out["spectral_type"].astype(str) + "__" + out["galaxy_population"].astype(str)
    return out


def build_xgb_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.0005)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def prepare_catboost_frames(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_cb = train_df.copy()
    test_cb = test_df.copy()
    categorical_cols = [c for c in train_cb.columns if train_cb[c].dtype == "object"]
    numeric_cols = [c for c in train_cb.columns if c not in categorical_cols]

    for col in numeric_cols:
        median = pd.to_numeric(train_cb[col], errors="coerce").median()
        train_cb[col] = pd.to_numeric(train_cb[col], errors="coerce").fillna(median)
        test_cb[col] = pd.to_numeric(test_cb[col], errors="coerce").fillna(median)

    for col in categorical_cols:
        mode = train_cb[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "missing"
        train_cb[col] = train_cb[col].astype(str).fillna(fill)
        test_cb[col] = test_cb[col].astype(str).fillna(fill)

    cat_idx = [train_cb.columns.get_loc(c) for c in categorical_cols]
    return train_cb, test_cb, cat_idx


def main() -> None:
    data_dir = os.environ.get("VERITY_DATA", "data")
    out_dir = os.environ.get("VERITY_OUT", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(str).to_numpy()
    y_idx = np.searchsorted(CLASS_ORDER, y_train)
    test_ids = test_df[ID_COL].to_numpy()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    train_features = add_features(X_train)
    test_features = add_features(test_df)

    xgb_preprocessor = build_xgb_preprocessor(train_features)
    X_train_xgb = xgb_preprocessor.fit_transform(train_features)
    X_test_xgb = xgb_preprocessor.transform(test_features)

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=900,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.95,
        colsample_bynode=0.8,
        colsample_bytree=0.95,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.0,
        tree_method="hist",
        random_state=SEED + 2,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_xgb, y_idx, sample_weight=sample_weight)
    xgb_probs = xgb_model.predict_proba(X_test_xgb)

    train_cb, test_cb, cat_idx = prepare_catboost_frames(train_features, test_features)
    train_pool = Pool(train_cb, y_idx, cat_features=cat_idx, weight=sample_weight)
    test_pool = Pool(test_cb, cat_features=cat_idx)

    cat_model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=1400,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=SEED,
        thread_count=-1,
        verbose=False,
        auto_class_weights="Balanced",
    )
    cat_model.fit(train_pool)
    cat_probs = cat_model.predict_proba(test_pool)

    probs = 0.5 * xgb_probs + 0.5 * cat_probs
    pred_idx = probs.argmax(axis=1).astype(int)
    preds = CLASS_ORDER[pred_idx]

    pd.DataFrame({ID_COL: test_ids, TARGET: preds}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
