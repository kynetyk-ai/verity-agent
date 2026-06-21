import os
import random
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
TARGET = "class"
ID_COL = "id"
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
MAG_COLS = ["u", "g", "r", "i", "z"]
CV_SPLITS = 4

random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")


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
    out["ug_plus_rz"] = (out["u"] - out["g"]) + (out["r"] - out["z"])
    out["ui_minus_gz"] = (out["u"] - out["i"]) - (out["g"] - out["z"])
    out["color_curvature"] = (out["u"] - out["g"]) - (out["g"] - out["r"])
    out["color_slope"] = (out["u"] - out["z"]) / 4.0
    out["ri_over_ug"] = (out["r"] - out["i"]) / (np.abs(out["u"] - out["g"]) + eps)
    out["iz_over_gr"] = (out["i"] - out["z"]) / (np.abs(out["g"] - out["r"]) + eps)
    out["urz_combo"] = (out["u"] - out["r"]) + (out["r"] - out["z"])
    out["gri_combo"] = (out["g"] - out["r"]) + (out["r"] - out["i"])

    out["mag_mean"] = out[MAG_COLS].mean(axis=1)
    out["mag_std"] = out[MAG_COLS].std(axis=1)
    out["mag_min"] = out[MAG_COLS].min(axis=1)
    out["mag_max"] = out[MAG_COLS].max(axis=1)
    out["mag_median"] = out[MAG_COLS].median(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]
    out["brightest_band"] = out[MAG_COLS].idxmin(axis=1)
    out["faintest_band"] = out[MAG_COLS].idxmax(axis=1)
    out["i_minus_z_abs"] = np.abs(out["i"] - out["z"])
    out["u_minus_r_abs"] = np.abs(out["u"] - out["r"])
    out["g_minus_i_abs"] = np.abs(out["g"] - out["i"])

    redshift = out["redshift"].astype(float)
    clipped_redshift = np.clip(redshift, a_min=-0.999999, a_max=None)
    out["redshift_abs"] = np.abs(redshift)
    out["redshift_sq"] = redshift ** 2
    out["redshift_cube"] = redshift ** 3
    out["redshift_sqrt_abs"] = np.sqrt(np.abs(redshift))
    out["redshift_log1p"] = np.log1p(clipped_redshift)
    out["redshift_sign"] = np.sign(redshift)
    out["redshift_is_high"] = (redshift > 1.0).astype(int)
    out["redshift_is_low"] = (redshift < 0.08).astype(int)
    out["redshift_is_very_high"] = (redshift > 2.2).astype(int)
    out["redshift_star_window"] = ((redshift >= -0.01) & (redshift <= 0.15)).astype(int)
    out["redshift_qso_window"] = ((redshift >= 0.8) & (redshift <= 3.5)).astype(int)
    redshift_bucket = pd.cut(
        redshift,
        bins=[-np.inf, 0.01, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2, 2.2, np.inf],
        labels=["ultra_low", "very_low", "low", "starish", "mid_low", "mid", "high", "very_high", "extreme"],
    )
    out["redshift_bucket"] = redshift_bucket.astype(object).fillna("missing")
    for band in MAG_COLS:
        out[f"redshift_{band}"] = redshift * out[band]
    out["redshift_mag_mean"] = redshift * out["mag_mean"]
    out["redshift_mag_range"] = redshift * out["mag_range"]
    out["redshift_color_ur"] = redshift * out["u_r_color"]
    out["redshift_color_gz"] = redshift * out["g_z_color"]
    out["redshift_color_ug"] = redshift * out["u_g_diff"]
    out["redshift_color_ri"] = redshift * out["r_i_diff"]
    out["redshift_over_mag_mean"] = redshift / (np.abs(out["mag_mean"]) + eps)

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
    out["spectral_redshift_bucket"] = out["spectral_type"].astype(str) + "__" + out["redshift_bucket"].astype(str)
    out["population_brightest"] = out["galaxy_population"].astype(str) + "__" + out["brightest_band"].astype(str)
    out["spectral_population_redshift"] = out["spectral_population"].astype(str) + "__" + out["redshift_bucket"].astype(str)
    out["spectral_population_quadrant"] = out["spectral_population"].astype(str) + "__" + out["sky_quadrant"].astype(str)
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_lgbm_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
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
        sparse_threshold=0.0,
    )


def prepare_catboost_frame(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    df = feature_df.copy()
    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
    for col in categorical_cols:
        df[col] = df[col].fillna("missing").astype(str)
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cat_idx = [df.columns.get_loc(c) for c in categorical_cols]
    return df, cat_idx


def build_xgb_models() -> list[XGBClassifier]:
    return [
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=1400,
            max_depth=8,
            learning_rate=0.03,
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
        ),
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=2.5,
            gamma=0.1,
            max_bin=256,
            tree_method="hist",
            random_state=SEED + 1,
            n_jobs=-1,
        ),
    ]


def build_lgbm_models() -> list[LGBMClassifier]:
    return [
        LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=1800,
            learning_rate=0.026,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.2,
            random_state=SEED + 2,
            n_jobs=-1,
            class_weight="balanced",
            verbosity=-1,
        ),
        LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=1300,
            learning_rate=0.036,
            num_leaves=255,
            max_depth=-1,
            min_child_samples=12,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=SEED + 3,
            n_jobs=-1,
            class_weight="balanced",
            verbosity=-1,
        ),
    ]


def build_cat_models() -> list[CatBoostClassifier]:
    return [
        CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=1600,
            depth=8,
            learning_rate=0.035,
            l2_leaf_reg=4.0,
            random_strength=0.8,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            random_seed=SEED + 4,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        ),
        CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=2200,
            depth=10,
            learning_rate=0.028,
            l2_leaf_reg=6.0,
            random_strength=1.2,
            bootstrap_type="Bernoulli",
            subsample=0.9,
            random_seed=SEED + 5,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        ),
    ]


def build_oof_stack(X_xgb, X_lgbm, X_cat, cat_idx, y_idx, sample_weight):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    model_specs = [
        ("xgb", m) for m in build_xgb_models()
    ] + [
        ("lgbm", m) for m in build_lgbm_models()
    ] + [
        ("cat", m) for m in build_cat_models()
    ]
    n_models = len(model_specs)
    n_classes = len(CLASS_ORDER)
    oof = np.zeros((len(y_idx), n_models * n_classes), dtype=np.float32)

    for tr_idx, va_idx in cv.split(X_xgb, y_idx):
        fold_weights = sample_weight[tr_idx]
        for model_idx, (kind, model) in enumerate(model_specs):
            if kind == "xgb":
                model.fit(X_xgb[tr_idx], y_idx[tr_idx], sample_weight=fold_weights)
                proba = model.predict_proba(X_xgb[va_idx]).astype(np.float32)
            elif kind == "lgbm":
                model.fit(X_lgbm[tr_idx], y_idx[tr_idx], sample_weight=fold_weights)
                proba = model.predict_proba(X_lgbm[va_idx]).astype(np.float32)
            else:
                model.fit(X_cat.iloc[tr_idx], y_idx[tr_idx], sample_weight=fold_weights, cat_features=cat_idx)
                proba = model.predict_proba(X_cat.iloc[va_idx]).astype(np.float32)
            start = model_idx * n_classes
            oof[va_idx, start : start + n_classes] = proba
    return oof, model_specs


def fit_predict_test(model_specs, X_xgb, X_lgbm, X_cat, cat_idx, y_idx, sample_weight, X_test_xgb, X_test_lgbm, X_test_cat):
    test_probas = []
    for kind, model in model_specs:
        if kind == "xgb":
            model.fit(X_xgb, y_idx, sample_weight=sample_weight)
            proba = model.predict_proba(X_test_xgb).astype(np.float32)
        elif kind == "lgbm":
            model.fit(X_lgbm, y_idx, sample_weight=sample_weight)
            proba = model.predict_proba(X_test_lgbm).astype(np.float32)
        else:
            model.fit(X_cat, y_idx, sample_weight=sample_weight, cat_features=cat_idx)
            proba = model.predict_proba(X_test_cat).astype(np.float32)
        test_probas.append(proba)
    return np.stack(test_probas, axis=1)


def tune_class_scaling(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    best_scale = np.ones(proba.shape[1], dtype=np.float32)
    best_score = balanced_accuracy_score(y_idx, np.argmax(proba, axis=1))
    grid = np.array([0.84, 0.9, 0.96, 1.0, 1.04, 1.1, 1.16], dtype=np.float32)

    improved = True
    while improved:
        improved = False
        for class_idx in range(proba.shape[1]):
            current_best = best_scale[class_idx]
            for val in grid:
                trial_scale = best_scale.copy()
                trial_scale[class_idx] = val
                pred = np.argmax(proba * trial_scale, axis=1)
                score = balanced_accuracy_score(y_idx, pred)
                if score > best_score + 1e-7:
                    best_score = score
                    current_best = val
                    best_scale = trial_scale
                    improved = True
            best_scale[class_idx] = current_best
    return best_scale


def tune_blend(meta_oof: np.ndarray, base_oof: np.ndarray, y_idx: np.ndarray) -> float:
    best_alpha = 0.8
    best_score = -1.0
    for alpha in [0.55, 0.65, 0.75, 0.8, 0.85, 0.9]:
        blend = alpha * meta_oof + (1.0 - alpha) * base_oof
        score = balanced_accuracy_score(y_idx, np.argmax(blend, axis=1))
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


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
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float32)

    train_features = add_features(X_train)
    test_features = add_features(test_df)

    xgb_preprocessor = build_xgb_preprocessor(train_features)
    lgbm_preprocessor = build_lgbm_preprocessor(train_features)

    X_train_xgb = xgb_preprocessor.fit_transform(train_features)
    X_test_xgb = xgb_preprocessor.transform(test_features)
    X_train_lgbm = lgbm_preprocessor.fit_transform(train_features)
    X_test_lgbm = lgbm_preprocessor.transform(test_features)
    if hasattr(X_train_xgb, "astype"):
        X_train_xgb = X_train_xgb.astype(np.float32)
        X_test_xgb = X_test_xgb.astype(np.float32)
    X_train_lgbm = np.asarray(X_train_lgbm, dtype=np.float32)
    X_test_lgbm = np.asarray(X_test_lgbm, dtype=np.float32)

    X_train_cat, cat_idx = prepare_catboost_frame(train_features)
    X_test_cat, _ = prepare_catboost_frame(test_features)

    oof_stack, model_specs = build_oof_stack(
        X_train_xgb,
        X_train_lgbm,
        X_train_cat,
        cat_idx,
        y_idx,
        sample_weight,
    )

    meta_model = LogisticRegression(
        max_iter=600,
        multi_class="multinomial",
        class_weight="balanced",
        C=1.8,
        n_jobs=-1,
        random_state=SEED,
    )
    meta_model.fit(oof_stack, y_idx)

    meta_oof = meta_model.predict_proba(oof_stack).astype(np.float32)
    base_oof = oof_stack.reshape(oof_stack.shape[0], len(model_specs), len(CLASS_ORDER)).mean(axis=1)
    blend_alpha = tune_blend(meta_oof, base_oof, y_idx)
    blended_oof = blend_alpha * meta_oof + (1.0 - blend_alpha) * base_oof
    class_scale = tune_class_scaling(blended_oof, y_idx)

    test_model_probas = fit_predict_test(
        model_specs,
        X_train_xgb,
        X_train_lgbm,
        X_train_cat,
        cat_idx,
        y_idx,
        sample_weight,
        X_test_xgb,
        X_test_lgbm,
        X_test_cat,
    )
    test_stack = test_model_probas.reshape(test_model_probas.shape[0], -1)
    meta_proba = meta_model.predict_proba(test_stack).astype(np.float32)
    base_avg = np.mean(test_model_probas, axis=1)
    proba = blend_alpha * meta_proba + (1.0 - blend_alpha) * base_avg
    proba = proba * class_scale

    pred_idx = np.argmax(proba, axis=1).astype(int)
    preds = CLASS_ORDER[pred_idx]

    pd.DataFrame({ID_COL: test_ids, TARGET: preds}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
