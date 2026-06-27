import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 2024
np.random.seed(SEED)

DATA = os.environ.get("VERITY_DATA", "data")
OUT = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT, exist_ok=True)

BANDS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]


def build_features(df, cat_values):
    X = pd.DataFrame(index=df.index)
    for c in ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]:
        X[c] = df[c].astype("float32")

    # Colour indices are the dominant photometric signal.
    for ia, a in enumerate(BANDS):
        for b in BANDS[ia + 1:]:
            X[f"{a}_{b}"] = (df[a] - df[b]).astype("float32")

    # A few low-noise curvature/summary colour features.
    X["ug_gr"] = (X["u_g"] - X["g_r"]).astype("float32")
    X["gr_ri"] = (X["g_r"] - X["r_i"]).astype("float32")
    X["ri_iz"] = (X["r_i"] - X["i_z"]).astype("float32")
    X["blue_red"] = ((df["u"] + df["g"] - df["i"] - df["z"]) / 2.0).astype("float32")

    mags = df[BANDS]
    X["mag_mean"] = mags.mean(axis=1).astype("float32")
    X["mag_std"] = mags.std(axis=1).astype("float32")
    X["mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")

    z = df["redshift"].astype("float64")
    zpos = np.clip(z.values, 0, None)
    X["redshift_log1p"] = np.log1p(zpos).astype("float32")
    X["redshift_sq"] = (z ** 2).astype("float32")
    X["redshift_sqrt"] = np.sqrt(zpos).astype("float32")
    X["redshift_inv1p"] = (1.0 / (1.0 + zpos)).astype("float32")
    X["redshift_neg"] = (z.values < 0).astype("int8")

    # Interactions with redshift help distinguish blue galaxies, QSOs, and stars.
    for c in ["u_g", "g_r", "r_i", "i_z", "mag_mean", "mag_range", "blue_red"]:
        X[f"{c}_x_z"] = (X[c].values * z.values).astype("float32")
        X[f"{c}_div_1pz"] = (X[c].values / (1.0 + zpos)).astype("float32")

    # Sky coordinates: keep raw and cyclic forms.
    X["alpha_sin"] = np.sin(np.deg2rad(df["alpha"])).astype("float32")
    X["alpha_cos"] = np.cos(np.deg2rad(df["alpha"])).astype("float32")
    X["delta_sin"] = np.sin(np.deg2rad(df["delta"])).astype("float32")
    X["delta_cos"] = np.cos(np.deg2rad(df["delta"])).astype("float32")
    X["abs_delta"] = np.abs(df["delta"]).astype("float32")

    for c in CAT_COLS:
        vals = cat_values[c]
        s = df[c].astype(str)
        mp = {v: i for i, v in enumerate(vals)}
        code = s.map(mp).fillna(-1).astype("int16")
        X[c + "_code"] = code
        for v in vals:
            oh = (s == v).astype("int8")
            X[c + "_" + v] = oh
            X[f"{c}_{v}_z"] = (oh.values * z.values).astype("float32")

    combo = df["spectral_type"].astype(str) + "__" + df["galaxy_population"].astype(str)
    vals = cat_values["combo"]
    mp = {v: i for i, v in enumerate(vals)}
    X["combo_code"] = combo.map(mp).fillna(-1).astype("int16")
    for v in vals:
        X["combo_" + v] = (combo == v).astype("int8")

    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_proba(probas, model, X):
    p = model.predict_proba(X)
    probas.append(np.asarray(p, dtype="float32"))


def main():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))

    cat_values = {
        c: sorted(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
        for c in CAT_COLS
    }
    cat_values["combo"] = sorted(
        pd.concat([
            train["spectral_type"].astype(str) + "__" + train["galaxy_population"].astype(str),
            test["spectral_type"].astype(str) + "__" + test["galaxy_population"].astype(str),
        ]).unique()
    )

    le = LabelEncoder()
    y = le.fit_transform(train["class"].astype(str))
    classes = list(le.classes_)

    Xtr = build_features(train, cat_values)
    Xte = build_features(test, cat_values)
    Xte = Xte[Xtr.columns]

    sw = compute_sample_weight("balanced", y)
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
    cw_map = {i: float(w) for i, w in enumerate(cw)}
    probas = []

    # LightGBM gives a useful independent boosted-tree view and is fast on this tabular data.
    try:
        from lightgbm import LGBMClassifier
        lgb_cfgs = [
            dict(n_estimators=1050, learning_rate=0.045, num_leaves=96, max_depth=-1,
                 min_child_samples=35, subsample=0.88, colsample_bytree=0.86,
                 reg_lambda=3.0, reg_alpha=0.05, random_state=101),
            dict(n_estimators=850, learning_rate=0.055, num_leaves=64, max_depth=-1,
                 min_child_samples=55, subsample=0.92, colsample_bytree=0.92,
                 reg_lambda=2.0, reg_alpha=0.0, random_state=303),
        ]
        for cfg in lgb_cfgs:
            m = LGBMClassifier(objective="multiclass", class_weight=cw_map, n_jobs=-1,
                               verbosity=-1, force_col_wise=True, **cfg)
            m.fit(Xtr, y)
            add_proba(probas, m, Xte)
            del m
    except Exception as e:
        print("lightgbm unavailable:", e)

    # XGBoost models: keep the strongest incumbent family, with complementary depths.
    try:
        import xgboost as xgb
        xgb_cfgs = [
            dict(n_estimators=800, max_depth=6, learning_rate=0.050, subsample=0.88,
                 colsample_bytree=0.86, min_child_weight=10, reg_lambda=3.0,
                 reg_alpha=0.08, random_state=13),
            dict(n_estimators=760, max_depth=7, learning_rate=0.045, subsample=0.90,
                 colsample_bytree=0.84, min_child_weight=5, reg_lambda=2.5,
                 reg_alpha=0.02, random_state=33),
        ]
        for cfg in xgb_cfgs:
            m = xgb.XGBClassifier(objective="multi:softprob", num_class=len(classes),
                                  tree_method="hist", eval_metric="mlogloss",
                                  n_jobs=-1, **cfg)
            m.fit(Xtr, y, sample_weight=sw)
            add_proba(probas, m, Xte)
            del m
    except Exception as e:
        print("xgboost unavailable:", e)

    # Robust sklearn fallback / diversity model, retained from the incumbent family.
    hgb_cfgs = [
        dict(max_iter=230, learning_rate=0.055, max_leaf_nodes=63,
             l2_regularization=0.02, min_samples_leaf=25, random_state=11),
    ]
    for cfg in hgb_cfgs:
        m = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                           max_bins=255, **cfg)
        m.fit(Xtr, y)
        add_proba(probas, m, Xte)
        del m

    if not probas:
        raise RuntimeError("No models trained")

    # Weighted blend: emphasize LightGBM/XGBoost, keep HGB as a small stabilizer.
    if len(probas) >= 5:
        weights = np.array([1.15, 1.05, 1.10, 1.10, 0.45], dtype="float64")[:len(probas)]
    else:
        weights = np.ones(len(probas), dtype="float64")
    avg = np.average(np.stack(probas, axis=0), axis=0, weights=weights)

    # Tuned toward balanced accuracy: reduce majority GALAXY prior and recover QSO/STAR recall.
    mult_map = {"GALAXY": 0.91, "QSO": 1.08, "STAR": 1.10}
    mult = np.array([mult_map.get(c, 1.0) for c in classes], dtype="float64")
    pred_idx = (avg * mult).argmax(axis=1)
    pred = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"id": test["id"].values, "class": pred})
    out.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    print("wrote", len(out), "rows")
    print(out["class"].value_counts())


if __name__ == "__main__":
    main()
