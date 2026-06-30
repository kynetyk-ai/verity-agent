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
Z_BINS = np.array([-10.0, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20,
                   0.30, 0.50, 0.80, 1.20, 1.60, 2.00, 2.60, 3.50, 10.0], dtype="float64")


def _safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace(".", "p").replace("-", "m")


def prepare_target_encoders(train, classes):
    """Smoothed class-prior features for coarse, stable astro groups."""
    df = train[["spectral_type", "galaxy_population", "redshift"]].copy()
    df["combo"] = df["spectral_type"].astype(str) + "__" + df["galaxy_population"].astype(str)
    df["zbin"] = pd.cut(df["redshift"], bins=Z_BINS, labels=False, include_lowest=True).fillna(-1).astype("int16")
    df["spec_z"] = df["spectral_type"].astype(str) + "__" + df["zbin"].astype(str)
    df["pop_z"] = df["galaxy_population"].astype(str) + "__" + df["zbin"].astype(str)
    df["combo_z"] = df["combo"].astype(str) + "__" + df["zbin"].astype(str)
    y = train["class"].astype(str).values
    global_p = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(0.0).values.astype("float64")
    enc = {"classes": list(classes), "global": global_p, "maps": {}}
    smooth = 80.0
    for col in ["spectral_type", "galaxy_population", "combo", "zbin", "spec_z", "pop_z", "combo_z"]:
        tmp = pd.DataFrame({"key": df[col].astype(str).values, "class": y})
        counts = pd.crosstab(tmp["key"], tmp["class"]).reindex(columns=classes, fill_value=0).astype("float64")
        n = counts.sum(axis=1).values[:, None]
        probs = (counts.values + smooth * global_p[None, :]) / (n + smooth)
        enc["maps"][col] = {k: probs[i].astype("float32") for i, k in enumerate(counts.index.astype(str))}
    return enc


def build_features(df, cat_values, encoders=None):
    X = pd.DataFrame(index=df.index)
    for c in ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]:
        X[c] = df[c].astype("float32")

    # All colour indices; photometric colours plus redshift carry most of the signal.
    for ia, a in enumerate(BANDS):
        for b in BANDS[ia + 1:]:
            X[f"{a}_{b}"] = (df[a] - df[b]).astype("float32")

    X["ug_gr"] = (X["u_g"] - X["g_r"]).astype("float32")
    X["gr_ri"] = (X["g_r"] - X["r_i"]).astype("float32")
    X["ri_iz"] = (X["r_i"] - X["i_z"]).astype("float32")
    X["blue_red"] = ((df["u"] + df["g"] - df["i"] - df["z"]) / 2.0).astype("float32")

    mags = df[BANDS]
    X["mag_mean"] = mags.mean(axis=1).astype("float32")
    X["mag_std"] = mags.std(axis=1).astype("float32")
    X["mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    X["mag_min"] = mags.min(axis=1).astype("float32")
    X["mag_max"] = mags.max(axis=1).astype("float32")
    # Rare bad/extreme magnitudes are informative outlier regimes; keep flags explicit.
    for b in BANDS:
        X[f"{b}_lt13"] = (df[b].values < 13.0).astype("int8")
        X[f"{b}_gt26"] = (df[b].values > 26.0).astype("int8")

    z = df["redshift"].astype("float64")
    zpos = np.clip(z.values, 0, None)
    X["redshift_log1p"] = np.log1p(zpos).astype("float32")
    X["redshift_sq"] = (z ** 2).astype("float32")
    X["redshift_sqrt"] = np.sqrt(zpos).astype("float32")
    X["redshift_inv1p"] = (1.0 / (1.0 + zpos)).astype("float32")
    X["redshift_neg"] = (z.values < 0).astype("int8")
    X["zbin"] = pd.cut(z, bins=Z_BINS, labels=False, include_lowest=True).fillna(-1).astype("int16").values

    for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "r_z", "mag_mean", "mag_range", "blue_red"]:
        X[f"{c}_x_z"] = (X[c].values * z.values).astype("float32")
        X[f"{c}_div_1pz"] = (X[c].values / (1.0 + zpos)).astype("float32")

    # Sky coordinates: raw and cyclic encodings.
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
            sv = _safe_name(v)
            X[c + "_" + sv] = oh
            X[f"{c}_{sv}_z"] = (oh.values * z.values).astype("float32")

    combo = df["spectral_type"].astype(str) + "__" + df["galaxy_population"].astype(str)
    vals = cat_values["combo"]
    mp = {v: i for i, v in enumerate(vals)}
    X["combo_code"] = combo.map(mp).fillna(-1).astype("int16")
    for v in vals:
        X["combo_" + _safe_name(v)] = (combo == v).astype("int8")

    if encoders is not None:
        zbin = pd.cut(df["redshift"], bins=Z_BINS, labels=False, include_lowest=True).fillna(-1).astype("int16").astype(str)
        groups = {
            "spectral_type": df["spectral_type"].astype(str),
            "galaxy_population": df["galaxy_population"].astype(str),
            "combo": combo.astype(str),
            "zbin": zbin,
            "spec_z": df["spectral_type"].astype(str) + "__" + zbin,
            "pop_z": df["galaxy_population"].astype(str) + "__" + zbin,
            "combo_z": combo.astype(str) + "__" + zbin,
        }
        classes = encoders["classes"]
        default = encoders["global"].astype("float32")
        for gname, keys in groups.items():
            mp = encoders["maps"][gname]
            arr = np.vstack([mp.get(str(k), default) for k in keys.values]).astype("float32")
            for j, cls in enumerate(classes):
                X[f"te_{gname}_{_safe_name(cls)}"] = arr[:, j]
            # compact contrast features; useful for separating STAR from low-z GALAXY.
            if len(classes) == 3:
                X[f"te_{gname}_qso_minus_gal"] = (arr[:, 1] - arr[:, 0]).astype("float32")
                X[f"te_{gname}_star_minus_gal"] = (arr[:, 2] - arr[:, 0]).astype("float32")

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
    encoders = prepare_target_encoders(train, classes)

    Xtr = build_features(train, cat_values, encoders)
    Xte = build_features(test, cat_values, encoders)
    Xte = Xte[Xtr.columns]

    sw = compute_sample_weight("balanced", y)
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
    cw_map = {i: float(w) for i, w in enumerate(cw)}
    probas = []
    model_names = []

    # LightGBM: strong and fast on the full tabular set.
    try:
        from lightgbm import LGBMClassifier
        lgb_cfgs = [
            dict(n_estimators=1200, learning_rate=0.040, num_leaves=96, max_depth=-1,
                 min_child_samples=32, subsample=0.90, colsample_bytree=0.88,
                 reg_lambda=2.8, reg_alpha=0.03, random_state=101),
            dict(n_estimators=1000, learning_rate=0.048, num_leaves=72, max_depth=-1,
                 min_child_samples=48, subsample=0.92, colsample_bytree=0.92,
                 reg_lambda=2.0, reg_alpha=0.0, random_state=303),
        ]
        for i, cfg in enumerate(lgb_cfgs):
            m = LGBMClassifier(objective="multiclass", class_weight=cw_map, n_jobs=-1,
                               verbosity=-1, force_col_wise=True, **cfg)
            m.fit(Xtr, y)
            add_proba(probas, m, Xte)
            model_names.append(f"lgb{i+1}")
            del m
    except Exception as e:
        print("lightgbm unavailable:", e)

    # XGBoost: complementary boosted-tree views; sample weights target balanced accuracy.
    try:
        import xgboost as xgb
        xgb_cfgs = [
            dict(n_estimators=850, max_depth=6, learning_rate=0.047, subsample=0.88,
                 colsample_bytree=0.86, min_child_weight=10, reg_lambda=3.0,
                 reg_alpha=0.08, random_state=13),
            dict(n_estimators=820, max_depth=7, learning_rate=0.043, subsample=0.90,
                 colsample_bytree=0.84, min_child_weight=5, reg_lambda=2.5,
                 reg_alpha=0.02, random_state=33),
            # Loss-guided trees gave the best minority-class balanced validation among single XGBs.
            dict(n_estimators=1100, max_depth=0, grow_policy="lossguide", max_leaves=96,
                 learning_rate=0.045, subsample=0.90, colsample_bytree=0.86,
                 min_child_weight=4, reg_lambda=3.0, reg_alpha=0.05, random_state=73),
        ]
        for i, cfg in enumerate(xgb_cfgs):
            m = xgb.XGBClassifier(objective="multi:softprob", num_class=len(classes),
                                  tree_method="hist", eval_metric="mlogloss",
                                  n_jobs=-1, **cfg)
            m.fit(Xtr, y, sample_weight=sw)
            add_proba(probas, m, Xte)
            model_names.append(f"xgb{i+1}")
            del m
    except Exception as e:
        print("xgboost unavailable:", e)

    # Small diversity model. It is slower per tree but stable and has a different bias.
    hgb_cfgs = [
        dict(max_iter=250, learning_rate=0.052, max_leaf_nodes=63,
             l2_regularization=0.02, min_samples_leaf=25, random_state=11),
    ]
    for cfg in hgb_cfgs:
        m = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                           max_bins=255, **cfg)
        m.fit(Xtr, y)
        add_proba(probas, m, Xte)
        model_names.append("hgb")
        del m

    if not probas:
        raise RuntimeError("No models trained")

    # Validation favored a stronger minority-class decision shift than the incumbent.
    base_weights = {
        "lgb1": 1.20, "lgb2": 1.08,
        "xgb1": 1.12, "xgb2": 1.12, "xgb3": 1.00,
        "hgb": 0.38,
    }
    weights = np.array([base_weights.get(n, 1.0) for n in model_names], dtype="float64")
    avg = np.average(np.stack(probas, axis=0), axis=0, weights=weights)

    # Decision multipliers tuned on stratified holdout for BALANCED accuracy.
    mult_map = {"GALAXY": 0.82, "QSO": 1.00, "STAR": 0.89}
    mult = np.array([mult_map.get(c, 1.0) for c in classes], dtype="float64")
    pred_idx = (avg * mult).argmax(axis=1)
    pred = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"id": test["id"].values, "class": pred})
    out.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    print("models", model_names)
    print("wrote", len(out), "rows")
    print(out["class"].value_counts())


if __name__ == "__main__":
    main()
