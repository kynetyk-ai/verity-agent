import os
import warnings

_THREADS = str(min(16, max(1, os.cpu_count() or 1)))
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _THREADS)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

SEED = 2026
np.random.seed(SEED)

DATA = os.environ.get("VERITY_DATA", "data")
OUT = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT, exist_ok=True)

BANDS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]
NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
Z_BINS = np.array([
    -10.0, 0.0, 0.015, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20,
    0.30, 0.50, 0.80, 1.20, 1.60, 2.00, 2.60, 3.50, 10.0
], dtype="float64")
COLOR_Z_BINS = [-10.0, 0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.80, 2.60, 10.0]


def _safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace(".", "p").replace("-", "m")


def _read_csv(path, has_target):
    dtype = {c: "float32" for c in NUM_COLS}
    dtype["id"] = "int64"
    dtype["spectral_type"] = "category"
    dtype["galaxy_population"] = "category"
    if has_target:
        dtype["class"] = "category"
    return pd.read_csv(path, dtype=dtype)


def _zbin_series(redshift):
    return pd.cut(redshift.astype("float64"), bins=Z_BINS, labels=False, include_lowest=True).fillna(-1).astype("int16")


def _rank_bins(s, q):
    # Deterministic approximate quantile bins.  Used only for low-cardinality class-rate groups;
    # computing per frame avoids storing fitted bin edges and remained robust in validation.
    return pd.qcut(s.astype("float64").rank(method="first"), q=q, labels=False, duplicates="drop").astype("int16").astype(str)


def _group_keys(df):
    spec = df["spectral_type"].astype(str)
    pop = df["galaxy_population"].astype(str)
    combo = spec + "__" + pop
    z = df["redshift"].astype("float64")
    zbin = _zbin_series(df["redshift"]).astype(str)
    keys = {
        "spectral_type": spec,
        "galaxy_population": pop,
        "combo": combo,
        "zbin": zbin,
        "spec_z": spec + "__" + zbin,
        "pop_z": pop + "__" + zbin,
        "combo_z": combo + "__" + zbin,
    }

    # Sky-local class-rate priors.  Position was a high-importance incumbent feature; these
    # smoothed encodings capture survey-region selection effects without using any test labels.
    alpha = df["alpha"].astype("float64")
    delta = df["delta"].astype("float64")
    ab10 = np.floor(alpha / 10.0).astype("int16").astype(str)
    db5 = np.floor((delta + 20.0) / 5.0).astype("int16").astype(str)
    ab20 = np.floor(alpha / 20.0).astype("int16").astype(str)
    db10 = np.floor((delta + 20.0) / 10.0).astype("int16").astype(str)
    ab5 = np.floor(alpha / 5.0).astype("int16").astype(str)
    db4 = np.floor((delta + 20.0) / 4.0).astype("int16").astype(str)
    keys["sky_10x5"] = ab10 + "_" + db5
    keys["spec_sky_10x5"] = spec + "__" + ab10 + "_" + db5
    keys["pop_sky_10x5"] = pop + "__" + ab10 + "_" + db5
    keys["sky_20x10"] = ab20 + "_" + db10
    keys["combo_sky_20x10"] = combo + "__" + ab20 + "_" + db10
    keys["sky_5x4"] = ab5 + "_" + db4

    # Coarse colour/redshift class-rate groups target the dominant low-z GALAXY/STAR and
    # mid-z GALAXY/QSO confusions under balanced accuracy.
    ug = df["u"].astype("float64") - df["g"].astype("float64")
    gr = df["g"].astype("float64") - df["r"].astype("float64")
    ri = df["r"].astype("float64") - df["i"].astype("float64")
    zc = pd.cut(z, bins=COLOR_Z_BINS, labels=False, include_lowest=True).fillna(-1).astype("int16").astype(str)
    ugc = _rank_bins(ug, 16)
    grc = _rank_bins(gr, 16)
    ric = _rank_bins(ri, 12)
    keys["z_ug"] = zc + "_" + ugc
    keys["z_gr"] = zc + "_" + grc
    keys["z_ri"] = zc + "_" + ric
    keys["spec_z_ug"] = spec + "__" + zc + "_" + ugc
    keys["pop_z_gr"] = pop + "__" + zc + "_" + grc
    return keys


def prepare_target_encoders(train, classes, smooth=85.0):
    y = train["class"].astype(str)
    global_counts = y.value_counts().reindex(classes).fillna(0.0).astype("float64").values
    global_p = global_counts / max(1.0, float(global_counts.sum()))
    enc = {"classes": list(classes), "global": global_p.astype("float32"), "maps": {}}
    keys = _group_keys(train)
    for name, key in keys.items():
        counts = pd.crosstab(key, y).reindex(columns=classes, fill_value=0).astype("float64")
        n = counts.sum(axis=1).values[:, None]
        probs = (counts.values + smooth * global_p[None, :]) / (n + smooth)
        enc["maps"][name] = pd.DataFrame(probs.astype("float32"), index=counts.index.astype(str), columns=classes)
    return enc


def make_cat_values(train, test):
    cat_values = {}
    for c in CAT_COLS:
        cat_values[c] = sorted(pd.concat([train[c], test[c]], ignore_index=True).astype(str).unique())
    tr_combo = train["spectral_type"].astype(str) + "__" + train["galaxy_population"].astype(str)
    te_combo = test["spectral_type"].astype(str) + "__" + test["galaxy_population"].astype(str)
    cat_values["combo"] = sorted(pd.concat([tr_combo, te_combo], ignore_index=True).astype(str).unique())
    return cat_values


def _add_target_encoding_features(feats, df, encoders):
    classes = encoders["classes"]
    default = encoders["global"].astype("float32")
    n = len(df)
    keys = _group_keys(df)
    for gname, key in keys.items():
        table = encoders["maps"][gname]
        idx = table.index.get_indexer(key.astype(str))
        arr = np.empty((n, len(classes)), dtype="float32")
        arr[:] = default
        ok = idx >= 0
        if ok.any():
            arr[ok] = table.values[idx[ok]]
        for j, cls in enumerate(classes):
            feats[f"te_{gname}_{_safe_name(cls)}"] = arr[:, j]
        if len(classes) == 3:
            feats[f"te_{gname}_qso_minus_gal"] = (arr[:, 1] - arr[:, 0]).astype("float32")
            feats[f"te_{gname}_star_minus_gal"] = (arr[:, 2] - arr[:, 0]).astype("float32")


def build_features(df, cat_values, encoders=None):
    feats = {}
    vals = {b: df[b].astype("float32").to_numpy(copy=False) for b in BANDS}
    z = df["redshift"].astype("float32").to_numpy(copy=False)
    z64 = z.astype("float64", copy=False)
    zpos = np.clip(z64, 0.0, None)

    for c in NUM_COLS:
        feats[c] = df[c].astype("float32").to_numpy(copy=False)

    for ia, a in enumerate(BANDS):
        for b in BANDS[ia + 1:]:
            feats[f"{a}_{b}"] = (vals[a] - vals[b]).astype("float32")

    feats["ug_gr"] = (feats["u_g"] - feats["g_r"]).astype("float32")
    feats["gr_ri"] = (feats["g_r"] - feats["r_i"]).astype("float32")
    feats["ri_iz"] = (feats["r_i"] - feats["i_z"]).astype("float32")
    feats["ug_ri"] = (feats["u_g"] - feats["r_i"]).astype("float32")
    feats["gr_iz"] = (feats["g_r"] - feats["i_z"]).astype("float32")
    feats["blue_red"] = ((vals["u"] + vals["g"] - vals["i"] - vals["z"]) * 0.5).astype("float32")
    feats["riz_curve"] = (feats["r_i"] + feats["i_z"] - feats["g_r"]).astype("float32")
    feats["ug_ratio"] = (feats["u_g"] / (np.abs(vals["g"]) + 1e-3)).astype("float32")
    feats["gr_ratio"] = (feats["g_r"] / (np.abs(vals["r"]) + 1e-3)).astype("float32")
    feats["ri_ratio"] = (feats["r_i"] / (np.abs(vals["i"]) + 1e-3)).astype("float32")
    feats["iz_ratio"] = (feats["i_z"] / (np.abs(vals["z"]) + 1e-3)).astype("float32")

    mag = np.vstack([vals[b] for b in BANDS]).T
    feats["mag_mean"] = mag.mean(axis=1).astype("float32")
    feats["mag_std"] = mag.std(axis=1).astype("float32")
    feats["mag_range"] = (mag.max(axis=1) - mag.min(axis=1)).astype("float32")
    feats["mag_min"] = mag.min(axis=1).astype("float32")
    feats["mag_max"] = mag.max(axis=1).astype("float32")
    feats["mag_u_center"] = (vals["u"] - feats["mag_mean"]).astype("float32")
    feats["mag_z_center"] = (vals["z"] - feats["mag_mean"]).astype("float32")
    for b in BANDS:
        feats[f"{b}_lt13"] = (vals[b] < 13.0).astype("int8")
        feats[f"{b}_gt26"] = (vals[b] > 26.0).astype("int8")

    feats["redshift_log1p"] = np.log1p(zpos).astype("float32")
    feats["redshift_sq"] = (z64 * z64).astype("float32")
    feats["redshift_sqrt"] = np.sqrt(zpos).astype("float32")
    feats["redshift_inv1p"] = (1.0 / (1.0 + zpos)).astype("float32")
    feats["redshift_neg"] = (z64 < 0.0).astype("int8")
    feats["zbin"] = _zbin_series(df["redshift"]).to_numpy(dtype="int16", copy=False)

    for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z", "mag_mean", "mag_range", "blue_red", "ug_gr", "gr_ri", "ri_iz"]:
        feats[f"{c}_x_z"] = (feats[c] * z64).astype("float32")
        feats[f"{c}_div_1pz"] = (feats[c] / (1.0 + zpos)).astype("float32")

    alpha = df["alpha"].astype("float64").to_numpy(copy=False)
    delta = df["delta"].astype("float64").to_numpy(copy=False)
    ar = np.deg2rad(alpha)
    dr = np.deg2rad(delta)
    feats["alpha_sin"] = np.sin(ar).astype("float32")
    feats["alpha_cos"] = np.cos(ar).astype("float32")
    feats["delta_sin"] = np.sin(dr).astype("float32")
    feats["delta_cos"] = np.cos(dr).astype("float32")
    feats["abs_delta"] = np.abs(delta).astype("float32")

    for c in CAT_COLS:
        s = df[c].astype(str)
        vals_cat = cat_values[c]
        mp = {v: i for i, v in enumerate(vals_cat)}
        feats[c + "_code"] = s.map(mp).fillna(-1).astype("int16").to_numpy()
        sv_arr = s.to_numpy()
        for v in vals_cat:
            oh = (sv_arr == v).astype("int8")
            sv = _safe_name(v)
            feats[c + "_" + sv] = oh
            feats[f"{c}_{sv}_z"] = (oh.astype("float32") * z).astype("float32")

    combo = df["spectral_type"].astype(str) + "__" + df["galaxy_population"].astype(str)
    vals_combo = cat_values["combo"]
    mp = {v: i for i, v in enumerate(vals_combo)}
    feats["combo_code"] = combo.map(mp).fillna(-1).astype("int16").to_numpy()
    combo_arr = combo.to_numpy()
    for v in vals_combo:
        feats["combo_" + _safe_name(v)] = (combo_arr == v).astype("int8")

    if encoders is not None:
        _add_target_encoding_features(feats, df, encoders)

    X = pd.DataFrame(feats, copy=False)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)
    return X


def _stratified_dev_sample(train, test):
    dev_n = int(os.environ.get("VERITY_DEV_SAMPLE", "0"))
    if dev_n > 0 and dev_n < len(train):
        train = train.groupby("class", group_keys=False, observed=True).apply(
            lambda x: x.sample(max(1, int(round(dev_n * len(x) / len(train)))), random_state=SEED)
        ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        test = test.sample(n=min(max(20, dev_n // 2), len(test)), random_state=SEED).reset_index(drop=True)
    return train, test


def main():
    train = _read_csv(os.path.join(DATA, "train.csv"), has_target=True)
    test = _read_csv(os.path.join(DATA, "test.csv"), has_target=False)
    train, test = _stratified_dev_sample(train, test)

    cat_values = make_cat_values(train, test)
    le = LabelEncoder()
    y = le.fit_transform(train["class"].astype(str))
    classes = list(le.classes_)
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    sample_weight = cw[y].astype("float32")

    enc = prepare_target_encoders(train, classes, smooth=85.0)
    Xtr = build_features(train, cat_values, enc)
    Xte = build_features(test, cat_values, enc).reindex(columns=Xtr.columns, fill_value=0.0)

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    n_threads = int(os.environ.get("VERITY_THREADS", _THREADS))
    lgb_est = int(os.environ.get("VERITY_LGB_ESTIMATORS", "560"))
    xgb_est = int(os.environ.get("VERITY_XGB_ESTIMATORS", "520"))

    probas = []
    weights = []

    lgb_configs = [
        dict(n_estimators=max(50, int(0.95 * lgb_est)), learning_rate=0.055, num_leaves=48,
             max_depth=-1, min_child_samples=90, subsample=0.98, colsample_bytree=0.95,
             reg_lambda=1.0, reg_alpha=0.0, random_state=SEED + 707),
        dict(n_estimators=lgb_est, learning_rate=0.060, num_leaves=64, max_depth=-1,
             min_child_samples=70, subsample=0.95, colsample_bytree=0.93, reg_lambda=1.6,
             reg_alpha=0.0, random_state=SEED + 303),
    ]
    for cfg, w in zip(lgb_configs, [0.23, 0.23]):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(classes),
            max_bin=255,
            subsample_freq=1,
            class_weight=class_weight,
            n_jobs=n_threads,
            force_col_wise=True,
            deterministic=True,
            verbosity=-1,
            **cfg,
        )
        model.fit(Xtr, y)
        probas.append(np.asarray(model.predict_proba(Xte), dtype="float32"))
        weights.append(w)
        del model

    xgb_configs = [
        dict(n_estimators=xgb_est, max_depth=6, learning_rate=0.055, subsample=0.92,
             colsample_bytree=0.88, min_child_weight=5.0, reg_lambda=3.0, reg_alpha=0.02,
             random_state=SEED + 919),
        dict(n_estimators=max(50, int(0.92 * xgb_est)), max_depth=5, learning_rate=0.065,
             subsample=0.96, colsample_bytree=0.94, min_child_weight=8.0, reg_lambda=1.5,
             reg_alpha=0.0, random_state=SEED + 1111),
    ]
    for cfg, w in zip(xgb_configs, [0.30, 0.24]):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes),
            tree_method="hist",
            max_bin=256,
            eval_metric="mlogloss",
            n_jobs=n_threads,
            verbosity=0,
            **cfg,
        )
        model.fit(Xtr, y, sample_weight=sample_weight, verbose=False)
        probas.append(np.asarray(model.predict_proba(Xte), dtype="float32"))
        weights.append(w)
        del model

    proba = np.average(np.stack(probas, axis=0), axis=0, weights=np.array(weights, dtype="float64")).astype("float32")

    # Conservative balanced-accuracy decision multipliers, validated across several stratified splits.
    prior_mult = {"GALAXY": 0.70, "QSO": 1.00, "STAR": 1.07}
    mult = np.array([prior_mult.get(c, 1.0) for c in classes], dtype="float32")
    pred_idx = (proba * mult[None, :]).argmax(axis=1)
    pred = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"id": test["id"].to_numpy(), "class": pred})
    out.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    print("rows", len(out), "features", Xtr.shape[1], "lgb_est", lgb_est, "xgb_est", xgb_est, "threads", n_threads)
    print("classes", classes, "multipliers", dict(zip(classes, mult.tolist())))
    print(out["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
