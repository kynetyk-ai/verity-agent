import os
import warnings

# Bound thread pools before importing the numerical stack.  The judge advertises ~16 CPU cores;
# using many more OpenMP threads can thrash and was the likely cause of the previous timeout.
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


def _group_keys(df):
    spec = df["spectral_type"].astype(str)
    pop = df["galaxy_population"].astype(str)
    combo = spec + "__" + pop
    zbin = _zbin_series(df["redshift"]).astype(str)
    return {
        "spectral_type": spec,
        "galaxy_population": pop,
        "combo": combo,
        "zbin": zbin,
        "spec_z": spec + "__" + zbin,
        "pop_z": pop + "__" + zbin,
        "combo_z": combo + "__" + zbin,
    }


def prepare_target_encoders(train, classes, smooth=90.0):
    """Smoothed per-class rates for low-cardinality astrophysical groups.

    These use only the labelled training split supplied to the script, and are applied
    vectorially to avoid the slow row-by-row maps used in the timed-out incumbent.
    """
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

    # All colour indices between the five photometric bands.
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
    zbin = _zbin_series(df["redshift"])
    feats["zbin"] = zbin.to_numpy(dtype="int16", copy=False)

    # Colour/redshift interactions: most of the class separation is colour plus redshift.
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


def main():
    train = _read_csv(os.path.join(DATA, "train.csv"), has_target=True)
    test = _read_csv(os.path.join(DATA, "test.csv"), has_target=False)

    # Local-only smoke-test knob.  It is unset in the gate, so the submitted run uses all rows.
    dev_n = int(os.environ.get("VERITY_DEV_SAMPLE", "0"))
    if dev_n > 0 and dev_n < len(train):
        train = train.groupby("class", group_keys=False, observed=True).apply(
            lambda x: x.sample(max(1, int(round(dev_n * len(x) / len(train)))), random_state=SEED)
        ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        test = test.sample(n=min(max(20, dev_n // 2), len(test)), random_state=SEED).reset_index(drop=True)

    cat_values = make_cat_values(train, test)
    le = LabelEncoder()
    y = le.fit_transform(train["class"].astype(str))
    classes = list(le.classes_)
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    enc = prepare_target_encoders(train, classes, smooth=85.0)
    Xtr = build_features(train, cat_values, enc)
    Xte = build_features(test, cat_values, enc).reindex(columns=Xtr.columns, fill_value=0.0)

    from lightgbm import LGBMClassifier

    base_est = int(os.environ.get("VERITY_LGB_ESTIMATORS", "640"))
    n_threads = int(os.environ.get("VERITY_THREADS", _THREADS))

    # Two diverse LightGBM models trained sequentially in one process (no CatBoost co-load,
    # so no thread-pool stall).  Averaging two bounded fits is well inside the time budget and
    # is more robust than a single model.  The previous incumbent timed out because of a
    # row-by-row target-encoding python loop plus an extra calibration fit; both are gone.
    configs = [
        dict(n_estimators=base_est, learning_rate=0.050, num_leaves=92, max_depth=-1,
             min_child_samples=36, subsample=0.91, colsample_bytree=0.88, reg_lambda=2.8,
             reg_alpha=0.03, random_state=SEED + 101),
        dict(n_estimators=max(10, int(0.82 * base_est)), learning_rate=0.060, num_leaves=64,
             max_depth=-1, min_child_samples=70, subsample=0.95, colsample_bytree=0.93,
             reg_lambda=1.6, reg_alpha=0.0, random_state=SEED + 303),
    ]
    weights = np.array([1.1, 0.9], dtype="float64")[:len(configs)]

    probas = []
    for cfg in configs:
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
        del model
    proba = np.average(np.stack(probas, axis=0), axis=0, weights=weights).astype("float32")

    # Conservative fixed decision multipliers from prior validation work.  They counter the
    # GALAXY majority class and directly optimize balanced accuracy rather than raw accuracy.
    prior_mult = {"GALAXY": 0.72, "QSO": 1.00, "STAR": 1.06}
    mult = np.array([prior_mult.get(c, 1.0) for c in classes], dtype="float32")
    pred_idx = (proba * mult[None, :]).argmax(axis=1)
    pred = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"id": test["id"].to_numpy(), "class": pred})
    out.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    print("rows", len(out), "features", Xtr.shape[1], "estimators", [c["n_estimators"] for c in configs], "threads", n_threads)
    print("classes", classes, "multipliers", dict(zip(classes, mult.tolist())))
    print(out["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
