import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

SEED = 2026
np.random.seed(SEED)

DATA = os.environ.get("VERITY_DATA", "data")
OUT = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT, exist_ok=True)

BANDS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]
Z_BINS = np.array([
    -10.0, 0.0, 0.015, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20,
    0.30, 0.50, 0.80, 1.20, 1.60, 2.00, 2.60, 3.50, 10.0
], dtype="float64")


def _safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace(".", "p").replace("-", "m")


def prepare_target_encoders(train, classes, smooth=80.0):
    """Smoothed class-prior features for non-unique astrophysical groups."""
    df = pd.DataFrame(index=train.index)
    df["spectral_type"] = train["spectral_type"].astype(str)
    df["galaxy_population"] = train["galaxy_population"].astype(str)
    df["combo"] = df["spectral_type"] + "__" + df["galaxy_population"]
    zbin = pd.cut(train["redshift"], bins=Z_BINS, labels=False, include_lowest=True).fillna(-1).astype("int16")
    df["zbin"] = zbin.astype(str)
    df["spec_z"] = df["spectral_type"] + "__" + df["zbin"]
    df["pop_z"] = df["galaxy_population"] + "__" + df["zbin"]
    df["combo_z"] = df["combo"] + "__" + df["zbin"]

    y = train["class"].astype(str).values
    global_p = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(0.0).values.astype("float64")
    enc = {"classes": list(classes), "global": global_p.astype("float32"), "maps": {}}
    for col in ["spectral_type", "galaxy_population", "combo", "zbin", "spec_z", "pop_z", "combo_z"]:
        tmp = pd.DataFrame({"key": df[col].values, "class": y})
        counts = pd.crosstab(tmp["key"], tmp["class"]).reindex(columns=classes, fill_value=0).astype("float64")
        n = counts.sum(axis=1).values[:, None]
        probs = (counts.values + smooth * global_p[None, :]) / (n + smooth)
        enc["maps"][col] = {str(k): probs[i].astype("float32") for i, k in enumerate(counts.index)}
    return enc


def build_features(df, cat_values, encoders=None):
    feats = {}
    n = len(df)
    for c in ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]:
        feats[c] = df[c].astype("float32").values

    vals = {b: df[b].astype("float32").values for b in BANDS}
    for ia, a in enumerate(BANDS):
        for b in BANDS[ia + 1:]:
            feats[f"{a}_{b}"] = (vals[a] - vals[b]).astype("float32")

    feats["ug_gr"] = (feats["u_g"] - feats["g_r"]).astype("float32")
    feats["gr_ri"] = (feats["g_r"] - feats["r_i"]).astype("float32")
    feats["ri_iz"] = (feats["r_i"] - feats["i_z"]).astype("float32")
    feats["blue_red"] = ((vals["u"] + vals["g"] - vals["i"] - vals["z"]) / 2.0).astype("float32")
    feats["ug_ratio"] = (feats["u_g"] / (np.abs(vals["g"]) + 1e-3)).astype("float32")
    feats["gr_ratio"] = (feats["g_r"] / (np.abs(vals["r"]) + 1e-3)).astype("float32")
    feats["riz_curve"] = (feats["r_i"] + feats["i_z"] - feats["g_r"]).astype("float32")

    mag = np.vstack([vals[b] for b in BANDS]).T
    feats["mag_mean"] = mag.mean(axis=1).astype("float32")
    feats["mag_std"] = mag.std(axis=1).astype("float32")
    feats["mag_range"] = (mag.max(axis=1) - mag.min(axis=1)).astype("float32")
    feats["mag_min"] = mag.min(axis=1).astype("float32")
    feats["mag_max"] = mag.max(axis=1).astype("float32")
    for b in BANDS:
        feats[f"{b}_lt13"] = (vals[b] < 13.0).astype("int8")
        feats[f"{b}_gt26"] = (vals[b] > 26.0).astype("int8")

    z = df["redshift"].astype("float64").values
    zpos = np.clip(z, 0.0, None)
    feats["redshift_log1p"] = np.log1p(zpos).astype("float32")
    feats["redshift_sq"] = (z * z).astype("float32")
    feats["redshift_sqrt"] = np.sqrt(zpos).astype("float32")
    feats["redshift_inv1p"] = (1.0 / (1.0 + zpos)).astype("float32")
    feats["redshift_neg"] = (z < 0).astype("int8")
    zbin = pd.cut(df["redshift"], bins=Z_BINS, labels=False, include_lowest=True).fillna(-1).astype("int16")
    feats["zbin"] = zbin.values

    for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "r_z", "mag_mean", "mag_range", "blue_red", "ug_gr", "gr_ri"]:
        feats[f"{c}_x_z"] = (feats[c] * z).astype("float32")
        feats[f"{c}_div_1pz"] = (feats[c] / (1.0 + zpos)).astype("float32")

    alpha_rad = np.deg2rad(df["alpha"].astype("float64").values)
    delta_rad = np.deg2rad(df["delta"].astype("float64").values)
    feats["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    feats["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    feats["delta_sin"] = np.sin(delta_rad).astype("float32")
    feats["delta_cos"] = np.cos(delta_rad).astype("float32")
    feats["abs_delta"] = np.abs(df["delta"].astype("float32").values)

    for c in CAT_COLS:
        s = df[c].astype(str)
        vals_cat = cat_values[c]
        mp = {v: i for i, v in enumerate(vals_cat)}
        feats[c + "_code"] = s.map(mp).fillna(-1).astype("int16").values
        for v in vals_cat:
            oh = (s.values == v).astype("int8")
            sv = _safe_name(v)
            feats[c + "_" + sv] = oh
            feats[f"{c}_{sv}_z"] = (oh.astype("float32") * z).astype("float32")

    combo = df["spectral_type"].astype(str) + "__" + df["galaxy_population"].astype(str)
    vals_combo = cat_values["combo"]
    mp = {v: i for i, v in enumerate(vals_combo)}
    feats["combo_code"] = combo.map(mp).fillna(-1).astype("int16").values
    for v in vals_combo:
        feats["combo_" + _safe_name(v)] = (combo.values == v).astype("int8")

    if encoders is not None:
        zbin_s = zbin.astype(str)
        groups = {
            "spectral_type": df["spectral_type"].astype(str),
            "galaxy_population": df["galaxy_population"].astype(str),
            "combo": combo.astype(str),
            "zbin": zbin_s,
            "spec_z": df["spectral_type"].astype(str) + "__" + zbin_s,
            "pop_z": df["galaxy_population"].astype(str) + "__" + zbin_s,
            "combo_z": combo.astype(str) + "__" + zbin_s,
        }
        classes = encoders["classes"]
        default = encoders["global"].astype("float32")
        for gname, keys in groups.items():
            mp = encoders["maps"][gname]
            arr = np.empty((n, len(classes)), dtype="float32")
            for i, k in enumerate(keys.values):
                arr[i] = mp.get(str(k), default)
            for j, cls in enumerate(classes):
                feats[f"te_{gname}_{_safe_name(cls)}"] = arr[:, j]
            if len(classes) == 3:
                feats[f"te_{gname}_qso_minus_gal"] = (arr[:, 1] - arr[:, 0]).astype("float32")
                feats[f"te_{gname}_star_minus_gal"] = (arr[:, 2] - arr[:, 0]).astype("float32")

    X = pd.DataFrame(feats)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def lightgbm_model(classes, class_weight, cfg):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        class_weight=class_weight,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
        subsample_freq=1,
        **cfg,
    )


def make_cat_values(train, test):
    cat_values = {c: sorted(pd.concat([train[c], test[c]], axis=0).astype(str).unique()) for c in CAT_COLS}
    cat_values["combo"] = sorted(pd.concat([
        train["spectral_type"].astype(str) + "__" + train["galaxy_population"].astype(str),
        test["spectral_type"].astype(str) + "__" + test["galaxy_population"].astype(str),
    ]).unique())
    return cat_values


def tune_multipliers(train, test, y, classes, cat_values, class_weight):
    """Tune class decision multipliers on a real holdout, bounded to one small LightGBM fit."""
    n = len(train)
    # Gate uses full-size data; local smoke tests use the same path with much smaller splits.
    if n < 3000:
        return np.ones(len(classes), dtype="float64")
    val_size = min(70000, max(15000, int(0.16 * n)))
    train_size = min(240000, n - val_size)
    if train_size < 2000 or val_size < 1000:
        return np.ones(len(classes), dtype="float64")

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=val_size, random_state=SEED + 17)
    idx_tr, idx_va = next(splitter.split(np.zeros(n, dtype="int8"), y))
    tr = train.iloc[idx_tr].copy()
    va = train.iloc[idx_va].copy()
    enc = prepare_target_encoders(tr, classes, smooth=90.0)
    Xtr = build_features(tr, cat_values, enc)
    Xva = build_features(va, cat_values, enc)[Xtr.columns]

    cfg = dict(
        n_estimators=int(os.environ.get("VERITY_CAL_ESTIMATORS", "360")),
        learning_rate=0.060,
        num_leaves=72,
        max_depth=-1,
        min_child_samples=45,
        subsample=0.92,
        colsample_bytree=0.90,
        reg_lambda=2.2,
        reg_alpha=0.02,
        max_bin=255,
        random_state=SEED + 31,
    )
    m = lightgbm_model(classes, class_weight, cfg)
    m.fit(Xtr, y[idx_tr])
    p = np.asarray(m.predict_proba(Xva), dtype="float32")
    yv = y[idx_va]
    del m, Xtr, Xva

    # Start from incumbent-like, conservative priors if the standard class names are present.
    prior_map = {"GALAXY": 0.72, "QSO": 1.00, "STAR": 1.06}
    mult = np.array([prior_map.get(c, 1.0) for c in classes], dtype="float64")

    def score(mm):
        return balanced_accuracy_score(yv, (p * mm[None, :]).argmax(axis=1))

    best = score(mult)
    # Coordinate descent in log-space.  Normalization is harmless because only ratios matter.
    for radius in [0.55, 0.35, 0.22, 0.14]:
        improved = True
        while improved:
            improved = False
            for j in range(len(classes)):
                cur = mult.copy()
                for fac in np.exp(np.linspace(-radius, radius, 17)):
                    cand = mult.copy()
                    cand[j] *= fac
                    cand /= np.exp(np.mean(np.log(np.clip(cand, 1e-6, None))))
                    s = score(cand)
                    if s > best + 1e-7:
                        best = s
                        cur = cand
                        improved = True
                mult = cur
    # Shrink toward the proven prior to reduce holdout-threshold variance.
    prior = np.array([prior_map.get(c, 1.0) for c in classes], dtype="float64")
    mult = np.exp(0.70 * np.log(np.clip(mult, 1e-6, None)) + 0.30 * np.log(np.clip(prior, 1e-6, None)))
    mult /= np.exp(np.mean(np.log(mult)))
    print("threshold_multipliers", dict(zip(classes, mult)), "holdout_bal_acc", round(float(best), 6))
    return mult.astype("float64")


def main():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))

    # Optional local smoke-test knob; unset in the gate, so the real run trains on all data.
    dev_n = int(os.environ.get("VERITY_DEV_SAMPLE", "0"))
    if dev_n > 0 and dev_n < len(train):
        train = train.groupby("class", group_keys=False).apply(
            lambda x: x.sample(max(1, int(round(dev_n * len(x) / len(train)))), random_state=SEED)
        ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        test = test.sample(n=min(dev_n // 2 + 10, len(test)), random_state=SEED).reset_index(drop=True)

    cat_values = make_cat_values(train, test)
    le = LabelEncoder()
    y = le.fit_transform(train["class"].astype(str))
    classes = list(le.classes_)
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    mult = tune_multipliers(train, test, y, classes, cat_values, class_weight)

    enc = prepare_target_encoders(train, classes, smooth=80.0)
    Xtr = build_features(train, cat_values, enc)
    Xte = build_features(test, cat_values, enc)[Xtr.columns]

    default_estimators = [780, 620]
    if "VERITY_LGB_ESTIMATORS" in os.environ:
        est = int(os.environ["VERITY_LGB_ESTIMATORS"])
        default_estimators = [est, max(10, int(0.8 * est))]

    configs = [
        dict(n_estimators=default_estimators[0], learning_rate=0.047, num_leaves=96, max_depth=-1,
             min_child_samples=34, subsample=0.91, colsample_bytree=0.88, reg_lambda=2.8,
             reg_alpha=0.03, max_bin=255, random_state=SEED + 101),
        dict(n_estimators=default_estimators[1], learning_rate=0.058, num_leaves=64, max_depth=-1,
             min_child_samples=70, subsample=0.95, colsample_bytree=0.93, reg_lambda=1.6,
             reg_alpha=0.00, max_bin=255, random_state=SEED + 303),
    ]
    weights = np.array([1.10, 0.90], dtype="float64")[:len(configs)]
    probas = []
    model_names = []
    for i, cfg in enumerate(configs):
        m = lightgbm_model(classes, class_weight, cfg)
        m.fit(Xtr, y)
        probas.append(np.asarray(m.predict_proba(Xte), dtype="float32"))
        model_names.append(f"lgb{i+1}")
        del m

    avg = np.average(np.stack(probas, axis=0), axis=0, weights=weights)
    pred_idx = (avg * mult[None, :]).argmax(axis=1)
    pred = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"id": test["id"].values, "class": pred})
    out.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    print("models", model_names)
    print("multipliers", dict(zip(classes, mult)))
    print("wrote", len(out), "rows")
    print(out["class"].value_counts())


if __name__ == "__main__":
    main()
