import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

SEED = 2024
np.random.seed(SEED)

DATA = os.environ.get("VERITY_DATA", "data")
OUT = os.environ.get("VERITY_OUT", "out")
os.makedirs(OUT, exist_ok=True)

CAT_COLS = ['spectral_type', 'galaxy_population']
BANDS = ['u', 'g', 'r', 'i', 'z']


def build_features(df, cat_values):
    X = pd.DataFrame(index=df.index)
    for c in ['u', 'g', 'r', 'i', 'z', 'redshift', 'alpha', 'delta']:
        X[c] = df[c].astype('float32')
    # photometric colour indices (all pairs) -- carry most of the signal
    for ia, a in enumerate(BANDS):
        for b in BANDS[ia + 1:]:
            X[f'{a}_{b}'] = (df[a] - df[b]).astype('float32')
    # magnitude summary statistics
    X['mag_mean'] = df[BANDS].mean(axis=1).astype('float32')
    X['mag_std'] = df[BANDS].std(axis=1).astype('float32')
    X['mag_range'] = (df[BANDS].max(axis=1) - df[BANDS].min(axis=1)).astype('float32')
    # redshift transforms (dominant signal, especially for QSO/STAR separation)
    z = np.clip(df['redshift'].values.astype('float64'), 0, None)
    X['redshift_log1p'] = np.log1p(z).astype('float32')
    X['redshift_sq'] = (df['redshift'] ** 2).astype('float32')
    X['redshift_sqrt'] = np.sqrt(z).astype('float32')
    # sky coordinates (cyclic for RA)
    X['alpha_sin'] = np.sin(np.deg2rad(df['alpha'])).astype('float32')
    X['alpha_cos'] = np.cos(np.deg2rad(df['alpha'])).astype('float32')
    X['delta_sin'] = np.sin(np.deg2rad(df['delta'])).astype('float32')
    X['delta_cos'] = np.cos(np.deg2rad(df['delta'])).astype('float32')
    X['abs_delta'] = np.abs(df['delta']).astype('float32')
    # categoricals: ordinal code + one-hot (categories shared train/test)
    for c in CAT_COLS:
        vals = cat_values[c]
        s = df[c].astype(str)
        mp = {v: i for i, v in enumerate(vals)}
        X[c + '_code'] = s.map(mp).fillna(-1).astype('int16')
        for v in vals:
            X[c + '_' + v] = (s == v).astype('int8')
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def main():
    train = pd.read_csv(os.path.join(DATA, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA, 'test.csv'))

    cat_values = {
        c: sorted(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
        for c in CAT_COLS
    }

    le = LabelEncoder()
    y = le.fit_transform(train['class'].astype(str))
    classes = list(le.classes_)

    Xtr = build_features(train, cat_values)
    Xte = build_features(test, cat_values)
    Xte = Xte[Xtr.columns]

    sw = compute_sample_weight('balanced', y)

    probas = []

    # ---- XGBoost models (balanced via sample weights) ----
    try:
        import xgboost as xgb
        xgb_cfgs = [
            dict(n_estimators=700, max_depth=6, learning_rate=0.055, subsample=0.86,
                 colsample_bytree=0.86, min_child_weight=12, reg_lambda=3.0,
                 reg_alpha=0.1, random_state=13),
            dict(n_estimators=900, max_depth=5, learning_rate=0.045, subsample=0.88,
                 colsample_bytree=0.88, min_child_weight=8, reg_lambda=2.0,
                 reg_alpha=0.05, random_state=21),
        ]
        for cfg in xgb_cfgs:
            m = xgb.XGBClassifier(objective='multi:softprob', num_class=len(classes),
                                  tree_method='hist', eval_metric='mlogloss',
                                  n_jobs=-1, **cfg)
            m.fit(Xtr, y, sample_weight=sw)
            probas.append(m.predict_proba(Xte))
            del m
    except Exception as e:
        print('xgboost unavailable:', e)

    # ---- HistGradientBoosting models (balanced class weight) ----
    hgb_cfgs = [
        dict(max_iter=220, learning_rate=0.055, max_leaf_nodes=63,
             l2_regularization=0.02, min_samples_leaf=25, random_state=11),
        dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=127,
             l2_regularization=0.08, min_samples_leaf=15, random_state=31),
    ]
    for cfg in hgb_cfgs:
        m = HistGradientBoostingClassifier(class_weight='balanced',
                                           early_stopping=False, max_bins=255, **cfg)
        m.fit(Xtr, y)
        probas.append(m.predict_proba(Xte))
        del m

    if not probas:
        raise RuntimeError('No models trained')

    avg = np.mean(probas, axis=0)

    # Modest class-recall multiplier (boost minority recall for balanced accuracy).
    # Keyed by class name so it is robust to label ordering.
    mult_map = {'GALAXY': 0.95, 'QSO': 1.05, 'STAR': 1.06}
    mult = np.array([mult_map.get(c, 1.0) for c in classes], dtype='float64')
    pred_idx = (avg * mult).argmax(axis=1)
    pred = le.inverse_transform(pred_idx)

    out = pd.DataFrame({'id': test['id'].values, 'class': pred})
    out.to_csv(os.path.join(OUT, 'predictions.csv'), index=False)
    print('wrote', len(out), 'rows; class dist:')
    print(out['class'].value_counts())


if __name__ == '__main__':
    main()
