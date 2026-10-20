"""Shared calculations for the three notebooks and dashboard; fitting lives in notebook 02."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

MODEL_NAMES = ['Rules', 'Isolation Forest', 'LOF', 'One-Class SVM', 'Hybrid']
HYBRID_WEIGHT = 0.5


def load_data(root):
    root = Path(root)
    df = pd.read_csv(root / 'data/synthetic/cardless_features.csv', parse_dates=['created_at'])
    df['redeemed_at'] = df.created_at + pd.to_timedelta(df.redeem_delay_min, unit='min')
    return df


def split_stage(df, stage, start_date):
    """Half-open day ranges: [0,90), [90,120), [120,180). B uses cash-out time."""
    if stage not in ('A', 'B'):
        raise ValueError('stage must be A or B')
    if stage == 'B':
        part = df.loc[df.redeemed == 1].copy()
        score_time = pd.to_datetime(part.redeemed_at)
    else:
        part = df.copy()
        score_time = pd.to_datetime(part.created_at)
    part['score_time'] = score_time
    start = pd.Timestamp(start_date)
    boundaries = [start + pd.Timedelta(days=n) for n in (0, 90, 120, 180)]
    return tuple(part[(score_time >= lo) & (score_time < hi)].sort_values('score_time')
                 for lo, hi in zip(boundaries[:-1], boundaries[1:]))


def metrics(y, scores, threshold):
    """Scam = 1; zero precision/recall/F1 when the respective denominator is zero."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Expected binary labels and finite scores')
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(threshold=float(threshold), precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                fpr=fp / (fp + tn) if fp + tn else 0.0, tp=tp, fp=fp, fn=fn, tn=tn)


def choose_threshold(y, scores):
    """Highest F1 on validation; tied F1 values choose the higher (fewer-alert) threshold."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if not len(y) or set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Validation requires both classes and finite scores')
    p, r, thresholds = precision_recall_curve(y, scores)
    f1 = np.divide(2*p[:-1]*r[:-1], p[:-1]+r[:-1], out=np.zeros(len(thresholds)),
                   where=p[:-1]+r[:-1] > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    threshold = float(thresholds[best])
    return threshold, metrics(y, scores, threshold)


def band_cutoffs(y, scores, threshold):
    """Existing UI policy only: Medium targets 80% recall, Critical 70% precision.

    High is exactly the saved alert threshold. Bands are ordered; they are not fraud probabilities.
    """
    p, r, t = precision_recall_curve(y, scores)
    medium = t[r[:-1] >= .8].max() if (r[:-1] >= .8).any() else t.min()
    critical = t[p[:-1] >= .7].min() if (p[:-1] >= .7).any() else t.max()
    return {'Medium': float(min(medium, threshold)), 'High': float(threshold),
            'Critical': float(max(critical, threshold))}


def risk_bands(scores, cutoffs):
    scores = np.asarray(scores)
    return np.select([scores >= cutoffs['Critical'], scores >= cutoffs['High'],
                      scores >= cutoffs['Medium']], ['Critical', 'High', 'Medium'], default='Low')


def fingerprint(root):
    """Reject stale fitted models when input data, configuration or fitting code changes."""
    root = Path(root)
    paths = ['data/synthetic/cardless_features.csv', 'config/generator.yaml',
             'src/models.py', 'src/core_evaluation.py', 'notebooks/02_fit_models.ipynb']
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode())
        path = root / name
        if path.suffix == '.ipynb':
            nb = json.loads(path.read_text())
            content = json.dumps([(c['cell_type'], c['source']) for c in nb['cells']], sort_keys=True).encode()
            digest.update(content)
        else:
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def check_fitted_bundle(bundle, root):
    if bundle['fingerprint'] != fingerprint(root):
        raise ValueError('Data, config or fitting code changed. Run notebook 02 again before notebook 03.')


def scenario_rates(part, predictions, stage):
    rows = []
    for name, pred in predictions.items():
        table = part[['scenario_type', 'is_scam']].copy()
        table['alert'] = np.asarray(pred, dtype=bool)
        for scenario, g in table.groupby('scenario_type'):
            rows.append(dict(stage=stage, model=name, scenario_type=scenario, n=len(g),
                             flagged=int(g.alert.sum()), rate=float(g.alert.mean()),
                             interpretation='Recall' if g.is_scam.iloc[0] else 'False-alarm rate'))
    return pd.DataFrame(rows)


def save_figure(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
