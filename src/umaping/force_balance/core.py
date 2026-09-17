"""Small, testable protocol and statistics primitives. No model training."""
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd

WEIGHTS = (0., .2, 1/3, .5, 2/3, .8, 1.)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Deadline:
    def __init__(self, protocol):
        self.compute = datetime.fromisoformat(protocol['stop_compute_utc'].replace('Z', '+00:00')).timestamp()
        self.new = datetime.fromisoformat(protocol['stop_new_experiments_utc'].replace('Z', '+00:00')).timestamp()
        self.hard = datetime.fromisoformat(protocol['hard_deadline_utc'].replace('Z', '+00:00')).timestamp()

    def check(self, new=False, report=False):
        limit = self.hard if report else (min(self.new, self.compute) if new else self.compute)
        if time.time() >= limit:
            raise TimeoutError('Original absolute deadline reached; it is never reset on resume')


def stratified_ids(labels, n, rng, exclude=()):
    """Equal allocation across actual time labels, without replacement."""
    labels = np.asarray(labels).astype(str)
    pools = [rng.permutation(np.flatnonzero((labels == group) & ~np.isin(np.arange(len(labels)), exclude))).tolist()
             for group in sorted(set(labels))]
    result = []
    while len(result) < min(n, sum(map(len, pools)) + len(result)):
        for pool in pools:
            if pool and len(result) < n:
                result.append(pool.pop())
    return np.asarray(result, dtype=int)


def select_weight(table):
    """Only complete, equally weighted two-teacher search results are eligible."""
    expected = {(t, w) for t in ('Uniform', 'FitGrid') for w in WEIGHTS}
    rows = table[table.scale == 2]
    actual = list(zip(rows.teacher, rows.w))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Selection requires all seven ratios for both teachers exactly once')
    if not np.isfinite(rows.macro_recall).all():
        raise ValueError('Nonfinite selection metric')
    scores = rows.groupby('w').macro_recall.mean().to_dict()
    # Rational candidate distances avoid 1/3 vs 2/3 floating-point asymmetry.
    chosen = min(scores, key=lambda w: (-scores[w], abs(Fraction(w).limit_denominator(15) - Fraction(1,2)), w))
    return dict(w=float(chosen), attraction_coefficient=2*(1-chosen), repulsion_coefficient=2*chosen,
                objective=float(scores[chosen]), scores={str(w):float(s) for w,s in scores.items()},
                rule='mean teacher macro Recall; exact ties nearest 0.5 then smaller w',
                scope='best among seven discrete candidates; confirmation not used')


def paired_bootstrap(left, right, n=2000, seed=20260917, check=lambda: None):
    """Paired cells within actual time strata, macro and micro reported separately."""
    keys = ['query_index', 'timepoint']
    if left.query_index.duplicated().any() or right.query_index.duplicated().any():
        raise ValueError('Duplicate query IDs')
    frame = left[keys+['recall']].merge(right[keys+['recall']], on=keys, how='outer',
                                      suffixes=('_left', '_right'), validate='one_to_one', indicator=True)
    if not (frame._merge == 'both').all():
        raise ValueError('Paired query IDs or time labels differ')
    frame['delta'] = frame.recall_left - frame.recall_right
    rng = np.random.default_rng(seed)
    groups = [x.delta.to_numpy() for _, x in frame.groupby('timepoint')]
    means = np.empty((n, len(groups)))
    for i in range(n):
        if i % 100 == 0:
            check()
        means[i] = [rng.choice(x, len(x), replace=True).mean() for x in groups]
    result = []
    for label, delta, samples in [('macro', np.mean([x.mean() for x in groups]), means.mean(1)),
                                   ('micro', frame.delta.mean(), np.average(means,axis=1,weights=list(map(len,groups))))]:
        lo, hi = np.quantile(samples, [.025,.975])
        result.append(dict(group=label,delta=float(delta),low=float(lo),high=float(hi),n=len(frame)))
    for j, (label, sub) in enumerate(frame.groupby('timepoint')):
        lo, hi = np.quantile(means[:,j], [.025,.975])
        result.append(dict(group=label,delta=float(sub.delta.mean()),low=float(lo),high=float(hi),n=len(sub)))
    return result


def metrics_from_neighbors(high_ids, high_dist, low_ids, low_dist, hd_radius, ld_radius):
    # Identical to fit_grid_experiment.metrics.query_metrics: binary NDCG@15,
    # density ratios use the SAME high-dimensional neighbor reference radii.
    relevant = (low_ids[:,:,None] == high_ids[:,None,:]).any(axis=2)
    recall = relevant.mean(1)
    discounts = 1 / np.log2(np.arange(2, high_ids.shape[1]+2))
    ndcg = (relevant * discounts).sum(1) / discounts.sum()
    high_ratio = high_dist[:,-1] / np.maximum(hd_radius[high_ids].mean(1), 1e-12)
    low_ratio = low_dist[:,-1] / np.maximum(ld_radius[high_ids].mean(1), 1e-12)
    density = np.abs(np.log(np.maximum(low_ratio,1e-12)) - np.log(np.maximum(high_ratio,1e-12)))
    return recall, ndcg, density
