"""同一queryでの差・時間近傍指標。未観測時間点の分類精度はN/A。"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from umaping.comparison.statistics import paired_statistics, holm
from umaping.graph import chunked_exact_knn
from umaping.evaluation.advanced import compute_per_query_diagnostics

def temporal_groups(labels, split):
    x = np.asarray(labels).astype(str)
    if not np.isin(x, split['query_timepoints']).all():
        raise ValueError('unexpected query timepoint')
    return dict(all=np.ones(len(x), bool), interpolation=x == split['interpolation_timepoint'],
                extrapolation=np.isin(x, split['extrapolation_timepoints']),
                **{f'timepoint:{label}': x == label for label in split['query_timepoints']})


def temporal_metrics(reference_ordinals, query_ordinals, neighbors, split):
    reference_ordinals = np.asarray(reference_ordinals, int)
    query_ordinals = np.asarray(query_ordinals, int)
    order = split['ordered_timepoints']
    labels = np.asarray(order)[query_ordinals-1]
    temporal_groups(labels, split)
    if not np.isin(np.asarray(order)[reference_ordinals-1], split['reference_timepoints']).all():
        raise ValueError('invalid reference ordinals')
    times = reference_ordinals[np.asarray(neighbors)]
    mae = np.abs(times - query_ordinals[:, None]).mean(axis=1)
    horizon = np.maximum(query_ordinals - split['last_observed_ordinal'], 0)
    m = split['interpolation_ordinal']
    bracket = np.any(times == m-1, axis=1) & np.any(times == m+1, axis=1)
    return pd.DataFrame(dict(timepoint=labels, timepoint_ordinal=query_ordinals, horizon=horizon,
        temporal_neighbor_mae=mae, excess_temporal_neighbor_mae=np.where(horizon > 0, mae-horizon, np.nan),
        temporal_bracketing_rate=np.where(query_ordinals == m, bracket.astype(float), np.nan),
        timepoint_label_accuracy=np.full(len(mae), np.nan)))


def aggregate_groups(frame, split=None):
    masks = temporal_groups(frame.timepoint, split) if 'timepoint' in frame else {'all': np.ones(len(frame), bool)}
    rows = []
    for group, mask in masks.items():
        sub = frame.loc[mask]
        row = dict(group=group, n_queries=len(sub))
        row.update(sub.select_dtypes(include='number').drop(columns=['query_index'], errors='ignore').mean().to_dict())
        if len(sub):
            row['recall_at_15_p05'] = float(sub.recall_at_15.quantile(.05))
            row['local_displacement_p95'] = float(sub.local_displacement.quantile(.95)) if 'local_displacement' in sub else np.nan
        row['label_accuracy_status'] = 'not_applicable_unseen_timepoints' if 'timepoint' in frame else 'not_requested'
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(uniform, fit, seed=0, split=None):
    if uniform.query_index.duplicated().any() or fit.query_index.duplicated().any():
        raise ValueError('duplicate query identity')
    cols = ['query_index', 'recall_at_15'] + (['timepoint'] if 'timepoint' in uniform else [])
    paired = uniform[cols].merge(fit[['query_index','recall_at_15']], on='query_index', suffixes=('_uniform','_fit_grid'), validate='one_to_one')
    if len(paired) != len(uniform) or len(paired) != len(fit):
        raise ValueError('different query identities')
    if 'timepoint' in fit:
        actual = fit.set_index('query_index').loc[paired.query_index, 'timepoint'].to_numpy()
        if not np.array_equal(actual, paired.timepoint):
            raise ValueError('paired timepoint mismatch')
    paired['delta_recall_at_15'] = paired.recall_at_15_fit_grid - paired.recall_at_15_uniform
    masks = temporal_groups(paired.timepoint, split) if 'timepoint' in paired else {'all': np.ones(len(paired), bool)}
    rows = []
    for group, mask in masks.items():
        sub = paired.loc[mask]
        if not len(sub):
            continue
        r = paired_statistics(sub.recall_at_15_fit_grid, sub.recall_at_15_uniform, 'recall_at_15', seed=seed)
        # Do not assert symmetry/exchangeability of biological query differences.
        r.pop('p_value', None); r.pop('test', None); r.pop('n_permutations', None)
        d = sub.delta_recall_at_15.to_numpy()
        nonzero = d[d != 0]
        r.update(group=group, fraction_worsened=float(np.mean(d < 0)),
                 paired_effect_size_dz=float(d.mean()/d.std(ddof=1)) if len(d)>1 and d.std(ddof=1)>0 else None,
                 conditional_sign_test_p=float(binomtest(int((nonzero>0).sum()), len(nonzero)).pvalue) if len(nonzero) else 1.,
                 test_assumption='conditional iid cells/non-tied signs; donor/time dependence not modeled',
                 inference_scope='fixed trained models and reference; cell bootstrap, not biological replication')
        rows.append(r)
    stats = pd.DataFrame(rows)
    stats['conditional_sign_test_p_holm'] = holm(stats.conditional_sign_test_p)
    return paired, stats


def query_metrics(name, prepared, outcome, cfg, query_indices=None):
    idx = np.arange(len(prepared.query_features)) if query_indices is None else np.asarray(query_indices)
    q = prepared.query_features[idx]
    ref, y = outcome.reference_embedding, outcome.query_embedding
    if len(prepared.reference_features) <= 15:
        raise ValueError('Recall@15 requires more than 15 reference cells')
    high_ids, high_dist = chunked_exact_knn(q, prepared.reference_features, k=15)
    low_ids, low_dist = chunked_exact_knn(y, ref, k=15)
    from umaping.graph import query_fuzzy_weights
    from umaping.umap_forces import find_ab_params
    weights, _, _ = query_fuzzy_weights(high_dist, eps=cfg.umap.eps)
    a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)
    # Use identical true neighbors for geometric diagnostics across methods.
    frame, _ = compute_per_query_diagnostics(name, q, prepared.reference_features, y, ref,
                                             list(high_ids), list(weights), a, b, ks=(15,))
    frame['query_index'] = idx
    # Ratios remove global scale; each query is compared with its HD neighbors' reference radii.
    _, hd_ref = chunked_exact_knn(prepared.reference_features, prepared.reference_features, k=16)
    _, ld_ref = chunked_exact_knn(ref, ref, k=16)
    eps = 1e-12
    high_ratio = high_dist[:,-1] / np.maximum(hd_ref[high_ids,-1].mean(1), eps)
    low_ratio = low_dist[:,-1] / np.maximum(ld_ref[high_ids,-1].mean(1), eps)
    frame['density_log_distortion'] = np.abs(np.log(np.maximum(low_ratio,eps)) - np.log(np.maximum(high_ratio,eps)))
    frame['collapse_rate'] = (low_ratio < .1).astype(float)
    center = ref.mean(0)
    radii = np.sort(np.linalg.norm(ref-center, axis=1))
    frame['global_periphery_percentile'] = np.searchsorted(radii, np.linalg.norm(y-center,axis=1), side='right') / len(ref)
    frame['latency_seconds'] = outcome.mean_query_latency_seconds
    if not np.isfinite(frame[['recall_at_15','density_log_distortion','local_displacement']]).all().all():
        raise FloatingPointError('non-finite embedding metrics')
    return frame, low_ids
