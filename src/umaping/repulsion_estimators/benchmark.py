"""Phase A: 固定query・全点和oracleによるteacher比較。"""
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .base import synchronize


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, Path):
            return str(v)
        return v
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def field_metrics(pred, exact):
    pred, exact = np.asarray(pred, float), np.asarray(exact, float)
    error = pred-exact
    norms, truth_norm = np.linalg.norm(pred, axis=-1), np.linalg.norm(exact, axis=-1)
    valid = (norms > 1e-12) & (truth_norm > 1e-12)
    cosine = np.sum(pred*exact, axis=-1) / np.maximum(norms*truth_norm, 1e-24)
    mse = float(np.mean(np.sum(error**2, axis=-1)))
    signal_rms = float(np.sqrt(np.mean(truth_norm**2)))
    relative_valid = truth_norm > max(1e-8, float(np.median(truth_norm))*1e-3)
    return dict(mse=mse, rmse=np.sqrt(mse), exact_signal_rms=signal_rms,
                normalized_rmse=np.sqrt(mse)/signal_rms if signal_rms > 1e-12 else np.nan,
                cosine=float(cosine[valid].mean()) if valid.any() else np.nan,
                cosine_valid_fraction=float(valid.mean()), magnitude_error=float(np.abs(norms-truth_norm).mean()),
                relative_magnitude_error=float((np.abs(norms-truth_norm)/np.maximum(truth_norm, 1e-24))[relative_valid].mean()) if relative_valid.any() else np.nan,
                relative_magnitude_valid_fraction=float(relative_valid.mean()))


def evaluate_teacher(name, teacher, anchor, t, y, exact, output, repetitions=50, batch_size=64):
    """全反復の予測を保存。乱数状態をリセットせず独立な連続サンプルを使う。"""
    repetitions = repetitions if teacher.stochastic else 1
    prediction_path = output / f'{name}_predictions.npy'
    if prediction_path.exists():
        raise FileExistsError(prediction_path)
    predictions = np.lib.format.open_memmap(prediction_path, mode='w+', shape=(repetitions, len(y), 2), dtype=np.float32)
    predictions[:] = np.nan
    diagnostics, weights, total_seconds = [], [], 0.
    for repetition in range(repetitions):
        for start in range(0, len(y), batch_size):
            stop = min(start+batch_size, len(y))
            synchronize(y.device)
            begin = time.perf_counter()
            try:
                value = teacher.estimate(y[start:stop], t[start:stop], anchor[start:stop])
                if value.shape != y[start:stop].shape or not torch.isfinite(value).all():
                    raise FloatingPointError('Non-finite/invalid teacher estimate')
            except Exception as exc:
                predictions.flush()
                write_json(output / f'{name}_failure.json', dict(repetition=repetition, query_start=start,
                           reason=f'{type(exc).__name__}: {exc}', diagnostics=teacher.diagnostics))
                raise
            synchronize(y.device)
            total_seconds += time.perf_counter()-begin
            predictions[repetition, start:stop] = value.cpu().numpy()
            diag = dict(teacher.diagnostics)
            if 'weights' in diag:
                weights.append(diag.pop('weights'))
            diagnostics.append(dict(repetition=repetition, n_queries=stop-start, **diag))
        if repetition == 0 or (repetition+1) % 10 == 0:
            print(f'Phase A {name}: {repetition+1}/{repetitions}', flush=True)
    predictions.flush()
    pd.DataFrame(diagnostics).to_csv(output / f'{name}_diagnostics.csv', index=False)
    if weights:
        np.save(output / f'{name}_weights.npy', np.concatenate(weights))
    truth = exact.cpu().numpy()
    metrics = field_metrics(predictions, np.broadcast_to(truth, predictions.shape))
    mean = predictions.astype(float).mean(0)
    bias_sq = np.sum((mean-truth)**2, -1)
    variance = np.sum((predictions-mean[None])**2, -1).mean(0)
    result = dict(method=name, status='success', stochastic=teacher.stochastic, repetitions=repetitions,
                  n_queries=len(y), **metrics, bias=float(np.sqrt(bias_sq).mean()),
                  bias_rmse=float(np.sqrt(bias_sq.mean())), variance=float(variance.mean()),
                  bias_squared_debiased=float((bias_sq-variance/max(1, repetitions-1)).mean()) if repetitions > 1 else np.nan,
                  runtime_seconds_per_query=total_seconds/(repetitions*len(y)))
    # biasは有限反復のsample-mean bias。debiased値は負にもなるのでclipしない。
    diag = pd.DataFrame(diagnostics)
    for column in ('entropy', 'mean_weight', 'ess', 'force_interactions'):
        if column in diag:
            result[column] = float(np.average(diag[column], weights=diag.n_queries))
    for column in ('max_probability', 'max_weight', 'outside_count'):
        if column in diag:
            result[column] = float(diag[column].max())
    if 'min_sampled_probability' in diag:
        result['min_sampled_probability'] = float(diag.min_sampled_probability.min())
    result['score_seconds_per_query'] = float(diag.score_seconds.sum()/(repetitions*len(y)))
    return result
