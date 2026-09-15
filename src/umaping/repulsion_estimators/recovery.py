"""中断runは読み取り専用。完了成果物とoptimizer付きsnapshotを検証して引き継ぐ。"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .benchmark import field_metrics, write_json


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


def save_snapshot(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def training_signature(cfg, hparams, selected, args):
    return dict(repulsion=asdict(cfg.repulsion), hparams=hparams, selected=selected,
                seed=args.seed, eval_every=args.eval_every, device=args.device)


def verify_initial(prior, initial, hp):
    checkpoint = torch.load(prior/'artifacts/initial_repulsion_field.pt', map_location='cpu', weights_only=False)
    if checkpoint['hparams'] != hp or set(checkpoint['state_dict']) != set(initial):
        raise ValueError('Recovery initialization architecture mismatch')
    for key in initial:
        if not torch.equal(initial[key], checkpoint['state_dict'][key]):
            raise ValueError(f'Recovery initialization mismatch: {key}')


@torch.no_grad()
def completed_results(prior, cfg, hp, selected, args, y, t, exact):
    from umaping.config import Config
    from umaping.models.repulsion import RepulsionField
    saved_cfg = Config.load(prior/'config/training.yaml')
    if asdict(saved_cfg.repulsion) != asdict(cfg.repulsion):
        raise ValueError('Recovery training settings differ from completed methods')
    path = prior/'metrics/trained_fields.csv'
    if not path.is_file():
        return {}
    try:
        records = pd.read_csv(path, float_precision='round_trip')
    except pd.errors.EmptyDataError:
        return {}
    if records.method.duplicated().any():
        raise ValueError('Duplicate completed method records')
    completed = {}
    for row in records.to_dict('records'):
        if row['status'] != 'success':
            continue
        method = row['method']
        if method not in selected or row['teacher_variant'] != selected[method]['name']:
            raise ValueError(f'Recovery teacher selection mismatch: {method}')
        folder = prior/'repulsion_training'/method
        required = ['repulsion_field.pt','validation_predictions.npy','validation.csv','losses.csv']
        for name in required:
            if not (folder/name).is_file():
                raise FileNotFoundError(f'Completed method missing artifact: {folder/name}')
        losses = pd.read_csv(folder/'losses.csv')
        if losses.step.tolist() != list(range(1,cfg.repulsion.steps+1)):
            raise ValueError(f'Completed method has incomplete loss history: {method}')
        checkpoint = torch.load(folder/'repulsion_field.pt', map_location='cpu', weights_only=False)
        if checkpoint['hparams'] != hp or checkpoint['teacher'] != selected[method]:
            raise ValueError(f'Completed checkpoint configuration mismatch: {method}')
        model = RepulsionField(**hp).to(args.device).eval()
        model.load_state_dict(checkpoint['state_dict'])
        predictions = torch.cat([model(y[s:s+256],t[s:s+256]) for s in range(0,len(y),256)]).cpu().numpy()
        stored = np.load(folder/'validation_predictions.npy', allow_pickle=False)
        if not np.isfinite(predictions).all() or not np.allclose(predictions,stored,rtol=1e-4,atol=1e-6):
            raise ValueError(f'Completed checkpoint/prediction mismatch: {method}')
        metric = field_metrics(stored,exact.cpu().numpy())
        for name, value in metric.items():
            if name not in row or not np.isclose(value,row[name],rtol=1e-8,atol=1e-12,equal_nan=True):
                raise ValueError(f'Completed checkpoint/exact metrics mismatch: {method}/{name}')
        completed[method] = {**row, 'reused_from':str(prior.resolve())}
    return completed


def load_snapshot(prior, method, signature):
    paths = sorted((prior/'repulsion_training'/method/'checkpoints').glob('step_*.pt'),reverse=True)
    if not paths:
        return None, None
    # 最新の確定snapshotだけ。壊れていれば黙って古い状態へ戻さない。
    path = paths[0]
    value = torch.load(path,map_location='cpu',weights_only=False)
    if value['signature'] != signature:
        raise ValueError(f'Recovery snapshot configuration mismatch: {path}')
    step = value['step']
    if not 0 < step <= signature['repulsion']['steps'] or len(value['training_losses']) != step:
        raise ValueError(f'Invalid checkpoint step/history: {path}')
    return value, path
