"""検証を完了してから独立したrunを作る。既存runには書き込まない。"""
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
from umaping.config import Config
from umaping.models.repulsion import RepulsionField
from umaping.repulsion_estimators.runner import sha, model_hparams, inspect_frozen
from umaping.data.preprocessing import load_prepared_dataset

HASHED = ('config.yaml', 'memory/reference_features.npy', 'memory/reference_trajectory.npz',
          'memory/retriever_keys.npy', 'checkpoints/retriever.pt', 'cache/prepared_dataset.npz')
UPSTREAM = HASHED + ('memory/graph_directed.npz', 'memory/graph_symmetric.npz',
                    'checkpoints/spectral_encoder.pt', 'memory/spectral_calibration.npz',
                    'cache/reference_spectral_embedding.npy')


def guard_output(output, *sources):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'既存出力は変更しません: {output}')
    resolved = output.resolve()
    for source in sources:
        source = Path(source).resolve()
        if source == resolved or source in resolved.parents or resolved in source.parents:
            raise ValueError(f'source/output overlap: {source}, {resolved}')
    return resolved


def convert_losses(path, expected_steps):
    table = pd.read_csv(path)
    if not {'step', 'loss'} <= set(table):
        raise ValueError('losses.csv requires step, loss')
    if not np.array_equal(table.step.to_numpy(), np.arange(1, expected_steps + 1)):
        raise ValueError('loss history must contain every step in order')
    if not np.isfinite(table.loss).all():
        raise ValueError('non-finite training loss')
    return {'losses': table.loss.tolist()}


def validate(source, benchmark):
    source, benchmark = Path(source).resolve(), Path(benchmark).resolve()
    manifest = json.loads((benchmark / 'config/manifest.json').read_text())
    origin = Path(manifest['args']['frozen_run'])
    recorded = {}
    for name, digest in manifest['sources'].items():
        relative = Path(name).relative_to(origin).as_posix()
        if '..' in Path(relative).parts or relative in recorded:
            raise ValueError('invalid/duplicate manifest source path')
        recorded[relative] = digest
    missing = [str(source / p) for p in UPSTREAM if not (source / p).is_file()]
    if missing:
        raise FileNotFoundError('必要な凍結成果物がありません:\n' + '\n'.join(missing))
    if not set(HASHED) <= recorded.keys():
        raise ValueError('manifest lacks required source hashes')
    for rel, digest in recorded.items():
        if sha(source / rel) != digest:
            raise ValueError(f'source hash mismatch: {source / rel}')
    cfg, trajectory, features, keys, _ = inspect_frozen(source, 'embryoid_body')
    prepared = load_prepared_dataset(source / 'cache/prepared_dataset.npz')
    if not np.array_equal(prepared.reference_features, features):
        raise ValueError('PreparedDataset/reference_features mismatch')
    if list(trajectory.positions.shape) != manifest['trajectory_shape']:
        raise ValueError('trajectory shape mismatch')
    if manifest['status'] != 'complete' or not manifest['source_hashes_unchanged']:
        raise ValueError('benchmark not complete or sources changed')
    spec = {'name': 'fit_grid_256', 'family': 'fit_grid', 'grid_size': 256}
    if manifest['selected'].get('fit_grid') != spec:
        raise ValueError('selected teacher is not fit_grid_256')
    checkpoint = benchmark / 'repulsion_training/fit_grid/repulsion_field.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ckpt['hparams'] != model_hparams(cfg) or ckpt.get('teacher') != spec:
        raise ValueError('incompatible FitGrid checkpoint architecture/teacher')
    model = RepulsionField(**model_hparams(cfg))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('non-finite FitGrid checkpoint')
    args = manifest['args']
    if (args.get('steps') not in (None, cfg.repulsion.steps)
            or args.get('batch_size') not in (None, cfg.repulsion.batch_size)
            or args.get('jitter_sigma') != cfg.repulsion.jitter_sigma
            or args['seed'] != cfg.seed):
        raise ValueError('benchmark training settings differ from source config')
    losses_path = benchmark / 'repulsion_training/fit_grid/losses.csv'
    losses = convert_losses(losses_path, cfg.repulsion.steps)
    hashes = {p: sha(source / p) for p in UPSTREAM}
    for optional in ('cache/reference_fitted_preprocessing.npz',):
        if (source / optional).is_file():
            hashes[optional] = sha(source / optional)
    from umaping.umap_forces import find_ab_params
    a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)
    force = manifest.get('force', {})
    if (not np.isclose(force.get('a', np.nan), a) or not np.isclose(force.get('b', np.nan), b)
            or force.get('epsilon') != .001 or force.get('clip') != cfg.flow.grad_clip):
        raise ValueError('benchmark force settings differ from source config')
    from umaping.inference import InferenceEngine
    InferenceEngine.load(source, torch.device('cpu'))
    return cfg, manifest, checkpoint, losses_path, losses, hashes


def copy_run(source, output, checkpoint, losses, provenance, hashes):
    """Copy only upstream/model/training artifacts; never stale evaluation outputs."""
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    output = guard_output(output, source, checkpoint.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.materialize-', dir=output.parent))
    try:
        for sub in ('memory', 'cache', 'checkpoints', 'metrics', 'figures', 'provenance'):
            (stage / sub).mkdir()
        for rel, digest in hashes.items():
            shutil.copy2(source / rel, stage / rel)
            if sha(stage / rel) != digest or sha(source / rel) != digest:
                raise ValueError(f'source changed during copy: {rel}')
        shutil.copy2(checkpoint, stage / 'checkpoints/repulsion_field.pt')
        if sha(stage / 'checkpoints/repulsion_field.pt') != provenance['checkpoint_sha256']:
            raise ValueError('checkpoint changed during copy')
        for name in ('retriever_training.json', 'spectral_training.json'):
            if (source / 'metrics' / name).is_file():
                shutil.copy2(source / 'metrics' / name, stage / 'metrics' / name)
        (stage / 'metrics/repulsion_training.json').write_text(json.dumps(losses, ensure_ascii=False))
        if 'losses_source' in provenance:
            shutil.copy2(provenance['losses_source'], stage / 'provenance/fit_grid_losses.csv')
            if sha(stage / 'provenance/fit_grid_losses.csv') != provenance['losses_sha256']:
                raise ValueError('losses changed during copy')
        metadata = dict(provenance, artifact_hashes=hashes,
                        stages_completed=['preprocess','graph','retriever','spectral','flow_dynamics','repulsion'])
        (stage / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        from umaping.utils.io import mark_done
        for directory, names in [('cache',['preprocess']), ('memory',['graph','flow_dynamics']),
                                 ('checkpoints',['retriever','spectral','repulsion'])]:
            for name in names:
                mark_done(stage / directory, name)
        # Reserve destination exclusively; a competing writer cannot be overwritten.
        output.mkdir(exist_ok=False)
        try:
            for child in stage.iterdir():
                shutil.move(str(child), output / child.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return output


def materialize(source, benchmark, output):
    guard_output(output, source, benchmark)
    cfg, manifest, ckpt, lossfile, losses, hashes = validate(source, benchmark)
    provenance = dict(repulsion_teacher='fit_grid', grid_size=256, source_run=str(Path(source).resolve()),
                      source_benchmark=str(Path(benchmark).resolve()), source_commit=manifest['git_commit'],
                      checkpoint_source=str(ckpt), checkpoint_sha256=sha(ckpt),
                      losses_source=str(lossfile), losses_sha256=sha(lossfile),
                      reused_artifacts=list(hashes), replaced_artifact='checkpoints/repulsion_field.pt')
    return copy_run(source, output, ckpt, losses, provenance, hashes)
