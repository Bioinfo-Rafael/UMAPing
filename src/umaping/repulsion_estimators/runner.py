"""凍結Embryo runからteacher単体評価→RepulsionField学習。上流を呼ばない。"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
import time
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from umaping.config import Config
from umaping.dynamics import ReferenceTrajectory
from umaping.models.repulsion import RepulsionField
from umaping.models.retriever import DualEncoder
from umaping.training.flow import train_repulsion_field, RepulsionTrainState
from umaping.umap_forces import find_ab_params
from . import UniformMC, DualImportance, DualTopL, BarnesHut, GridField, TrainingQueries, exact_field, hubness_vector
from .base import synchronize
from .benchmark import evaluate_teacher, field_metrics, write_json
from .recovery import (atomic_json, save_snapshot, training_signature, verify_initial,
                       completed_results, load_snapshot)

METHODS = ('uniform_mc', 'dual_raw_is', 'dual_hub_is', 'dual_topl', 'barnes_hut', 'fit_grid')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_frozen(run_dir, dataset):
    run = Path(run_dir).resolve()
    required = ['config.yaml', 'memory/reference_trajectory.npz', 'memory/retriever_keys.npy', 'checkpoints/retriever.pt']
    features = run / 'memory/reference_features.npy'
    cache = run / 'cache/prepared_dataset.npz'
    if not features.is_file():
        required.append('cache/prepared_dataset.npz')
    missing = [str(run/p) for p in required if not (run/p).is_file()]
    if missing:
        raise FileNotFoundError('凍結成果物がありません。自動生成・別dataset代用はしません:\n'+'\n'.join(missing))
    cfg = Config.load(run/'config.yaml')
    if dataset not in ('embryoid_body', 'embryo') or cfg.dataset.name != dataset:
        raise ValueError(f'Embryo dataset identity mismatch: requested={dataset}, stored={cfg.dataset.name}')
    trajectory = ReferenceTrajectory.load(run/'memory/reference_trajectory.npz')
    p, ts = trajectory.positions, trajectory.times
    if p.ndim != 3 or p.shape[2] != 2 or p.shape[0] != len(ts) or len(ts) < 2 or p.shape[1] < 1:
        raise ValueError('Expected frozen trajectory (T>=2,N>=1,2)')
    if not np.isfinite(p).all() or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('Invalid trajectory values/times')
    if features.is_file():
        x = np.load(features, allow_pickle=False)
        required.append('memory/reference_features.npy')
    else:
        with np.load(cache, allow_pickle=False) as stored:
            x = stored['reference_features'].copy()
    keys = np.load(run/'memory/retriever_keys.npy', allow_pickle=False)
    if x.shape != (p.shape[1], cfg.dataset.input_dim) or len(keys) != len(x) or not np.isfinite(x).all():
        raise ValueError('Frozen feature/key/trajectory ordering or shape mismatch')
    if cfg.repulsion.weight_by_row_mass:
        required.append('memory/graph_symmetric.npz')
    if cache.is_file() and 'cache/prepared_dataset.npz' not in required:
        required.append('cache/prepared_dataset.npz')
    manifest = {str(run/f): sha(run/f) for f in required}
    return cfg, trajectory, x, keys, manifest


@torch.no_grad()
def dual_embeddings(run, x, keys, device):
    checkpoint = torch.load(Path(run)/'checkpoints/retriever.pt', map_location='cpu', weights_only=False)
    model = DualEncoder(**checkpoint['hparams']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    q = []
    for start in range(0, len(x), 256):
        features = torch.as_tensor(x[start:start+256], dtype=torch.float32, device=device)
        q.append(model.encode_query(features).cpu().numpy())
        computed = model.encode_key(features).cpu().numpy()
        if not np.allclose(computed, keys[start:start+256], rtol=1e-4, atol=1e-5):
            raise ValueError('retriever_keys.npy does not match frozen DualEncoder/reference ordering')
    return np.concatenate(q), model.temperature


def create_teacher(spec, common, q, keys, hub, temperature, args):
    method = spec['family']
    if method == 'uniform_mc':
        return UniformMC(**common, samples=64)
    dual = dict(queries=q, keys=keys, temperature=temperature, proposal_temperature=args.proposal_temperature,
                mixture=spec.get('mixture', 1.), score_batch=args.score_batch)
    if method in ('dual_raw_is', 'dual_hub_is'):
        return DualImportance(**common, **dual, samples=64, hubness=hub if method == 'dual_hub_is' else None, beta=args.beta)
    if method == 'dual_topl':
        return DualTopL(**common, **dual, top_l=32, tail_samples=32)
    if method == 'barnes_hut':
        return BarnesHut(**common, theta=spec['theta'])
    if method == 'fit_grid':
        return GridField(**common, grid_size=spec['grid_size'], jitter_sigma=args.jitter_sigma)
    raise ValueError(method)


def specs(args):
    result = [dict(name=m, family=m) for m in METHODS[:4]]
    if args.mixture_ablation:
        result += [dict(name=f'{m}_mix{args.mixture_lambda:g}', family=m, mixture=args.mixture_lambda) for m in ('dual_raw_is', 'dual_hub_is')]
    result += [dict(name=f'barnes_hut_{theta:g}', family='barnes_hut', theta=theta) for theta in args.thetas]
    result += [dict(name=f'fit_grid_{size}', family='fit_grid', grid_size=size) for size in args.grid_sizes]
    if len({s['name'] for s in result}) != len(result):
        raise ValueError('Duplicate estimator variants would overwrite outputs')
    return result


def select_teachers(rows, variants):
    successful = {r['method']: r for r in rows if r['status'] == 'success' and np.isfinite(r['mse'])}
    selected = {}
    for family in METHODS:
        candidates = [s for s in variants if s['family'] == family and s['name'] in successful and s.get('mixture', 1.) == 1.]
        if not candidates:
            continue
        # 単体MSEの最良から10%以内の候補で最速。選択評価への適合はreportで明示。
        best = min(successful[s['name']]['mse'] for s in candidates)
        candidates = [s for s in candidates if successful[s['name']]['mse'] <= 1.1*best+1e-15]
        selected[family] = min(candidates, key=lambda s: successful[s['name']]['runtime_seconds_per_query'])
    return selected


def model_hparams(cfg):
    return dict(embedding_dim=2, hidden_dim=cfg.repulsion.hidden_dim, n_residual_blocks=cfg.repulsion.n_residual_blocks,
                time_embed_dim=cfg.repulsion.time_embed_dim, activation=cfg.repulsion.activation)


def train_models(output, cfg, trajectory, common, q, keys, hub, temperature, args, validation, selected, row_mass, metadata):
    anchor, t, y, exact = validation
    hp = model_hparams(cfg)
    torch.manual_seed(args.seed)
    initial = RepulsionField(**hp).state_dict()
    reused = {}
    if args.reuse_completed:
        verify_initial(args.from_run, initial, hp)
        reused = completed_results(args.from_run, cfg, hp, selected, args, y, t, exact)
    torch.save({'hparams': hp, 'state_dict': initial}, output/'artifacts/initial_repulsion_field.pt')
    metadata['artifact_hashes']['initial_repulsion_field.pt'] = sha(output/'artifacts/initial_repulsion_field.pt')
    metadata['reused_methods'] = list(reused)
    atomic_json(output/'config/manifest.json', metadata)
    sampler = TrainingQueries(trajectory, args.device, cfg.repulsion.batch_size, cfg.repulsion.jitter_sigma, args.seed+2000)
    rows = []
    for method in METHODS:
        print(f'Phase B {method}', flush=True)
        folder = output/'repulsion_training'/method
        if method in reused:
            shutil.copytree(args.from_run/'repulsion_training'/method, folder)
            rows.append(reused[method])
            pd.DataFrame(rows).to_csv(output/'metrics/trained_fields.csv', index=False)
            print(f'  REUSED completed {method}; no training', flush=True)
            continue
        folder.mkdir()
        if method not in selected:
            rows.append(dict(method=method, status='skipped', reason='Phase A failed; see estimator metrics'))
            continue
        losses, validations, diagnostics = [], [], []
        begin = time.perf_counter()
        try:
            teacher = create_teacher(selected[method], common, q, keys, hub, temperature, args)
            model = RepulsionField(**hp)
            model.load_state_dict(copy.deepcopy(initial))
            model.to(args.device)
            signature = training_signature(cfg, hp, selected[method], args)
            snapshot, snapshot_path = (load_snapshot(args.from_run, method, signature)
                                       if args.reuse_completed else (None, None))
            resume_state, optimizer_state = None, None
            prior_train_seconds, prior_checkpoint_seconds = 0., 0.
            if snapshot is not None:
                model.load_state_dict(snapshot['state_dict'])
                optimizer_state = snapshot['optimizer_state']
                resume_state = RepulsionTrainState(step=snapshot['step'], losses=snapshot['training_losses'])
                teacher.rng.bit_generator.state = snapshot['teacher_numpy_rng']
                teacher.generator.set_state(snapshot['teacher_torch_rng'])
                torch.set_rng_state(snapshot['torch_rng'])
                if args.device == 'cuda':
                    torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
                losses, validations, diagnostics = snapshot['losses'], snapshot['validations'], snapshot['diagnostics']
                prior_train_seconds = snapshot['train_seconds']
                prior_checkpoint_seconds = snapshot.get('checkpoint_seconds', 0.)
                print(f'  RESUME {method} from step={resume_state.step}: {snapshot_path}', flush=True)
            synchronize(args.device)
            teacher_build_seconds = time.perf_counter()-begin
            training_begin = time.perf_counter()
            validation_seconds = 0.
            checkpoint_seconds = 0.
            def callback(step, loss, current):
                nonlocal validation_seconds
                losses.append(dict(step=step, loss=loss))
                if step % args.eval_every == 0 or step == cfg.repulsion.steps or step == 0:
                    synchronize(args.device)
                    evaluation_begin = time.perf_counter()
                    current.eval()
                    with torch.no_grad():
                        prediction = torch.cat([current(y[s:s+256], t[s:s+256]) for s in range(0, len(y), 256)])
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError('Non-finite learned field on exact validation queries')
                    synchronize(args.device)
                    elapsed = prior_train_seconds+time.perf_counter()-training_begin-validation_seconds-checkpoint_seconds
                    metric = field_metrics(prediction.cpu().numpy(), exact.cpu().numpy())
                    validations.append(dict(step=step, elapsed_training_seconds=elapsed, **metric))
                    if step > 0:
                        diag = {k:v for k,v in teacher.diagnostics.items() if k != 'weights'}
                        diagnostics.append(dict(step=step, **diag))
                    current.train()
                    validation_seconds += time.perf_counter()-evaluation_begin
                    print(f'  {method} step={step} exact RMSE={metric["rmse"]:.6g}', flush=True)
            def checkpoint_callback(state, current, optimizer):
                nonlocal checkpoint_seconds
                if state.step % args.eval_every and state.step != cfg.repulsion.steps:
                    return
                synchronize(args.device)
                begin_checkpoint = time.perf_counter()
                elapsed = prior_train_seconds+begin_checkpoint-training_begin-validation_seconds-checkpoint_seconds
                path = folder/'checkpoints'/f'step_{state.step:08d}.pt'
                save_snapshot(path, dict(signature=signature, step=state.step,
                    state_dict=current.state_dict(), optimizer_state=optimizer.state_dict(),
                    training_losses=state.losses, losses=losses, validations=validations, diagnostics=diagnostics,
                    teacher_numpy_rng=teacher.rng.bit_generator.state, teacher_torch_rng=teacher.generator.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [],
                    train_seconds=elapsed, checkpoint_seconds=prior_checkpoint_seconds+checkpoint_seconds,
                    validation_sha256=metadata['artifact_hashes']['validation.npz']))
                pd.DataFrame(losses).to_csv(folder/'losses.csv', index=False)
                pd.DataFrame(validations).to_csv(folder/'validation.csv', index=False)
                metadata['progress'] = dict(method=method, step=state.step, checkpoint=str(path))
                atomic_json(output/'config/manifest.json', metadata)
                checkpoint_seconds += time.perf_counter()-begin_checkpoint
                print(f'  CHECKPOINT {method} step={state.step}', flush=True)
            if resume_state is None:
                callback(0, np.nan, model)
                losses.clear()
            elif snapshot['validation_sha256'] != metadata['artifact_hashes']['validation.npz']:
                raise ValueError('Snapshot validation hash mismatch')
            else:
                (folder/'checkpoints').mkdir(exist_ok=True)
                shutil.copy2(snapshot_path, folder/'checkpoints'/snapshot_path.name)
            train_repulsion_field(model, trajectory, common['a'], common['b'], cfg.repulsion, row_mass,
                                  torch.device(args.device), seed=args.seed, progress=False, grad_clip=cfg.flow.grad_clip,
                                  teacher=teacher, query_sampler=sampler, callback=callback, resume_state=resume_state,
                                  optimizer_state=optimizer_state, checkpoint_callback=checkpoint_callback)
            synchronize(args.device)
            train_seconds = prior_train_seconds+time.perf_counter()-training_begin-validation_seconds-checkpoint_seconds
            model.eval()
            with torch.no_grad():
                prediction = torch.cat([model(y[s:s+256], t[s:s+256]) for s in range(0, len(y), 256)])
                # Warmup済みの同一バッチサイズによるforwardのみの平均時間。
                synchronize(args.device)
                begin_forward = time.perf_counter()
                for _ in range(10):
                    for s in range(0, len(y), 256):
                        model(y[s:s+256], t[s:s+256])
                synchronize(args.device)
                forward_seconds = (time.perf_counter()-begin_forward)/(10*len(y))
            np.save(folder/'validation_predictions.npy', prediction.cpu().numpy())
            torch.save({'hparams': hp, 'state_dict': model.cpu().state_dict(), 'teacher': selected[method]}, folder/'repulsion_field.pt')
            loss_array = np.array([r['loss'] for r in losses])
            window = min(100, len(loss_array))
            late = loss_array[-max(1, len(loss_array)//5):]
            rolling = pd.Series(loss_array).rolling(window, min_periods=2).std()
            for record, value in zip(losses, rolling):
                record['rolling_std'] = value
            threshold = .5 * validations[0]['rmse']
            reached = next((v for v in validations if v['step'] > 0 and v['rmse'] <= threshold), None)
            row = dict(method=method, status='success', teacher_variant=selected[method]['name'],
                       resumed_from_step=snapshot['step'] if snapshot else 0,
                       checkpoint_seconds=prior_checkpoint_seconds+checkpoint_seconds,
                       teacher_build_seconds=teacher_build_seconds, train_seconds=train_seconds,
                       inference_seconds_per_query=forward_seconds, rolling_std_mean=float(rolling.mean()),
                       late_loss_cv=float(late.std()/max(abs(late.mean()), 1e-12)),
                       convergence_rmse_threshold=threshold,
                       first_step_half_initial_rmse=reached['step'] if reached else np.nan,
                       seconds_to_half_initial_rmse=reached['elapsed_training_seconds'] if reached else np.nan,
                       best_validation_rmse=min(v['rmse'] for v in validations), **field_metrics(prediction.cpu().numpy(), exact.cpu().numpy()))
            rows.append(row)
        except Exception as exc:
            rows.append(dict(method=method, status='failed', reason=f'{type(exc).__name__}: {exc}'))
            print(f'  FAILED {method}: {exc}', flush=True)
        finally:
            pd.DataFrame(losses).to_csv(folder/'losses.csv', index=False)
            pd.DataFrame(validations).to_csv(folder/'validation.csv', index=False)
            pd.DataFrame(diagnostics).to_csv(folder/'teacher_diagnostics.csv', index=False)
            pd.DataFrame(rows).to_csv(output/'metrics/trained_fields.csv', index=False)
    return rows


def run(args):
    run_dir, output = args.frozen_run.resolve(), args.output_dir.resolve()
    if output.exists() or output == run_dir or run_dir in output.parents or output in run_dir.parents:
        raise ValueError(f'新規かつ入力と重ならないoutput-dirが必要です: {output}')
    cfg, trajectory, x, keys, manifest = inspect_frozen(run_dir, args.dataset)
    variants = specs(args)
    if args.steps is not None:
        cfg.repulsion.steps = args.steps
    if args.batch_size is not None:
        cfg.repulsion.batch_size = args.batch_size
    cfg.repulsion.teacher_negative_samples = 64
    args.jitter_sigma = cfg.repulsion.jitter_sigma
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.from_run and (output == args.from_run.resolve() or args.from_run.resolve() in output.parents or output in args.from_run.resolve().parents):
        raise ValueError('Output overlaps previous benchmark')
    output.mkdir(parents=True)
    for folder in ('config', 'estimator_benchmark', 'repulsion_training', 'metrics', 'figures', 'artifacts'):
        (output/folder).mkdir()
    shutil.copy2(run_dir/'config.yaml', output/'config/source_config.yaml')
    cfg.save(output/'config/training.yaml')
    repo = Path(__file__).resolve().parents[3]
    metadata = dict(timestamp=datetime.now(timezone.utc).isoformat(), dataset=args.dataset,
                    input_dim=x.shape[1], n_reference=len(x), trajectory_shape=list(trajectory.positions.shape),
                    trajectory_times=trajectory.times.tolist(), sources=manifest, seed=args.seed, args=vars(args),
                    git_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(),
                    status='running', device=args.device, numpy=np.__version__, torch=torch.__version__,
                    temporal_caveat='BH/grid interpolate checkpoint FIELDS; exact oracle interpolates reference POSITIONS. Temporal approximation error remains.')
    write_json(output/'config/manifest.json', metadata)
    try:
        if args.stage == 'train':
            if args.from_run is None:
                raise ValueError('--stage train requires --from-run (completed Phase A)')
            prior = json.loads((args.from_run/'config/manifest.json').read_text())
            allowed = ('estimators_complete','complete','completed_with_failures')
            if args.reuse_completed:
                allowed += ('running','interrupted','failed')
            if prior['sources'] != manifest or prior['status'] not in allowed:
                raise ValueError('Phase A source hashes/status mismatch')
            for key in ('proposal_temperature', 'beta', 'hubness_k', 'seed', 'jitter_sigma', 'thetas', 'grid_sizes', 'mixture_ablation', 'mixture_lambda'):
                if prior['args'][key] != vars(args)[key]:
                    raise ValueError(f'Phase A settings mismatch: {key}')
            for name in ('dual_queries.npy','dual_keys.npy','hubness.npy','validation.npz'):
                path = args.from_run/'artifacts'/name
                if sha(path) != prior['artifact_hashes'][name]:
                    raise ValueError(f'Phase A cached artifact hash mismatch: {name}')
            q = np.load(args.from_run/'artifacts/dual_queries.npy', allow_pickle=False)
            hub = np.load(args.from_run/'artifacts/hubness.npy', allow_pickle=False)
            if not np.array_equal(keys, np.load(args.from_run/'artifacts/dual_keys.npy', allow_pickle=False)):
                raise ValueError('Phase A frozen keys mismatch')
            temperature = prior['dual_temperature']
            metadata.update(dual_cache_seconds=0., hubness_build_seconds=0., cache_reused_from=str(args.from_run.resolve()))
        else:
            begin = time.perf_counter()
            q, temperature = dual_embeddings(run_dir, x, keys, args.device)
            q_tensor = torch.as_tensor(q, device=args.device)
            k_tensor = torch.as_tensor(keys, device=args.device)
            synchronize(args.device)
            metadata['dual_cache_seconds'] = time.perf_counter()-begin
            begin = time.perf_counter()
            hub = hubness_vector(q_tensor, k_tensor, temperature, args.hubness_k).cpu().numpy()
            synchronize(args.device)
            metadata['hubness_build_seconds'] = time.perf_counter()-begin
        metadata['dual_temperature'] = temperature
        np.save(output/'artifacts/dual_queries.npy', q)
        np.save(output/'artifacts/dual_keys.npy', keys)
        np.save(output/'artifacts/hubness.npy', hub)
        a, b = find_ab_params(cfg.umap.spread, cfg.umap.min_dist)
        common = dict(trajectory=trajectory, a=a, b=b, device=args.device, clip=cfg.flow.grad_clip, seed=args.seed+3000)
        metadata['force'] = dict(a=a, b=b, epsilon=1e-3, clip=cfg.flow.grad_clip, normalization='1/N, self included')
        if args.stage == 'train':
            shutil.copytree(args.from_run/'estimator_benchmark', output/'estimator_benchmark', dirs_exist_ok=True)
            shutil.copy2(args.from_run/'artifacts/validation.npz', output/'artifacts/validation.npz')
            with np.load(output/'artifacts/validation.npz') as saved:
                anchor = torch.as_tensor(saved['anchor'], device=args.device)
                t = torch.as_tensor(saved['t'], device=args.device)
                y = torch.as_tensor(saved['y'], device=args.device)
                exact = torch.as_tensor(saved['exact'], device=args.device)
            rows = json.loads((args.from_run/'metrics/estimator_metrics.json').read_text())
            metadata['phase_a_source'] = str(args.from_run.resolve())
        else:
            anchor, t, y = TrainingQueries(trajectory, args.device, args.validation_queries, cfg.repulsion.jitter_sigma, args.seed+1000)(0)
            print(f'Phase A exact oracle: {len(y)} queries, N={len(x)}', flush=True)
            begin = time.perf_counter()
            exact = exact_field(trajectory, y, t, a, b, cfg.flow.grad_clip)
            synchronize(args.device)
            metadata['exact_oracle_seconds'] = time.perf_counter()-begin
            np.savez(output/'artifacts/validation.npz', anchor=anchor.cpu().numpy(), t=t.cpu().numpy(), y=y.cpu().numpy(), exact=exact.cpu().numpy())
            rows = []
            for index, spec in enumerate(variants):
                print(f'Phase A build {spec["name"]}', flush=True)
                begin = time.perf_counter()
                teacher = None
                try:
                    teacher = create_teacher(spec, {**common, 'seed': args.seed+4000+index}, q, keys, hub, temperature, args)
                    synchronize(args.device)
                    setup = time.perf_counter()-begin
                    row = evaluate_teacher(spec['name'], teacher, anchor, t, y, exact, output/'estimator_benchmark', args.repetitions, args.eval_batch_size)
                    row.update(build_seconds=setup, family=spec['family'])
                    if isinstance(teacher, GridField):
                        np.savez(output/'artifacts'/f'{spec["name"]}_field.npz', fields=teacher.fields.cpu().numpy(), lower=teacher.lower, upper=teacher.upper)
                except Exception as exc:
                    row = dict(method=spec['name'], family=spec['family'], status='failed', reason=f'{type(exc).__name__}: {exc}')
                    print(f'FAILED {spec["name"]}: {exc}', flush=True)
                finally:
                    del teacher
                rows.append(row)
                write_json(output/'metrics/estimator_metrics.json', rows)
                pd.DataFrame(rows).to_csv(output/'metrics/estimator_metrics.csv', index=False)
        write_json(output/'metrics/estimator_metrics.json', rows)
        pd.DataFrame(rows).to_csv(output/'metrics/estimator_metrics.csv', index=False)
        selected = select_teachers(rows, variants)
        metadata['selected'] = selected
        # Phase A gate: 全点和・Uniformの有限性、失敗・低ESS/巨大weightを必ず記録してから学習。
        uniform = next((r for r in rows if r['method'] == 'uniform_mc' and r['status'] == 'success'), None)
        if not torch.isfinite(exact).all() or uniform is None:
            raise FloatingPointError('Phase A gate failed: exact oracle or Uniform invalid; training prohibited')
        issues = []
        for row in rows:
            if row['status'] != 'success':
                issues.append(f"{row['method']}: {row.get('reason')}")
            elif row.get('ess', 64) < 6.4 or row.get('max_weight', 1) > 100:
                issues.append(f"{row['method']}: low ESS / weight explosion: ESS={row.get('ess')}, max={row.get('max_weight')}")
        metadata['phase_a_issues'] = issues
        write_json(output/'metrics/phase_a_review.json', dict(uniform_rmse=uniform['rmse'], selected=selected, issues=issues,
                    note='低ESSや有限な大誤差は隠さず学習比較に残す。NaN/support喪失したteacherだけskip。'))
        print('Phase A reviewed:', json.dumps(issues, ensure_ascii=False), flush=True)
        if args.reuse_completed:
            review = json.loads((args.from_run/'metrics/phase_a_review.json').read_text())
            if review['selected'] != selected:
                raise ValueError('Interrupted Phase A review/selection mismatch')
        metadata['artifact_hashes'] = {p.name: sha(p) for p in (output/'artifacts').iterdir() if p.is_file()}
        metadata['phase_a_complete'] = True
        atomic_json(output/'config/manifest.json', metadata)
        trained = []
        if args.stage != 'estimators':
            row_mass = np.ones(len(x), dtype=np.float32)
            if cfg.repulsion.weight_by_row_mass:
                row_mass = np.asarray(sp.load_npz(run_dir/'memory/graph_symmetric.npz').sum(1)).ravel()
            trained = train_models(output, cfg, trajectory, common, q, keys, hub, temperature, args, (anchor,t,y,exact), selected, row_mass, metadata)
        metadata['status'] = 'estimators_complete' if args.stage == 'estimators' else 'complete'
        metadata['all_six_training_succeeded'] = len(trained) == 6 and all(r['status'] == 'success' for r in trained)
        if args.stage != 'estimators' and not metadata['all_six_training_succeeded']:
            metadata['status'] = 'completed_with_failures'
        write_json(output/'metrics/trained_fields.json', trained)
        from .report import render
        render(output, metadata, rows, trained)
    except KeyboardInterrupt:
        metadata.update(status='interrupted', reason='KeyboardInterrupt')
        raise
    except Exception as exc:
        metadata.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata['source_hashes_unchanged'] = all(sha(p) == h for p,h in manifest.items())
        metadata['artifact_hashes'] = {p.name: sha(p) for p in (output/'artifacts').iterdir() if p.is_file()}
        atomic_json(output/'config/manifest.json', metadata)
        if not metadata['source_hashes_unchanged']:
            raise RuntimeError('Frozen source changed during experiment')
    print(f'実行終了 ({metadata["status"]}): {output}\nレポート: {output / "report.md"}', flush=True)
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-run', type=Path, required=True)
    p.add_argument('--dataset', choices=['embryoid_body','embryo'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stage', choices=['all','estimators','train'], default='all')
    p.add_argument('--from-run', type=Path)
    p.add_argument('--reuse-completed', action='store_true', help='中断runの完了手法を検証してコピーし、未完了手法だけ実行')
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--validation-queries', type=int, default=1000)
    p.add_argument('--repetitions', type=int, default=50)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--score-batch', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--proposal-temperature', type=float, default=1.)
    p.add_argument('--hubness-k', type=int, default=32)
    p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--mixture-ablation', action='store_true')
    p.add_argument('--mixture-lambda', type=float, default=.9)
    p.add_argument('--thetas', nargs='+', type=float, default=[.2,.5,.8])
    p.add_argument('--grid-sizes', nargs='+', type=int, default=[64,128,256])
    args = p.parse_args(argv)
    if args.reuse_completed and (args.stage != 'train' or args.from_run is None):
        p.error('--reuse-completed requires --stage train --from-run')
    for key in ('validation_queries','repetitions','eval_batch_size','score_batch','eval_every','steps','batch_size'):
        value = getattr(args, key)
        if value is not None and value < 1:
            p.error(f'{key} must be positive')
    if args.repetitions < 2:
        p.error('At least two repetitions required to estimate variance')
    if not 0 <= args.mixture_lambda < 1:
        p.error('mixture-lambda must be in [0,1); pure lambda=1 remains in the main comparison')
    run(args)


if __name__ == '__main__':
    main()
