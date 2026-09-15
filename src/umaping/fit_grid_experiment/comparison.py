"""読み取り専用の二runから、全手法の同一query比較を新規出力する。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from umaping.config import Config
from umaping.data.preprocessing import load_prepared_dataset
from umaping.dynamics import ReferenceTrajectory
from umaping.inference import InferenceEngine
from umaping.baselines import embed_all_queries
from umaping.experiments import baselines as baseline
from umaping.repulsion_estimators import TrainingQueries, exact_field
from umaping.repulsion_estimators.benchmark import field_metrics, write_json
from umaping.umap_forces import find_ab_params
from .materialize import UPSTREAM, guard_output, sha, copy_run
from .metrics import query_metrics, aggregate_groups, paired_table, temporal_metrics


def compare(uniform, fit_grid, output, device, standard_suite=False, temporal=None, include_baselines=True):
    uniform, fit_grid = Path(uniform).resolve(), Path(fit_grid).resolve()
    output = guard_output(output, uniform, fit_grid)
    hashes = {p: sha(uniform / p) for p in UPSTREAM}
    for rel, digest in hashes.items():
        if sha(fit_grid / rel) != digest:
            raise ValueError(f'paired upstream hash mismatch: {rel}')
    models = {name: sha(run / 'checkpoints/repulsion_field.pt') for name, run in [('uniform_mc',uniform),('fit_grid',fit_grid)]}
    cfg = Config.load(uniform / 'config.yaml')
    prepared = load_prepared_dataset(uniform / 'cache/prepared_dataset.npz')
    trajectory = ReferenceTrajectory.load(uniform / 'memory/reference_trajectory.npz')
    if cfg.dataset.name != 'embryoid_body':
        raise ValueError('this command requires embryoid_body, never a substitute dataset')
    if temporal is None and (uniform / 'temporal_split.json').is_file():
        temporal = json.loads((uniform / 'temporal_split.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('figures','embeddings','exact_field'):
        (output / folder).mkdir()
    status = dict(status='running', source_runs=[str(uniform),str(fit_grid)], upstream_hashes=hashes,
                  model_hashes=models, seed=cfg.seed, device=str(device), reused_baseline_results=False,
                  reference_trustworthiness='identical reference; consistency check only',
                  standard_field_json='legacy noisy teacher diagnostic, NOT the primary exact field metric')
    write_json(output / 'manifest.json', status)
    try:
        if standard_suite:
            from umaping.pipeline import run_evaluation, run_analysis, run_advanced_analysis
            # Existing runs remain read-only, even when they already contain evaluation files.
            for name, source in [('uniform_mc',uniform),('fit_grid',fit_grid)]:
                ckpt = source / 'checkpoints/repulsion_field.pt'
                losses = json.loads((source / 'metrics/repulsion_training.json').read_text())
                workspace = copy_run(source, output / f'{name}_evaluation', ckpt, losses,
                                     dict(source_run=str(source), checkpoint_sha256=sha(ckpt)), hashes)
                for fn in (run_evaluation, run_analysis, run_advanced_analysis):
                    print(f'{name}: {fn.__name__}', flush=True)
                    fn(workspace, device)
        rows, frames, field_rows = [], {}, []
        a,b = find_ab_params(cfg.umap.spread,cfg.umap.min_dist)
        sampler = TrainingQueries(trajectory, device, 1000, cfg.repulsion.jitter_sigma, cfg.seed+9100)
        anchor,t,y = sampler(0)
        exact = exact_field(trajectory,y,t,a,b,clip=cfg.flow.grad_clip)
        np.savez_compressed(output / 'exact_field/validation.npz', anchor_idx=anchor.cpu().numpy(),
                            t=t.cpu().numpy(), y=y.cpu().numpy(), exact=exact.cpu().numpy())

        def record(outcome):
            if isinstance(outcome, baseline.BaselineUnavailable):
                rows.append(dict(method=outcome.name, status='unavailable',reason=outcome.reason))
                pd.DataFrame(rows).to_csv(output / 'aggregate.csv',index=False)
                return
            idx = np.asarray(outcome.extra.get('query_subset_indices', np.arange(len(prepared.query_features))), int)
            if not np.isfinite(outcome.query_embedding).all():
                raise FloatingPointError(f'{outcome.name}: non-finite coordinates')
            np.savez_compressed(output / 'embeddings' / f'{outcome.name}.npz', query_index=idx,
                                reference=outcome.reference_embedding, query=outcome.query_embedding)
            frame, neighbors = query_metrics(outcome.name,prepared,outcome,cfg,idx)
            if 'per_query_latencies' in outcome.extra:
                frame['latency_seconds'] = outcome.extra['per_query_latencies']
                frame['latency_scope'] = 'per_query'
            else:
                frame['latency_scope'] = 'method_mean_only' if outcome.mean_query_latency_seconds is not None else 'unavailable'
            if temporal is not None:
                ro = np.asarray(temporal['reference_ordinals'])
                qo = np.asarray(temporal['query_ordinals'])[idx]
                frame = pd.concat([frame,temporal_metrics(ro,qo,neighbors,temporal)],axis=1)
                cell_type = temporal.get('cell_type_column')
                if cell_type:
                    # String labels: deterministic majority vote; alphabetical tie-break.
                    ref_labels = prepared.reference_labels[cell_type].astype(str)
                    true = prepared.query_labels[cell_type][idx].astype(str)
                    predictions = []
                    for ids in neighbors:
                        values,counts = np.unique(ref_labels[ids],return_counts=True)
                        predictions.append(values[counts.argmax()])
                    frame['cell_type_knn_accuracy'] = (np.asarray(predictions) == true).astype(float)
            frames[outcome.name] = frame
            frame.to_csv(output / f'per_query_{outcome.name}.csv',index=False)
            agg = aggregate_groups(frame,temporal)
            agg['method'] = outcome.name; agg['status']='success'
            agg['fit_time_seconds'] = outcome.fit_time_seconds
            rows.extend(agg.to_dict('records'))
            pd.DataFrame(rows).to_csv(output / 'aggregate.csv',index=False)

        for name,run in [('uniform_mc',uniform),('fit_grid',fit_grid)]:
            print(f'{name}: full OOS inference ({len(prepared.query_features)} queries)',flush=True)
            engine = InferenceEngine.load(run,device,seed=cfg.seed)
            with torch.no_grad():
                pred = torch.cat([engine.repulsion_field(y[i:i+128],t[i:i+128]) for i in range(0,len(y),128)])
            np.save(output / 'exact_field' / f'{name}.npy',pred.cpu().numpy())
            field_rows.append(dict(method=name,**field_metrics(pred.cpu().numpy(),exact.cpu().numpy())))
            emb,ids,weights,latency,_ = embed_all_queries(engine,prepared.query_features)
            outcome = baseline.BaselineResult(name,trajectory.positions[-1],emb,
                         mean_query_latency_seconds=float(latency.mean()),extra={'per_query_latencies':latency})
            record(outcome)
            frames[name]['latency_seconds'] = latency
            frames[name].to_csv(output / f'per_query_{name}.csv',index=False)
        pd.DataFrame(field_rows).to_csv(output / 'exact_field/metrics.csv',index=False)
        paired,stats = paired_table(frames['uniform_mc'],frames['fit_grid'],cfg.seed,temporal)
        paired.to_csv(output / 'paired_per_query.csv',index=False)
        stats.to_csv(output / 'bootstrap.csv',index=False)
        write_json(output / 'bootstrap.json',stats.to_dict('records'))
        if include_baselines:
            # All adapters are reused as-is; unavailable packages are never installed.
            functions = [
                ('standard_umap',lambda: baseline.run_standard_umap_baseline(prepared,cfg)),
                ('reduced_repulsion_umap',lambda: baseline.run_reduced_repulsion_umap_baseline(prepared,cfg)),
                ('weighted_knn',lambda: baseline.run_weighted_knn_baseline(prepared,standard_ref,cfg)),
                ('parametric_umap',lambda: baseline.run_parametric_umap_baseline(prepared,cfg)),
                ('numap',lambda: baseline.run_numap_baseline(prepared,cfg,device=device)),
                ('param_repulsor',lambda: baseline.run_param_repulsor_baseline(prepared,cfg)),
            ]
            standard_ref = None
            for name,fn in functions:
                print(f'baseline: {name}',flush=True)
                try:
                    if name == 'weighted_knn' and standard_ref is None:
                        result = baseline.BaselineUnavailable(name,'standard UMAP reference unavailable')
                    else:
                        result = fn()
                    if name == 'standard_umap' and isinstance(result,baseline.BaselineResult):
                        standard_ref = result.reference_embedding
                    record(result)
                except Exception as exc:
                    rows.append(dict(method=name,status='failed',reason=f'{type(exc).__name__}: {exc}'))
            for name,options in [('fit_grid_oracle_neighbors',dict(neighbor_source='oracle')),
                                 ('no_repulsion',dict(use_repulsion=False)),
                                 ('exact_repulsion_diagnostic',dict(repulsion_mode='exact'))]:
                idx = np.arange(len(prepared.query_features))
                if name == 'exact_repulsion_diagnostic':
                    idx = np.random.default_rng(cfg.seed).choice(len(idx),size=min(cfg.eval.repulsion_oracle_query_subset,len(idx)),replace=False)
                    write_json(output / 'oracle_query_indices.json',idx.tolist())
                print(f'baseline: {name}, queries={len(idx)}',flush=True)
                engine = InferenceEngine.load(fit_grid,device,seed=cfg.seed,**options)
                emb,_,_,latency,_ = embed_all_queries(engine,prepared.query_features[idx])
                record(baseline.BaselineResult(name,trajectory.positions[-1],emb,
                    mean_query_latency_seconds=float(latency.mean()),extra={'query_subset_indices':idx.tolist(),'per_query_latencies':latency}))
        aggregate = pd.DataFrame(rows)
        aggregate.to_csv(output / 'aggregate.csv',index=False)
        aggregate.to_csv(output / 'accuracy_runtime.csv',index=False)
        from .report import make_report
        make_report(output,aggregate,paired,stats,pd.DataFrame(field_rows),frames,temporal)
        for rel,digest in hashes.items():
            if sha(uniform / rel) != digest or sha(fit_grid / rel) != digest:
                raise ValueError(f'source changed during evaluation: {rel}')
        for name,run in [('uniform_mc',uniform),('fit_grid',fit_grid)]:
            if sha(run / 'checkpoints/repulsion_field.pt') != models[name]:
                raise ValueError('model changed during evaluation')
        status['status'] = 'complete'; status['source_hashes_unchanged'] = True
    except BaseException as exc:
        status.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'manifest.json',status)
    return output
