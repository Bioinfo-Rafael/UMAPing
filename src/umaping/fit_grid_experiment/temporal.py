"""実ラベル監査→参照のみ前処理→凍結上流一組→matched B_phi学習。"""
from pathlib import Path
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from .materialize import guard_output, copy_run, UPSTREAM, sha, model_hparams
from umaping.repulsion_estimators.benchmark import write_json


def chronological_order(labels):
    """数値・day_3・E8.5・非重複数値窓・ISO日時のみ自動解釈。"""
    import re
    from datetime import datetime
    numeric = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    parsed = []
    for label in labels:
        match = re.fullmatch(rf"(?:day[_ ]?|time[_ ]?|E)?({numeric})", label, re.IGNORECASE)
        if match:
            parsed.append((float(match[1]), float(match[1]), label, 'numeric_time'))
            continue
        match = re.fullmatch(rf"({numeric})\s*[-–]\s*({numeric})", label)
        if match and float(match[1]) < float(match[2]):
            parsed.append((float(match[1]), float(match[2]), label, 'numeric_window'))
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?", label):
            try:
                dt = datetime.fromisoformat(label)
                value = dt.toordinal() * 86400 + dt.hour * 3600 + dt.minute * 60 + dt.second
                parsed.append((value,value,label,'ISO_datetime'))
                continue
            except ValueError:
                pass
        return None, 'ambiguous temporal label; supply explicit group_order'
    if len({item[3] for item in parsed}) != 1:
        return None, 'mixed time formats; supply explicit group_order'
    parsed.sort(key=lambda item:item[0])
    if any(a[1] >= b[0] for a,b in zip(parsed,parsed[1:])):
        return None, 'equal/overlapping temporal values; supply explicit group_order'
    return [item[2] for item in parsed], parsed[0][3]


def audit_labels(labels, column, group_order=None):
    values = np.asarray(labels)
    if pd.isna(values).any():
        raise ValueError('missing timepoint label')
    values = values.astype(str)
    unique, counts = np.unique(values, return_counts=True)
    if np.isin(unique, ['', 'nan', 'None', '<NA>']).any():
        raise ValueError('empty/missing timepoint label')
    actual = dict(zip(unique.tolist(), counts.tolist()))
    if group_order is not None:
        order = [str(x) for x in group_order]
        if len(set(order)) != len(order) or set(order) != set(unique):
            raise ValueError('group_order must contain every actual label exactly once')
        evidence = 'explicit group_order'
    else:
        order, evidence = chronological_order(unique.tolist())
    reason = None if order else evidence
    if len(unique) < 5:
        reason = f'at least five timepoints required; observed {len(unique)}'
    return dict(grouping_column=column, cells_per_timepoint=actual, ordered_timepoints=order,
                n_timepoints=len(unique), valid_temporal_split=len(unique)>=5 and order is not None,
                reason=reason, chronology_evidence=evidence)


def resolve_column(columns, explicit=None):
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(f'grouping column absent: {explicit}; columns={columns}')
        return explicit
    candidates = {'time','day','sample_labels','timepoint','time_point','stage'}
    found = [column for column in columns if column.lower() in candidates]
    if len(found) != 1:
        raise ValueError(f'ambiguous grouping column; specify --group-column. columns={columns}')
    return found[0]


def load_labels(path, column):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'実データがありません（downloadしません）: {path.resolve()}')
    if path.suffix == '.npz':
        from umaping.data.preprocessing import load_prepared_dataset
        prepared = load_prepared_dataset(path)
        column = resolve_column(list(prepared.reference_labels), column)
        return np.concatenate([prepared.reference_labels[column],prepared.query_labels[column]]), column
    from umaping.data._scrna_common import load_anndata_any_format
    adata = load_anndata_any_format(path)
    column = resolve_column(adata.obs.columns.tolist(), column)
    return adata.obs[column].to_numpy(), column


def audit_file(path, column, group_order=None):
    labels, column = load_labels(path,column)
    report = audit_labels(labels,column,group_order)
    report['source'] = str(Path(path).resolve()); report['source_sha256'] = sha(path)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def temporal_split(labels, cell_ids, column, group_order=None):
    audit = audit_labels(labels,column,group_order)
    if not audit['valid_temporal_split']:
        raise ValueError(audit['reason'])
    cell_ids = np.asarray(cell_ids).astype(str)
    if len(cell_ids) != len(labels) or len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError('cell IDs must be unique and match rows')
    order = audit['ordered_timepoints']
    m = min(max(int(np.floor(.6 * len(order))), 2), len(order)-2)
    reference_ordinals = list(range(1,m)) + [m+1]
    query_ordinals = [m] + list(range(m+2,len(order)+1))
    mapping = {label:i+1 for i,label in enumerate(order)}
    ordinals = np.array([mapping[str(x)] for x in labels])
    ref = np.flatnonzero(np.isin(ordinals,reference_ordinals))
    query = np.flatnonzero(np.isin(ordinals,query_ordinals))
    from umaping.data.preprocessing import assert_no_overlap
    assert_no_overlap(ref,query); assert_no_overlap(cell_ids[ref],cell_ids[query])
    return dict(audit, reference_indices=ref.tolist(),query_indices=query.tolist(),
                reference_cell_ids=cell_ids[ref].tolist(),query_cell_ids=cell_ids[query].tolist(),
                reference_ordinals=ordinals[ref].tolist(),query_ordinals=ordinals[query].tolist(),
                reference_timepoints=[order[i-1] for i in reference_ordinals],
                query_timepoints=[order[i-1] for i in query_ordinals],
                interpolation_timepoint=order[m-1], last_observed_timepoint=order[m],
                extrapolation_timepoints=order[m+1:], interpolation_ordinal=m, last_observed_ordinal=m+1,
                selection_rule='T>=5; m=clip(floor(0.6*T),2,T-2); ref=1..m-1,m+1; query=m,m+2..T')


def train_pair(source, fit_output, cfg, trajectory, w, device, seed):
    from umaping.models.repulsion import RepulsionField
    from umaping.repulsion_estimators import TrainingQueries, UniformMC, GridField
    from umaping.training.flow import train_repulsion_field
    from umaping.umap_forces import find_ab_params
    from umaping.graph import row_degree
    from umaping.utils.io import mark_done
    a,b = find_ab_params(cfg.umap.spread,cfg.umap.min_dist)
    torch.manual_seed(seed)
    hp = model_hparams(cfg)
    initial = copy.deepcopy(RepulsionField(**hp).state_dict())
    initial_path = source / 'checkpoints/repulsion_initial.pt'
    if initial_path.exists() or (source / 'checkpoints/repulsion_field.pt').exists():
        raise FileExistsError('refuse retraining/overwriting existing repulsion checkpoints')
    torch.save(dict(hparams=hp,state_dict=initial),initial_path)
    hashes = {p:sha(source / p) for p in UPSTREAM}
    optional = 'cache/reference_fitted_preprocessing.npz'
    if (source/optional).is_file():
        hashes[optional]=sha(source/optional)
    rows = []
    for method in ('uniform_mc','fit_grid'):
        begin = time.perf_counter()
        common = dict(trajectory=trajectory,a=a,b=b,clip=cfg.flow.grad_clip,device=device,seed=seed+3000)
        teacher = (UniformMC(**common,samples=64) if method == 'uniform_mc' else
                   GridField(**common,grid_size=256,jitter_sigma=cfg.repulsion.jitter_sigma))
        build = time.perf_counter()-begin
        torch.manual_seed(seed)
        model = RepulsionField(**hp)
        model.load_state_dict(copy.deepcopy(initial))
        sampler = TrainingQueries(trajectory,device,cfg.repulsion.batch_size,cfg.repulsion.jitter_sigma,seed+2000)
        begin = time.perf_counter()
        state = train_repulsion_field(model,trajectory,a,b,cfg.repulsion,row_degree(w),device,
                   seed=seed,teacher=teacher,query_sampler=sampler,grad_clip=cfg.flow.grad_clip)
        if not np.isfinite(state.losses).all() or any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError(f'{method} seed={seed} non-finite model/loss')
        elapsed = time.perf_counter()-begin
        provenance = dict(repulsion_teacher=method,grid_size=256 if method=='fit_grid' else None,
                          matched_seed=seed,initial_checkpoint_sha256=sha(initial_path),
                          query_sequence='TrainingQueries(seed=matched_seed+2000, step)',
                          teacher_build_seconds=build,train_seconds=elapsed)
        if method=='uniform_mc':
            target=source
            torch.save(dict(hparams=hp,state_dict=model.state_dict()),source/'checkpoints/repulsion_field.pt')
            write_json(source/'metrics/repulsion_training.json',dict(losses=state.losses))
            metadata=json.loads((source/'metadata.json').read_text());metadata.update(provenance)
            write_json(source/'metadata.json',metadata)
            mark_done(source/'checkpoints','repulsion')
        else:
            # Temporary checkpoint lives in this newly-created uniform run, with explicit provenance.
            ckpt=source/'checkpoints/fit_grid_trained.pt'
            torch.save(dict(hparams=hp,state_dict=model.state_dict(),teacher=dict(name='fit_grid_256',family='fit_grid',grid_size=256)),ckpt)
            provenance.update(checkpoint_sha256=sha(ckpt),source_run=str(source))
            target=copy_run(source,fit_output,ckpt,dict(losses=state.losses),provenance,hashes)
            (target/'temporal_split.json').write_bytes((source/'temporal_split.json').read_bytes())
        loss=pd.Series(state.losses)
        pd.DataFrame(dict(step=np.arange(1,len(loss)+1),loss=loss,rolling_std=loss.rolling(100,min_periods=2).std())).to_csv(target/'metrics/losses.csv',index=False)
        rows.append(dict(method=method,seed=seed,build_seconds=build,train_seconds=elapsed))
    for rel,digest in hashes.items():
        if sha(source/rel)!=digest or sha(fit_output/rel)!=digest:
            raise ValueError('upstream changed during matched training')
    return rows


def run_temporal(args):
    from umaping.config import Config
    from umaping.data._scrna_common import load_anndata_any_format, prepare_continuous_scrna_dataset
    from umaping.data.preprocessing import save_prepared_dataset
    from umaping.pipeline import (init_run_dir,stage_graph,stage_retriever,stage_spectral,stage_flow_dynamics)
    from umaping.utils.io import mark_done
    from umaping.utils.seed import set_seed
    from .comparison import compare
    if not args.seeds or args.seeds[0]!=0 or len(set(args.seeds))!=len(args.seeds) or any(s<0 for s in args.seeds):
        raise ValueError('unique nonnegative matched seeds, seed 0 first, required')
    audit=audit_file(args.data,args.group_column,args.group_order)
    if not audit['valid_temporal_split']:
        raise ValueError(audit['reason'])
    if args.data.suffix=='.npz':
        raise ValueError('temporal preprocessing requires raw annotated counts; easy-split PCA cache cannot be reused')
    args.group_column = audit['grouping_column']
    adata=load_anndata_any_format(args.data)
    split=temporal_split(adata.obs[args.group_column].to_numpy(),adata.obs_names.to_numpy(),args.group_column,args.group_order)
    if args.cell_type_column and (args.cell_type_column==args.group_column or args.cell_type_column not in adata.obs):
        raise ValueError('cell-type annotation must be a separate existing column')
    split['cell_type_column']=args.cell_type_column
    root=args.run_root.resolve(); u=root/'temporal_holdout_uniform'; f=root/'temporal_holdout_fit_grid'; c=root/'comparisons/temporal_holdout'
    planned=[u,f,c]
    for seed in args.seeds[1:]:
        planned.extend([root/f'temporal_seed_{seed}_uniform',root/f'temporal_seed_{seed}_fit_grid',root/'comparisons'/f'temporal_seed_{seed}'])
    for path in planned:
        guard_output(path,args.data)
    cfg=Config.load(args.config)
    if cfg.dataset.name!='embryoid_body':
        raise ValueError('expected embryoid_body config')
    cfg.seed=0;cfg.repulsion.teacher_negative_samples=64
    cfg.dataset.params.update(grouping_column=args.group_column,group_order=split['ordered_timepoints'],query_groups=split['query_timepoints'])
    preprocessing={}
    prepared=prepare_continuous_scrna_dataset(adata,args.group_column,tuple(split['query_timepoints']),
                 cfg.dataset.params.get('n_hvg',2000),cfg.dataset.params.get('n_pcs',50),0,
                 label_columns=(args.cell_type_column,) if args.cell_type_column else (), artifacts=preprocessing)
    cfg.dataset.input_dim=prepared.input_dim
    device=torch.device(args.device)
    u.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',audit=audit,seeds=args.seeds,all_queries_each_seed=True,
                  source_raw_sha256=audit['source_sha256'],temporal_split=split)
    try:
        set_seed(0); layout=init_run_dir(cfg,u)
        write_json(u/'temporal_split.json',split)
        pd.DataFrame([dict(timepoint=label, n_cells=split['cells_per_timepoint'][label], ordinal=i+1,
                          role='reference' if label in split['reference_timepoints'] else 'interpolation' if label==split['interpolation_timepoint'] else 'extrapolation',
                          horizon=max(0,i+1-split['last_observed_ordinal']))
                      for i,label in enumerate(split['ordered_timepoints'])]).to_csv(u/'metrics/temporal_split.csv',index=False)
        write_json(u/'temporal_manifest.json',manifest)
        save_prepared_dataset(u/'cache/prepared_dataset.npz',prepared)
        np.save(u/'memory/reference_features.npy',prepared.reference_features)
        np.savez_compressed(u/'cache/reference_fitted_preprocessing.npz',**preprocessing)
        mark_done(u/'cache','preprocess')
        mu,w=stage_graph(cfg,layout,prepared)
        stage_retriever(cfg,layout,prepared,mu,device)
        _,_,embedding=stage_spectral(cfg,layout,prepared,w,device)
        trajectory=stage_flow_dynamics(cfg,layout,embedding,w,device)
        training_rows=train_pair(u,f,cfg,trajectory,w,device,0)
        compare(u,f,c,device,standard_suite=True,temporal=split)
        summaries=[]
        for seed in args.seeds:
            if seed==0:
                comparison=c
            else:
                source=root/f'temporal_seed_{seed}_uniform'; fit=root/f'temporal_seed_{seed}_fit_grid'
                # Copy upstream only; no checkpoint from seed 0 may be mistaken for seed N.
                source.mkdir(parents=True,exist_ok=False)
                for directory in ('memory','cache','checkpoints','metrics','figures'):
                    (source/directory).mkdir()
                import shutil
                for rel in UPSTREAM:
                    shutil.copy2(u/rel,source/rel)
                shutil.copy2(u/'cache/reference_fitted_preprocessing.npz',source/'cache/reference_fitted_preprocessing.npz')
                write_json(source/'metadata.json',dict(source_run=str(u),matched_seed=seed))
                write_json(source/'temporal_split.json',split)
                training_rows.extend(train_pair(source,fit,cfg,trajectory,w,device,seed))
                comparison=root/'comparisons'/f'temporal_seed_{seed}'
                compare(source,fit,comparison,device,temporal=split,include_baselines=False)
            table=pd.read_csv(comparison/'aggregate.csv')
            table=table[table.method.isin(['uniform_mc','fit_grid'])].copy();table['seed']=seed;summaries.append(table)
        all_seeds=pd.concat(summaries,ignore_index=True)
        all_seeds.to_csv(c/'matched_seed_results.csv',index=False)
        all_seeds.groupby(['method','group'])[['recall_at_15','temporal_neighbor_mae','density_log_distortion']].agg(['mean','std']).to_csv(c/'matched_seed_summary.csv')
        pd.DataFrame(training_rows).to_csv(c/'matched_training_runtime.csv',index=False)
        if sha(args.data)!=audit['source_sha256']:
            raise ValueError('raw data changed during experiment')
        manifest.update(status='complete',source_raw_unchanged=True)
    except BaseException as exc:
        manifest.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(u/'temporal_manifest.json',manifest)
