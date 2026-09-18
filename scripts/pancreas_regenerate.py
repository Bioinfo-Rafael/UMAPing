"""Explicitly authorized coordinate regeneration; never trains a neural network.

Called only by pancreas_visual_review.py --regenerate. Uses the frozen
PreparedDataset, existing reference trajectory, and checkpoints. New UMAP
fit/transform and fresh metrics are stored outside the original run.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def log(event, **fields):
    print(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),event=event,**fields),ensure_ascii=False),flush=True)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    temp=Path(str(path)+'.tmp'); temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); os.replace(temp,path)


def atomic_npz(path,**arrays):
    temp=Path(str(path)+'.tmp.npz'); np.savez_compressed(temp,**arrays); os.replace(temp,path)


def require(ok,message):
    if not ok: raise ValueError(message)


def regenerate(run,output,device='cpu',threads=1,max_seconds=7200):
    # Make the checked-out code authoritative even if .venv is editable-installed
    # against the untouched source worktree.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    import torch
    from umaping.baselines import run_standard_umap
    from umaping.config import Config
    from umaping.data.preprocessing import load_prepared_dataset
    from umaping.evaluation.advanced import multi_k_recall_and_ndcg
    from umaping.inference import InferenceEngine
    from umaping.utils.seed import set_seed

    run=Path(run).resolve(); output=Path(output).resolve()
    require(not output.exists(),'Regeneration output already exists; refusing overwrite')
    require(run!=output and run not in output.parents,'Regeneration may not write inside the original run')
    require(threads>=1 and np.isfinite(max_seconds) and max_seconds>0,'Invalid threads or max_seconds')
    output.mkdir(parents=True); (output/'embeddings').mkdir()
    begin=time.perf_counter(); phase=['audit']; stopped=threading.Event()
    source_files=['config.yaml','cache/prepared_dataset.npz','memory/reference_features.npy',
                  'memory/reference_trajectory.npz','memory/retriever_keys.npy','memory/spectral_calibration.npz',
                  'checkpoints/retriever.pt','checkpoints/spectral_encoder.pt','checkpoints/repulsion_field.pt']
    manifest=dict(status='started',source_run=str(run),started_utc=datetime.now(timezone.utc).isoformat(),
                  neural_network_training=False,preprocessing_refit=False,reference_trajectory_recomputed=False,
                  standard_umap_refit=True,umaping_query_reinference=True,old_metrics_used_for_new_coordinates=False,
                  device=device,threads=threads,max_seconds=max_seconds,sources={},steps_completed=[])
    for package in ('numpy','scipy','scikit-learn','torch','umap-learn','pandas'):
        manifest.setdefault('versions',{})[package]=importlib.metadata.version(package)
    def check():
        if time.perf_counter()-begin>=max_seconds: raise TimeoutError('Regeneration time limit reached')
    def timeout(signum,frame): raise TimeoutError('Regeneration time limit reached')
    def heartbeat():
        while not stopped.wait(30): log('heartbeat',phase=phase[0],elapsed_seconds=round(time.perf_counter()-begin,1))
    def save(): write_json(output/'regeneration.json',manifest)
    thread=threading.Thread(target=heartbeat,daemon=True)
    old_handler=signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max_seconds); thread.start()
    log('regeneration_start',source=str(run),output=str(output),device=device)
    try:
        missing=[str(run/p) for p in source_files if not (run/p).is_file()]
        require(not missing,'Required frozen inputs missing: '+', '.join(missing))
        for rel in source_files:
            check(); manifest['sources'][rel]=digest(run/rel)
        cfg=Config.load(run/'config.yaml')
        require(cfg.dataset.name=='pancreas' and cfg.dataset.params.get('batch_correction') is True,
                'Regeneration is restricted to batch-corrected pancreas')
        require(cfg.umap.embedding_dim==2,'This visual review requires 2D embeddings')
        prepared=load_prepared_dataset(run/'cache/prepared_dataset.npz')
        ref=prepared.reference_features; query=prepared.query_features
        require(len(ref)>=30 and len(query)>0,'Recall@5/15/30 needs >=30 reference cells and nonempty queries')
        require(np.isfinite(ref).all() and np.isfinite(query).all(),'Nonfinite prepared features')
        require(np.array_equal(ref,np.load(run/'memory/reference_features.npy')),'Prepared/reference memory mismatch')
        for split,features,labels in [('reference',ref,prepared.reference_labels),('query',query,prepared.query_labels)]:
            require({'celltype','tech'}<=set(labels),f'{split}: missing labels')
            require(all(len(labels[k])==len(features) for k in ('celltype','tech')),f'{split}: label lengths differ')
        # IDs are bound to source rows BEFORE generating coordinates, not guessed
        # for pre-existing arrays. Each query result is placed at its known ID.
        ref_index=np.arange(len(ref),dtype=np.int64); query_index=np.arange(len(query),dtype=np.int64)
        pd.DataFrame(dict(reference_index=ref_index,celltype=prepared.reference_labels['celltype'],
                          tech=prepared.reference_labels['tech'])).to_csv(output/'reference_ids.csv',index=False)
        pd.DataFrame(dict(query_index=query_index,celltype=prepared.query_labels['celltype'],
                          tech=prepared.query_labels['tech'])).to_csv(output/'query_ids.csv',index=False)
        old_metrics=run/'metrics/advanced_per_query.csv'
        if old_metrics.exists():
            manifest['sources']['metrics/advanced_per_query.csv']=digest(old_metrics)
            old=pd.read_csv(old_metrics)
            for name in ('standard_umap','ours_full'):
                sub=old[old.method==name]
                require(not sub.query_index.duplicated().any(),f'{name}: duplicate source metric IDs')
                require(np.issubdtype(sub.query_index.dtype,np.integer),f'{name}: noninteger source IDs')
                sub=sub.set_index('query_index').sort_index()
                require(np.array_equal(sub.index,query_index),f'{name}: source metric ID set differs')
                require(np.array_equal(sub.label.astype(str),np.asarray(prepared.query_labels['celltype']).astype(str)),
                        f'{name}: source metric labels differ at the same IDs')
            manifest['source_metric_identity_verified']=True
        torch.set_num_threads(threads); set_seed(cfg.seed)
        target=torch.device(device)
        require(target.type=='cpu' or (target.type=='cuda' and torch.cuda.is_available()),'Requested device unavailable')
        manifest.update(n_reference=len(ref),n_query=len(query),seed=cfg.seed,
            umap_config=dict(n_neighbors=cfg.umap.n_neighbors,min_dist=cfg.umap.min_dist,spread=cfg.umap.spread,
                             embedding_dim=cfg.umap.embedding_dim,seed=cfg.seed),
            umap_baseline='Existing umaping.baselines.run_standard_umap: fit(reference), transform(query); library defaults retained',
            umaping_config=dict(n_steps=cfg.flow.n_steps,initial_alpha=cfg.flow.initial_alpha,grad_clip=cfg.flow.grad_clip,
                                negative_sample_rate=cfg.umap.negative_sample_rate,balance_w=.5,balance_scale=2),
            id_contract='IDs assigned from PreparedDataset before computation; exact query loop IDs; standard UMAP preserves input-row order')
        manifest['steps_completed'].append('input_and_id_audit'); save()
        phase[0]='umaping_checkpoint_load'; log('umaping_checkpoint_load')
        engine=InferenceEngine.load(run,target,seed=cfg.seed)
        require(np.array_equal(engine.reference_features,ref),'Engine reference features differ')
        require(engine.trajectory.positions.shape[1:]==(len(ref),2),'Reference trajectory shape differs')
        ours_ref=np.asarray(engine.trajectory.positions[-1],dtype=np.float32).copy()
        ours_query=np.full((len(query),2),np.nan,dtype=np.float32); seconds=np.full(len(query),np.nan)
        phase[0]='umaping_query_inference'; log('umaping_inference_start',n_queries=len(query))
        def sync():
            if target.type=='cuda': torch.cuda.synchronize(target)
        try:
            for i,x in enumerate(query):
                check(); sync(); started=time.perf_counter()
                result=engine.embed_one(x,deadline_check=check)
                sync(); seconds[i]=time.perf_counter()-started; ours_query[i]=result.embedding
                if (i+1)%50==0 or i+1==len(query):
                    atomic_npz(output/'ours_partial.npz',query=ours_query,query_index=query_index,seconds=seconds,
                               prepared_dataset_sha256=np.array(manifest['sources']['cache/prepared_dataset.npz']))
                    log('umaping_progress',completed=i+1,total=len(query),inference_seconds=float(np.nansum(seconds)))
        finally:
            atomic_npz(output/'ours_partial.npz',query=ours_query,query_index=query_index,seconds=seconds,
                       prepared_dataset_sha256=np.array(manifest['sources']['cache/prepared_dataset.npz']))
        require(np.isfinite(ours_query).all(),'Incomplete/nonfinite UMAPing embeddings')
        packs={'ours_full':(ours_ref,ours_query)}
        manifest['umaping_seconds']=float(seconds.sum()); manifest['steps_completed'].append('umaping_inference'); save()
        # Immediately preserve the completed UMAPing result before the UMAP fit.
        def save_embedding(name,reference,queries):
            require(reference.shape==(len(ref),2) and queries.shape==(len(query),2),'Wrong embedding shape')
            require(np.isfinite(reference).all() and np.isfinite(queries).all(),'Nonfinite embedding coordinates')
            atomic_npz(output/'embeddings'/f'{name}.npz',reference=reference,query=queries,
                       reference_index=ref_index,query_index=query_index,
                       reference_celltype=np.asarray(prepared.reference_labels['celltype']).astype(str),
                       query_celltype=np.asarray(prepared.query_labels['celltype']).astype(str),
                       reference_tech=np.asarray(prepared.reference_labels['tech']).astype(str),
                       query_tech=np.asarray(prepared.query_labels['tech']).astype(str),
                       prepared_dataset_sha256=np.array(manifest['sources']['cache/prepared_dataset.npz']))
        save_embedding('ours_full',ours_ref,ours_query)
        del engine
        phase[0]='standard_umap_fit_transform'; log('standard_umap_fit_transform_start',n_reference=len(ref),n_query=len(query))
        check(); started=time.perf_counter()
        standard_ref,standard_query=run_standard_umap(ref,query,**manifest['umap_config'])
        check(); manifest['standard_umap_seconds']=time.perf_counter()-started
        packs['standard_umap']=(standard_ref,standard_query); save_embedding('standard_umap',standard_ref,standard_query)
        manifest['steps_completed'].append('standard_umap_fit_transform'); save()
        log('standard_umap_fit_transform_complete',seconds=manifest['standard_umap_seconds'])
        phase[0]='fresh_recall_evaluation'; frames=[]
        for name in ('standard_umap','ours_full'):
            check(); log('fresh_metrics_start',method=name)
            yref,yquery=packs[name]
            recalls,ndcg=multi_k_recall_and_ndcg(query,ref,yquery,yref,ks=(5,10,15,30))
            frame=pd.DataFrame(dict(method=name,query_index=query_index,label=prepared.query_labels['celltype'],
                                    **{f'recall_at_{k}':v for k,v in recalls.items()},ndcg=ndcg))
            require(np.isfinite(frame.filter(regex='recall_at_|ndcg')).all().all(),'Nonfinite fresh metrics')
            frames.append(frame); frame.to_csv(output/f'{name}_fresh_metrics.csv',index=False)
            log('fresh_metrics_complete',method=name,recall_at_5=float(recalls[5].mean()),recall_at_15=float(recalls[15].mean()))
        pd.concat(frames,ignore_index=True).to_csv(output/'advanced_per_query.csv',index=False)
        manifest['steps_completed'].append('fresh_metrics')
        phase[0]='source_integrity_check'
        for rel,expected in manifest['sources'].items():
            check(); require(digest(run/rel)==expected,f'Original source changed: {rel}')
        manifest.update(status='complete',source_artifacts_unchanged=True)
        log('regeneration_complete',output=str(output))
        return output
    except Exception as exc:
        manifest.update(status='failed',error=str(exc)); log('regeneration_failed',error=str(exc)); raise
    finally:
        stopped.set(); thread.join(timeout=1)
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        manifest.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-begin)
        save()
