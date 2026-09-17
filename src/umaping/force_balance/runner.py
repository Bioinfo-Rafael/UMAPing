"""Read-only frozen artifacts; all new outputs isolated under one run ID."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from umaping.config import Config
from umaping.data.preprocessing import load_prepared_dataset
from umaping.inference import InferenceEngine
from umaping.graph import chunked_exact_knn
from umaping.fit_grid_experiment.materialize import UPSTREAM
from .core import (WEIGHTS, Deadline, write_json, utcnow, stratified_ids,
                   select_weight, paired_bootstrap, metrics_from_neighbors)

SECTIONS = {'search':'01_balance_search','confirmation':'02_selected_ratio_confirmation',
            'transfer':'03_transfer_to_existing_split','forces':'04_force_contributions'}
REQUIRED = tuple(p for p in UPSTREAM if 'graph_' not in p and 'reference_spectral_embedding' not in p)
RUNS = {'temporal0':('temporal_holdout_uniform','temporal_holdout_fit_grid'),
        'temporal1':('temporal_seed_1_uniform','temporal_seed_1_fit_grid'),
        'existing':('main','fit_grid')}


def sha(path, check=lambda: None):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while chunk:=f.read(4*1024*1024):
            check(); h.update(chunk)
    return h.hexdigest()


def inspect(root, deadline):
    inventory=[]; contexts={}
    for context,names in RUNS.items():
        paths=[root/'runs/embryoid_body'/name for name in names]
        ok=True
        for run in paths:
            for rel in REQUIRED+('checkpoints/repulsion_field.pt',):
                exists=(run/rel).is_file(); ok &= exists
                inventory.append(dict(context=context,run=str(run),artifact=rel,exists=exists))
        if not ok:
            continue
        hashes=[]
        for run in paths:
            hashes.append({p:sha(run/p,deadline.check) for p in REQUIRED})
        if hashes[0]!=hashes[1]:
            raise ValueError(f'{context}: teacher upstream artifacts differ')
        metadata=[json.loads((p/'metadata.json').read_text()) for p in paths]
        if context.startswith('temporal'):
            expected=int(context[-1])
            if any(m.get('matched_seed')!=expected for m in metadata):
                raise ValueError(f'{context}: training seed metadata mismatch')
        contexts[context]=dict(paths=paths,hashes=hashes[0],metadata=metadata,
            model_hashes=[sha(p/'checkpoints/repulsion_field.pt',deadline.check) for p in paths])
    if 'temporal0' in contexts and 'temporal1' in contexts:
        if contexts['temporal0']['hashes']!=contexts['temporal1']['hashes']:
            raise ValueError('seed 1 upstream differs: shared query confirmation not safe')
    return inventory,contexts


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def atomic_npz(path, **arrays):
    temp=path.with_suffix('.tmp.npz'); np.savez(temp,**arrays); os.replace(temp,path)


class Experiment:
    def __init__(self,args):
        self.args=args; self.root=Path(args.source_root).resolve(); self.out=Path(args.output_root).resolve()
        self.protocol_path=Path(args.protocol).resolve()
        self.protocol=json.loads(self.protocol_path.read_text()); self.deadline=Deadline(self.protocol)
        self.run_id=self.protocol['run_id']; self.work=self.out/'experiments/force_balance'/self.run_id
        self.report=self.out/'Final_analysis'/('force_balance_'+self.run_id)
        for p in [self.work/'configs',self.work/'cache',self.work/'logs',self.report/'99_provenance']:
            p.mkdir(parents=True,exist_ok=True)
        for section in SECTIONS.values(): (self.report/section).mkdir(exist_ok=True)
        saved=self.work/'configs/protocol.json'
        if saved.exists() and json.loads(saved.read_text())!=self.protocol:
            raise ValueError('Existing original protocol differs; deadlines cannot reset')
        write_json(saved,self.protocol)
        self.device=torch.device(args.device); self.summaries=[]; self.pairs=[]; self.force_rows=[]
        self.reference_metrics={}
        self.status={'status':'started','execution_started_utc':utcnow(),'protocol':self.protocol,
                     'command':sys.argv,'completed_conditions':[],'optional_D':'not run: A–C priority',
                     'optional_E':'not run: A–C priority'}
        code_root=Path(__file__).resolve().parents[3]
        self.status['code_commit']=subprocess.run(['git','rev-parse','HEAD'],cwd=code_root,
            capture_output=True,text=True,timeout=10).stdout.strip()

    def audit(self):
        self.deadline.check()
        inventory,self.contexts=inspect(self.root,self.deadline)
        pd.DataFrame(inventory).to_csv(self.report/'99_provenance/input_inventory.csv',index=False)
        write_json(self.report/'99_provenance/artifacts.json',{
            k:dict(v,paths=list(map(str,v['paths']))) for k,v in self.contexts.items()})
        gpu={}
        for name,cmd in [('memory',['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv']),
                         ('processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])]:
            try: gpu[name]=subprocess.run(cmd,text=True,capture_output=True,timeout=10).stdout
            except (FileNotFoundError,subprocess.TimeoutExpired) as e: gpu[name]=str(e)
        write_json(self.report/'99_provenance/gpu_inventory.json',gpu)
        if 'temporal0' not in self.contexts:
            raise FileNotFoundError('Temporal seed 0 frozen inputs missing; see input_inventory.csv. No upstream reconstruction or data substitution performed.')
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')

    def load_context(self,key):
        context=self.contexts[key]; path=context['paths'][0]
        prepared=load_prepared_dataset(path/'cache/prepared_dataset.npz')
        ref=np.load(path/'memory/reference_features.npy')
        if not np.array_equal(ref,prepared.reference_features):
            raise ValueError('PreparedDataset/reference memory mismatch')
        split=None
        if key.startswith('temporal'):
            split=json.loads((path/'temporal_split.json').read_text())
            labels=np.asarray(prepared.query_labels[split['grouping_column']]).astype(str)
            qids=np.asarray(split['query_cell_ids']).astype(str)
            rids=np.asarray(split['reference_cell_ids']).astype(str)
            expected=np.asarray(split['ordered_timepoints'])[np.asarray(split['query_ordinals'])-1]
            if not np.array_equal(labels,expected) or len(qids)!=len(labels):
                raise ValueError('Temporal labels/IDs do not match prepared rows')
            if len(set(qids))!=len(qids) or np.intersect1d(qids,rids).size:
                raise ValueError('Temporal cell ID overlap or duplicates')
        else:
            cfg=Config.load(path/'config.yaml')
            column=cfg.dataset.params.get('grouping_column','stage')
            if column not in prepared.query_labels:
                raise ValueError(f'Existing split actual grouping column missing: {column}')
            labels=np.asarray(prepared.query_labels[column]).astype(str)
            # Existing PreparedDataset has no cell IDs: identity is hash + row index.
            qids=np.array([context['hashes']['cache/prepared_dataset.npz']+':'+str(i) for i in range(len(labels))])
        return prepared,labels,qids,split

    def fix_ids(self):
        prepared,labels,qids,split=self.load_context('temporal0')
        self.temporal=(prepared,labels,qids,split)
        rng=np.random.default_rng(self.protocol['sampling_seed'])
        selection=stratified_ids(labels,self.protocol['selection_n'],rng)
        confirmation=stratified_ids(labels,self.protocol['confirmation_n'],rng,selection)
        if np.intersect1d(selection,confirmation).size: raise AssertionError('ID overlap')
        self.ids={'search':selection,'confirmation':confirmation}
        self.datasets={'temporal0':self.temporal}
        if 'existing' in self.contexts:
            data=self.load_context('existing'); self.datasets['existing']=data
            self.ids['transfer']=stratified_ids(data[1],self.protocol['transfer_n'],rng)
        for stage,idx in self.ids.items():
            data=self.datasets['existing'] if stage=='transfer' else self.temporal
            frame=pd.DataFrame(dict(query_index=idx,query_id=data[2][idx],timepoint=data[1][idx]))
            dest=self.work/'configs'/f'{stage}_ids.csv'
            if dest.exists():
                old=pd.read_csv(dest,dtype={'query_id':str,'timepoint':str})
                pd.testing.assert_frame_equal(old,frame,check_dtype=False)
            else: frame.to_csv(dest,index=False)
        rule_path=self.work/'configs/selection_rule.json'
        if not rule_path.exists():
            write_json(rule_path,{'rule':self.protocol['selection_rule'],'weights':WEIGHTS,
                       'fixed_before_evaluation_utc':utcnow(),'sampling_seed':self.protocol['sampling_seed']})

    def metric_cache(self,context,idx):
        data=self.datasets['existing'] if context=='existing' else self.temporal
        prepared=data[0]; path=self.contexts[context]['paths'][0]
        from umaping.dynamics import ReferenceTrajectory
        begin=time.perf_counter(); self.deadline.check()
        hi,hd=chunked_exact_knn(prepared.query_features[idx],prepared.reference_features,k=15)
        self.deadline.check()
        key=(self.contexts[context]['hashes']['memory/reference_trajectory.npz'],
             self.contexts[context]['hashes']['memory/reference_features.npy'])
        if key not in self.reference_metrics:
            reference=ReferenceTrajectory.load(path/'memory/reference_trajectory.npz').positions_at(1.)
            _,hr=chunked_exact_knn(prepared.reference_features,prepared.reference_features,k=16)
            self.deadline.check()
            _,lr=chunked_exact_knn(reference,reference,k=16)
            self.reference_metrics[key]=(hr[:,-1],lr[:,-1],reference)
        hr,lr,reference=self.reference_metrics[key]
        self.deadline.check()
        return (hi,hd,hr,lr,reference),time.perf_counter()-begin

    def condition(self,stage,context,teacher,w,scale,idx,metric_cache, *, pilot=False):
        self.deadline.check(new=True)
        teacher_index=0 if teacher=='Uniform' else 1
        source=self.contexts[context]['paths'][teacher_index]
        name=f'{context}_{teacher}_w{w:.12g}_scale{scale:g}'
        if pilot: name='pilot_'+str(time.time_ns())+'_'+name
        cache=self.work/'cache'/stage/name; cache.mkdir(parents=True,exist_ok=True)
        signature={'context':context,'teacher':teacher,'w':w,'scale':scale,'query_indices':idx.tolist(),
                   'upstream':self.contexts[context]['hashes'],'model':self.contexts[context]['model_hashes'][teacher_index]}
        import umaping.inference as inference_module
        signature['inference_code_sha256']=sha(Path(inference_module.__file__))
        sig=cache/'signature.json'
        if sig.exists() and json.loads(sig.read_text())!=signature: raise ValueError('Cache identity mismatch')
        write_json(sig,signature)
        data=self.datasets['existing'] if context=='existing' else self.temporal
        prepared,labels,qids,_=data
        n=len(idx); y=np.full((n,2),np.nan,dtype=np.float32); elapsed=np.full(n,np.nan)
        state=cache/'partial.npz'
        if state.exists():
            with np.load(state) as old:
                if not np.array_equal(old['query_indices'],idx): raise ValueError('Partial query identity mismatch')
                y=old['embedding']; elapsed=old['seconds']
        engine=InferenceEngine.load(source,self.device,seed=0)
        engine.cfg.balance_w=w; engine.cfg.balance_scale=scale
        force_path=cache/'force_rows.json'; rows=json.loads(force_path.read_text()) if force_path.exists() else []
        common_force_ids=set(self.ids['search'][:8])
        def callback(i):
            def log(step,t,a,r,v,cfg):
                if step not in {0,1,10,50,100,cfg.n_steps-1}: return
                an=float(torch.linalg.vector_norm(a)); rn=float(torch.linalg.vector_norm(r))
                wa=cfg.balance_scale*(1-cfg.balance_w)*an; wr=cfg.balance_scale*cfg.balance_w*rn
                rows.append(dict(context=context,teacher=teacher,w=w,scale=scale,query_index=int(i),step=step,
                    optimization_time=t,attraction_norm=an,repulsion_norm=rn,weighted_attraction_norm=wa,
                    weighted_repulsion_norm=wr,repulsion_fraction=wr/(wa+wr) if wa+wr>0 else None,
                    zero_denominator=wa+wr==0,preclip_norm=float(torch.linalg.vector_norm(v)),
                    clipped=bool((v.abs()>cfg.grad_clip).any()) if cfg.grad_clip is not None else False,
                    finite=bool(torch.isfinite(v).all())))
            return log
        try:
            for j,i in enumerate(idx):
                if np.isfinite(elapsed[j]): continue
                self.deadline.check(); sync(self.device); start=time.perf_counter()
                result=engine.embed_one(prepared.query_features[i],deadline_check=self.deadline.check,
                    force_callback=callback(i) if stage=='search' and i in common_force_ids else None)
                sync(self.device); elapsed[j]=time.perf_counter()-start; y[j]=result.embedding
                if (j+1)%25==0:
                    atomic_npz(state,embedding=y,seconds=elapsed,query_indices=idx)
                    write_json(force_path,rows)
        finally:
            atomic_npz(state,embedding=y,seconds=elapsed,query_indices=idx)
            write_json(force_path,rows)
        if not np.isfinite(y).all() or not np.isfinite(elapsed).all(): raise ValueError('Incomplete condition')
        high_ids,high_dist,hr,lr,reference=metric_cache
        self.deadline.check(); begin=time.perf_counter()
        low_ids,low_dist=chunked_exact_knn(y,reference,k=15)
        recall,ndcg,density=metrics_from_neighbors(high_ids,high_dist,low_ids,low_dist,hr,lr)
        eval_seconds=time.perf_counter()-begin
        frame=pd.DataFrame(dict(query_index=idx,query_id=qids[idx],timepoint=labels[idx],
              recall=recall,ndcg=ndcg,density_log_distortion=density,zero_recall=recall==0,seconds=elapsed))
        if pilot:
            self.status['pilot_condition_evaluation_seconds']=eval_seconds
        if not np.isfinite(frame[['recall','ndcg','density_log_distortion']]).all().all():
            raise FloatingPointError('Nonfinite metrics')
        if not pilot:
            target=self.report/SECTIONS[stage]
            frame.to_csv(target/f'metrics_{name}.csv',index=False)
            masks={'micro':np.ones(n,bool),**{g:labels[idx]==g for g in sorted(set(labels[idx]))}}
            for group,mask in masks.items():
                sub=frame.loc[mask]
                self.summaries.append(dict(stage=stage,context=context,teacher=teacher,w=w,scale=scale,group=group,n=len(sub),
                    recall=float(sub.recall.mean()),ndcg=float(sub.ndcg.mean()),density=float(sub.density_log_distortion.mean()),
                    zero_recall=float(sub.zero_recall.mean()),seconds=float(sub.seconds.sum()),
                    macro_recall=float(frame.groupby('timepoint').recall.mean().mean()),evaluation_seconds=eval_seconds))
            self.force_rows+=rows
            self.status['completed_conditions'].append(f'{stage}/{name}')
            self.save()
        return frame

    def save(self):
        write_json(self.report/'99_provenance/execution.json',self.status)
        if self.summaries: pd.DataFrame(self.summaries).to_csv(self.report/'99_provenance/all_summary.csv',index=False)
        if self.force_rows: pd.DataFrame(self.force_rows).to_csv(self.report/SECTIONS['forces']/'force_contributions_temporal_seed0.csv',index=False)
        if self.pairs: pd.DataFrame(self.pairs).to_csv(self.report/'02_selected_ratio_confirmation/paired_recall_all_splits_seeds.csv',index=False)

    def compare(self,frames,stage,context,selected):
        comparisons=[]
        for teacher in ('Uniform','FitGrid'):
            comparisons.append((f'{teacher} selected minus current',(teacher,selected,2),(teacher,.5,2)))
            if context=='temporal0':
                for w,scale in [(0.,1),(0.,2),(1.,1),(1.,2)]:
                    comparisons.append((f'{teacher} selected minus endpoint w={w} scale={scale}',(teacher,selected,2),(teacher,w,scale)))
        comparisons.append(('FitGrid minus Uniform selected',('FitGrid',selected,2),('Uniform',selected,2)))
        for label,left,right in comparisons:
            for row in paired_bootstrap(frames[left],frames[right],self.protocol['bootstrap_iterations'],
                                        self.protocol['sampling_seed'],lambda:self.deadline.check(report=True)):
                self.pairs.append(dict(stage=stage,context=context,comparison=label,**row))
        self.save()

    def execute(self):
        self.audit(); self.fix_ids()
        selection=self.ids['search']; pilot=selection[:min(100,len(selection))]
        # Same real queries, old A+R expression and new default, before searching.
        from unittest.mock import patch
        checks=[]
        engine=InferenceEngine.load(self.contexts['temporal0']['paths'][0],self.device,seed=0)
        for i in pilot[:3]:
            x=self.temporal[0].query_features[i]
            current=engine.embed_one(x,deadline_check=self.deadline.check).embedding
            with patch('umaping.inference.balanced_force',lambda a,r,w,scale:a+r):
                old=engine.embed_one(x,deadline_check=self.deadline.check).embedding
            np.testing.assert_array_equal(current,old)
            checks.append(dict(query_index=int(i),identical=True,max_absolute_difference=float(np.max(np.abs(current-old)))))
        write_json(self.report/'99_provenance/current_reproduction_real_queries.json',checks)
        del engine
        cache,metric_seconds=self.metric_cache('temporal0',pilot)
        p=self.condition('search','temporal0','Uniform',.5,2,pilot,cache,pilot=True)
        per_query=float(p.seconds.mean())
        self.status['pilot']={'n':len(p),'inference_seconds':float(p.seconds.sum()),
                              'evaluation_seconds':self.status['pilot_condition_evaluation_seconds'],
                              'metric_preparation_seconds':metric_seconds,'seconds_per_query':per_query}
        write_json(self.report/'99_provenance/pilot.json',self.status['pilot'])
        # Common counts across conditions. Reduce extra seed before confirmation count.
        remaining=min(self.deadline.compute,self.deadline.new)-time.time()
        estimate=per_query*(18*len(selection)+16*len(self.ids['confirmation'])+4*len(self.ids.get('transfer',[])))
        self.status['pilot']['projected_inference_seconds']=estimate
        budget_path=self.work/'configs/budget_adjustment.json'
        saved_budget=json.loads(budget_path.read_text()) if budget_path.exists() else None
        if saved_budget is not None:
            self.ids['confirmation']=np.asarray(saved_budget['confirmation_query_indices'],dtype=int)
            if saved_budget['skip_seed1']:
                self.status['seed1_skipped']='original pilot budget decision preserved on resume'
        elif estimate>remaining*.8:
            self.status['seed1_skipped']='pilot budget: extra seed removed before reducing confirmation size'
        # Reduction is fixed before search metrics, shared across every confirmation condition.
        main_estimate=per_query*(18*len(selection)+12*len(self.ids['confirmation'])+4*len(self.ids.get('transfer',[])))
        if saved_budget is None and main_estimate>remaining*.8:
            target=max(2,int((remaining*.8/per_query-18*len(selection)-4*len(self.ids.get('transfer',[])))/12))
            self.ids['confirmation']=self.ids['confirmation'][:target]
        if saved_budget is None:
            write_json(budget_path,dict(reason='measured pilot',confirmation_n=len(self.ids['confirmation']),
                confirmation_query_indices=self.ids['confirmation'].tolist(),projected_seconds=main_estimate,
                remaining_seconds=remaining,skip_seed1='seed1_skipped' in self.status))
        effective=self.ids['confirmation']
        pd.DataFrame(dict(query_index=effective,query_id=self.temporal[2][effective],timepoint=self.temporal[1][effective])).to_csv(
            self.work/'configs/effective_confirmation_ids.csv',index=False)
        self.save()
        cache,_=self.metric_cache('temporal0',selection)
        for teacher in ('Uniform','FitGrid'):
            for w in WEIGHTS: self.condition('search','temporal0',teacher,w,2,selection,cache)
            for w in (0.,1.): self.condition('search','temporal0',teacher,w,1,selection,cache)
        table=pd.DataFrame(self.summaries)
        result=select_weight(table[(table.stage=='search') & (table.group=='micro')])
        chosen_path=self.work/'configs/selection.json'
        if chosen_path.exists() and json.loads(chosen_path.read_text())!=result: raise ValueError('Selection changed on resume')
        write_json(chosen_path,result); write_json(self.report/'01_balance_search/selection.json',result)
        self.status['selected']=result; self.save()
        from .report import render
        render(self)
        chosen=result['w']
        contexts=['temporal0']
        if 'temporal1' in self.contexts and 'seed1_skipped' not in self.status: contexts.append('temporal1')
        elif 'temporal1' not in self.contexts: self.status['seed1_skipped']='existing seed 1 frozen inputs missing'
        if 'existing' in self.contexts: contexts.append('existing')
        else: self.status['transfer_incomplete']='existing split inputs missing'
        for context in contexts:
            stage='transfer' if context=='existing' else 'confirmation'; idx=self.ids[stage]
            cache,_=self.metric_cache(context,idx); frames={}
            for teacher in ('Uniform','FitGrid'):
                conditions=[(.5,2),(chosen,2)]
                if context=='temporal0': conditions += [(0.,1),(0.,2),(1.,1),(1.,2)]
                for w,scale in dict.fromkeys(conditions):
                    frames[(teacher,w,scale)]=self.condition(stage,context,teacher,w,scale,idx,cache)
            self.compare(frames,stage,context,chosen); render(self)
        # Re-check frozen sources before claiming completion.
        _,after=inspect(self.root,self.deadline)
        for context,before in self.contexts.items():
            if after[context]['hashes']!=before['hashes'] or after[context]['model_hashes']!=before['model_hashes']:
                raise ValueError(f'Source artifacts changed during inference: {context}')
        self.status['source_artifacts_unchanged']=True
        self.status['status']='complete_A_B_C' if 'existing' in self.contexts else 'partial_A_B_missing_C'
        self.save(); render(self)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True); parser.add_argument('--output-root',required=True)
    parser.add_argument('--protocol',required=True); parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args(); torch.set_num_threads(args.threads)
    experiment=Experiment(args)
    lock=experiment.work/'execution.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Existing execution lock: inspect owning PID before retry: {lock}')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    def timeout(signum,frame): raise TimeoutError('Absolute compute timeout')
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(.01,experiment.deadline.compute-time.time()))
    try:
        experiment.execute()
    except Exception as exc:
        experiment.status.update(status='incomplete',error=str(exc),traceback=traceback.format_exc())
        print(traceback.format_exc(),file=sys.stderr)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        experiment.status['execution_ended_utc']=utcnow()
        start=__import__('datetime').datetime.fromisoformat(experiment.protocol['started_utc'].replace('Z','+00:00')).timestamp()
        experiment.status['elapsed_since_original_start_seconds']=time.time()-start
        experiment.save()
        try:
            # The compute timeout no longer applies to reporting; the original hard deadline does.
            signal.setitimer(signal.ITIMER_REAL,max(.01,experiment.deadline.hard-time.time()))
            from .report import render
            render(experiment)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            lock.unlink(missing_ok=True)
    return 0 if experiment.status['status']=='complete_A_B_C' else 2
