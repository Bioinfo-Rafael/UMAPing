#!/usr/bin/env python3
"""Run fixed synthetic protocol. Every completed stage is reusable."""
import os
os.environ.setdefault('TQDM_DISABLE','1');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('NUMBA_NUM_THREADS','2')
import argparse,copy,datetime,hashlib,json,logging,pickle,platform,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.sparse import save_npz
from scipy.sparse.csgraph import connected_components
from geometry import generate,graph_audit,STRUCTURES
from evaluate import evaluate,summarize
from umaping.config import Config
from umaping.data.preprocessing import PreparedDataset,save_prepared_dataset
from umaping.pipeline import init_run_dir,stage_graph,stage_retriever,stage_spectral,stage_flow_dynamics
from umaping.repulsion_estimators import TrainingQueries,UniformMC,GridField
from umaping.models.repulsion import RepulsionField
from umaping.repulsion_estimators.runner import model_hparams
from umaping.training.flow import train_repulsion_field,RepulsionTrainState
from umaping.graph import row_degree
from umaping.umap_forces import find_ab_params
from umaping.inference import InferenceEngine
from umaping.utils.seed import set_seed
from umaping.utils.io import atomic_torch_save,collect_package_versions

BASE=ROOT/'SyntheticAnalysis'
def js(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str));tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(message): print(now(),message,flush=True)
def timed(run,key,fn):
    begin=time.monotonic();log('START '+key);value=fn();elapsed=time.monotonic()-begin
    with (run/'logs/timing.jsonl').open('a') as f:f.write(json.dumps(dict(stage=key,seconds=elapsed,finished=now()))+'\n')
    log(f'DONE {key}: {elapsed:.1f} seconds');return value


def initialize(args):
    if args.run:run=Path(args.run).resolve()
    else:
        run=BASE/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True,exist_ok=False)
        for sub in ['data','models','embeddings','metrics','figures','logs','configs']: (run/sub).mkdir()
        for p in (BASE/'configs').glob('*'):shutil.copy2(p,run/'configs'/p.name)
        js(run/'status.json',dict(started=now(),started_epoch=time.time(),completed=[],status='created'))
        js(run/'logs/environment.json',dict(platform=platform.platform(),python=sys.version,versions=collect_package_versions(),threads=2,device='cpu',base_commit=os.popen(f'git -C {ROOT} rev-parse HEAD').read().strip()))
        shutil.copytree(BASE/'scripts',run/'logs/source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
        js(run/'logs/source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/umaping').rglob('*.py')})
    status=json.loads((run/'status.json').read_text());remaining=int(status['started_epoch']+11.5*3600-time.time())
    if remaining<=0 and args.phase=='train':raise RuntimeError('Original 11.5 hour deadline expired')
    if args.phase=='train':
        def deadline(*_):raise TimeoutError('11.5 hour deadline reached; completed artifacts are retained')
        signal.signal(signal.SIGALRM,deadline);signal.alarm(max(1,remaining))
    lock=run/'execution.lock'
    if lock.exists():
        pid=int(lock.read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'run already active: PID {pid}')
    lock.write_text(str(os.getpid()));return run


def prepare(run):
    if (run/'data/prepared.done').exists():return
    df,x,A=generate();df.to_csv(run/'data/all_points.csv',index=False);np.savez_compressed(run/'data/isometric_map.npz',A=A,x=x)
    audits={}
    for condition in ['disconnected','sparse_bridge']:
        frame=df[df.structure<3].copy() if condition=='disconnected' else df.copy();frame=frame.reset_index(drop=True)
        xx=x[frame.id.to_numpy()];ref=np.flatnonzero(frame.split=='reference');q=np.flatnonzero(frame.split=='query')
        p=run/'data'/condition;p.mkdir(exist_ok=True);frame.to_csv(p/'points.csv',index=False);np.savez_compressed(p/'arrays.npz',x=xx,z=frame[['z0','z1']].to_numpy(),reference=ref,query=q,ids=frame.id.to_numpy())
        frame.groupby(['structure','branch','split']).size().rename('n').to_csv(p/'split_counts.csv')
        for scope,ind in [('all',np.arange(len(frame))),('reference',ref)]:
            audit,idx,dist,g=graph_audit(frame.iloc[ind].reset_index(drop=True),xx[ind]);audits[condition+'_'+scope]=audit
            np.savez_compressed(p/f'{scope}_knn.npz',indices=idx,distances=dist);save_npz(p/f'{scope}_knn_graph.npz',g)
            expected=3 if condition=='disconnected' else 1
            assert audit['components']==expected,(condition,scope,audit)
            assert audit['components_without_bridge']==3
            assert audit['unexpected_cross_structure_edges']==0
            assert audit['wrong_turn_edges']==audit['wrong_branch_edges']==0
        pca=PCA(2,svd_solver='full').fit(xx[ref]);y=pca.transform(xx)
        # Orthogonal Procrustes for sanity only; no plotting/evaluation query fitting.
        zr=frame[['z0','z1']].to_numpy();zm=zr[ref].mean(0);ym=y[ref].mean(0)
        u,_,vt=np.linalg.svd((y[ref]-ym).T@(zr[ref]-zm));aligned=(y-ym)@(u@vt)+zm
        sanity=dict(orthonormal_error=float(np.max(abs(A.T@A-np.eye(2)))),pca_recovery_max_error=float(np.max(abs(aligned-zr))),pca_explained_variance=float(pca.explained_variance_ratio_.sum()),n_internal=6000,n_bridge=len(frame)-6000,n_reference=len(ref),n_query=len(q))
        assert sanity['pca_recovery_max_error']<1e-9
        js(p/'sanity.json',sanity);np.savez_compressed(p/'pca.npz',reference=y[ref],query=y[q],mean=pca.mean_,components=pca.components_)
        pcadir=run/'models'/condition;pcadir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pcadir/'pca_model.npz',mean=pca.mean_,components=pca.components_)
    js(run/'metrics/input_graph_audit.json',audits)
    from plots import truth_figures
    truth_figures(run)
    (run/'data/prepared.done').write_text(now())
    freeze_protocol(run)


def freeze_protocol(run):
    path=run/'logs/pretraining_frozen_manifest.json'
    if path.exists():return
    sources=list((BASE/'scripts').glob('*.py'))+list((run/'configs').glob('*'))+list((run/'data').rglob('*'))
    js(path,dict(frozen_utc=now(),sha256={str(p):sha(p) for p in sources if p.is_file()}))


def verify_frozen(run):
    path=run/'logs/pretraining_frozen_manifest.json'
    if path.exists():
        for name,digest in json.loads(path.read_text())['sha256'].items():
            if '/data/' in name or '/configs/' in name or name.endswith(('geometry.py','evaluate.py')):
                p=Path(name);p=p if p.is_absolute() else ROOT/p
                if sha(p)!=digest:raise ValueError(f'Frozen experiment input changed: {p}')
    for name,digest in json.loads((run/'logs/source_hashes.json').read_text()).items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Existing UMAPing implementation changed: {name}')


def allow_expansion(run):
    status=json.loads((run/'status.json').read_text())
    if not {'disconnected/seed_0','sparse_bridge/seed_0'}<=set(status['completed']) or not (run/'report.md').exists():
        raise RuntimeError('Complete seed 0, figures and report before extending')
    timing=pd.read_json(run/'logs/timing.jsonl',lines=True)
    first=timing[timing.stage.str.contains('/0/')]
    measured=float(first.seconds.sum());remaining=status['started_epoch']+11.5*3600-time.time()
    needed=4*measured+1800
    if needed>=remaining:raise RuntimeError('Conservative three-seed forecast exceeds remaining deadline')
    path=run/'logs/seed_expansion_decision.json'
    if not path.exists():
        js(path,dict(decided_utc=now(),seed0_pair_compute_seconds=measured,estimated_remaining_compute_seconds=2*measured,safety_factor=2,report_reserve_seconds=1800,conservative_remaining_seconds=needed,deadline_remaining_seconds=remaining,decision='extend to matched seeds 1,2 in fixed order',selection_basis='runtime only; all executed seeds retained',mandatory_report_completed=True))


def train_teacher(run,condition,seed,method,cfg,layout,trajectory,w,initial):
    hp=model_hparams(cfg);a,b=find_ab_params(cfg.umap.spread,cfg.umap.min_dist);device=torch.device('cpu')
    out=run/'models'/condition/f'seed_{seed}'/method;out.mkdir(exist_ok=True)
    final=out/'repulsion_field.pt';model=RepulsionField(**hp)
    if final.exists():model.load_state_dict(torch.load(final,weights_only=False)['state_dict']);return model
    model.load_state_dict(initial['state_dict']);common=dict(trajectory=trajectory,a=a,b=b,device=device,clip=cfg.flow.grad_clip,seed=seed+3000)
    teacher=timed(run,f'{condition}/{seed}/{method}/teacher_build',lambda:UniformMC(**common,samples=64) if method=='uniform' else GridField(**common,grid_size=256,jitter_sigma=cfg.repulsion.jitter_sigma))
    resume=None;optimizer=None;checkpoint=out/'resume.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,weights_only=False);model.load_state_dict(ck['state_dict']);resume=RepulsionTrainState(step=ck['step'],losses=ck['losses']);optimizer=ck['optimizer'];teacher.rng.bit_generator.state=ck['teacher_rng'];torch.set_rng_state(ck['torch_rng'])
    else:set_seed(seed+1000)
    sampler=TrainingQueries(trajectory,device,cfg.repulsion.batch_size,cfg.repulsion.jitter_sigma,seed+2000)
    def checkpoint_cb(state,m,opt):
        if state.step%250==0 or state.step==cfg.repulsion.steps:
            atomic_torch_save(dict(step=state.step,losses=state.losses,state_dict=m.state_dict(),optimizer=opt.state_dict(),teacher_rng=teacher.rng.bit_generator.state,torch_rng=torch.get_rng_state()),checkpoint)
            log(f'{condition} seed {seed} {method}: step {state.step}/{cfg.repulsion.steps}, loss {state.losses[-1]:.6g}')
    state=timed(run,f'{condition}/{seed}/{method}/training',lambda:train_repulsion_field(model,trajectory,a,b,cfg.repulsion,row_degree(w),device,seed=seed,teacher=teacher,query_sampler=sampler,grad_clip=cfg.flow.grad_clip,progress=False,resume_state=resume,optimizer_state=optimizer,checkpoint_callback=checkpoint_cb))
    assert np.isfinite(state.losses).all()
    atomic_torch_save(dict(hparams=hp,state_dict=model.state_dict()),final)
    pd.DataFrame(dict(step=np.arange(1,len(state.losses)+1),loss=state.losses)).to_csv(out/'losses.csv',index=False)
    js(out/'provenance.json',dict(initial_sha256=sha(layout['checkpoints']/'repulsion_initial.pt'),trajectory_sha256=sha(layout['memory']/'reference_trajectory.npz'),retriever_sha256=sha(layout['checkpoints']/'retriever.pt'),spectral_sha256=sha(layout['checkpoints']/'spectral_encoder.pt'),query_sequence_seed=seed+2000,teacher_rng_seed=seed+3000,teacher=method,teacher_diagnostics=teacher.diagnostics))
    return model


def train_condition(run,condition,seed):
    cfg=Config.load(run/'configs/model.yaml');cfg.seed=seed;device=torch.device('cpu')
    d=run/'data'/condition;frame=pd.read_csv(d/'points.csv');arr=np.load(d/'arrays.npz');ref,q=arr['reference'],arr['query'];x=arr['x'].astype(np.float32)
    modeldir=run/'models'/condition/f'seed_{seed}';modeldir.mkdir(parents=True,exist_ok=True)
    embdir=run/'embeddings'/condition/f'seed_{seed}';embdir.mkdir(parents=True,exist_ok=True)
    prepared=PreparedDataset(x[ref],x[q],{'id':frame.id.to_numpy()[ref]},{'id':frame.id.to_numpy()[q]},50)
    layout=init_run_dir(cfg,modeldir/'shared')
    if not (layout['memory']/'reference_features.npy').exists():
        np.save(layout['memory']/'reference_features.npy',x[ref]);save_prepared_dataset(layout['cache']/'prepared_dataset.npz',prepared)
    set_seed(seed+100);mu,w=timed(run,f'{condition}/{seed}/graph',lambda:stage_graph(cfg,layout,prepared))
    nc,lab=connected_components(w,directed=False);js(modeldir/'shared/input_graph_components.json',dict(components=int(nc),sizes=np.bincount(lab).tolist(),self_excluded_neighbors=15))
    set_seed(seed+200);timed(run,f'{condition}/{seed}/retriever',lambda:stage_retriever(cfg,layout,prepared,mu,device))
    set_seed(seed+300);_,_,y0=timed(run,f'{condition}/{seed}/spectral',lambda:stage_spectral(cfg,layout,prepared,w,device))
    set_seed(seed+400);trajectory=timed(run,f'{condition}/{seed}/trajectory',lambda:stage_flow_dynamics(cfg,layout,y0,w,device))
    p=layout['checkpoints']/'repulsion_initial.pt'
    if not p.exists():
        set_seed(seed);atomic_torch_save(dict(hparams=model_hparams(cfg),state_dict=RepulsionField(**model_hparams(cfg)).state_dict()),p)
    initial=torch.load(p,weights_only=False)
    for method in ['uniform','fitgrid']:
        model=train_teacher(run,condition,seed,method,cfg,layout,trajectory,w,initial)
        output=embdir/f'{method}.npz'
        if output.exists():continue
        # Engine reuses the same persisted upstream; only its repulsion field is swapped.
        shared_rep=layout['checkpoints']/'repulsion_field.pt'
        if not shared_rep.exists():shutil.copy2(modeldir/'uniform/repulsion_field.pt',shared_rep)
        engine=InferenceEngine.load(layout['root'],device,seed=seed);engine.repulsion_field=model.eval()
        partial=embdir/f'{method}_partial.npz'
        yq=np.full((len(q),2),np.nan,np.float32);ids=np.full((len(q),15),-1,np.int64);lat=np.zeros(len(q));done=0
        if partial.exists():
            prev=np.load(partial);yq=prev['query'];ids=prev['retrieved'];lat=prev['latency'];done=int(prev['done'])
        def inference():
            for i in range(done,len(q)):
                begin=time.perf_counter();r=engine.embed_one(x[q[i]]);lat[i]=time.perf_counter()-begin;yq[i]=r.embedding;ids[i]=r.neighbor_ids
                if (i+1)%100==0 or i+1==len(q):
                    tmp=partial.with_name(partial.stem+'_tmp.npz');np.savez_compressed(tmp,query=yq,retrieved=ids,latency=lat,done=i+1);tmp.replace(partial);log(f'{condition} seed {seed} {method}: query {i+1}/{len(q)}')
            np.savez_compressed(output,reference=trajectory.positions[-1],query=yq,retrieved=ids,latency=lat)
        timed(run,f'{condition}/{seed}/{method}/inference',inference)
    if not (embdir/'umap.npz').exists():
        import umap
        def baseline():
            modelpath=modeldir/'standard_umap.pkl'
            if modelpath.exists():
                with modelpath.open('rb') as f:reducer=pickle.load(f)
                yr=reducer.embedding_
            else:
                reducer=umap.UMAP(n_neighbors=15,min_dist=.1,spread=1.,n_components=2,metric='euclidean',random_state=seed,transform_seed=seed,n_epochs=500,n_jobs=1,negative_sample_rate=5)
                yr=reducer.fit_transform(x[ref])
                with modelpath.open('wb') as f:pickle.dump(reducer,f)
            yq=reducer.transform(x[q]);np.savez_compressed(embdir/'umap.npz',reference=yr,query=yq)
            nc,lab=connected_components(reducer.graph_,directed=False);js(modeldir/'standard_umap_graph.json',dict(components=int(nc),sizes=np.bincount(lab).tolist(),n_neighbors_including_self=15,epochs=500,transform_epochs=166))
            save_npz(modeldir/'standard_umap_graph.npz',reducer.graph_)
        timed(run,f'{condition}/{seed}/standard_umap',baseline)
    if not (embdir/'pca.npz').exists():shutil.copy2(d/'pca.npz',embdir/'pca.npz')
    timed(run,f'{condition}/{seed}/evaluate',lambda:evaluate_condition(run,condition,seed))
    status=json.loads((run/'status.json').read_text());key=f'{condition}/seed_{seed}'
    if key not in status['completed']:status['completed'].append(key)
    status['updated']=now();status['status']='trained';js(run/'status.json',status)


def evaluate_condition(run,condition,seed):
    d=run/'data'/condition;frame=pd.read_csv(d/'points.csv');a=np.load(d/'arrays.npz');emb=run/'embeddings'/condition/f'seed_{seed}';out=run/'metrics'/condition/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for method in ['pca','umap','uniform','fitgrid']:
        path=emb/f'{method}.npz'
        if not path.exists():continue
        v=np.load(path);points=evaluate(frame,a['reference'],a['query'],v['reference'],v['query']);points.to_csv(out/f'{method}_points.csv.gz',index=False)
        summary=summarize(points);summary['method']=method;summary['condition']=condition;summary['seed']=seed;summaries.append(summary)
        if method=='pca':assert points.recall15.min()==1 and points.abs_log_radius15.max()<1e-8
    pd.concat(summaries,ignore_index=True).to_csv(out/'summary.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--phase',choices=['prepare','train','evaluate','plot'],default='prepare');p.add_argument('--seeds',type=int,nargs='+',default=[0]);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    run=initialize(args);print('RUN_DIR='+str(run),flush=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(run/'logs/pipeline.log'),logging.StreamHandler()])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    try:
        verify_frozen(run)
        prepare(run)
        freeze_protocol(run)
        if args.phase=='train':
            if args.seeds not in ([0],[1,2],[0,1,2]):raise ValueError('Use seed 0 first, then fixed seeds 1 2; no seed selection')
            for seed in args.seeds:
                if seed==1:allow_expansion(run)
                for condition in ['disconnected','sparse_bridge']:train_condition(run,condition,seed)
        if args.phase=='evaluate':
            for seed in args.seeds:
                for condition in ['disconnected','sparse_bridge']:evaluate_condition(run,condition,seed)
        if args.phase in ['train','evaluate','plot']:
            from plots import all_figures
            from report import write_report
            from audit_results import audit
            all_figures(run);audit(run);write_report(run)
    finally:(run/'execution.lock').unlink(missing_ok=True)
if __name__=='__main__':main()
