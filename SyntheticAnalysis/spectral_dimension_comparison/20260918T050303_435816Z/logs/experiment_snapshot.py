#!/usr/bin/env python3
"""Reference-only spectral ablation; no UMAP coordinate supervision."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,time,datetime,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'SyntheticAnalysis/scripts'))
import run as core
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from umaping.models.spectral import SpectralEncoderNet
from umaping.models.mlp import batched_forward
from umaping.graph import row_degree,upper_triangular_edges,normalized_laplacian
from umaping.dynamics import ReferenceTrajectory
from evaluate import evaluate,summarize
from geometry import neighbors
BASE=ROOT/'SyntheticAnalysis/runs/20260918T043524_656398Z'
HOME=ROOT/'SyntheticAnalysis/spectral_dimension_comparison'
CONDITIONS=['disconnected','sparse_bridge'];METHODS=['baseline_3_to_2','direct_2','spectral_100_pca_2'];TIMES=[0,50,100,150,200]
DEVICE=torch.device('cpu')

class FrozenMap:
    def __init__(self,model,x,t,coef,projection,mean,scale):
        self.model=model.eval();self.x=x;self.t=t;self.coef=coef;self.projection=projection;self.mean=mean;self.scale=scale
        self.nn=NearestNeighbors(n_neighbors=15).fit(x)
    def trivial(self,x):
        d,i=self.nn.kneighbors(np.atleast_2d(x));weights=1/np.maximum(d,1e-8);weights/=weights.sum(1,keepdims=True)
        return (self.t[i]*weights[:,:,None]).sum(1)
    def embed(self,x):
        raw=batched_forward(self.model,np.atleast_2d(x),DEVICE).astype(float)
        return ((raw-self.trivial(x)@self.coef)@self.projection-self.mean)*self.scale
    def embed_one(self,x):return self.embed(x)[0]

def train_spectral(out,x,w,mode,cfg):
    r=2 if mode=='direct_2' else 100;n=len(x);degree=row_degree(w)
    nc,labels=connected_components(w,directed=False)
    t=np.zeros((n,nc))
    for c in range(nc):
        mask=labels==c;t[mask,c]=np.sqrt(degree[mask]);t[:,c]*=np.sqrt(n)/np.linalg.norm(t[:,c])
    hp=dict(input_dim=50,hidden_dims=cfg.spectral.hidden_dims,raw_output_dim=r,activation=cfg.spectral.activation)
    core.set_seed(300);model=SpectralEncoderNet(**hp)
    path=out/'checkpoints/spectral_encoder.pt';calpath=out/'memory/custom_calibration.npz'
    if path.exists() and calpath.exists():
        model.load_state_dict(torch.load(path,weights_only=False)['state_dict']);v=np.load(calpath)
        fm=FrozenMap(model,x,t,v['coef'],v['projection'],v['mean'],float(v['scale']))
        return fm,v['reference']
    xt=torch.tensor(x);tt=torch.tensor(t,dtype=torch.float32);inv=torch.tensor(1/np.sqrt(degree),dtype=torch.float32)
    row,col,weight=upper_triangular_edges(w);row=torch.tensor(row);col=torch.tensor(col);weight=torch.tensor(weight,dtype=torch.float32)
    rng=np.random.default_rng(0);opt=torch.optim.Adam(model.parameters(),lr=cfg.spectral.lr,weight_decay=cfg.spectral.weight_decay);logs=[]
    # Divide both original loss terms by r: their relative coefficient stays 1.
    for step in range(cfg.spectral.steps):
        z=model(xt);idx=rng.integers(0,len(row),size=min(cfg.spectral.edge_batch_size,len(row)))
        diff=z[row[idx]]*inv[row[idx],None]-z[col[idx]]*inv[col[idx],None]
        energy=(weight[idx]*diff.square().sum(1)).mean();gram=z.T@z/n
        ortho=(gram-torch.eye(r)).square().sum();trivial=(tt.T@z/n).square().sum()
        loss=(energy+cfg.spectral.orthogonality_weight*ortho+trivial)/r
        opt.zero_grad();loss.backward();opt.step()
        logs.append(dict(step=step+1,dirichlet_raw=energy.item(),orthogonality_raw=ortho.item(),trivial_raw=trivial.item(),loss=loss.item(),dirichlet_per_dim=energy.item()/r,orthogonality_per_dim=ortho.item()/r,trivial_per_dim=trivial.item()/r))
        if (step+1)%250==0:core.log(f'{mode} spectral {step+1}/2000 loss {loss.item():.5g}')
    pd.DataFrame(logs).to_csv(out/'spectral_losses.csv',index=False)
    raw=batched_forward(model,x,DEVICE).astype(float);coef=t.T@raw/n;corrected=raw-t@coef
    pca=PCA(2,svd_solver='full').fit(corrected) if r==100 else None
    proj=pca.components_.T if pca is not None else np.eye(2)
    y=corrected@proj;mean=y.mean(0);scale=cfg.spectral.calibration_target_scale/max(abs(y-mean).max(),1e-12);yr=(y-mean)*scale
    l=normalized_laplacian(w,degree);h=corrected.T@(l@corrected)/n
    eig=np.linalg.eigvalsh(np.cov(corrected,rowvar=False))[::-1]
    diagnostics=dict(raw_dim=r,components=nc,raw_gram_eigenvalues=np.linalg.eigvalsh(raw.T@raw/n).tolist(),covariance_eigenvalues=eig.tolist(),explained_variance_ratio=(eig/eig.sum()).tolist(),top_relative_gaps=((eig[:min(10,r-1)]-eig[1:min(11,r)])/np.maximum(eig[:min(10,r-1)],1e-15)).tolist(),H_eigenvalues=np.linalg.eigvalsh((h+h.T)/2).tolist(),raw_trivial_leakage=float(np.linalg.norm(t.T@raw/n)**2/r),corrected_trivial_leakage=float(np.linalg.norm(t.T@corrected/n)**2/r),orthogonality_error=float(np.linalg.norm(raw.T@raw/n-np.eye(r))**2/r),centered_2d_eigenvalues=np.linalg.eigvalsh(np.cov(y,rowvar=False)).tolist(),scale=float(scale),post_pca_whitening=False)
    # Fixed reference bootstrap diagnoses PCA subspace sensitivity, never selects settings.
    if r==100:
        boot=[];rr=np.random.default_rng(987)
        for _ in range(20):
            sample=corrected[rr.integers(n,size=n)];q=PCA(2,svd_solver='full').fit(sample).components_.T
            sv=np.linalg.svd(proj.T@q,compute_uv=False);boot.append(np.degrees(np.arccos(np.clip(sv,0,1))).tolist())
        diagnostics['bootstrap_principal_angles_degrees']=boot
    core.js(out/'diagnostics.json',diagnostics)
    core.atomic_torch_save(dict(hparams=hp,state_dict=model.state_dict()),path)
    np.savez_compressed(calpath,coef=coef,projection=proj,mean=mean,scale=scale,reference=yr,trivial_vectors=t)
    fm=FrozenMap(model,x,t,coef,proj,mean,scale)
    assert np.max(abs(fm.embed(x)-yr))<1e-5
    return fm,yr

def setup(run):
    for p in ['configs','data','models','embeddings','metrics','figures','logs']:(run/p).mkdir(parents=True,exist_ok=True)
    if not (run/'configs/protocol.json').exists():
        shutil.copy2(BASE/'configs/model.yaml',run/'configs/model.yaml');shutil.copytree(BASE/'data',run/'data',dirs_exist_ok=True)
        core.js(run/'configs/protocol.json',dict(seed=0,baseline=str(BASE),methods=METHODS,new_methods=METHODS[1:],data_conditions=CONDITIONS,steps=2000,loss='(edge_energy + gram_error + trivial_projection_error)/raw_dim',orthogonality_sampling='exact full reference',trivial_extension='fixed reference inverse-distance 15-neighbor interpolation in input space',pca='centered covariance, no whitening, top variance 2',bootstrap_seed=987,pair_seed=12345,pair_count=20000,started=time.time()))
        core.js(run/'logs/source_data_hashes.json',{str(p.relative_to(BASE)):core.sha(p) for p in (BASE/'data').rglob('*') if p.is_file()})
        shutil.copy2(__file__,run/'logs/experiment_snapshot.py')
    for name,digest in json.loads((run/'logs/source_data_hashes.json').read_text()).items():assert core.sha(run/name)==digest

def execute(run,condition):
    cfg=core.Config.load(run/'configs/model.yaml');arr=np.load(run/'data'/condition/'arrays.npz');ref,q=arr['reference'],arr['query'];x=arr['x'].astype(np.float32)
    old=BASE/'models'/condition/'seed_0/shared';engine=core.InferenceEngine.load(old,DEVICE,seed=0)
    w=sp.load_npz(old/'memory/graph_symmetric.npz')
    out=run/'embeddings'/condition;out.mkdir(exist_ok=True)
    # Reuse persisted retrieval IDs, derive their same input distances once.
    ids=np.load(BASE/'embeddings'/condition/'seed_0/fitgrid.npz')['retrieved'];dist=np.sqrt(((x[q,None]-x[ref][ids])**2).sum(2))
    np.savez_compressed(run/'data'/condition/'shared_retrieval.npz',indices=ids,distances=dist)
    for m in ['umap','pca']:shutil.copy2(BASE/'embeddings'/condition/'seed_0'/f'{m}.npz',out/f'{m}.npz')
    for mode in METHODS:
        if (out/f'{mode}.npz').exists():continue
        core.log(f'START {condition}/{mode}');start=time.time()
        if mode=='baseline_3_to_2':
            engine=core.InferenceEngine.load(old,DEVICE,seed=0)
            ck=torch.load(BASE/'models'/condition/'seed_0/fitgrid/repulsion_field.pt',weights_only=False);engine.repulsion_field.load_state_dict(ck['state_dict'])
            trajectory=engine.trajectory
        else:
            local=run/'models'/condition/mode;local.mkdir(parents=True,exist_ok=True)
            layout=core.init_run_dir(cfg,local/'shared')
            for rel in ['checkpoints/retriever.pt','checkpoints/repulsion_initial.pt','memory/graph_symmetric.npz','memory/retriever_keys.npy','memory/reference_features.npy']:
                dest=layout['root']/rel
                if not dest.exists():shutil.copy2(old/rel,dest)
            fm,yr=core.timed(run,f'{condition}/{mode}/spectral',lambda:train_spectral(layout['root'],x[ref],w,mode,cfg))
            trajectory=core.timed(run,f'{condition}/{mode}/dynamics',lambda:core.stage_flow_dynamics(cfg,layout,yr,w,DEVICE))
            initial=torch.load(old/'checkpoints/repulsion_initial.pt',weights_only=False)
            # Reuse teacher routine with separate storage for each changed trajectory.
            teacherroot=run/'models'/condition/mode/'teacher';(teacherroot/'models'/condition/'seed_0').mkdir(parents=True,exist_ok=True);(teacherroot/'logs').mkdir(exist_ok=True)
            model=core.train_teacher(teacherroot,condition,0,'fitgrid',cfg,layout,trajectory,w,initial)
            engine.spectral_embedder=fm;engine.trajectory=trajectory;engine.repulsion_field=model.eval()
        # This engine's only changed inputs are the spectral map, trajectory and its new FitGrid network.
        qr=np.empty((5,len(q),2),np.float32)
        for i in range(len(q)):
            engine.retrieve_neighbors=lambda _,i=i:(ids[i],dist[i])
            result=engine.embed_one(x[q[i]],return_trajectory=True);qr[:,i]=result.trajectory[TIMES]
            if (i+1)%400==0:core.log(f'{condition}/{mode} query {i+1}/{len(q)}')
        rr=trajectory.positions[TIMES]
        if mode=='baseline_3_to_2':
            prior=np.load(BASE/'embeddings'/condition/'seed_0/fitgrid.npz');assert np.max(abs(prior['query']-qr[-1]))<1e-5
        np.savez_compressed(out/f'{mode}.npz',reference=rr[-1],query=qr[-1],reference_stages=rr,query_stages=qr,times=np.array(TIMES)/200,retrieved=ids)
        core.js(out/f'{mode}_timing.json',dict(seconds=time.time()-start))
        core.log(f'DONE {condition}/{mode} {time.time()-start:.1f}s')

def analysis(run):
    summaries=[];large=[];ranks=[]
    for condition in CONDITIONS:
        frame=pd.read_csv(run/'data'/condition/'points.csv');a=np.load(run/'data'/condition/'arrays.npz');ref,q=a['reference'],a['query'];z=a['z'];out=run/'metrics'/condition;out.mkdir(exist_ok=True)
        pairs={};rng=np.random.default_rng(12345)
        for split,ind in [('reference',ref),('query',q)]:
            for group,mask in [('all',np.ones(len(frame),bool)),('A_spiral',frame.structure==0),('B_tree',frame.structure==1)]:
                subset=ind[np.asarray(mask)[ind]];ij=rng.choice(subset,(20000,2));ij=ij[ij[:,0]!=ij[:,1]];pairs[split+'_'+group]=ij
        np.savez_compressed(out/'distance_pairs.npz',**pairs)
        for method in ['pca','umap']+METHODS:
            v=np.load(run/'embeddings'/condition/f'{method}.npz');stages=range(5) if 'reference_stages' in v else [4]
            for stage in stages:
                yr=v['reference_stages'][stage] if len(stages)>1 else v['reference'];yq=v['query_stages'][stage] if len(stages)>1 else v['query'];y=np.zeros_like(z);y[ref]=yr;y[q]=yq
                pts=evaluate(frame,ref,q,yr,yq);pts.to_csv(out/f'{method}_stage{stage}_points.csv.gz',index=False)
                sm=summarize(pts);sm['method']=method;sm['condition']=condition;sm['stage']=stage/4;summaries.append(sm)
                for split,ind in [('reference',ref),('query',q)]:
                    src=ind[frame.structure.to_numpy()[ind]==0];cand=ref[frame.structure.to_numpy()[ref]==0]
                    ex=np.searchsorted(cand,src) if split=='reference' else None
                    iz,_=neighbors(z[src],z[cand],400,ex);iy,_=neighbors(y[src],y[cand],400,ex)
                    for k in [5,15,50,100,200,400]:large.append(dict(condition=condition,method=method,stage=stage/4,split=split,k=k,recall=np.mean([len(set(i[:k])&set(j[:k]))/k for i,j in zip(iz,iy)])))
                    for group in ['all','A_spiral','B_tree']:
                        ij=pairs[split+'_'+group];dz=np.linalg.norm(z[ij[:,0]]-z[ij[:,1]],axis=1);dy=np.linalg.norm(y[ij[:,0]]-y[ij[:,1]],axis=1)
                        ranks.append(dict(condition=condition,method=method,stage=stage/4,split=split,group=group,pairs=len(ij),spearman=float(spearmanr(dz,dy).statistic)))
    pd.concat(summaries).to_csv(run/'metrics/summary.csv',index=False);pd.DataFrame(large).to_csv(run/'metrics/spiral_recall.csv',index=False);pd.DataFrame(ranks).to_csv(run/'metrics/distance_rank.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--phase',choices=['train','analyze'],default='train');args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    run=Path(args.run).resolve() if args.run else HOME/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    setup(run);core.log('RUN_DIR='+str(run))
    if args.phase=='train':
        for c in CONDITIONS:execute(run,c)
    analysis(run)
    from figures import render,report
    render(run);report(run)
if __name__=='__main__':main()
