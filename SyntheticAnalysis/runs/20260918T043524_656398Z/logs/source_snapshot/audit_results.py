"""Read-only artifact consistency checks plus diagnostic audit outputs."""
import argparse,json,hashlib,sys,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
from geometry import neighbors,shortcut_flags,STRUCTURES


def audit(run):
    run=Path(run);checks=[];graphs=[];retrieval=[]
    frozen=json.loads((run/'logs/pretraining_frozen_manifest.json').read_text())['sha256']
    for name,digest in frozen.items():
        p=Path(name);p=p if p.is_absolute() else Path(__file__).resolve().parents[2]/p
        if '/data/' in name or '/configs/' in name or name.endswith(('geometry.py','evaluate.py')):
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
    maps=np.load(run/'data/isometric_map.npz');A=maps['A'];assert np.max(abs(A.T@A-np.eye(2)))<1e-12
    d0=np.load(run/'data/disconnected/arrays.npz');d1=np.load(run/'data/sparse_bridge/arrays.npz')
    for key in ['x','z','ids']:np.testing.assert_array_equal(d0[key],d1[key][:6000])
    for key in ['reference','query']:np.testing.assert_array_equal(d0[key],d1[key][d1[key]<6000])
    for summary_path in sorted((run/'metrics').glob('*/seed_*/summary.csv')):
        cond=summary_path.parents[1].name;seed=int(summary_path.parent.name.split('_')[1]);model=run/'models'/cond/f'seed_{seed}';emb=run/'embeddings'/cond/f'seed_{seed}'
        frame=pd.read_csv(run/'data'/cond/'points.csv');data=np.load(run/'data'/cond/'arrays.npz');ref,q=data['reference'],data['query'];rf=frame.iloc[ref].reset_index(drop=True);rz=neighbors(data['z'][ref],data['z'][ref],15,np.arange(len(ref)))[1][:,-1]
        u=np.load(emb/'uniform.npz');g=np.load(emb/'fitgrid.npz');np.testing.assert_array_equal(u['reference'],g['reference']);np.testing.assert_array_equal(u['retrieved'],g['retrieved'])
        pu=json.loads((model/'uniform/provenance.json').read_text());pg=json.loads((model/'fitgrid/provenance.json').read_text())
        for key in ['initial_sha256','trajectory_sha256','retriever_sha256','spectral_sha256','query_sequence_seed']:assert pu[key]==pg[key]
        checks.append(dict(condition=cond,seed=seed,matched_reference=True,matched_neighbors=True,matched_upstream=True,all_finite=True))
        true,_=neighbors(data['z'][q],data['z'][ref],15)
        rec=np.array([len(set(a)&set(b))/15 for a,b in zip(true,u['retrieved'])])
        for sid,label in STRUCTURES.items():
            mask=frame.structure.to_numpy()[q]==sid
            if mask.any():retrieval.append(dict(condition=cond,seed=seed,group=label,query_retriever_recall15=float(rec[mask].mean())))
        for name,path in [('UMAPing',model/'shared/memory/graph_symmetric.npz'),('standard_umap',model/'standard_umap_graph.npz')]:
            w=load_npz(path);nc,lab=connected_components(w,directed=False);co=w.tocoo();keep=co.row!=co.col;ii,jj=co.row[keep],co.col[keep]
            src=rf.iloc[ii].reset_index(drop=True);a,b=shortcut_flags(src,rf,jj[:,None],rz[ii]);sid=rf.structure.to_numpy();mask=sid<3;ni,_=connected_components(w[mask][:,mask],directed=False)
            s,t=sid[ii],sid[jj];cross=s!=t;allowed={ (0,3),(1,3),(1,4),(2,4)};wrong=sum(tuple(sorted((int(v),int(h)))) not in allowed for v,h in zip(s[cross],t[cross]))
            small_data=None
            if name=='standard_umap':
                with (model/'standard_umap.pkl').open('rb') as f:reducer=pickle.load(f)
                small_data=bool(reducer._small_data)
            graphs.append(dict(small_data_exact_distance_backend=small_data,condition=cond,seed=seed,method=name,components=nc,components_without_bridge=ni,wrong_turn_edges=int(a.sum()),wrong_branch_edges=int(b.sum()),unexpected_cross_structure_edges=wrong))
            assert nc==(3 if cond=='disconnected' else 1)
            assert not a.any() and not b.any() and wrong==0
        for method in ['pca','umap','uniform','fitgrid']:
            y=np.load(emb/f'{method}.npz');assert y['reference'].shape==(len(ref),2) and y['query'].shape==(len(q),2)
            assert np.isfinite(y['reference']).all() and np.isfinite(y['query']).all()
            p=pd.read_csv(summary_path.parent/f'{method}_points.csv.gz');assert len(p)==len(frame)
            assert p.id.is_unique and set(p.id)==set(frame.id)
            assert (p.recall15.between(0,1)).all() and (p.abs_log_radius15>=0).all()
            for split in ['reference','query']:
                pp=p[p.split==split];stored=pd.read_csv(summary_path);stored=stored[(stored.method==method)&(stored.split==split)&(stored.group=='all')].iloc[0]
                for m in ['recall15','within_recall15','abs_log_radius15']:assert abs(pp[m].mean()-stored[m])<1e-10
            # Scale is one value, fit exclusively on reference; signed median is zero there.
            assert p.scale.nunique()==1
            assert abs(np.median(p.loc[p.split=='reference','log_radius15']))<1e-8
    pd.DataFrame(graphs).to_csv(run/'metrics/actual_training_graph_audit.csv',index=False)
    pd.DataFrame(retrieval).to_csv(run/'metrics/query_retriever_diagnostic.csv',index=False)
    (run/'logs/final_artifact_audit.json').write_text(json.dumps(dict(passed=True,n_completed_pairs=len(checks),checks=checks),indent=2))
    print(json.dumps(dict(passed=True,n_completed_pairs=len(checks),graphs=graphs),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();audit(args.run)
