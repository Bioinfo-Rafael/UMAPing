"""Numerical invariants and baseline diagnostics; no fitting to truth."""
import json,sys,shutil
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp
from experiment import BASE,CONDITIONS,METHODS,DEVICE,train_spectral,core
from umaping.models.mlp import batched_forward
from umaping.graph import normalized_laplacian,row_degree

def audit(run):
    checks=[]
    for c in CONDITIONS:
        a=np.load(run/'data'/c/'arrays.npz');x=a['x'].astype(np.float32);ref,q=a['reference'],a['query'];old=BASE/'models'/c/'seed_0/shared'
        engine=core.InferenceEngine.load(old,DEVICE,seed=0);w=sp.load_npz(old/'memory/graph_symmetric.npz');lap=normalized_laplacian(w,row_degree(w))
        raw=batched_forward(engine.spectral_embedder.encoder,x[ref],DEVICE).astype(float);cal=engine.spectral_embedder.calibration;white=raw@cal.whitening
        diag=dict(raw_dim=3,covariance_eigenvalues=np.linalg.eigvalsh(np.cov(raw,rowvar=False))[::-1].tolist(),raw_gram_eigenvalues=np.linalg.eigvalsh(raw.T@raw/len(ref)).tolist(),H_whitened_per_N_eigenvalues=np.linalg.eigvalsh(white.T@(lap@white)/len(ref)).tolist(),calibration_discarded_eigenvalue=cal.discarded_eigenvalue,calibration_retained_eigenvalues=cal.retained_eigenvalues.tolist())
        p=run/'models'/c/'baseline_3_to_2';p.mkdir(exist_ok=True)
        core.js(p/'diagnostics.json',diag);shutil.copy2(old/'metrics/spectral_training.json',p/'spectral_training.json')
        shared=np.load(run/'data'/c/'shared_retrieval.npz');base=np.load(BASE/'embeddings'/c/'seed_0/fitgrid.npz')
        assert np.array_equal(shared['indices'],base['retrieved'])
        hashes=[]
        for m in METHODS:
            v=np.load(run/'embeddings'/c/f'{m}.npz')
            assert v['reference_stages'].shape==(5,len(ref),2) and v['query_stages'].shape==(5,len(q),2)
            assert np.isfinite(v['reference_stages']).all() and np.isfinite(v['query_stages']).all()
            assert np.array_equal(v['retrieved'],shared['indices'])
            if m=='baseline_3_to_2':
                assert np.max(abs(v['query']-base['query']))<1e-5
                checks.append(dict(condition=c,method=m,baseline_query_max_error=float(np.max(abs(v['query']-base['query'])))))
                continue
            root=run/'models'/c/m/'shared';cfg=core.Config.load(run/'configs/model.yaml')
            fm,yr=train_spectral(root,x[ref],w,m,cfg);cv=np.load(root/'memory/custom_calibration.npz');t=cv['trivial_vectors'];n=len(ref)
            assert np.max(abs(t.T@t/n-np.eye(t.shape[1])))<1e-10
            assert np.max(abs(lap@t))<1e-5
            assert np.max(abs(yr-v['reference_stages'][0]))<1e-5
            # Each unseen point is independent of the other query points in a batch.
            chunk=x[q[:17]];batch=fm.embed(chunk);one=np.stack([fm.embed_one(xx) for xx in chunk]);rev=fm.embed(chunk[::-1])[::-1]
            err=float(np.max(abs(batch-one)));assert err<1e-4,err
            assert np.max(abs(batch-rev))<1e-4
            for rel in ['checkpoints/retriever.pt','memory/graph_symmetric.npz','memory/retriever_keys.npy','memory/reference_features.npy','checkpoints/repulsion_initial.pt']:assert core.sha(root/rel)==core.sha(old/rel)
            prov=json.loads((run/'models'/c/m/'teacher/models'/c/'seed_0/fitgrid/provenance.json').read_text());h=core.sha(root/'memory/reference_trajectory.npz');assert prov['trajectory_sha256']==h;hashes.append(h)
            d=json.loads((root/'diagnostics.json').read_text());assert d['corrected_trivial_leakage']<1e-20
            if m=='direct_2':assert cv['projection'].shape==(2,2) and np.array_equal(cv['projection'],np.eye(2))
            else:
                raw=batched_forward(fm.model,x[ref],DEVICE).astype(float);corrected=raw-t@cv['coef'];cov=np.cov(corrected,rowvar=False);ev,vec=np.linalg.eigh(cov);proj=vec[:,-2:];saved=cv['projection'];assert np.linalg.norm(proj@proj.T-saved@saved.T)<1e-7
            checks.append(dict(condition=c,method=m,query_batch_max_error=err,trivial_laplacian_max_error=float(abs(lap@t).max()),trajectory_sha256=h,shared_retriever_sha256=core.sha(root/'checkpoints/retriever.pt'),shared_graph_sha256=core.sha(root/'memory/graph_symmetric.npz'),teacher_matches_trajectory=True))
        assert len(set(hashes))==2
    core.js(run/'logs/artifact_audit.json',dict(passed=True,checks=checks))
    print(json.dumps(checks,indent=2))
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(2);audit(Path(sys.argv[1]).resolve())
