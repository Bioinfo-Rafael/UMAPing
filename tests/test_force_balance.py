"""Protocol tests use synthetic fixtures only, never reported as scientific results."""
from datetime import datetime, timezone, timedelta
import dataclasses
import json
import numpy as np
import pandas as pd
import pytest
import torch
from umaping.inference import balanced_force, InferenceConfig
from umaping.force_balance.core import (WEIGHTS, Deadline, stratified_ids, select_weight,
                                         paired_bootstrap, metrics_from_neighbors, write_json)


def test_force_coefficients_endpoints_and_no_double_rate():
    a=torch.tensor([3.,-7.]); r=torch.tensor([-2.,11.]) # already scaled R
    assert torch.equal(balanced_force(a,r),a+r)
    for w in WEIGHTS:
        torch.testing.assert_close(balanced_force(a,r,w),2*((1-w)*a+w*r))
    assert torch.equal(balanced_force(a,r,0),2*a)
    assert torch.equal(balanced_force(a,r,1),2*r)
    assert torch.equal(balanced_force(a,r,0,1),a)
    assert torch.equal(balanced_force(a,r,1,1),r)
    assert torch.equal(balanced_force(a,torch.full_like(r,float('nan')),0,1),a)
    with pytest.raises(ValueError): balanced_force(a,r,float('nan'))


def test_current_reproduction_and_clip_order(monkeypatch):
    from test_inference import _build_toy_engine
    import umaping.inference as module
    engine,features=_build_toy_engine(11)
    q=features[0]+.13
    before=engine.trajectory.positions.copy()
    engine.cfg.balance_w=.5
    actual=engine.embed_one(q,return_trajectory=True)
    with monkeypatch.context() as m:
        m.setattr(module,'balanced_force',lambda a,r,w,scale:a+r)
        old=engine.embed_one(q,return_trajectory=True)
    np.testing.assert_array_equal(actual.trajectory,old.trajectory)
    np.testing.assert_array_equal(before,engine.trajectory.positions)
    for w,scale in [(0.,1.),(0.,2.),(1.,1.),(1.,2.),(.2,2.)]:
        engine.cfg.balance_w=w; engine.cfg.balance_scale=scale
        recorded=[]
        result=engine.embed_one(q,True,force_callback=lambda e,t,a,r,v,c:recorded.append((e,v.detach().numpy().copy())))
        for e,v in recorded:
            expected=result.trajectory[e]+engine.cfg.initial_alpha*(1-e/engine.cfg.n_steps)*np.clip(v,-engine.cfg.grad_clip,engine.cfg.grad_clip)
            np.testing.assert_allclose(result.trajectory[e+1],expected,atol=2e-6)
        again=engine.embed_one(q)
        np.testing.assert_array_equal(result.embedding,again.embedding)
    def stop(): raise TimeoutError('test deadline')
    with pytest.raises(TimeoutError): engine.embed_one(q,deadline_check=stop)


def test_sampling_reproducible_and_disjoint_actual_labels():
    labels=np.array(['4-5']*20+['8-9']*22)
    rng=np.random.default_rng(42); s=stratified_ids(labels,12,rng); c=stratified_ids(labels,16,rng,s)
    np.testing.assert_array_equal(s,stratified_ids(labels,12,np.random.default_rng(42)))
    assert not np.intersect1d(s,c).size
    assert (labels[s]=='4-5').sum()==6 and (labels[c]=='8-9').sum()==8
    assert len(stratified_ids(labels,100,np.random.default_rng(42)))==42


def test_selection_complete_only_and_ties():
    table=pd.DataFrame([dict(teacher=t,w=w,scale=2,macro_recall=.1) for t in ('Uniform','FitGrid') for w in WEIGHTS])
    assert select_weight(table)['w']==.5
    table.loc[table.w==0,'macro_recall']=.2
    assert select_weight(table)['w']==0
    with pytest.raises(ValueError): select_weight(table.iloc[1:])
    table.macro_recall=.1
    table.loc[table.w.isin([1/3,2/3]),'macro_recall']=.3
    assert select_weight(table)['w']==1/3


def test_paired_bootstrap_respects_ids_and_macro():
    a=pd.DataFrame(dict(query_index=[0,1,2],timepoint=['4-5','8-9','8-9'],recall=[1.,.2,.2]))
    b=a.copy(); b.recall=0
    rows=paired_bootstrap(a,b.iloc[::-1],200,seed=4)
    assert rows[0]['delta']==pytest.approx(.6)
    assert rows[1]['delta']==pytest.approx(1.4/3)
    assert rows[0]['low']==rows[0]['high']==pytest.approx(.6)
    with pytest.raises(ValueError): paired_bootstrap(a,b.iloc[:2],10)
    b.loc[0,'timepoint']='wrong'
    with pytest.raises(ValueError): paired_bootstrap(a,b,10)


def test_cached_metrics_match_existing_definition():
    from umaping.graph import chunked_exact_knn
    from umaping.evaluation.advanced import multi_k_recall_and_ndcg
    from umaping.fit_grid_experiment.metrics import query_metrics
    from umaping.data.preprocessing import PreparedDataset
    from umaping.config import Config
    from types import SimpleNamespace
    rng=np.random.default_rng(8)
    ref=rng.normal(size=(40,5)).astype('float32'); q=rng.normal(size=(4,5)).astype('float32')
    yref=rng.normal(size=(40,2)).astype('float32'); y=rng.normal(size=(4,2)).astype('float32')
    hi,hd=chunked_exact_knn(q,ref,15); li,ld=chunked_exact_knn(y,yref,15)
    _,hr=chunked_exact_knn(ref,ref,16); _,lr=chunked_exact_knn(yref,yref,16)
    actual=metrics_from_neighbors(hi,hd,li,ld,hr[:,-1],lr[:,-1])
    prepared=PreparedDataset(ref,q,{}, {},5)
    outcome=SimpleNamespace(reference_embedding=yref,query_embedding=y,mean_query_latency_seconds=.1)
    expected,_=query_metrics('fixture',prepared,outcome,Config.load('configs/mock.yaml'))
    for arr,col in zip(actual,['recall_at_15','ndcg','density_log_distortion']):
        np.testing.assert_allclose(arr,expected[col],rtol=1e-6)


def test_deadlines_and_atomic_save(tmp_path):
    now=datetime.now(timezone.utc)
    protocol={k:(now+timedelta(seconds=-1)).isoformat() for k in ['stop_compute_utc','stop_new_experiments_utc','hard_deadline_utc']}
    d=Deadline(protocol)
    for kwargs in ({},{'new':True},{'report':True}):
        with pytest.raises(TimeoutError): d.check(**kwargs)
    target=tmp_path/'checkpoint.json'; write_json(target,{'done':[1,2]})
    assert json.loads(target.read_text())=={'done':[1,2]}
    assert not target.with_suffix('.json.tmp').exists()


def test_condition_saves_partial_and_resumes_same_ids(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_inference import _build_toy_engine
    from umaping.force_balance.runner import Experiment
    from umaping.data.preprocessing import PreparedDataset
    from umaping.graph import chunked_exact_knn
    engine,ref=_build_toy_engine(18)
    q=ref[:3]+.2; labels=np.array(['4-5','8-9','4-5']); ids=np.array(['c0','c1','c2'])
    prepared=PreparedDataset(ref,q,{}, {},ref.shape[1])
    now=datetime.now(timezone.utc)
    protocol=dict(run_id='fixture_only',base_commit='test',started_utc=now.isoformat(),
        stop_compute_utc=(now+timedelta(minutes=5)).isoformat(),
        stop_new_experiments_utc=(now+timedelta(minutes=4)).isoformat(),
        hard_deadline_utc=(now+timedelta(minutes=6)).isoformat())
    config=tmp_path/'protocol.json'; write_json(config,protocol)
    exp=Experiment(SimpleNamespace(source_root=str(tmp_path),output_root=str(tmp_path/'outputs'),protocol=str(config),device='cpu'))
    exp.contexts={'temporal0':dict(paths=[tmp_path,tmp_path],hashes={'fixture':'hash'},model_hashes=['a','b'])}
    exp.temporal=(prepared,labels,ids,None); exp.datasets={}; exp.ids={'search':np.arange(3)}
    yref=engine.trajectory.positions_at(1.)
    hi,hd=chunked_exact_knn(q,ref,15); _,hr=chunked_exact_knn(ref,ref,16); _,lr=chunked_exact_knn(yref,yref,16)
    cache=(hi,hd,hr[:,-1],lr[:,-1],yref)
    monkeypatch.setattr('umaping.force_balance.runner.InferenceEngine.load',lambda *a,**kw:engine)
    original=engine.embed_one; count=[0]
    def interrupted(*a,**kw):
        count[0]+=1
        if count[0]==2: raise TimeoutError('test interruption')
        return original(*a,**kw)
    monkeypatch.setattr(engine,'embed_one',interrupted)
    with pytest.raises(TimeoutError): exp.condition('search','temporal0','Uniform',.5,2,np.arange(3),cache)
    checkpoint=next((exp.work/'cache').rglob('partial.npz'))
    with np.load(checkpoint) as data:
        assert np.isfinite(data['seconds']).sum()==1
        np.testing.assert_array_equal(data['query_indices'],[0,1,2])
    monkeypatch.setattr(engine,'embed_one',original)
    result=exp.condition('search','temporal0','Uniform',.5,2,np.arange(3),cache)
    assert result.query_id.tolist()==ids.tolist()
    assert len(exp.status['completed_conditions'])==1
    assert np.isfinite(result.recall).all()
    from umaping.force_balance.report import render
    render(exp)
    assert (exp.report/'README_ja.md').is_file()
    assert list(exp.report.rglob('*.png')) and list(exp.report.rglob('*.pdf'))
