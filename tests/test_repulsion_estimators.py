"""合成データだけで数式・公平性・保存・fail-fastを検証する。"""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from umaping.config import Config, DatasetConfig, RepulsionConfig
from umaping.dynamics import ReferenceTrajectory
from umaping.models.repulsion import RepulsionField
from umaping.models.retriever import DualEncoder
from umaping.repulsion_estimators import (
    UniformMC, DualImportance, DualTopL, BarnesHut, GridField, TrainingQueries, exact_field, hubness_vector)
from umaping.repulsion_estimators.benchmark import field_metrics
from umaping.repulsion_estimators.runner import main, inspect_frozen, sha
from umaping.training.flow import train_repulsion_field
from umaping.umap_forces import g_minus


@pytest.fixture(autouse=True)
def limit_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def small():
    rng = np.random.default_rng(2)
    positions = rng.normal(size=(12, 2)).astype(np.float32)
    trajectory = ReferenceTrajectory(np.array([0.,1.]), np.stack([positions, positions+.2]))
    q = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(12,4)),dtype=torch.float32),dim=1)
    k = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(12,4)),dtype=torch.float32),dim=1)
    common = dict(trajectory=trajectory, a=1.5, b=.9, seed=123)
    return common, q, k


@pytest.mark.parametrize('family', ['uniform','raw','hub','mixture','topl'])
def test_stochastic_unbiased_and_reproducible(small, family):
    common, q, k = small
    dual = dict(queries=q, keys=k, temperature=1., score_batch=128)
    h = hubness_vector(q, k, 1., topk=4, chunk=3)
    def make():
        if family == 'uniform':
            return UniformMC(**common)
        if family == 'topl':
            return DualTopL(**common, **dual, top_l=5, tail_samples=7)
        return DualImportance(**common, **dual, hubness=h if family == 'hub' else None,
                              mixture=.9 if family == 'mixture' else 1.)
    teacher = make()
    y = torch.tensor([[.3,-.1]]).repeat(2500,1)
    t, anchors = torch.full((len(y),), .3), torch.zeros(len(y),dtype=torch.long)
    pred = teacher.estimate(y,t,anchors)
    truth = exact_field(common['trajectory'], y[:1], t[:1], common['a'],common['b'])[0]
    tolerance = 5 * pred.std(0)/np.sqrt(len(pred)) + 2e-5
    assert torch.all(torch.abs(pred.mean(0)-truth) < tolerance)
    assert pred.shape == y.shape and torch.isfinite(pred).all()
    torch.testing.assert_close(make().estimate(y,t,anchors), pred, rtol=0,atol=0)
    if family in ('raw','hub','mixture'):
        assert 1 <= teacher.diagnostics['ess'] <= 64.0001
        assert teacher.diagnostics['min_sampled_probability'] > 0


def test_hubness_is_dual_only_and_matches_dense_tiny(small):
    _, q, k = small
    expected = (q.double()@k.double().T/.1).topk(4,dim=0).values.mean(0)
    actual = hubness_vector(q,k,.1,topk=4,chunk=3)
    torch.testing.assert_close(actual, expected)


def test_importance_probabilities_and_support_loss(small):
    common,q,k = small
    teacher = DualImportance(**common,queries=q,keys=k,temperature=.5,proposal_temperature=2.,mixture=.9)
    expected = .1/len(k)+.9*torch.softmax(q[[0]].double()@k.double().T,dim=1)
    torch.testing.assert_close(teacher.probabilities(torch.tensor([0])),expected)
    bad = DualImportance(**common,queries=q,keys=k,temperature=.1,proposal_temperature=1e-9)
    with pytest.raises(FloatingPointError,match='support'):
        bad.probabilities(torch.tensor([0]))


def test_topl_all_references_includes_anchor(small):
    common,q,k = small
    teacher = DualTopL(**common,queries=q,keys=k,top_l=100)
    anchors,t,y = TrainingQueries(common['trajectory'],'cpu',5,.1,1)(0)
    torch.testing.assert_close(teacher.estimate(y,t,anchors),exact_field(common['trajectory'],y,t,1.5,.9),atol=2e-7,rtol=2e-6)


def test_barnes_hut_small_theta_exact_at_checkpoints(small):
    common,_,_ = small
    teacher = BarnesHut(**common,theta=0)
    y = torch.tensor([[.3,.7],[.01,-.01],[0.,0.]])
    t, anchors = torch.tensor([0.,1.,0.]),torch.arange(3)
    torch.testing.assert_close(teacher.estimate(y,t,anchors),exact_field(common['trajectory'],y,t,1.5,.9),atol=1e-7,rtol=1e-6)
    assert teacher.diagnostics['force_interactions'] == 12


def test_barnes_hut_actual_tree_and_time_interpolation():
    rng=np.random.default_rng(5)
    p=rng.normal(size=(100,2)).astype(np.float32)
    trajectory=ReferenceTrajectory(np.array([0.,1.]),np.stack([p,p+2]))
    y=torch.tensor([[4.,3.]])
    t=torch.tensor([.4])
    teacher=BarnesHut(trajectory,1.5,.9,theta=0)
    expected=.6*exact_field(trajectory,y,torch.tensor([0.]),1.5,.9)+.4*exact_field(trajectory,y,torch.tensor([1.]),1.5,.9)
    torch.testing.assert_close(teacher.estimate(y,t,torch.tensor([0])),expected,atol=1e-7,rtol=1e-6)
    approximate=BarnesHut(trajectory,1.5,.9,theta=.8)
    approximate.estimate(y,t,torch.tensor([0]))
    assert approximate.diagnostics['force_interactions'] < 200


def test_grid_resolution_orientation_linear_convolution_and_bounds():
    trajectory=ReferenceTrajectory(np.array([0.,1.]),np.zeros((2,1,2),dtype=np.float32))
    y=torch.tensor([[.6,.0],[.41,.71],[-.33,.48]])
    t=torch.tensor([0.,.4,1.]); anchor=torch.zeros(3,dtype=torch.long)
    exact=exact_field(trajectory,y,t,1.5,.9)
    coarse=GridField(trajectory,1.5,.9,grid_size=16,jitter_sigma=.2)
    fine=GridField(trajectory,1.5,.9,grid_size=129,jitter_sigma=.2)
    c,f=coarse.estimate(y,t,anchor),fine.estimate(y,t,anchor)
    assert (f-exact).square().mean() < (c-exact).square().mean()
    assert f[0,0] > 0 and abs(f[0,1]) < 1e-6
    torch.testing.assert_close(f[0],exact[0],atol=1e-5,rtol=1e-5)
    # Far edge is finite and outward: opposite boundary must not wrap into near-field.
    edge=torch.tensor([[1.18,.0]])
    pred=fine.estimate(edge,t[:1],anchor[:1])
    torch.testing.assert_close(pred,exact_field(trajectory,edge,t[:1],1.5,.9),atol=.003,rtol=.003)
    with pytest.raises(ValueError,match='outside'):
        fine.estimate(torch.tensor([[100.,0.]]),t[:1],anchor[:1])
    assert fine.diagnostics['outside_count'] == 1


def test_exact_oracle_reuses_clipped_force_and_is_chunk_invariant(small):
    common,_,_=small
    anchors,t,y=TrainingQueries(common['trajectory'],'cpu',7,.1,7)(0)
    positions=common['trajectory'].positions_at_many(np.repeat(t.numpy(),12),np.tile(np.arange(12),len(y))).reshape(len(y),12,2)
    expected=g_minus(y[:,None],torch.tensor(positions,dtype=torch.float32),1.5,.9,clip=.3).mean(1)
    torch.testing.assert_close(exact_field(common['trajectory'],y,t,1.5,.9,clip=.3,query_chunk=2,reference_chunk=3),expected,atol=1e-7,rtol=1e-6)


def test_training_query_sequence_independent_of_teacher_randomness(small):
    common,_,_=small
    sampler=TrainingQueries(common['trajectory'],'cpu',8,.1,10)
    first=sampler(3)
    UniformMC(**common).estimate(first[2],first[1],first[0])
    torch.randn(500)
    for x,y in zip(first,sampler(3)):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert not torch.equal(first[2],sampler(4)[2])


def test_training_interface_reproducible_and_keeps_frozen_inputs(small):
    common,_,_=small
    trajectory=common['trajectory']
    before=trajectory.positions.copy()
    cfg=RepulsionConfig(hidden_dim=8,n_residual_blocks=1,time_embed_dim=4,steps=3,batch_size=8)
    initial=RepulsionField(hidden_dim=8,n_residual_blocks=1,time_embed_dim=4)
    finals=[]
    for _ in range(2):
        model=copy.deepcopy(initial)
        state=train_repulsion_field(model,trajectory,1.5,.9,cfg,np.ones(12),torch.device('cpu'),progress=False,
                                    teacher=UniformMC(**common),query_sampler=TrainingQueries(trajectory,'cpu',8,.1,9))
        assert len(state.losses)==3
        finals.append(model.state_dict())
    for k in finals[0]:
        torch.testing.assert_close(finals[0][k],finals[1][k],rtol=0,atol=0)
    np.testing.assert_array_equal(before,trajectory.positions)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in this environment')
def test_cuda_estimators(small):
    common,q,k=small
    common={**common,'device':'cuda'}
    anchor,t,y=TrainingQueries(common['trajectory'],'cuda',4,.1,3)(0)
    for cls,kw in [(UniformMC,{}),(DualImportance,dict(queries=q,keys=k)),(DualTopL,dict(queries=q,keys=k)),(BarnesHut,{}),(GridField,dict(grid_size=16))]:
        value=cls(**common,**kw).estimate(y,t,anchor)
        assert value.is_cuda and value.shape==(4,2) and torch.isfinite(value).all()


def make_frozen_fixture(root):
    run=root/'synthetic_frozen'
    for part in ('memory','checkpoints'):
        (run/part).mkdir(parents=True)
    cfg=Config(dataset=DatasetConfig(name='embryoid_body',input_dim=4),
               repulsion=RepulsionConfig(hidden_dim=8,n_residual_blocks=1,time_embed_dim=4,steps=3,batch_size=4))
    cfg.save(run/'config.yaml')
    rng=np.random.default_rng(4)
    x=rng.normal(size=(40,4)).astype(np.float32)
    np.save(run/'memory/reference_features.npy',x)
    p=rng.normal(size=(40,2)).astype(np.float32)
    ReferenceTrajectory(np.array([0.,1.]),np.stack([p,p+.05])).save(run/'memory/reference_trajectory.npz')
    hp=dict(input_dim=4,hidden_dims=[8],retrieval_dim=4,temperature=1.)
    model=DualEncoder(**hp).eval()
    torch.save(dict(hparams=hp,state_dict=model.state_dict()),run/'checkpoints/retriever.pt')
    with torch.no_grad():
        np.save(run/'memory/retriever_keys.npy',model.encode_key(torch.tensor(x)).numpy())
    return run


def test_missing_actual_data_fails_without_creating_output(tmp_path):
    with pytest.raises(FileNotFoundError,match='reference_trajectory'):
        main(['--frozen-run',str(tmp_path/'missing'),'--dataset','embryoid_body','--output-dir',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()


def test_full_synthetic_phase_a_before_b_artifacts_plots_no_upstream(tmp_path,monkeypatch):
    import umaping.dynamics
    def forbidden(*args,**kwargs):
        raise AssertionError('Upstream trajectory generation prohibited')
    monkeypatch.setattr(umaping.dynamics,'simulate_reference_dynamics',forbidden)
    import umaping.training.flow
    monkeypatch.setattr(umaping.training.flow,'simulate_reference_dynamics',forbidden)
    run=make_frozen_fixture(tmp_path)
    original={str(p):sha(p) for p in run.rglob('*') if p.is_file()}
    out=tmp_path/'benchmark'
    command=['--frozen-run',str(run),'--dataset','embryoid_body','--output-dir',str(out),'--device','cpu',
             '--validation-queries','8','--repetitions','3','--eval-every','1','--thetas','.5','--grid-sizes','16']
    main(command)
    meta=json.loads((out/'config/manifest.json').read_text())
    assert meta['all_six_training_succeeded'] and meta['source_hashes_unchanged']
    assert original=={str(p):sha(p) for p in run.rglob('*') if p.is_file()}
    assert len(pd.read_csv(out/'metrics/trained_fields.csv'))==6
    assert len(list((out/'figures').glob('*.png')))>=12
    assert (out/'artifacts/validation.npz').is_file()
    assert (out/'artifacts/hubness.npy').is_file()
    with pytest.raises(ValueError,match='output-dir'):
        main(command)
    with pytest.raises(ValueError,match='identity'):
        inspect_frozen(run,'embryo')
    assert '同じvalidation' in (out/'report.md').read_text()


def test_default_uniform_training_matches_original_rng_and_update(small):
    common,_,_=small
    trajectory=common['trajectory']
    cfg=RepulsionConfig(steps=3,batch_size=5,teacher_negative_samples=7,lr=.001)
    initial=RepulsionField(hidden_dim=8,n_residual_blocks=1,time_embed_dim=4)
    expected=copy.deepcopy(initial)
    actual=copy.deepcopy(initial)
    rng=np.random.default_rng(19)
    from umaping.dynamics import TorchTrajectoryView
    view=TorchTrajectoryView(trajectory,torch.device('cpu'))
    optimizer=torch.optim.Adam(expected.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    torch.manual_seed(37)
    losses=[]
    for _ in range(cfg.steps):
        anchor=torch.tensor(rng.integers(0,12,size=cfg.batch_size))
        t=torch.tensor(rng.uniform(0,1,size=cfg.batch_size),dtype=torch.float32)
        base=view.positions_at_many(t,anchor)
        y=base+torch.randn_like(base)*cfg.jitter_sigma
        idx=torch.tensor(rng.integers(0,12,size=cfg.batch_size*cfg.teacher_negative_samples))
        neg=view.positions_at_many(t.repeat_interleave(cfg.teacher_negative_samples),idx).reshape(cfg.batch_size,cfg.teacher_negative_samples,2)
        target=g_minus(y[:,None],neg,1.5,.9,clip=4.).mean(1)
        loss=(expected(y,t)-target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    torch.manual_seed(37)
    state=train_repulsion_field(actual,trajectory,1.5,.9,cfg,np.ones(12),torch.device('cpu'),seed=19,progress=False)
    assert state.losses==losses
    for key,value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key],value,rtol=0,atol=0)


def test_phase_a_then_training_reuses_validation_and_can_replot(tmp_path, monkeypatch):
    from umaping.repulsion_estimators.report import render
    run=make_frozen_fixture(tmp_path)
    a,b=tmp_path/'phase_a',tmp_path/'phase_b'
    common=['--frozen-run',str(run),'--dataset','embryoid_body','--device','cpu','--validation-queries','8',
            '--repetitions','2','--eval-every','1','--thetas','.5','--grid-sizes','16']
    main(common+['--stage','estimators','--output-dir',str(a)])
    assert not list((a/'repulsion_training').glob('*/repulsion_field.pt'))
    validation_hash=sha(a/'artifacts/validation.npz')
    import umaping.repulsion_estimators.runner as runner
    def no_recompute(*args, **kwargs):
        raise AssertionError('Phase B must reuse cached Dual embeddings and hubness')
    monkeypatch.setattr(runner, 'dual_embeddings', no_recompute)
    monkeypatch.setattr(runner, 'hubness_vector', no_recompute)
    main(common+['--stage','train','--from-run',str(a),'--output-dir',str(b)])
    assert sha(b/'artifacts/validation.npz')==validation_hash
    meta=json.loads((b/'config/manifest.json').read_text())
    assert meta['all_six_training_succeeded']
    assert meta['cache_reused_from'] == str(a.resolve())
    figures=tmp_path/'replot'
    render(b,meta,json.loads((b/'metrics/estimator_metrics.json').read_text()),
           json.loads((b/'metrics/trained_fields.json').read_text()),figures)
    assert len(list((figures/'figures').glob('*.png')))>=12
    with pytest.raises(FileExistsError):
        render(b,meta,[],[],figures)
    (a/'artifacts/hubness.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact hash'):
        main(common+['--stage','train','--from-run',str(a),'--output-dir',str(tmp_path/'tampered')])


def test_failed_teacher_preserves_partial_predictions_and_reason(tmp_path,small):
    from umaping.repulsion_estimators.benchmark import evaluate_teacher
    common,q,k=small
    teacher=DualImportance(**common,queries=q,keys=k,proposal_temperature=1e-9)
    anchor,t,y=TrainingQueries(common['trajectory'],'cpu',4,.1,1)(0)
    truth=exact_field(common['trajectory'],y,t,1.5,.9)
    with pytest.raises(FloatingPointError):
        evaluate_teacher('collapsed',teacher,anchor,t,y,truth,tmp_path,repetitions=2)
    assert np.isnan(np.load(tmp_path/'collapsed_predictions.npy')).all()
    assert 'support' in (tmp_path/'collapsed_failure.json').read_text()
