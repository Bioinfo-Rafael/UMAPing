"""合成小データによる成果物保護・時間holdout・対応評価の検証。"""
import json
import numpy as np
import pandas as pd
import pytest
import torch
from umaping.config import Config
from umaping.fit_grid_experiment.materialize import (
    UPSTREAM, HASHED, guard_output, materialize, convert_losses, sha, model_hparams)
from umaping.models.repulsion import RepulsionField


@pytest.fixture
def frozen(tmp_path):
    from umaping.pipeline import run_training
    cfg = Config.load('configs/mock.yaml')
    cfg.dataset.params.update(n_reference=40, n_query=8)
    cfg.retriever.steps = cfg.spectral.steps = cfg.repulsion.steps = 2
    cfg.flow.n_steps = 2
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    source = tmp_path / 'source'
    run_training(cfg, source, torch.device('cpu'))
    torch.set_num_threads(before)
    # This remains a synthetic fixture, never an actual Embryoid experiment.
    cfg.dataset.name = 'embryoid_body'
    cfg.save(source / 'config.yaml')
    benchmark = tmp_path / 'benchmark'
    (benchmark / 'config').mkdir(parents=True)
    folder = benchmark / 'repulsion_training/fit_grid'
    folder.mkdir(parents=True)
    spec = dict(name='fit_grid_256', family='fit_grid', grid_size=256)
    hp = model_hparams(cfg)
    torch.save(dict(hparams=hp, state_dict=RepulsionField(**hp).state_dict(), teacher=spec),
               folder / 'repulsion_field.pt')
    pd.DataFrame(dict(step=[1, 2], loss=[.4, .2])).to_csv(folder / 'losses.csv', index=False)
    m = dict(args=dict(frozen_run='/remote/run', steps=None, batch_size=None,
                      jitter_sigma=cfg.repulsion.jitter_sigma, seed=0),
             sources={f'/remote/run/{p}': sha(source / p) for p in HASHED},
             status='complete', source_hashes_unchanged=True, git_commit='synthetic',
             selected={'fit_grid': spec}, trajectory_shape=[3, 40, 2])
    from umaping.umap_forces import find_ab_params
    a,b=find_ab_params(cfg.umap.spread,cfg.umap.min_dist)
    m['force']=dict(a=a,b=b,epsilon=.001,clip=cfg.flow.grad_clip)
    (benchmark / 'config/manifest.json').write_text(json.dumps(m))
    return source, benchmark


def test_guards_and_loss_conversion(tmp_path):
    with pytest.raises(ValueError, match='overlap'):
        guard_output(tmp_path / 'source/new', tmp_path / 'source')
    with pytest.raises(ValueError, match='overlap'):
        guard_output(tmp_path / 'new', tmp_path / 'new/child')
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(FileExistsError):
        guard_output(existing)
    loss = tmp_path / 'loss.csv'
    loss.write_text('step,loss,rolling_std\n1,0.5,\n2,0.125,0.2\n')
    assert convert_losses(loss, 2) == {'losses': [.5, .125]}
    with pytest.raises(ValueError, match='every step'):
        convert_losses(loss, 3)


def test_materialization_and_hash_failure(frozen, tmp_path):
    source, benchmark = frozen
    before = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'promoted'
    materialize(source, benchmark, output)
    for rel in UPSTREAM:
        assert sha(source / rel) == sha(output / rel)
        assert not (output / rel).is_symlink()
    assert json.loads((output / 'metrics/repulsion_training.json').read_text())['losses'] == [.4, .2]
    assert not (output / 'metrics/embedding.json').exists()
    assert before == {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        materialize(source, benchmark, output)
    (source / 'memory/reference_features.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        materialize(source, benchmark, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_incompatible_checkpoint(frozen, tmp_path):
    source, benchmark = frozen
    path = benchmark / 'repulsion_training/fit_grid/repulsion_field.pt'
    ckpt = torch.load(path, weights_only=True)
    ckpt['hparams']['hidden_dim'] += 1
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='incompatible'):
        materialize(source, benchmark, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()

from umaping.fit_grid_experiment.temporal import audit_labels, temporal_split
from umaping.fit_grid_experiment.metrics import temporal_metrics, aggregate_groups, paired_table


@pytest.mark.parametrize('labels,order', [
    (['0-1','2-3','4-5','6-7','8-9'],['0-1','2-3','4-5','6-7','8-9']),
    (['day_10','day_0','day_3','day_7','day_20'],['day_0','day_3','day_7','day_10','day_20']),
    (['E9.5','E8.5','E10.5','E9.0','E10.0'],['E8.5','E9.0','E9.5','E10.0','E10.5']),
    ([str(i) for i in range(10,0,-1)],[str(i) for i in range(1,11)]),
    (['2020-01-01','2020-03-01','2020-02-01','2020-05-01','2020-04-01'],
     ['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01'])])
def test_chronology_and_dynamic_split(labels,order):
    labels=np.repeat(labels,3)
    split=temporal_split(labels,np.arange(len(labels)), 'time')
    assert split['ordered_timepoints']==order
    m=min(max(int(.6*len(order)),2),len(order)-2)
    assert split['interpolation_timepoint']==order[m-1]
    assert split['last_observed_timepoint']==order[m]
    assert split['reference_timepoints']==order[:m-1]+[order[m]]
    assert split['query_timepoints']==[order[m-1]]+order[m+1:]
    assert not set(split['reference_indices']) & set(split['query_indices'])
    assert len(split['reference_indices'])+len(split['query_indices'])==len(labels)
    assert len(split['query_cell_ids'])==len(set(split['query_cell_ids']))
    for label in split['query_timepoints']:
        assert set(np.flatnonzero(labels==label)) <= set(split['query_indices'])


def test_chronology_ambiguity_and_insufficient_resolution():
    assert not audit_labels(['a','b','c','d','e'],'time')['valid_temporal_split']
    assert not audit_labels(['0','1','2','3'],'time')['valid_temporal_split']
    assert not audit_labels(['0-3','2-4','5-6','7-8','9-10'],'time')['valid_temporal_split']
    with pytest.raises(ValueError,match='every actual label'):
        audit_labels(['a','b','c','d','e'],'time',['a','a','b','c','d'])
    assert audit_labels(['a','b','c','d','e'],'time',['b','a','c','d','e'])['valid_temporal_split']
    with pytest.raises(ValueError,match='at least five'):
        temporal_split(['1','2','3','4'],np.arange(4),'time')
    with pytest.raises(ValueError,match='unique'):
        temporal_split(['1','2','3','4','5'],['same']*5,'time')


@pytest.mark.parametrize('n',[5,7,10,12])
def test_temporal_metrics_actual_labels_and_horizons(n):
    labels=[f'day_{i*3}' for i in range(n)]
    split=temporal_split(labels,np.arange(n),'day')
    ref=np.asarray(split['reference_ordinals']);query=np.asarray(split['query_ordinals'])
    m=split['interpolation_ordinal']
    bracket=[int(np.flatnonzero(ref==m-1)[0]),int(np.flatnonzero(ref==m+1)[0])]
    ids=np.tile(bracket,(len(query),1))
    frame=temporal_metrics(ref,query,ids,split)
    assert frame.timepoint.tolist()==split['query_timepoints']
    assert frame.horizon.tolist()==[0]+list(range(1,n-m))
    assert frame.temporal_bracketing_rate.iloc[0]==1
    assert frame.timepoint_label_accuracy.isna().all()
    assert (frame.excess_temporal_neighbor_mae.iloc[1:]==1).all()
    frame['query_index']=np.arange(len(frame));frame['recall_at_15']=.5
    agg=aggregate_groups(frame,split)
    assert set(agg.group)=={'all','interpolation','extrapolation'}|{f'timepoint:{label}' for label in split['query_timepoints']}
    assert set(agg.label_accuracy_status)=={'not_applicable_unseen_timepoints'}
    other=frame.copy();other['recall_at_15']+=.1
    paired,stats=paired_table(frame,other,seed=7,split=split)
    paired2,stats2=paired_table(frame,other,seed=7,split=split)
    pd.testing.assert_frame_equal(stats,stats2)
    assert np.allclose(paired.delta_recall_at_15,.1)


def test_comparison_outputs_and_read_only_sources(frozen,tmp_path):
    from umaping.fit_grid_experiment.comparison import compare
    source,benchmark=frozen;fit=tmp_path/'fit';materialize(source,benchmark,fit)
    before={str(p):sha(p) for run in (source,fit) for p in run.rglob('*') if p.is_file()}
    out=tmp_path/'comparison'
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        compare(source,fit,out,torch.device('cpu'),include_baselines=False)
    finally:
        torch.set_num_threads(old)
    assert json.loads((out/'manifest.json').read_text())['status']=='complete'
    assert len(list((out/'figures').glob('*.png')))==6
    assert len(pd.read_csv(out/'paired_per_query.csv'))==8
    assert before=={str(p):sha(p) for run in (source,fit) for p in run.rglob('*') if p.is_file()}


def test_temporal_preprocessing_and_matched_training(tmp_path,monkeypatch):
    import anndata as ad
    from types import SimpleNamespace
    from umaping.fit_grid_experiment.temporal import run_temporal
    from umaping.fit_grid_experiment import comparison
    rng=np.random.default_rng(2)
    adata=ad.AnnData(rng.poisson(3,size=(50,30)).astype(np.float32))
    adata.obs['time']=np.repeat(['0-1','2-3','4-5','6-7','8-9'],10)
    adata.obs_names=[f'cell_{i}' for i in range(50)]
    raw=tmp_path/'synthetic.h5ad';adata.write_h5ad(raw)
    cfg=Config.load('configs/mock.yaml');cfg.dataset.name='embryoid_body'
    cfg.dataset.params={'n_hvg':20,'n_pcs':4};cfg.dataset.input_dim=4
    cfg.retriever.steps=cfg.spectral.steps=cfg.repulsion.steps=2;cfg.flow.n_steps=2
    config=tmp_path/'config.yaml';cfg.save(config)
    # Production comparisons are separately exercised; test the entire matched-training orchestration.
    actual_compare=comparison.compare
    def small_compare(u,f,o,device,**kwargs):
        return actual_compare(u,f,o,device,temporal=kwargs.get('temporal'),include_baselines=False)
    monkeypatch.setattr(comparison,'compare',small_compare)
    args=SimpleNamespace(data=raw,group_column='time',group_order=None,config=config,
                         run_root=tmp_path/'runs',device='cpu',seeds=[0,1,2],cell_type_column=None)
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        run_temporal(args)
    finally:
        torch.set_num_threads(old)
    u=args.run_root/'temporal_holdout_uniform';f=args.run_root/'temporal_holdout_fit_grid'
    split=json.loads((u/'temporal_split.json').read_text())
    assert split['query_timepoints']==['4-5','8-9']
    for rel in UPSTREAM:
        assert sha(u/rel)==sha(f/rel)
    for seed in [1,2]:
        other=args.run_root/f'temporal_seed_{seed}_uniform'
        for rel in UPSTREAM:
            assert sha(u/rel)==sha(other/rel)
    assert len(pd.read_csv(args.run_root/'comparisons/temporal_holdout/matched_seed_results.csv'))==30
    fitted=np.load(u/'cache/reference_fitted_preprocessing.npz')
    assert fitted['reference_indices'].tolist()==split['reference_indices']
    assert json.loads((u/'temporal_manifest.json').read_text())['status']=='complete'


def test_query_only_preprocessing_changes_do_not_change_reference(tmp_path):
    import anndata as ad
    from umaping.data._scrna_common import prepare_continuous_scrna_dataset
    rng=np.random.default_rng(5)
    data=ad.AnnData(rng.poisson(5,size=(50,40)).astype(np.float32))
    data.obs['time']=np.repeat(['0-1','2-3','4-5','6-7','8-9'],10)
    options=dict(grouping_column='time',query_groups=('4-5','8-9'),n_hvg=20,n_pcs=4,seed=0)
    artifacts={}
    before=prepare_continuous_scrna_dataset(data,**options,artifacts=artifacts)
    changed=data.copy();changed.X[np.isin(changed.obs.time,options['query_groups'])]*=np.arange(1,41)
    after_artifacts={}
    after=prepare_continuous_scrna_dataset(changed,**options,artifacts=after_artifacts)
    np.testing.assert_array_equal(before.reference_features,after.reference_features)
    np.testing.assert_array_equal(artifacts['hvg_genes'],after_artifacts['hvg_genes'])
    np.testing.assert_array_equal(artifacts['pca_components'],after_artifacts['pca_components'])


def test_baseline_registry_and_standard_suite_in_new_workspaces(frozen,tmp_path,monkeypatch):
    from umaping.fit_grid_experiment.comparison import compare
    from umaping.experiments import baselines
    import umaping.pipeline as pipeline
    source,benchmark=frozen;fit=tmp_path/'fit';materialize(source,benchmark,fit)
    calls=[]
    def fake_suite(run,device):
        calls.append(run)
        assert run not in (source,fit)
    for name in ['run_evaluation','run_analysis','run_advanced_analysis']:
        monkeypatch.setattr(pipeline,name,fake_suite)
    for name in ['run_parametric_umap_baseline','run_numap_baseline','run_param_repulsor_baseline']:
        monkeypatch.setattr(baselines,name,lambda *a,_name=name,**kw:baselines.BaselineUnavailable(_name,'synthetic unavailable'))
    out=tmp_path/'comparison'
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        compare(source,fit,out,torch.device('cpu'),standard_suite=True)
    finally:
        torch.set_num_threads(old)
    assert len(calls)==6
    table=pd.read_csv(out/'aggregate.csv')
    assert (table.status=='unavailable').sum()==3
    assert {'standard_umap','reduced_repulsion_umap','weighted_knn','exact_repulsion_diagnostic','fit_grid_oracle_neighbors','no_repulsion'}<=set(table.method)
    assert json.loads((out/'oracle_query_indices.json').read_text())==np.random.default_rng(0).choice(8,size=8,replace=False).tolist()
