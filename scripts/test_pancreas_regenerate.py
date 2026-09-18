"""Small regenerated-coordinate integration test; no scientific data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from pancreas_regenerate import regenerate, digest
from pancreas_visual_review import load_embedding, load_prepared, load_metrics


def make_frozen_fixture(root,n_query=6):
    from test_inference import test_saved_model_can_be_reloaded_and_used_for_single_point_inference
    from umaping.config import Config
    from umaping.data.preprocessing import PreparedDataset,save_prepared_dataset
    # Existing inference fixture saves a small model and validates reload.
    test_saved_model_can_be_reloaded_and_used_for_single_point_inference(root)
    run=Path(root)/'run'; cfg=Config.load(run/'config.yaml')
    cfg.dataset.name='pancreas'; cfg.dataset.params={'batch_correction':True,'query_tech':['smartseq2','celseq2']}
    cfg.save(run/'config.yaml'); (run/'cache').mkdir(); (run/'metrics').mkdir()
    ref=np.load(run/'memory/reference_features.npy'); query=np.random.default_rng(2026).normal(size=(n_query,ref.shape[1])).astype('float32')
    labels=np.resize(np.array(['alpha','beta','ductal']),n_query)
    prepared=PreparedDataset(ref,query,{'celltype':np.resize(np.array(['alpha','beta','ductal']),len(ref)),
        'tech':np.resize(np.array(['celseq','in_drop']),len(ref))},
        {'celltype':labels,'tech':np.resize(np.array(['smartseq2','celseq2']),n_query)},ref.shape[1])
    save_prepared_dataset(run/'cache/prepared_dataset.npz',prepared)
    # Deliberately absurd old Recall values make accidental reuse obvious.
    pd.concat([pd.DataFrame(dict(method=m,query_index=np.arange(n_query),label=labels,recall_at_5=-123.,recall_at_15=-123.))
               for m in ('ours_full','standard_umap')]).to_csv(run/'metrics/advanced_per_query.csv',index=False)
    (run/'metadata.json').write_text('{"repulsion_teacher":"uniform_mc","fixture_only":true}')
    return run


def test_regenerate_new_metrics_ids_and_source_unchanged(tmp_path):
    run=make_frozen_fixture(tmp_path)
    before={p:digest(p) for p in run.rglob('*') if p.is_file()}
    output=regenerate(run,tmp_path/'regenerated',max_seconds=180)
    metrics=load_metrics(output/'advanced_per_query.csv'); prepared=load_prepared(run/'cache/prepared_dataset.npz')
    for method in ('standard_umap','ours_full'):
        data,checks=load_embedding(output/'embeddings'/f'{method}.npz',prepared,metrics[method],digest(run/'cache/prepared_dataset.npz'))
        assert len(data['query'])==6 and checks['prepared_hash_embedded']
        assert (metrics[method].recall_at_15>=0).all() # not the -123 old values
    assert all(digest(p)==h for p,h in before.items())
    import json
    manifest=json.loads((output/'regeneration.json').read_text())
    assert manifest['status']=='complete' and manifest['source_artifacts_unchanged']
    assert manifest['neural_network_training'] is False
    assert manifest['old_metrics_used_for_new_coordinates'] is False
    with pytest.raises(ValueError,match='already exists'): regenerate(run,output)


def test_timeout_preserves_failed_manifest(tmp_path):
    run=make_frozen_fixture(tmp_path)
    with pytest.raises(TimeoutError): regenerate(run,tmp_path/'timeout',max_seconds=.01)
    import json
    record=json.loads((tmp_path/'timeout/regeneration.json').read_text())
    assert record['status']=='failed' and 'time limit' in record['error']
