"""Identity regression tests. Synthetic examples are never scientific results."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('pancreas_review',Path(__file__).with_name('pancreas_visual_review.py'))
review=importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.annotations={s:pd.DataFrame({'celltype':['alpha','beta','ductal'],'tech':['a','b','c']},index=pd.Index([0,1,2],name='index')) for s in ('reference','query')}
        self.metrics=pd.DataFrame({'label':['alpha','beta','ductal'],'recall_at_5':[.2,.4,.6],'recall_at_15':[.1,.2,.3]},index=pd.Index([0,1,2],name='query_index'))
        self.y=np.array([[1.,2.],[3.,4.],[5.,6.]])

    def tearDown(self): self.tmp.cleanup()

    def pack(self,**changes):
        order=np.array([2,0,1]); data=dict(reference=self.y[order],query=self.y[order],reference_index=order,query_index=order,prepared_dataset_sha256=np.array('verified'))
        data.update(changes); p=self.root/'embedding.npz'; np.savez(p,**data); return p

    def test_shuffled_coordinates_align_by_explicit_id(self):
        frames,check=review.load_embedding(self.pack(),self.annotations,self.metrics,'verified')
        np.testing.assert_array_equal(frames['query'][['x','y']],self.y)
        self.assertTrue(check['explicit_reference_ids'])

    def test_missing_reference_ids_is_not_guessed(self):
        p=self.root/'legacy.npz'; np.savez(p,reference=self.y,query=self.y,query_index=np.arange(3))
        with self.assertRaisesRegex(ValueError,'reference_index'): review.load_embedding(p,self.annotations,self.metrics,'verified')

    def test_duplicate_and_out_of_range_id_rejected(self):
        for index in (np.array([0,0,2]),np.array([0,1,3])):
            with self.assertRaises(ValueError): review.load_embedding(self.pack(query_index=index),self.annotations,self.metrics,'verified')

    def test_hash_or_labels_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'hash mismatch'): review.load_embedding(self.pack(),self.annotations,self.metrics,'wrong')
        m=self.metrics.copy(); m.loc[0,'label']='gamma'
        with self.assertRaisesRegex(ValueError,'celltype'): review.load_embedding(self.pack(),self.annotations,m,'verified')

    def test_metric_rows_are_joined_by_id(self):
        rows=[]
        for method in review.METHODS:
            f=self.metrics.reset_index().copy(); f['method']=method
            rows.append(f.sample(frac=1,random_state=len(method)))
        p=self.root/'metrics.csv'; pd.concat(rows).to_csv(p,index=False)
        frames=review.load_metrics(p)
        self.assertTrue(frames['ours_full'].index.equals(frames['standard_umap'].index))
        wrong=pd.concat(rows); wrong.loc[wrong.method=='ours_full','query_index']=[3,4,5]; wrong.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'different query ID'): review.load_metrics(p)

    def test_existing_output_rejected(self):
        args=review.parser().parse_args(['--run',str(self.root),'--output',str(self.root)])
        with self.assertRaisesRegex(ValueError,'Output already exists'): review.Review(args)


def make_visual_fixture(root):
    """Full-format visual fixture with intentional row permutations and all legends."""
    root=Path(root); run=root/'fixture_run'; (run/'cache').mkdir(parents=True); (run/'metrics').mkdir(); (run/'embeddings').mkdir()
    celltypes=['acinar','activated_stellate','alpha','beta','delta','ductal','gamma','endothelial','epsilon','macrophage','mast','quiescent_stellate','schwann']
    ref_labels=np.repeat(celltypes,100); query_labels=np.concatenate([np.repeat(t,110 if i<7 else 5) for i,t in enumerate(celltypes)])
    rng=np.random.default_rng(491)
    reference_tech=np.resize(np.array(['in_drop','celseq']),len(ref_labels)); query_tech=np.resize(np.array(['smartseq2','celseq2']),len(query_labels))
    np.savez(run/'cache/prepared_dataset.npz',reference_features=np.zeros((len(ref_labels),2)),query_features=np.zeros((len(query_labels),2)),
             reference_label__celltype=ref_labels,query_label__celltype=query_labels,reference_label__tech=reference_tech,query_label__tech=query_tech)
    (run/'metadata.json').write_text('{"repulsion_teacher":"uniform_mc"}')
    (run/'config.yaml').write_text('dataset:\n  name: pancreas\n  params:\n    batch_correction: true\n    query_tech: [smartseq2, celseq2]\n')
    h=review.sha(run/'cache/prepared_dataset.npz'); metrics=[]
    centers={t:np.array([np.cos(i/len(celltypes)*2*np.pi),np.sin(i/len(celltypes)*2*np.pi)])*7 for i,t in enumerate(celltypes)}
    for n,method in enumerate(review.METHODS):
        data={}
        for split,labels in [('reference',ref_labels),('query',query_labels)]:
            y=np.array([centers[t] for t in labels])+rng.normal(0,.5,(len(labels),2))
            if n: y=y@np.array([[0.,-1.],[1.,0.]])+np.array([3.,-4.])
            order=rng.permutation(len(labels)); data[split]=y[order]; data[split+'_index']=order
        data['prepared_dataset_sha256']=np.array(h); np.savez(run/'embeddings'/f'{method}.npz',**data)
        metrics.append(pd.DataFrame(dict(method=method,query_index=np.arange(len(query_labels)),label=query_labels,
                        recall_at_5=rng.integers(0,6,len(query_labels))/5,recall_at_15=rng.integers(0,16,len(query_labels))/15)).sample(frac=1,random_state=n))
    pd.concat(metrics).to_csv(run/'metrics/advanced_per_query.csv',index=False)
    expr=[]
    for split,labels in [('reference',ref_labels),('query',query_labels)]:
        f=pd.DataFrame(dict(split=split,index=np.arange(len(labels)),total_counts=10000))
        for gene in review.GENES: f[gene]=rng.integers(0,100,len(labels)).astype(float)
        expr.append(f)
    pd.concat(expr).sample(frac=1,random_state=42).to_csv(run/'markers.csv',index=False)
    review.json_write(run/'markers.csv.json',{'prepared_dataset_sha256':h})
    return run


if __name__=='__main__': unittest.main()
