"""Bounded unit tests of saved-artifact parsing/statistics; no experimental code.

All fixtures are hand-constructed toy tables in a temporary directory. They are
never used as scientific results or copied into final_analysis.
Run: .venv/bin/python -m unittest discover -s scripts -p 'test_final_saved_analysis.py'
"""
import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SPEC=importlib.util.spec_from_file_location("saved_analysis",Path(__file__).with_name("final_saved_analysis.py"))
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SavedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        args=argparse.Namespace(root=self.root,output=self.root/"out",max_seconds=600,
                                bootstrap=2000,seed=71,bootstrap_memory_mb=1)
        self.app=M.Analysis(args)
        self.comp=self.root/"runs/embryoid_body/comparisons/temporal_seed_1"
        self.comp.mkdir(parents=True)
        self.write_json(self.comp/"manifest.json",dict(status="complete",seed=0,source_hashes_unchanged=True))
        self.stage=np.array(["4-5"]*4+["8-9"]*6)
        self.ref=np.array(["0-1","2-3","6-7"])
        self.split=dict(ordered_timepoints=["0-1","2-3","4-5","6-7","8-9"],
            reference_ordinals=[1,2,4],query_ordinals=[3]*4+[5]*6,
            reference_timepoints=["0-1","2-3","6-7"],query_timepoints=["4-5","8-9"],
            interpolation_timepoint="4-5",extrapolation_timepoints=["8-9"],
            cells_per_timepoint={"0-1":1,"2-3":1,"4-5":4,"6-7":1,"8-9":6})
        for run in ("temporal_seed_1_uniform","temporal_seed_1_fit_grid"):
            base=self.root/"runs/embryoid_body"/run
            (base/"cache").mkdir(parents=True)
            self.write_json(base/"metadata.json",dict(matched_seed=1,seed=0))
            self.write_json(base/"temporal_split.json",self.split)
            np.savez(base/"cache/prepared_dataset.npz",reference_label__stage=self.ref,query_label__stage=self.stage)
        rows=[]
        for method,offset in (("uniform_mc",0),("fit_grid",.02)):
            f=pd.DataFrame(dict(method=method,query_index=np.arange(10),timepoint=self.stage,
                                recall_at_15=np.arange(10)/15+offset,ndcg=np.arange(10)/20+offset,
                                density_log_distortion=1+np.arange(10)/20-offset))
            f.sample(frac=1,random_state=7).to_csv(self.comp/f"per_query_{method}.csv",index=False)
            for group,mask in (("all",np.ones(10,bool)),("interpolation",self.stage=="4-5"),("extrapolation",self.stage=="8-9")):
                rows.append(dict(method=method,group=group,n_queries=int(mask.sum()),
                                 **f.loc[mask,M.METRICS].mean().to_dict()))
        pd.DataFrame(rows).to_csv(self.comp/"aggregate.csv",index=False)

    def tearDown(self):
        M.plt.close("all")
        self.tmp.cleanup()

    def write_json(self,path,data):
        path.write_text(json.dumps(data))

    def load(self):
        self.app.load_context("temporal1","temporal_seed_1","temporal_seed_1_uniform","temporal_seed_1_fit_grid",1,True)
        return self.app.contexts["temporal1"]

    def test_shuffled_ids_seed_and_strata(self):
        ctx=self.load()
        self.assertEqual(ctx["seed"],1)
        self.assertEqual(ctx["evaluation_seed"],0)
        self.assertEqual(set(ctx["frames"]),{"uniform_mc","fit_grid"})
        self.assertTrue(np.array_equal(ctx["frames"]["fit_grid"].stage,self.stage))
        self.assertEqual(len(self.app.checks),18)
        labels=np.array(["a","a","b"])
        values=np.array([[1.,2.],[1.,2.],[10.,20.]])
        ci,_=self.app.bootstrap(values,labels,42)
        np.testing.assert_allclose(ci,[[4.,8.],[4.,8.]])

    def test_duplicate_and_missing_ids_rejected(self):
        path=self.comp/"per_query_fit_grid.csv"
        f=pd.read_csv(path)
        f.loc[0,"query_index"]=f.loc[1,"query_index"]
        f.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"Duplicate query"):
            self.load()
        f.drop(index=0).to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"Missing full-query"):
            self.load()

    def test_stage_mismatch_rejected(self):
        path=self.comp/"per_query_fit_grid.csv"
        f=pd.read_csv(path)
        f.loc[0,"timepoint"]="0-1"
        f.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"Stage mismatch"):
            self.load()

    def test_wrong_training_seed_rejected(self):
        path=self.root/"runs/embryoid_body/temporal_seed_1_fit_grid/metadata.json"
        self.write_json(path,dict(matched_seed=0))
        with self.assertRaisesRegex(ValueError,"Training seed mismatch"):
            self.load()

    def test_aggregate_disagreement_rejected(self):
        path=self.comp/"aggregate.csv"
        f=pd.read_csv(path)
        f.loc[0,"recall_at_15"]+=.1
        f.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"Aggregate mismatch"):
            self.load()

    def test_paired_signs_and_finalization_after_deadline(self):
        ctx=self.load()
        self.app.paired(ctx)
        for row in self.app.pairs:
            self.assertAlmostEqual(row["mean_improvement"],.02)
            self.assertAlmostEqual(row["fraction_improved"],1)
            self.assertAlmostEqual(row["ci_low"],.02)
        self.app.start-=601
        with self.assertRaises(M.BudgetExceeded):
            self.app.check_time()
        self.app.status="time_budget_reached"
        self.app.finalize()
        self.assertTrue((self.app.out/"02_temporal_holdout/paired_improvement_seed_1.png").is_file())
        meta=json.loads((self.app.out/"99_provenance/analysis_manifest.json").read_text())
        self.assertEqual(meta["html_missing_links"],[])
        self.assertTrue(meta["used_sources_unchanged"])

    def test_forbidden_inputs_and_nonoverwrite(self):
        p=self.root/"reference_trajectory.npz"
        np.savez(p,x=[1])
        with self.assertRaisesRegex(ValueError,"Forbidden"):
            self.app.track(p)
        with self.assertRaisesRegex(ValueError,"overwrite"):
            M.Analysis(self.app.args)

    def test_embedding_label_join_and_colors(self):
        ctx=self.load()
        folder=self.comp/"embeddings"
        folder.mkdir()
        ids=np.array([9,1,8,2,7,3,6,4,5,0])
        for m in ("uniform_mc","fit_grid"):
            np.savez(folder/f"{m}.npz",query_index=ids,reference=np.array([[0.,0.],[1.,1.],[2.,0.]]),query=np.stack([ids/10,ids/20],axis=1))
        self.app.embeddings(ctx)
        table=pd.read_csv(self.app.out/"02_temporal_holdout/embedding_by_stage.csv")
        self.assertEqual(set(table.stage),set(self.split["ordered_timepoints"]))
        self.assertTrue(table.groupby("stage").color.nunique().eq(1).all())
        self.assertEqual(table[table.method=="fit_grid"].query_count.sum(),10)
        self.app.embeddings(ctx,highlight=True)

    def test_recall_k_validation_and_standard_deduplication(self):
        ctx=self.load()
        ctx["frames"]["standard_umap"]=ctx["frames"]["uniform_mc"].copy()
        for method in ("uniform_mc","fit_grid"):
            tables=[]
            for source,target in ((method,"ours_full"),("standard_umap","standard_umap")):
                f=ctx["frames"][source].reset_index()[["query_index","recall_at_15","stage"]].rename(columns={"stage":"label"})
                f["method"]=target
                for k in (5,10,30):
                    f[f"recall_at_{k}"]=f.recall_at_15
                tables.append(f)
            path=self.comp/f"{method}_evaluation/metrics/advanced_per_query.csv"
            path.parent.mkdir(parents=True)
            pd.concat(tables).to_csv(path,index=False)
        self.app.recall_k(ctx)
        result=pd.read_csv(self.app.out/"02_temporal_holdout/recall_at_k.csv")
        self.assertEqual(len(result),3*3*4)
        self.assertFalse(result.duplicated(["method","group","k"]).any())

    def test_field_regions_use_vector_rmse_and_remain_separate(self):
        self.load()
        base=self.comp/"exact_field"
        base.mkdir()
        exact=np.ones((1000,2))
        t=np.linspace(0,1,1000)
        np.savez(base/"validation.npz",exact=exact,t=t)
        metrics=[]
        for method,offset in (("uniform_mc",.2),("fit_grid",.1)):
            np.save(base/f"{method}.npy",exact+offset)
            metrics.append(dict(method=method,rmse=np.sqrt(2)*offset,cosine=1))
            path=self.comp/f"{method}_evaluation/metrics/advanced_analysis.json"
            path.parent.mkdir(parents=True)
            item=dict(n_eval_points=300,vs_exact=dict(b_phi=dict(mse_mean=offset**2,cosine_mean=.5)))
            self.write_json(path,dict(field_denoising=dict(per_t={"t=0.00":item,"t=1.00":item},mixed=item)))
        pd.DataFrame(metrics).to_csv(base/"metrics.csv",index=False)
        self.app.field_data()
        self.assertEqual(set(self.app.near.n_points),{1000})
        self.assertEqual(set(self.app.grid.n_points),{300})
        self.assertEqual(int(self.app.binned[self.app.binned.method=="uniform_mc"].n_points.sum()),1000)
        self.assertAlmostEqual(self.app.near.iloc[0].rmse,np.sqrt(2)*.2)
        for kind in ("near","near_time","grid_time","region","distribution"):
            self.app.field_plot(kind)
        table=pd.read_csv(self.app.out/"04_learned_field/field_accuracy_by_evaluation_region.csv")
        self.assertEqual(set(table.region),{"near reference trajectory","spatial grid"})
        np.testing.assert_allclose(table.fit_grid_over_uniform_rmse,.5)

    def test_five_method_and_temporal_auxiliary_plots(self):
        ctx=self.load()
        for method in M.METHODS:
            if method not in ctx["frames"]:
                ctx["frames"][method]=ctx["frames"]["uniform_mc"].copy()
            f=ctx["frames"][method]
            f["temporal_neighbor_mae"]=1.
            f["excess_temporal_neighbor_mae"]=np.where(f.stage=="8-9",0.,np.nan)
            f["temporal_bracketing_rate"]=np.where(f.stage=="4-5",1.,np.nan)
        self.app.contexts["temporal0"]=ctx
        self.app.performance(ctx)
        self.app.distributions(ctx)
        self.app.distributions(ctx,zero=True)
        self.app.temporal_metric("diagnostics")
        self.app.temporal_metric("recall_at_15")
        self.app.temporal_metric("density_log_distortion")


if __name__=="__main__":
    unittest.main()
