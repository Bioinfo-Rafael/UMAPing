#!/usr/bin/env python3
"""A+B only: retain the original A/B geometry, remove C and bridge_BC before fitting."""
import argparse,json,logging,shutil,time
from pathlib import Path
import run as core
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.sparse import save_npz,load_npz
from scipy.sparse.csgraph import connected_components
from geometry import generate,graph_audit
import plots
from report import table


def prepare(run):
    if (run/'data/prepared.done').exists():return
    original,x,A=generate();keep=original.structure.isin([0,1,3]).to_numpy()
    df=original[keep].reset_index(drop=True);features=x[keep]
    assert set(df.structure)=={0,1,3} and len(df[df.structure<3])==3999
    df.to_csv(run/'data/all_points.csv',index=False)
    np.savez_compressed(run/'data/isometric_map.npz',A=A,x=features,ids=df.id.to_numpy())
    audits={}
    for condition in ['disconnected','sparse_bridge']:
        mask=(df.structure<3).to_numpy() if condition=='disconnected' else np.ones(len(df),bool)
        frame=df[mask].reset_index(drop=True);xx=features[mask]
        ref=np.flatnonzero(frame.split=='reference');q=np.flatnonzero(frame.split=='query');z=frame[['z0','z1']].to_numpy()
        out=run/'data'/condition;out.mkdir(exist_ok=True)
        frame.to_csv(out/'points.csv',index=False);frame.groupby(['structure','branch','split']).size().rename('n').to_csv(out/'split_counts.csv')
        np.savez_compressed(out/'arrays.npz',x=xx,z=z,reference=ref,query=q,ids=frame.id.to_numpy())
        for scope,ids in [('all',np.arange(len(frame))),('reference',ref)]:
            audit,idx,dist,g=graph_audit(frame.iloc[ids].reset_index(drop=True),xx[ids]);audits[condition+'_'+scope]=audit
            assert audit['components']==(2 if condition=='disconnected' else 1)
            assert audit['components_without_bridge']==2
            assert audit['wrong_turn_edges']==audit['wrong_branch_edges']==audit['unexpected_cross_structure_edges']==0
            np.savez_compressed(out/f'{scope}_knn.npz',indices=idx,distances=dist);save_npz(out/f'{scope}_knn_graph.npz',g)
        pca=PCA(2,svd_solver='full').fit(xx[ref]);y=pca.transform(xx)
        u,_,vt=np.linalg.svd((y[ref]-y[ref].mean(0)).T@(z[ref]-z[ref].mean(0)))
        aligned=(y-y[ref].mean(0))@(u@vt)+z[ref].mean(0)
        error=float(np.max(abs(aligned-z)));assert error<1e-9
        core.js(out/'sanity.json',dict(n_internal=3999,n_bridge=int((frame.structure>=3).sum()),n_reference=len(ref),n_query=len(q),orthonormal_error=float(np.max(abs(A.T@A-np.eye(2)))),pca_recovery_max_error=error,pca_explained_variance=float(pca.explained_variance_ratio_.sum())))
        np.savez_compressed(out/'pca.npz',reference=y[ref],query=y[q],mean=pca.mean_,components=pca.components_)
        model=run/'models'/condition;model.mkdir(exist_ok=True);np.savez_compressed(model/'pca_model.npz',mean=pca.mean_,components=pca.components_)
    core.js(run/'metrics/input_graph_audit.json',audits)
    plots.truth_figures(run)
    (run/'data/prepared.done').write_text(core.now());core.freeze_protocol(run)


def audit(run):
    core.verify_frozen(run);records=[]
    d=np.load(run/'data/disconnected/arrays.npz');b=np.load(run/'data/sparse_bridge/arrays.npz')
    for key in ['x','z','ids']:np.testing.assert_array_equal(d[key],b[key][:3999])
    for key in ['reference','query']:np.testing.assert_array_equal(d[key],b[key][b[key]<3999])
    for condition in ['disconnected','sparse_bridge']:
        frame=pd.read_csv(run/'data'/condition/'points.csv');assert not frame.structure.isin([2,4]).any()
        emb=run/'embeddings'/condition/'seed_0';models=run/'models'/condition/'seed_0'
        u=np.load(emb/'uniform.npz');f=np.load(emb/'fitgrid.npz')
        np.testing.assert_array_equal(u['reference'],f['reference']);np.testing.assert_array_equal(u['retrieved'],f['retrieved'])
        pu=json.loads((models/'uniform/provenance.json').read_text());pf=json.loads((models/'fitgrid/provenance.json').read_text())
        for key in ['initial_sha256','trajectory_sha256','retriever_sha256','spectral_sha256','query_sequence_seed']:assert pu[key]==pf[key]
        for method in ['pca','umap','uniform','fitgrid']:
            y=np.load(emb/f'{method}.npz');assert np.isfinite(y['reference']).all() and np.isfinite(y['query']).all()
            points=pd.read_csv(run/'metrics'/condition/'seed_0'/f'{method}_points.csv.gz');assert len(points)==len(frame) and points.id.is_unique
            assert points.scale.nunique()==1 and abs(points[points.split=='reference'].log_radius15.median())<1e-8
            if method=='pca':assert points.recall15.min()==1 and points.abs_log_radius15.max()<1e-8
        for method,path in [('umaping',models/'shared/memory/graph_symmetric.npz'),('umap',models/'standard_umap_graph.npz')]:
            g=load_npz(path);nc,_=connected_components(g,directed=False);assert nc==(2 if condition=='disconnected' else 1)
            records.append(dict(condition=condition,method=method,components=int(nc)))
    core.js(run/'logs/ab_artifact_audit.json',dict(passed=True,shared_internal_count=3999,excluded_structure_ids=[2,4],matched_seed=0,graphs=records))


def report(run):
    summary=pd.read_csv(run/'metrics/all_summary.csv');primary=summary[(summary.split=='query')&(summary.group=='internal')]
    status=json.loads((run/'status.json').read_text());elapsed=time.time()-status['started_epoch']
    lines=['# A+Bのみ：螺旋・分岐木の比較（seed 0）',
           'C（楕円）とB–C橋を学習入力から削除し、PCA・通常UMAP・UMAPing Uniform/FitGridを再学習しました。A/Bの座標・ID・split・等長50D写像は前回と同一です。前回の図からCを非表示にしただけの結果ではありません。',
           '内部点はA 2,001点＋B 1,998点＝3,999点、reference 2,666／query 1,333。sparse_bridgeにはA–B橋240点（reference160／query80）のみ追加します。両条件で内部点・splitは共通。seed 0の1回ずつです。',
           '学習条件と評価式は[README](../../README.md)およびconfigsを継承。Uniform/FitGridはtrajectory・Retriever・Spectral・初期値・学習位置列を共有しteacherだけ変更。係数は引力1＋反発1、FitGrid256、Uniform64。通常UMAPはreferenceのみfitしてqueryをtransform。今回は両条件のreference数が4096未満で通常UMAPの距離全計算経路を使用。',
           '入力kNN検査は橋なし2成分／橋あり1成分、橋除去後2成分。事前規則で別の巻き・枝への近道および意図しない構造間接触0。PCAによる元形状回復と、両teacherの上流一致も検証しました。',
           '## 共通内部queryの比較',table(primary,['condition','method','recall5','recall15','within_recall15','log_radius15','abs_log_radius15']),
           '半径誤差はreference全体から求めた単一倍率を補正した自然対数比です。負は圧縮、正は膨張。群別・query別の再スケーリングはせず、島の大きさだけで改善とはしません。',
           '## 構造別・reference/query別',table(summary[summary.group.isin(['A_spiral','B_tree','bridge_AB'])],['condition','method','split','group','within_recall15','log_radius15','abs_log_radius15']),
           '## 橋の保持',table(summary[(summary.condition=='sparse_bridge')&(summary.group=='all')],['method','split','bridge_edge_loss15','bridge_cross_edge_loss15','wrong_turn15','wrong_branch15']),
           '## 図', '![正解](figures/01_ground_truth_geometry.png)','![入力グラフ](figures/02_input_reference_graph.png)']
    for condition in ['disconnected','sparse_bridge']:
        lines.append('### '+condition)
        for stem in ['10_structure','11_branch','12_progress','13_density','14_error','15_recall','20_progress_spiral_zoom','21_progress_branch_zoom','23_branch_tree_full']+(['22_progress_bridge_zoom'] if condition=='sparse_bridge' else []):
            lines.append(f'![{stem}](figures/{condition}/seed_0/{stem}.png)')
    lines+=['## 再実行',f'```sh\n.venv/bin/python SyntheticAnalysis/scripts/run_ab.py --run {run.relative_to(core.ROOT)} --phase train\n.venv/bin/python SyntheticAnalysis/scripts/run_ab.py --run {run.relative_to(core.ROOT)} --phase plot\n```',
            f'所要時間（run開始から）{elapsed/60:.2f}分。保存先はdata/models/embeddings/metrics/figures/logsに分離。PNGのみ保存、PDFなし。既存runは変更していません。',
            'この無ノイズ等長写像条件・1 seedの観測であり、安定した優位性や高難度データへの一般化、引力過多を原因として断定しません。']
    (run/'report.md').write_text('\n\n'.join(lines)+'\n');status.update(status='complete',wall_seconds=elapsed,report_updated=core.now());core.js(run/'status.json',status)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run');parser.add_argument('--phase',choices=['prepare','train','plot'],default='prepare');args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);plots.SAVE_PDF=False
    run=core.initialize(args);print('RUN_DIR='+str(run),flush=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(run/'logs/pipeline.log'),logging.StreamHandler()]);logging.getLogger('fontTools').setLevel(logging.WARNING)
    try:
        if not args.run:
            protocol=json.loads((run/'configs/protocol.json').read_text());protocol.update(variant='AB_only',learning_seeds=[0],internal_n=3999,bridge_n=240,excluded_structure_ids=[2,4],export_formats=['png'],expansion_rule='No extension: user requested one seed',geometry_rule='Filter fixed original data to IDs of A, B, bridge_AB; retain coordinates, IDs, split, linear map')
            core.js(run/'configs/protocol.json',protocol)
        assert json.loads((run/'configs/protocol.json').read_text()).get('variant')=='AB_only'
        core.verify_frozen(run);prepare(run)
        if args.phase=='train':
            for condition in ['disconnected','sparse_bridge']:core.train_condition(run,condition,0)
        if args.phase in ['train','plot']:
            plots.all_figures(run)
            for condition in ['disconnected','sparse_bridge']:
                plots.panels(run,condition,0,'branch',23,subset=lambda d:d.structure==1,suffix='_tree_full')
            audit(run);report(run)
    finally:(run/'execution.lock').unlink(missing_ok=True)
if __name__=='__main__':main()
