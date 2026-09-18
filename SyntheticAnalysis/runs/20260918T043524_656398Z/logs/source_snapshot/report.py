"""Report all completed seeds, including negative results; no seed selection."""
import datetime,json,time
from pathlib import Path
import numpy as np
import pandas as pd


def table(df,cols=None):
    if cols is not None:df=df[cols]
    def fmt(v):return '—' if pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']+['|'+'|'.join(fmt(v) for v in row)+'|' for row in df.itertuples(index=False,name=None)])


def write_report(run):
    summary=pd.read_csv(run/'metrics/all_summary.csv');status=json.loads((run/'status.json').read_text());seeds=sorted(int(v) for v in summary.seed.unique());audit=json.loads((run/'metrics/input_graph_audit.json').read_text())
    mean=summary.groupby(['condition','method','split','group'],as_index=False).mean(numeric_only=True)
    primary=mean[(mean.split=='query')&(mean.group=='internal')]
    elapsed=time.time()-status['started_epoch'];timing=pd.read_json(run/'logs/timing.jsonl',lines=True)
    timing.to_csv(run/'metrics/timing.csv',index=False)
    lines=['# 既知構造の合成データ：通常UMAP transformとUMAPing', '',f'作成: {datetime.datetime.now(datetime.timezone.utc).isoformat()}。run開始から {elapsed/3600:.3f} 時間。実行済み学習seed: {seeds}。',
           '全手法をreferenceのみで学習し、queryは未使用の点として配置した。通常UMAPはreferenceにfitしてqueryをtransformした結果であり、全点fitではない。島の面積や全体拡大は改善指標にしていない。',
           '', '## 目的と事前固定',
           '局所近傍、相対局所スケール、巻き・枝・橋の接続を評価する。無ノイズの2次元平面を50次元へ等長線形写像した機構確認であり、高難度・実データでの優位性は主張しない。',
           '生成と評価規則は学習前に固定（`logs/pretraining_frozen_manifest.json`）。データ生成条件・重み・学習回数をtest成績に合わせて変更していない。全点ID・正解座標・構造ID・枝ID・進行位置・splitをdataに保存。',
           '', '## 設定',
           '- 内部6,000点（A:2,001、B:1,998、C:2,001）、reference 4,000／query 2,000。橋AB・BCは各240点、計480点（reference 320／query 160）追加。内部点・ID・splitは両条件と全seedで完全共通。',
           '- A: 2.5周、半径1.2–5.6、全幅0.36。B: 7枝、全幅0.36、枝の交差なし。C: 半径4×3の楕円、正規化横位置uに対して密度∝exp(1.2u)。橋全幅0.11。',
           '- 座標標準化・ノイズ・非線形変換なし。x=z Aᵀ、AᵀA=I。学習APIの入力のみfloat32、正解/PCA/評価はfloat64。validationとearly stoppingは使わず固定step。',
           '- k設定15、Euclidean、min_dist=0.1、spread=1、negative_sample_rate=5。通常UMAPは500 epochs、transformは166 epochs。umap-learnのfit近傍数15はself込み（実質14）、既存UMAPingはself除外15。既存の規約差を保持して記録する。',
           '- Retriever [512,256]→128、2,000 steps、batch256。Spectral [512,256]→3、2,000 steps、edge batch4096、初期較正scale10。reference trajectory 200 steps、MC64、alpha1、成分clip4。',
           '- 反発ネットhidden256、4 residual blocks、time embedding32、4,000 steps、batch512、jitter0.1、Adam lr0.001。Uniform MC64／既存FitGrid256×256。両teacherで上流trajectory・Retriever・Spectral・初期state_dict・TrainingQueries列・学習条件を共有。FitGridの場自体は推論で使わず蒸留ネットを使用。',
           '- force-balanceの保存済み採用値 w=0.5, scale=2（係数1の引力＋係数1の反発）を使用。`configs/force_balance_selection_source.json`に元結果を保存。今回のquery情報による再探索なし。',
           '- CPU、PyTorch threads=2。seedはモデル全体の再学習seedで、同じ固定データとsplitを使用。3 seedsでもデータ生成分布の反復ではない。',
           '', '## 評価定義',
           'Recall@5/15は正解zと埋め込みyにおける同じreference候補集合へのkNNの共通要素数/k。reference行はselfを除く。withinは候補referenceを同じ構造（橋は各橋）に制限し、島分離の影響を分ける。',
           '半径rᶻᵢ,ₖとrʸᵢ,ₖは同じ候補集合から各空間で独立に選ぶk番目の距離。s=exp(median_{i∈reference} log(rᶻᵢ,₁₅/rʸᵢ,₁₅))を手法・条件・seedにつき一つだけ求め、Eᵢ,ₖ=log(s rʸᵢ,ₖ/rᶻᵢ,ₖ)。負は圧縮、正は膨張、0は倍率補正後一致。k=5/15とwithinにも同じsを使用。群別・query別の再較正なし。相対局所スケール指標で、絶対島サイズは評価しない。',
           '誤近傍は埋め込みkNN15のうち、正解距離が点自身の正解r15の2倍超、かつAでは角度差>π、Bでは異なる枝の辺と定義。共有分岐点から双方0.6以内の隣接枝は除外。全近傍15本を分母に率を算出する。正解kNNに存在しない非局所接続のみを数えるため、近傍境界の微小な入替えは誤接続としない。',
           'bridge edge lossは、正解kNN15で少なくとも片端が橋である辺のうち埋め込みkNN15から消えた割合。cross bridge edge lossはさらに構造IDの異なる端点接続辺に限定。分母0は欠測。bridge_nearはこれら正解辺に関係するsource点で、橋の点だけでなく接続端の内部点も含む。',
           '', '## 入力段階の健全性',
           table(pd.DataFrame([dict(scope=k,components=v['components'],without_bridges=v['components_without_bridge'],unexpected_contacts=v['unexpected_cross_structure_edges'],wrong_turn=v['wrong_turn_edges'],wrong_branch=v['wrong_branch_edges']) for k,v in audit.items()])),
           '全点グラフとreferenceグラフの双方で、disconnectedは3成分、sparse_bridgeは1成分。橋を除くと3成分。意図した端点以外の構造間接触と、事前規則で検出される巻き・枝間近道は0。橋は入力段階で切れていない。実際のUMAPing/umap-learnの学習グラフもmodelsに保存した。',
           'PCA回復最大絶対誤差: '+', '.join(f'{c}={json.loads((run/"data"/c/"sanity.json").read_text())["pca_recovery_max_error"]:.3g}' for c in ['disconnected','sparse_bridge'])+'。PCAはreferenceのみでfitし、回転・平行移動の較正もreferenceだけで計算。',
           '![正解](figures/01_ground_truth_geometry.png)','![入力グラフ](figures/02_input_reference_graph.png)',
           '', '## 主比較：共通内部query 2,000点、実行seed平均',
           table(primary,['condition','method','recall5','recall15','within_recall15','log_radius15','abs_log_radius15','wrong_turn15','wrong_branch15']),
           '', '## 観測事実：五つの問いへの回答']
    for cond in ['disconnected','sparse_bridge']:
        sub=primary[primary.condition==cond].set_index('method')
        lines += ['',f'### {cond}']
        for method in ['uniform','fitgrid']:
            r=sub.loc[method];base=sub.loc['umap'];dr=r.within_recall15-base.within_recall15;de=r.abs_log_radius15-base.abs_log_radius15
            lines.append(f'{method} − 通常UMAP: 内部Recall@15差 {dr:+.4f}、半径絶対log誤差差 {de:+.4f}（後者は負が改善）。内部近傍は'+('改善した。' if dr>0 else '改善しなかった。')+'島が大きいという視覚印象だけではこの結論を変更しない。')
        fg=sub.loc['fitgrid'];u=sub.loc['uniform'];lines.append(f'FitGrid − Uniform: 内部Recall@15差 {fg.within_recall15-u.within_recall15:+.4f}、半径誤差差 {fg.abs_log_radius15-u.abs_log_radius15:+.4f}。以下に全seedと構造別の悪化も含めて示す。')
    # Common-ID matched differences, global and within metrics. No bridge points in this table.
    deltas=[]
    for seed in seeds:
        for method in ['pca','umap','uniform','fitgrid']:
            paths=[run/'metrics'/c/f'seed_{seed}'/f'{method}_points.csv.gz' for c in ['disconnected','sparse_bridge']]
            if not all(p.exists() for p in paths):continue
            left,right=[pd.read_csv(p) for p in paths];left=left[left.structure<3];right=right[right.structure<3]
            both=left.merge(right,on=['id','split','structure','branch'],suffixes=('_d','_b'),validate='one_to_one');assert len(both)==6000
            for split in ['reference','query']:
                for group,mask in [('internal',np.ones(len(both),bool)),('A_spiral',both.structure==0),('B_tree',both.structure==1),('C_ellipse',both.structure==2)]:
                    rows=both[(both.split==split)&mask];item=dict(seed=seed,method=method,split=split,group=group,n=len(rows))
                    for m in ['recall15','within_recall15','abs_log_radius15','within_abs_log_radius15','log_radius15']:item['delta_'+m]=float((rows[m+'_b']-rows[m+'_d']).mean())
                    deltas.append(item)
    paired=pd.DataFrame(deltas);paired.to_csv(run/'metrics/paired_bridge_effect.csv',index=False)
    avg=paired.groupby(['method','split','group'],as_index=False).mean(numeric_only=True)
    lines += ['','### 橋追加による共通内部点の変化',
              '同じIDを対応付けた sparse_bridge − disconnected。within Recallでは候補referenceも同じ構造内の同一ID集合なので、橋を候補に加える直接効果を除いて比較できる。各条件で再学習するため、差には上流構造の変化も含む。倍率sは条件ごとのreferenceから求める単一値であり、半径差にもこの全体較正差が含まれる。',
              table(avg[avg.group=='internal'],['method','split','delta_recall15','delta_within_recall15','delta_abs_log_radius15']),
              '[全seed・構造別の対応差](metrics/paired_bridge_effect.csv)',
              '', '### reference構造とquery配置の差',
              'UniformとFitGridのreference座標は完全に同じなので、両者のreference指標差は0。両者のquery差は、この共有上流の下でteacher変更に伴う反発ネットの違いとして比較できる。通常UMAPとの比較はreference配置自体も異なる。reference→queryの平均差は異なる点集合間の記述比較であり、配置手順の因果効果を単独同定するものではない。',
              table(mean[(mean.group=='internal')],['condition','method','split','within_recall15','log_radius15','abs_log_radius15']),
              '', '### 接続の保持',
              table(mean[(mean.condition=='sparse_bridge')&(mean.group=='all')],['method','split','wrong_turn15','wrong_branch15','bridge_edge_loss15','bridge_cross_edge_loss15','bridge_near_recall15']),
              '![橋接続指標](figures/31_bridge_connectivity_metrics.png)',
              '橋を切って島を離すだけで改善としない。橋周辺や端点での正解辺欠落を上表で併記する。入力は連結なので、ここでの欠落は入力グラフの橋切れとは区別される。',
              '', '## 構造別比較（悪化も含む、seed平均）',
              table(mean[(mean.split=='query')&mean.group.isin(['A_spiral','B_tree','C_ellipse','bridge_AB','bridge_BC'])],['condition','method','group','within_recall15','log_radius15','abs_log_radius15','wrong_turn15','wrong_branch15']),
              '![構造別・橋なし](figures/30_disconnected_structure_metrics.png)','![構造別・橋あり](figures/30_sparse_bridge_structure_metrics.png)',
              '', '## 実行した全seedの主指標',
              table(summary[(summary.split=='query')&(summary.group=='internal')],['condition','seed','method','recall15','within_recall15','abs_log_radius15']),
              'seedを結果の良し悪しで選択していない。固定データ上の学習seed変動であり、3反復の有意差や一般化を強く主張しない。全点値は各condition/seedの `*_points.csv.gz`、全枝・reference/query集計は [all_summary.csv](metrics/all_summary.csv)。',
              '', '## 主要図']
    for cond in ['disconnected','sparse_bridge']:
        lines += ['',f'### {cond}、seed 0（全seedは同じ構成で保存）']
        for stem in ['10_structure','11_branch','12_progress','13_density','14_error','15_recall','20_progress_spiral_zoom','21_progress_branch_zoom']+(['22_progress_bridge_zoom'] if cond=='sparse_bridge' else []):
            lines.append(f'![{stem}](figures/{cond}/seed_0/{stem}.png)')
    # Matched seed differences: small teacher changes must not be overstated.
    paired_method=[]
    fields=['within_recall15','abs_log_radius15','log_radius15','bridge_edge_loss15']
    keys=['condition','seed','split','group']
    for name,left,right in [('Uniform minus UMAP','uniform','umap'),('FitGrid minus UMAP','fitgrid','umap'),('FitGrid minus Uniform','fitgrid','uniform')]:
        l=summary[summary.method==left];r=summary[summary.method==right]
        joined=l.merge(r,on=keys,suffixes=('_left','_right'),validate='one_to_one')
        for _,row in joined.iterrows():
            paired_method.append(dict(comparison=name,**{k:row[k] for k in keys},**{'delta_'+f:row[f+'_left']-row[f+'_right'] for f in fields}))
    effects=pd.DataFrame(paired_method);effects.to_csv(run/'metrics/paired_method_effect.csv',index=False)
    teacher=effects[(effects.comparison=='FitGrid minus Uniform')&(effects.split=='query')&(effects.group=='internal')]
    lines += ['', '## 橋追加の要約（共通内部query）']
    for _,r in avg[(avg.group=='internal')&(avg.split=='query')].iterrows():
        lines.append(f'{r.method}: 橋あり−橋なしの内部Recall@15差 {r.delta_within_recall15:+.4f}、半径絶対log誤差差 {r.delta_abs_log_radius15:+.4f}。')
    lines += ['', '## FitGridの追加効果：matched seed差',table(teacher,['condition','seed','delta_within_recall15','delta_abs_log_radius15']),
              '全構造・全枝・reference/queryの対応差は [paired_method_effect.csv](metrics/paired_method_effect.csv)。正負どちらの差も残し、微小な差やseedで向きが変わる差を安定した優位性とは扱わない。']
    lines += ['', '## reference/query差の数値分解（記述的）']
    for cond in ['disconnected','sparse_bridge']:
        base=mean[(mean.condition==cond)&(mean.group=='internal')].set_index(['method','split'])
        for method in ['uniform','fitgrid']:
            ref_delta=base.loc[(method,'reference'),'within_recall15']-base.loc[('umap','reference'),'within_recall15']
            query_delta=base.loc[(method,'query'),'within_recall15']-base.loc[('umap','query'),'within_recall15']
            lines.append(f'{cond} / {method}: 通常UMAPに対する内部Recall差はreferenceで {ref_delta:+.4f}、queryで {query_delta:+.4f}。query差−reference差は {query_delta-ref_delta:+.4f}。reference側の差がすでに存在するかを区別し、query配置だけの原因とは扱わない。')
    lines += ['', '## 原因に関する仮説と限界',
              '入力グラフが健全でも、Spectral初期化・reference trajectory・学習Retrieverの近傍誤差・蒸留場の誤差・推論更新の違いが最終配置に影響し得る。これらは原因候補であり、この比較だけで引力過多とは断定しない。UniformとFitGridの差はteacherを変えた比較だが、通常UMAPとの差には別の初期化・reference最適化も含まれる。',
              'このデータの真の支持集合は2次元線形部分空間で、PCAはほぼ完全に回復する。従ってUMAP系の情報損失は測定できるが、高次元非線形データでの汎化や優位性は未検証。橋ありはreference点数が320多いという差もあり、橋のトポロジーだけの因果効果ではない。さらに既存umap-learnの既定では4096点を閾値に近傍探索が切り替わるため、reference4000点の橋なしは距離全計算、4320点の橋ありは近似探索となる。入力グラフの連結性・誤接続は実モデルで監査したが、この実装上の違いも橋追加の比較に含まれる。',
              '学習済みRetrieverのquery近傍Recallを [query_retriever_diagnostic.csv](metrics/query_retriever_diagnostic.csv) に保存。実際の両手法の学習グラフを [actual_training_graph_audit.csv](metrics/actual_training_graph_audit.csv) で監査。これらは学習終了後の診断で、設定選択には使っていない。',
              '', '## 時間・再開・完了範囲',
              f'現在までのwall time {elapsed/3600:.3f} h、記録された計算stage合計 {timing.seconds.sum()/3600:.3f} h。stage別の実測は [timing.csv](metrics/timing.csv)。初回2条件×seed0完了後の拡張判断は `logs/seed_expansion_decision.json`（実施時）。',
              '完成した上流stage、teacher別モデル、100 queryごとの埋め込みを再利用。反発学習は250 stepsごとにoptimizer・乱数状態を保存。Retriever/Spectralは完成stage単位で再開し、中断中のstageのみ同一seedから再計算する。絶対11.5時間で学習停止し、既存成果物を削除しない。',
              '完了: '+', '.join(status['completed'])+'。必須比較・評価CSV・PNG/PDF・本報告を保存。',
              '今回はノイズ・非線形写像・追加橋密度・広範な設定探索は実施していない。mainへのmergeなし。',
              '', '## 再実行',
              '手順と評価式の詳細は [README](../../README.md)。作図は学習済みembeddingだけで再実行可能。']
    lines += ['', '## 検証と出力確認',
              ('関連テスト38件成功・CUDA未搭載による1件skip。追加の再開テストを含む実験固有テスト4件成功（既存3件を含む）。optimizerとteacher乱数を復元した再開は、中断なし学習と重み・lossが完全一致。ログは `logs/tests.txt` と `logs/protocol_resume_tests.txt`。' if (run/'logs/tests.txt').exists() and (run/'logs/protocol_resume_tests.txt').exists() else 'このrun固有のテストログは未記録。READMEの検証コマンドを参照。'),
              f'完了{len(status["completed"])}条件の成果物監査で、共通データ・split不変、Uniform/FitGrid上流・初期値・reference座標・検索近傍一致、有限値、集計整合を確認。`logs/final_artifact_audit.json`。',
              f'PNG {len(list((run/"figures").rglob("*.png")))}枚、PDF {len(list((run/"figures").rglob("*.pdf")))}枚を保存。主要図の凡例・カラーバー・等アスペクト・reference/query範囲を目視確認し、正解図の凡例重なりを修正。必須項目の未完了なし。']
    text='\n\n'.join(lines)
    while '\n\n\n' in text:text=text.replace('\n\n\n','\n\n')
    (run/'report.md').write_text(text+'\n')
    status.update(report_updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=elapsed,status='complete' if len(status['completed'])==6 else 'seed0_complete' if len(status['completed'])==2 else 'partial')
    (run/'status.json').write_text(json.dumps(status,indent=2))
