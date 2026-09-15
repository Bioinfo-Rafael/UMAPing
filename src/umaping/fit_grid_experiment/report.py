"""保存済み表から図と日本語報告を再生成する。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from umaping.comparison.reporting import markdown_table
from umaping.repulsion_estimators.benchmark import write_json


def make_report(output, aggregate, paired, stats, fields, frames, temporal=None):
    output=Path(output)
    success=aggregate[(aggregate.status=='success') & (aggregate.group=='all')].copy()
    def save(name):
        plt.tight_layout();plt.savefig(output/'figures'/f'{name}.png',dpi=160);plt.close()
    def bars(column,name,ylabel):
        fig,ax=plt.subplots(figsize=(10,5))
        ax.bar(success.method,success[column]);ax.tick_params(axis='x',labelrotation=35);ax.set_ylabel(ylabel)
        save(name)
    bars('recall_at_15','final_neighborhood_recall','Query-to-reference Recall@15')
    plt.figure(figsize=(7,4));plt.hist(paired.delta_recall_at_15,bins=31);plt.axvline(0,color='black',ls='--')
    plt.xlabel('Paired Recall@15: FitGrid - Uniform');save('paired_recall_difference')
    row=stats[stats.group=='all'].iloc[0]
    plt.figure(figsize=(6,4));plt.bar(['Improved','Tied','Worsened'],[row.fraction_ours_better,row.fraction_tied,row.fraction_worsened]);plt.ylim(0,1)
    save('improved_tied_worsened')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,col in zip(axes,['density_log_distortion','global_periphery_percentile']):
        ax.bar(success.method,success[col]);ax.set_ylabel(col);ax.tick_params(axis='x',labelrotation=60)
    save('local_density_periphery')
    fig,ax=plt.subplots(figsize=(8,5))
    for _,r in success.iterrows():
        if pd.notna(r.latency_seconds) and r.latency_seconds>0:
            ax.scatter(r.latency_seconds,r.recall_at_15);ax.annotate(r.method,(r.latency_seconds,r.recall_at_15),fontsize=8)
    ax.set_xscale('log');ax.set_xlabel('Embedding seconds/query (fit excluded)');ax.set_ylabel('Recall@15')
    save('accuracy_vs_latency')
    component=fields[['method','rmse','cosine','magnitude_error']].merge(success[['method','recall_at_15','latency_seconds']],on='method')
    component.to_csv(output/'component_vs_end_to_end.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,2));ax.axis('off')
    display=component.copy()
    for col in display.columns[1:]:
        display[col]=display[col].map(lambda x:f'{x:.6g}')
    ax.table(cellText=display.values,colLabels=display.columns,loc='center')
    save('component_vs_end_to_end')
    text=['# FitGrid最終埋め込み比較', '',
          '元runは読み取り専用。全手法は同じPreparedDatasetを使用。参照軌道生成は変更していない。',
          '主指標はquery→固定referenceのRecall@15。reference trustworthinessは同一参照の整合性確認のみ。',
          '通常評価のfield.jsonは従来のMC teacher診断。主たる場の評価はexact_field/validation.npzの全点和1000点。',
          'benchmarkのUniformモデルとmainのUniformモデルは別学習であり、混同しない。以下は今回実際に読み込んだモデルの比較。',
          '',markdown_table(component), '', '## 最終埋め込み（全query）',markdown_table(success),
          '', '## 対応比較',markdown_table(stats),
          'CIは固定モデル・参照に条件付けたcell bootstrap（10,000回）。生物学的反復やdonor依存を補正していない。',
          '符号検定は独立な非同点queryの正負確率を仮定した参考値。Holm補正を併記し、仮定の成立を断言しない。',
          '効果量は平均Recall差（主）とpaired dz。CIが0を含む場合、改善の証拠は不確定。',
          '', '## 密度・周辺・collapseの定義',
          '密度歪みは、各空間のqueryの15近傍半径をそのHD真近傍のreference側15近傍半径平均で割り、両空間のlog比の差の絶対値。',
          'global_periphery_percentileは全reference重心からの半径の参照内百分位。未観測時間ラベルでclass条件付き重心を作らない。',
          'collapse_rateは2Dの正規化15近傍半径が0.1未満のquery割合。NDCGは既存距離rank保持指標。',
          '標準/reduced UMAP adapterがquery時間を返さない場合はN/A。fit込み時間をquery latencyに代入しない。',
          'exact_repulsion_diagnosticは配備用手法ではない。oracle_query_indices.jsonの固定subsetのみ。',
          '', '## 判定']
    if row.ci_low>0:
        text.append('この固定モデル・参照とcell bootstrapの条件では、FitGridの平均Recall改善CIは0より大きい。生物学的一般化は未検証。')
    elif row.ci_high<0:
        text.append('この固定モデル・参照ではFitGridの平均Recallは低下した。component改善だけでは最終埋め込み改善にならない。')
    else:
        text.append('平均Recall差の95%CIは0を含むため、最終埋め込み改善の証拠は不確定。')
    if temporal is not None:
        text.extend(['','## 時間holdout',markdown_table(pd.DataFrame([
            dict(timepoint=label,n_cells=temporal['cells_per_timepoint'][label],
                 role='reference' if label in temporal['reference_timepoints'] else 'interpolation' if label==temporal['interpolation_timepoint'] else 'extrapolation')
            for label in temporal['ordered_timepoints']])),
            f"選択規則: {temporal['selection_rule']}",
            '時間点分類精度はN/A（queryの時間ラベルはreferenceに存在しない）。別annotationがあればcell-type精度を別列で報告。',
            'temporal-neighbor MAEの単位は年代順の位置差。実時間間隔が不均等でも日数誤差とは解釈しない。',
            'excess MAEはlast observedからの不可避なordinal horizonを差し引く。',
            '',markdown_table(aggregate)])
        horizon_rows=[]
        for name,frame in frames.items():
            for label in temporal['query_timepoints']:
                sub=frame[frame.timepoint==label]
                if not len(sub):continue
                horizon_rows.append(dict(method=name,timepoint=label,horizon=int(sub.horizon.iloc[0]),
                    recall_at_15=sub.recall_at_15.mean(),temporal_neighbor_mae=sub.temporal_neighbor_mae.mean(),
                    excess_temporal_neighbor_mae=sub.excess_temporal_neighbor_mae.mean()))
        horizon=pd.DataFrame(horizon_rows);horizon.to_csv(output/'temporal_horizon.csv',index=False)
        monotonic=[]
        for name,sub in horizon[horizon.horizon>0].groupby('method',sort=False):
            sub=sub.sort_values('horizon')
            monotonic.append(dict(method=name,n_horizons=len(sub),
                mae_monotonic_nondecreasing=bool(np.all(np.diff(sub.temporal_neighbor_mae)>=0)) if len(sub)>1 else None,
                reason='one horizon: monotonicity not assessable' if len(sub)<2 else None))
        write_json(output/'horizon_monotonicity.json',monotonic)
        for col in ['recall_at_15','temporal_neighbor_mae','excess_temporal_neighbor_mae']:
            fig,ax=plt.subplots(figsize=(9,5))
            for name,sub in horizon.groupby('method',sort=False):
                sub=sub.set_index('timepoint').reindex(temporal['query_timepoints'])
                ax.plot(temporal['query_timepoints'],sub[col],marker='o',label=name)
            ax.set_xlabel('Actual timepoint label');ax.set_ylabel(col);ax.legend(fontsize=7)
            save(f'temporal_{col}')
        text.extend(['','外挿時間点が1つだけの場合、horizonに対する単調性は判定不可。'])
    unavailable=aggregate[aggregate.status!='success']
    text.extend(['','## unavailable / failed',markdown_table(unavailable) if len(unavailable) else 'なし。'])
    (output/'report.md').write_text('\n\n'.join(text),encoding='utf-8')
