"""Generate only figures backed by completed conditions, plus a Japanese report."""
import json
from pathlib import Path
import shlex
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import utcnow


def interpret(exp):
    """Conservative descriptive decisions; a CI containing zero is not equivalence."""
    if not exp.pairs or 'selected' not in exp.status:
        return ('判定保留: 実データの確認比較が未完了。',
                '採用比率・FitGridの追加効果・各力の必要性は未検証です。',
                ['未知点推論時の引力・反発の係数比を比較する実験を実装した。',
                 '実データで完了した確認結果がないため、性能改善は未検証である。',
                 '構成要素の必要性は、端点比較と別seed・別splitへの転用を終えてから判断する。'])
    pairs=pd.DataFrame(exp.pairs); macro=pairs[pairs.group=='macro']
    primary=macro[macro.context=='temporal0']
    gains=primary[primary.comparison.str.endswith('selected minus current')]
    fg=primary[primary.comparison=='FitGrid minus Uniform selected']
    ends=primary[primary.comparison.str.contains('minus endpoint')]
    if len(gains)!=2 or len(ends)!=8 or not len(fg):
        return ('判定保留: 主seedの必須paired比較が不足。','実験完了まで採用構成を変更しません。',
                ['確認比較の一部を実行した。','主seedの必須比較はまだ揃っていない。','改善成立と構成の必要性は判定保留である。'])
    directions=macro[macro.comparison.str.endswith('selected minus current')].groupby('context').delta.mean()
    opposite=(directions>0).any() and (directions<0).any()
    if opposite:
        decision='E: seedまたはsplitで平均改善の方向が変わる。条件依存であり普遍的な最適比率ではない。'
    elif (ends.delta<=0).any():
        decision='C候補: 少なくとも一つの係数1/2端点が主seedの観測平均で選択比率以上。簡素化候補だが同等性検定はしていない。'
    elif not (gains.delta>0).all():
        decision='Dまたは改善不成立: 両teacherの確認用改善を再現できていない。'
    elif 0<exp.status['selected']['w']<1 and (ends.low>0).all() and float(fg.iloc[0].low)>0:
        decision='A: 主seedの確認集合では中間比率の端点超過と調整済みUniformに対するFitGridの追加効果を支持。'
    elif float(fg.iloc[0].low)<=0<=float(fg.iloc[0].high):
        decision='B: 両teacherの観測平均は改善。FitGrid–Uniform差の95%区間は0を含み、追加効果の証拠は限定的（同等性の証明ではない）。'
    else:
        decision='条件付き改善: 両teacherの観測平均は改善するが、組合せとFitGridの追加効果を一括して支持する証拠は不十分。'
    recommendation='主seedの差は図表の95%区間と併読してください。density/NDCGとのトレードオフ、別seed、別splitの確認が揃わない範囲では採用を限定します。'
    return decision,recommendation,[decision, '細胞bootstrapの区間は固定モデル下の評価であり、生物学的反復や学習seed間の不確実性とは異なる。',
                                   'この比率は離散候補から選んだもので、参照軌道は固定し、他のseedとsplitで再調整していない。']


def save(fig, path, frame):
    fig.tight_layout(); fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig)
    frame.to_csv(path.with_suffix('.csv'),index=False)


def render(exp):
    exp.deadline.check(report=True)
    root=exp.report; rows=pd.DataFrame(exp.summaries)
    if len(rows):
        search=rows[(rows.stage=='search') & (rows.scale==2) & (rows.group!='micro')]
        if len(search):
            groups=sorted(search.group.unique()); fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),4),squeeze=False)
            for ax,group in zip(axes[0],groups):
                for teacher,s in search[search.group==group].groupby('teacher'):
                    s=s.sort_values('w'); ax.plot(s.w,s.recall,'o-',label=teacher)
                ax.set(title=f'Selection only | time {group}',xlabel='w (coefficient balance)',ylabel='Recall@15'); ax.legend()
            save(fig,root/'01_balance_search/balance_search_recall_temporal_seed0',search)
        for stage,section,stem in [('confirmation','02_selected_ratio_confirmation','selected_ratio_confirmation'),
                                   ('transfer','03_transfer_to_existing_split','ratio_transfer')]:
            sub=rows[(rows.stage==stage)&(rows.scale==2)&(rows.group!='micro')]
            chosen=exp.status.get('selected',{}).get('w')
            sub=sub[sub.w.isin([.5,chosen])]
            for context,c in sub.groupby('context'):
                groups=sorted(c.group.unique()); fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),4),squeeze=False)
                for ax,group in zip(axes[0],groups):
                    g=c[c.group==group]; x=np.arange(len(g))
                    ax.bar(x,g.recall,color=['#276D9A' if t=='Uniform' else '#CA7639' for t in g.teacher])
                    ax.set_xticks(x,[f'{t}\nw={w:.3g}' for t,w in zip(g.teacher,g.w)])
                    ax.set(title=f'{context} | time {group}',ylabel='Recall@15')
                save(fig,root/section/f'{stem}_recall_{context}',c)
        runtime=rows[rows.group=='micro']
        fig,ax=plt.subplots(figsize=(6,4))
        for (stage,teacher),s in runtime.groupby(['stage','teacher']):
            ax.scatter(s.seconds/s.n,s.macro_recall,label=f'{stage} {teacher}',alpha=.7)
        ax.set(xlabel='Measured inference seconds / query',ylabel='Macro Recall@15'); ax.legend(fontsize=7)
        save(fig,root/'02_selected_ratio_confirmation/performance_vs_runtime_recall_all_splits_seeds',runtime)
    if exp.pairs:
        paired=pd.DataFrame(exp.pairs)
        for context,p in paired[paired.group=='macro'].groupby('context'):
            fig,ax=plt.subplots(figsize=(9,max(3,.4*len(p))))
            y=np.arange(len(p)); ax.hlines(y,p.low*100,p.high*100); ax.scatter(p.delta*100,y)
            ax.set_yticks(y,p.comparison); ax.axvline(0,color='grey',lw=1)
            ax.set(xlabel='Macro Recall difference (percentage points), paired cell bootstrap 95% CI',title=context)
            section='03_transfer_to_existing_split' if context=='existing' else '02_selected_ratio_confirmation'
            save(fig,root/section/f'paired_recall_improvement_{context}',p)
    if exp.force_rows:
        force=pd.DataFrame(exp.force_rows)
        cols=['weighted_attraction_norm','weighted_repulsion_norm','repulsion_fraction','clipped']
        agg=force.groupby(['context','teacher','w','scale','step'],as_index=False)[cols].mean()
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for ax,col in zip(axes.flat,cols):
            for (teacher,w,scale),s in agg.groupby(['teacher','w','scale']):
                ax.plot(s.step,s[col],label=f'{teacher} w={w:.3g} s={scale:g}',alpha=.7)
            ax.set(xlabel='Embedding optimization step',ylabel=col)
        axes[0,0].legend(fontsize=5,ncol=2)
        save(fig,root/'04_force_contributions/effective_force_contributions_temporal_seed0',agg)
    selected=exp.status.get('selected')
    decision,recommendation,conclusions=interpret(exp)
    text=['# 引力・反発バランス改善実験', '',f"状態: **{exp.status['status']}**。未完了条件を成功とは扱いません。",'',
          f"ブランチ: `experiments/force-balance-rescue`。基準commit: `{exp.protocol['base_commit']}`。",
          f"実行コードcommit: `{exp.status.get('code_commit','未記録')}`。",
          f"開始: {exp.protocol['started_utc']}。終了: {exp.status.get('execution_ended_utc','実行中')}。",
          f"着手からの経過秒: {exp.status.get('elapsed_since_original_start_seconds','実行中')}。絶対終了: {exp.protocol['hard_deadline_utc']}。",'',
          '## 実行コマンド','```sh','python -m umaping.force_balance '+shlex.join(exp.status['command'][1:]),'```','',
          '## 実装上の事実',
          '`A` はpairwise成分clipping後に近傍重みを掛けて足した引力です。`R` は学習済み反発ネット出力に既存のnegative_sample_rateとquery近傍重み総和を掛けた値です。係数の二重適用はありません。',
          '`v = 2*((1-w)*A + w*R)` の後に現行と同じ成分別合力clippingと学習率を適用します。w=0.5はA+Rを再現します。比率は係数比であり、実効ベクトルノルム比ではありません。端点2A/2Rと係数1のA/Rを区別します。',
          '参照軌道は固定です。時間tは最適化時間であり生物学的時間ではありません。FitGridの場を直接推論に使用せず、学習済みB_phiだけを使用します。',
          '主指標は実際のheld-out時間群のRecall@15の等重み平均です。NDCGは既存の最終埋め込み評価と同じbinary NDCG@15です。densityは既存query_metricsの、入力空間の正解近傍を共通基準とした対数半径比歪みです。',
          '95%区間は時間群内のpaired cell bootstrapです。training seed間変動とは別で、生物学的反復ではありません。seed 1は反発ネットの学習seedのみで、上流は共通です。今回の確認集合は研究全体で完全未使用の独立テストではありません。','',
          '## 選択',json.dumps(selected,ensure_ascii=False) if selected else '未選択。探索未完了のため比率を決定していません。',
          '選択規則: 7候補それぞれについてUniform/FitGridのmacro Recall平均を最大化。完全同点なら0.5に近い候補、さらに同点なら小さいw。確認集合や別seed/splitで再選択しません。','',
          '## 確認・転用結果']
    if len(rows):
        sub=rows[(rows.stage!='search')&(rows.group=='micro')]
        text+=['| split | teacher | w | scale | n | macro Recall | micro Recall | NDCG | density | zero Recall |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in sub.itertuples():
            text.append(f'|{r.context}|{r.teacher}|{r.w:.5g}|{r.scale:g}|{r.n}|{r.macro_recall:.5f}|{r.recall:.5f}|{r.ndcg:.5f}|{r.density:.5f}|{r.zero_recall:.5f}|')
    else: text+=['新規実験の測定結果はありません。既存解析の値を今回の改善実験結果に代用していません。']
    text+=['','## 完了・未完了',f"完了条件数: {len(exp.status['completed_conditions'])}。詳細は99_provenance/execution.json。",
           exp.status.get('error',''),f"D: {exp.status['optional_D']}。E: {exp.status['optional_E']}。",
           'Standard UMAPは今回同じ調整予算で探索していません。比較しない場合に優越性を主張しません。',
           '', '## 結果の解釈と未検証事項',
           decision,recommendation,
           '係数が非ゼロであることだけでは実質的寄与の証拠にしません。力のノルム・寄与割合・clipping率は04_force_contributionsのCSVと図に記録します。寄与割合の分母0は欠測として明示します。',
           '既存splitの学習seedがmetadataに無い場合は不明とします。確認用の改善、seed間再現性、split間転用はそれぞれ区別して判断します。',
           '', '## 発表で使える結論（日本語3文）',
           *[f'{i+1}. {sentence}' for i,sentence in enumerate(conclusions)],'']
    (root/'README_ja.md').write_text('\n'.join(text))
