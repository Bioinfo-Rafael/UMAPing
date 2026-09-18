"""PNG comparisons and evidence-linked report; no parameter selection."""
import sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'SyntheticAnalysis/scripts'))
from plots import COLORS,BRANCH_COLORS,bounds
from report import table
CONDITIONS=['disconnected','sparse_bridge'];METHODS=['baseline_3_to_2','direct_2','spectral_100_pca_2']
TITLES={'truth':'Ground truth','umap':'UMAP fit / transform','baseline_3_to_2':'baseline 3 → 2','direct_2':'direct 2','spectral_100_pca_2':'100 → PCA 2','pca':'Input PCA (control)'}
plt.rcParams.update({'font.size':9,'figure.dpi':120,'savefig.bbox':'tight'})
def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=140);plt.close(fig)
def colors(frame,kind):
    if kind=='structure':
        mapping={0:('A spiral',COLORS[0]),1:('B tree',COLORS[1]),3:('A–B bridge',COLORS[3])}
        return np.array([mapping[s][1] for s in frame.structure]),[Line2D([],[],marker='o',ls='',color=c,label=l) for s,(l,c) in mapping.items() if s in set(frame.structure)]
    if kind=='branch':
        keys=[]
        for s,b in zip(frame.structure,frame.branch):keys.append('A spiral' if s==0 else f'B branch {b}' if s==1 else 'A–B bridge')
        labels=['A spiral']+[f'B branch {b}' for b in range(7)]+['A–B bridge'];mapping={k:BRANCH_COLORS[i] for i,k in enumerate(labels)};labels=[k for k in labels if k in set(keys)]
        return np.array([mapping[k] for k in keys]),[Line2D([],[],marker='o',ls='',color=mapping[k],label=k) for k in labels]
    return frame.progress.to_numpy(),None

def render(run):
    for condition in CONDITIONS:
        frame=pd.read_csv(run/'data'/condition/'points.csv');a=np.load(run/'data'/condition/'arrays.npz');ref,q=a['reference'],a['query'];v={m:np.load(run/'embeddings'/condition/f'{m}.npz') for m in ['umap','pca']+METHODS};dest=run/'figures'/condition
        def coords(m,stage):
            if m=='truth':return a['z']
            y=np.empty_like(a['z']);vv=v[m]
            for key,ind in [('reference',ref),('query',q)]:y[ind]=vv[key+'_stages'][stage] if key+'_stages' in vv else vv[key]
            return y
        for phase,stage in [('initial',0),('final',4)]:
            for ki,kind in enumerate(['structure','branch','progress']):
                for subset,label in [(np.ones(len(frame),bool),'all')]+([(frame.structure==0,'spiral'),(frame.structure==1,'tree')] if phase=='final' and kind=='progress' else []):
                    col=['truth','umap']+METHODS;c,handles=colors(frame,kind)
                    fig,axes=plt.subplots(2,5,figsize=(19,8),layout='constrained')
                    for j,m in enumerate(col):
                        y=coords(m,stage);lo,hi=bounds(y[subset])
                        for row,ind in enumerate([ref,q]):
                            ids=ind[np.asarray(subset)[ind]];ax=axes[row,j];kw=dict(c=c[ids],s=3,alpha=.85)
                            if kind=='progress':kw.update(cmap='viridis',vmin=0,vmax=1)
                            im=ax.scatter(*y[ids].T,**kw);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');ax.set_title(TITLES[m]+(' (final control)' if phase=='initial' and m=='umap' else ''))
                            if j==0:ax.set_ylabel(['Reference','Query'][row])
                    if handles:fig.legend(handles=handles,loc='outside lower center',ncol=min(9,len(handles)),frameon=False)
                    else:fig.colorbar(im,ax=axes,orientation='horizontal',shrink=.5,pad=.03,label='A: arclength; B: within-branch progress; bridge: arclength')
                    fig.suptitle(f'{condition} | {phase} | {kind} | {label}')
                    save(fig,dest/f'{10 if phase=="initial" else 20}{ki}_{phase}_{kind}_{label}.png')
        # Fixed axes across time for each method; separate reference and query figures.
        for split,ind in [('reference',ref),('query',q)]:
            for kind in ['structure','branch','progress']:
                c,handles=colors(frame,kind);fig,axes=plt.subplots(3,5,figsize=(19,10),layout='constrained')
                for row,m in enumerate(METHODS):
                    allcoords=np.concatenate([coords(m,s) for s in range(5)]);lo,hi=bounds(allcoords)
                    for stage in range(5):
                        ax=axes[row,stage];kw=dict(c=c[ind],s=2)
                        if kind=='progress':kw.update(cmap='viridis',vmin=0,vmax=1)
                        im=ax.scatter(*coords(m,stage)[ind].T,**kw);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');ax.set_title(f'{stage*25}%')
                        if stage==0:ax.set_ylabel(TITLES[m])
                if handles:fig.legend(handles=handles,loc='outside lower center',ncol=min(9,len(handles)),frameon=False)
                else:fig.colorbar(im,ax=axes,orientation='horizontal',shrink=.5,pad=.03,label='Within-structure / within-branch progress [0,1]')
                fig.suptitle(f'{condition} | {split} | dynamics | {kind} | fixed per-method limits')
                save(fig,dest/f'30_dynamics_{split}_{kind}.png')
        fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained');c,h=colors(frame,'branch')
        for j,m in enumerate(['truth','pca']):
            y=coords(m,4);lo,hi=bounds(y)
            for row,ind in enumerate([ref,q]):
                axes[row,j].scatter(*y[ind].T,c=c[ind],s=3);axes[row,j].set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',title=TITLES[m]);axes[row,0].set_ylabel(['Reference','Query'][row])
        fig.legend(handles=h,loc='outside lower center',ncol=5,frameon=False);fig.suptitle(condition+' | isometric input control');save(fig,dest/'01_pca_control.png')
    summary=pd.read_csv(run/'metrics/summary.csv');large=pd.read_csv(run/'metrics/spiral_recall.csv');rank=pd.read_csv(run/'metrics/distance_rank.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,c in enumerate(CONDITIONS):
        for col,split in enumerate(['reference','query']):
            ax=axes[row,col]
            for m in ['umap']+METHODS:
                p=large[(large.condition==c)&(large.split==split)&(large.method==m)&(large.stage==1)];ax.plot(p.k,p.recall,'o-',label=TITLES[m])
            ax.set(xscale='log',ylim=(0,1.02),xlabel='k (same-spiral reference candidates)',ylabel='Recall@k',title=c+' | '+split);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);save(fig,run/'figures/40_spiral_recall_curves.png')
    for split in ['reference','query']:
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for row,c in enumerate(CONDITIONS):
            for col,(metric,title) in enumerate([('within_recall15','Within Recall@15 ↑'),('abs_log_radius15','Absolute log radius error ↓'),('spearman','Distance rank correlation ↑')]):
                ax=axes[row,col];data=rank if metric=='spearman' else summary;groups=['A_spiral','B_tree'];xx=np.arange(2)
                for j,m in enumerate(['umap']+METHODS):
                    pp=data[(data.condition==c)&(data.split==split)&(data.method==m)&(data.stage==1)].set_index('group');ax.bar(xx+(j-1.5)*.18,[pp.loc[g,metric] for g in groups],.18,label=TITLES[m])
                ax.set(xticks=xx,xticklabels=['Spiral','Tree'],title=c+' | '+title)
        axes[0,0].legend(fontsize=8);fig.suptitle(split);save(fig,run/'figures'/f'41_metrics_{split}.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,c in enumerate(CONDITIONS):
        for col,metric in enumerate(['within_recall15','abs_log_radius15','wrong_turn15']):
            ax=axes[row,col]
            for m in METHODS:
                for split,style in [('reference','-'),('query','--')]:
                    p=summary[(summary.condition==c)&(summary.method==m)&(summary.split==split)&(summary.group=='internal')];ax.plot(p.stage,p[metric],style,marker='o',label=TITLES[m]+' '+split)
            ax.set(title=c+' | '+metric,xlabel='Dynamics time');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);save(fig,run/'figures/42_stage_metrics.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,c in enumerate(CONDITIONS):
        for m in METHODS[1:]:
            root=run/'models'/c/m/'shared';loss=pd.read_csv(root/'spectral_losses.csv');diag=json.loads((root/'diagnostics.json').read_text())
            for field,style in [('dirichlet_per_dim','-'),('orthogonality_per_dim','--'),('trivial_per_dim',':')]:axes[row,0].plot(loss.step,loss[field],style,label=TITLES[m]+' '+field)
            ev=np.array(diag['covariance_eigenvalues']);axes[row,1].plot(np.arange(1,len(ev)+1),ev,'o-',ms=2,label=TITLES[m]);hv=diag['H_eigenvalues'];axes[row,2].plot(np.arange(1,len(hv)+1),hv,'o-',ms=2,label=TITLES[m])
        axes[row,0].set(yscale='log',title=c+' | training terms per dimension');axes[row,1].set(yscale='log',title='Corrected covariance eigenvalues');axes[row,2].set(title='H spectrum (diagnostic only)');axes[row,0].legend(fontsize=6);axes[row,1].legend(fontsize=8)
    save(fig,run/'figures/43_spectral_diagnostics.png')

def report(run):
    s=pd.read_csv(run/'metrics/summary.csv');r=pd.read_csv(run/'metrics/distance_rank.csv');k=pd.read_csv(run/'metrics/spiral_recall.csv');protocol=json.loads((run/'configs/protocol.json').read_text())
    lines=['# Spectral output dimension comparison: A+B, seed 0',
    '既存A+B runのデータ・ID・split・50D等長写像をそのままコピー。条件はdisconnectedとsparse_bridge。正解座標・構造ラベルは評価と作図だけに使用し、通常UMAPの座標や軌道を教師にしません。baseline_3_to_2は既存FitGrid結果を使用。通常UMAPと入力PCAの比較座標も再利用しました。',
    '''## 採用候補と結論

暫定的な追加検証候補はdirect_2。ただし置き換え採用を決める結果ではありません。内部queryの最終Within Recall@15と平均半径誤差は両条件で小幅改善した一方、橋ありの木のRecallは悪化し、木の距離順位相関は両条件で約0.91から約0.81へ低下しました。螺旋・木・橋あり/なしのすべてで一貫した改善ではありません。

100次元＋PCAは今回の採用候補にしません。最終局所Recallが両条件で低下し、上位PCA固有値の近接とbootstrapの大きな部分空間変動が観測されました。100次元の直交制約自体も十分には満たされておらず、「全100軸が等分散になった」とは解釈しません。

初期の改善がそのまま最終に残るという結果ではありません。direct_2は橋なしの初期Recallを改善しましたが、橋ありでは初期Recallが低下し、最終のみ小幅改善しました。全方式で25%時点のRecallが大きく低下し、その後回復します。初期からの重なりと更新中の追加変形の両方が残り、Spectral変更だけで変形・重なりを解消できたとは結論しません。''',
    '## 固定した数式・実装差',
    r'グラフは $L=I-D^{-1/2}WD^{-1/2}$。連結成分ごとに $T_{ic}=\sqrt{N}\sqrt{d_i}1[i\in c]/\sqrt{\sum_{j\in c}d_j}$ とおくと $T^TT/N=I$、$LT=0$。次数が不均一なため単なる平均0では自明モードを除けません。成分は入力グラフだけから求め、構造ラベルを使いません。',
    r'新2方式は出力 $Z=f_\theta(X_{ref})\in\mathbb R^{N\times r}$ に対して $[E_{edge}+\|Z^TZ/N-I_r\|_F^2+\|T^TZ/N\|_F^2]/r$ を最小化。$E_{edge}=\mathrm{mean}_{(i,j)\sim upper(W)}w_{ij}\|z_i/\sqrt{d_i}-z_j/\sqrt{d_j}\|^2$ は既存と同じ一様edge sampling4096本。各項をrで割り、既存のenergy対orthogonalityの係数比1を保ちつつ次元の総和増大を抑えます。直交性は新2方式で同じfull-reference Gramを用い、baselineのedge由来unique-node proxyとの違いもあります。従って出力次元だけの単独介入ではありません。',
    r'学習後、referenceだけから $B=T^TZ/N$ を求め、$Z_c=Z-TB$ として残存自明成分を除去。queryでは入力50Dの固定reference15近傍から逆距離重みでTを補間して $f_\theta(x)-t(x)B$ を適用。この補間はSpectral較正のためだけに使い、Dual Encoderの引力近傍は変更しません。queryバッチの平均・共分散・query-query graphは使いません。',
    '- **direct_2**: r=2。自明成分補正後の2列をそのまま使い、追加モードを捨てず、白色化・Hによる選択もしません。\n- **spectral_100_pca_2**: r=100。補正後referenceの中心化共分散にPCAをfitし、分散最大の2軸を採用。完全白色化なし。queryに同じprojectionを固定適用。Hは診断として記録するだけで軸選択に使いません。\n- **baseline_3_to_2**: 既存3出力、soft Gram制約、完全白色化、Hの最小1モードを捨て次の2モードを選択。既存モデルは再学習しません。',
    r'2D較正は全方式でreference平均 $\mu$ と単一倍率 $s=10/\max_{i,d}|y_{id}-\mu_d|$。queryにも $(y-\mu)s$ を固定適用。hidden layers512/256 GELU、Adam lr0.001、Spectral2000 stepsは共通。新2方式のhidden layers初期値・edge抽選seedも共通。test成績で係数を調整していません。',
    '同じdata条件では既存Dual Encoder・入力グラフ・保存済み検索IDを共有。独自reference dynamics200 steps、係数、乱数seedを維持。各新trajectoryごとに既存FitGrid256 teacherを再構築し、新しい反発ネットを共通初期値から4000 steps学習。teacher/学習位置サンプルのseedは既存と同じ。異なるtrajectoryの反発ネットは流用しません。',
    '## 評価定義',
    '既存Recall@5/15・構造内Recall・誤った巻き/枝・橋の欠落を再利用。半径誤差は、同じreference候補集合でz/yそれぞれのkNN半径を求め、referenceのmedian(log(rz15/ry15))から単一倍率sを決定しlog(s·ry/rz)を記録。負が圧縮、正が膨張。各stageごとにreferenceのみから倍率を決める相対局所スケール指標であり、島面積の評価ではありません。reference自身はself除外。構造/枝/橋・reference/query別に全stageを保存。',
    '螺旋内Recall@50/100/200/400も同じ構造のreference候補に対して計算。距離順位は全体・螺旋・木ごと、reference同士/query同士の固定seed12345による最大20,000点対（self除外）を全方式・全stageで共有しSpearman相関を計算。点対はmetrics/<condition>/distance_pairs.npzに保存。',
    '## 最終内部query比較',table(s[(s.split=='query')&(s.group=='internal')&(s.stage==1)],['condition','method','recall5','recall15','within_recall15','abs_log_radius15','wrong_turn15','wrong_branch15']),
    '## 初期→最終：reference/queryと構造別',table(s[(s.group.isin(['A_spiral','B_tree','bridge_AB']))&(s.stage.isin([0,1]))],['condition','method','stage','split','group','within_recall15','abs_log_radius15']),
    '## 大域距離順位（最終）',table(r[r.stage==1],['condition','method','split','group','spearman']),
    '## 螺旋の大きいk（最終query）',table(k[(k.stage==1)&(k.split=='query')],['condition','method','k','recall']),
    '## 橋の保持（最終）',table(s[(s.condition=='sparse_bridge')&(s.stage==1)&(s.group=='all')],['method','split','bridge_edge_loss15','bridge_cross_edge_loss15','wrong_turn15','wrong_branch15']),
    '## 観測と解釈']
    for c in CONDITIONS:
        lines.append('### '+c)
        for m in METHODS[1:]:
            vals=s[(s.condition==c)&(s.method==m)&(s.split=='query')&(s.group=='internal')].set_index('stage');base=s[(s.condition==c)&(s.method==METHODS[0])&(s.split=='query')&(s.group=='internal')].set_index('stage')
            ini=vals.loc[0,'within_recall15']-base.loc[0,'within_recall15'];fin=vals.loc[1,'within_recall15']-base.loc[1,'within_recall15']
            lines.append(f'**{m}**: baseline比の内部query Within Recall@15差は初期 {ini:+.4f} → 最終 {fin:+.4f}。最終の局所半径絶対誤差差は {vals.loc[1,"abs_log_radius15"]-base.loc[1,"abs_log_radius15"]:+.4f}（負が改善）。')
            parts=[]
            for g in ['A_spiral','B_tree']:
                def sel(method):return s[(s.condition==c)&(s.method==method)&(s.split=='query')&(s.group==g)&(s.stage==1)].iloc[0]
                delta=sel(m).within_recall15-sel(METHODS[0]).within_recall15
                def rho(method):return r[(r.condition==c)&(r.method==method)&(r.split=='query')&(r.group==g)&(r.stage==1)].iloc[0].spearman
                parts.append(f'{g}: Within Recall@15差 {delta:+.4f}、距離順位相関差 {rho(m)-rho(METHODS[0]):+.4f}')
            lines.append('；'.join(parts)+'。')
            d=json.loads((run/'models'/c/m/'shared/diagnostics.json').read_text())
            if m=='direct_2':lines.append(f'direct_2の補正前自明成分量/r={d["raw_trivial_leakage"]:.5g}、補正後={d["corrected_trivial_leakage"]:.3g}。2D共分散固有値={d["centered_2d_eigenvalues"]}、Gram誤差/r={d["orthogonality_error"]:.4g}。正の2固有値は全体collapseがないことを示しますが、局所重なりを否定しません。')
            else:
                angles=np.array(d['bootstrap_principal_angles_degrees'])[:,-1]
                lines.append(f'100D PCA: Gram誤差/r={d['orthogonality_error']:.4f}、上位3共分散固有値={d["covariance_eigenvalues"][:3]}、上位2寄与率合計={sum(d["explained_variance_ratio"][:2]):.4f}、相対gap λ1–λ2/λ1={d["top_relative_gaps"][0]:.4f}、λ2–λ3/λ2={d["top_relative_gaps"][1]:.4f}。固定20回reference bootstrapの2D部分空間最大主角度中央値={np.median(angles):.2f}°、最大={angles.max():.2f}°。これはsampling感度でありseed安定性の検証ではありません。')
    lines+=['仮説：橋なしで各連結成分の自明モードを除くと成分間の平行移動を表す自由度も除かれるため、初期の成分間重なりに影響した可能性があります。この比較だけで原因は確定できません。次元・自明モード処理・Gram推定の変更が組になった比較です。初期からの幾何変形とdynamicsによる追加変形はstage図・指標から区別し、原因がDual EncoderやFitGrid、引力過多であるとは断定しません。1 seed・無ノイズ等長写像の機構確認に限られます。',
    '## 図',
    '各条件ディレクトリに初期/最終5列比較、reference/query別の0/25/50/75/100%推移、螺旋/木拡大図、入力PCA対照を保存。色対応・凡例は共通。初期図の通常UMAP列は再利用した最終比較座標と明記。推移図では同一方式の全時刻・reference/queryを含む軸範囲を固定。',
    '![Recall曲線](figures/40_spiral_recall_curves.png)','![query指標](figures/41_metrics_query.png)','![stage指標](figures/42_stage_metrics.png)','![Spectral診断](figures/43_spectral_diagnostics.png)']
    for c in CONDITIONS:
        lines += [f'### {c}',f'![初期](figures/{c}/102_initial_progress_all.png)',f'![最終](figures/{c}/202_final_progress_all.png)',f'![reference推移](figures/{c}/30_dynamics_reference_branch.png)',f'![query推移](figures/{c}/30_dynamics_query_branch.png)']
    lines += ['## 再実行・保存',f'```sh\n.venv/bin/python SyntheticAnalysis/spectral_dimension_comparison/scripts/experiment.py\n# 完成モデル/埋め込みを再利用して再開\n.venv/bin/python SyntheticAnalysis/spectral_dimension_comparison/scripts/experiment.py --run {run.relative_to(ROOT)}\n# 評価・作図・報告のみ\n.venv/bin/python SyntheticAnalysis/spectral_dimension_comparison/scripts/experiment.py --run {run.relative_to(ROOT)} --phase analyze\n```',
    'configs/固定設定、data/元データと共有検索結果、models/新Spectral・較正・trajectory・各FitGrid反発モデル、embeddings/各stage reference/query、metrics/点別・集計・固定点対、figures/PNG、logs/監査・ソース・所要時間。Spectralは完成モデル単位、反発ネットは250stepsチェックポイント単位、queryは完成方式単位で再利用。',
    f'実行開始から報告まで {(time.time()-protocol["started"])/60:.2f} 分。必須2データ条件×2新方式と既存baseline比較を完了。未実施は追加seedによる再現性検証。main merge・既存run上書きはありません。']
    (run/'README.md').write_text('\n\n'.join(lines)+'\n')
