# 既知構造の合成データによるUMAPing比較

通常umap-learnの **reference fit → query transform** と、UMAPing Uniform／FitGrid、PCAを比較します。目的は未知点の局所近傍・局所的な広がり・内部構造の保持です。島の面積そのものを改善とは扱いません。無ノイズ・等長線形写像の機構確認であり、高難度データへの優位性は主張しません。

## 保存構造

```text
SyntheticAnalysis/
  README.md
  scripts/                    # 実験固有の生成・実行・評価・作図・報告
  configs/                    # 学習前に固定した設定と採用係数の出典
  runs/<UTC timestamp>/
    data/                     # ID、正解座標、構造・枝・進行位置、共通split、50D写像
    models/<condition>/seed_N/# shared上流 + uniform / fitgridチェックポイント
    embeddings/<condition>/seed_N/
    metrics/<condition>/seed_N/
    figures/<condition>/seed_N/# 全手法×reference/query、拡大図、PNGとPDF
    logs/                     # 時間、環境、ソースhash、実行前固定manifest
    configs/                  # run固有の凍結設定コピー
    report.md
```

既存UMAPingのgraph、Retriever、Spectral、reference dynamics、repulsion学習、UniformMC、GridField、InferenceEngineを直接importして再利用します。既存コードへの変更なし。通常UMAPは既存baselineと同じfit/transform方式を用い、モデル保存とepochs明記のため実験runnerで直接呼びます。

## データとsplit

- 固定data seed 20260918。A: arclengthにほぼ一様な幅0.36の2.5周螺旋（2,001点）、B: 枝が交差しない幅0.36の7枝の木（1,998点）、C: 半径4×3の楕円（2,001点）。楕円内の密度は正規化横位置u∈[-1,1]に対して `p∝exp(1.2u)`。正規化定数は固定で、相対解析密度も保存。
- disconnectedとsparse_bridgeで内部6,000点の座標・IDは完全共通。橋はAB、BC各240点、全幅0.11。元データに橋を追加するだけで、学習結果に応じて生成を変更しない。
- 構造・木の各枝・各橋の空間順に並べた3点組から1 queryを固定乱数で選び、残り2 reference。楕円は20列の蛇行順で全領域を覆う。内部reference 4,000／query 2,000、橋reference 320／query 160。Aの巻き境界は進行位置順の3点組で近傍層化される。枝別・構造別件数を保存。
- 全学習seedでデータとsplitは同一。全手法にも共通。validationは使わず既存標準stepで学習。query正解は評価・作図だけに使用。
- `A` は50×2の固定直交列行列。行ベクトル表記で `X=Z A.T`、`A.T A=I`。標準化なし、ノイズなし。学習コードへ渡すときだけfloat32。PCAはreferenceだけでfitし、正解回復はreferenceから求める剛体変換で検証。
- 学習前に全点/referenceの入力kNN（self除外15）を検査。連結成分、構造間辺、橋削除後の成分、別の巻き/枝への近道を保存。期待成分数と不正接触0をassertしてから学習する。umap-learnのk=15はself込みで、既存UMAPingはself除外15という規約差を変更せず記録。

## 固定したモデル設定

`configs/model.yaml`は既存`experiments/configs/embryoid_body.yaml`の学習・推論設定をそのまま用い、datasetだけ置き換えています。FitGrid256、UniformMC64は既存matched teacher実験を継承。Uniform/FitGridでreference trajectory、Retriever、Spectral、repulsion初期state_dict、stepごとの学習位置・時刻・jitter列、optimizer設定を共通化し、teacherだけ変更します。seed 0/1/2では上流も再学習します。

force-balanceの採用結果は sibling worktree `UMAPing-pancreas-review/Final_analysis/force_balance_20260917T124700Z/01_balance_search/selection.json` から確認・コピー。採用値は `w=0.5, scale=2`、すなわち引力係数1・反発係数1で、元のInferenceEngineの `A+R` と一致します。test成績を使った係数探索はしません。

通常UMAP: Euclidean、k15、min_dist0.1、spread1、negative_sample_rate5、500 epochs（通常の小規模データ既定と同じ）、transform166 epochs、random_state/transform_seed=学習seed。全点fitは実施しません。既存umap-learn既定の4096点閾値により、reference4000点の橋なしは距離全計算、4320点の橋ありは近似近傍探索となります。実際の学習グラフも監査し、このbackend差と点数差は橋比較の限界として報告します。結果を見てbackendを変更しません。PCAはfloat64のfull SVD。

## 評価式（学習前固定）

候補reference集合をC_iとし、reference自身の評価ではselfを必ず除外します。queryは全referenceが候補です。

1. **Recall@k**: `|N_z(i,k) ∩ N_y(i,k)| / k`、k=5,15。globalは全reference候補、withinは同じ構造ID内だけの候補（橋も各橋を別構造）。構造・枝・橋、reference/queryを別集計。
2. **局所半径**: 同じ候補C_iから、zとyそれぞれで独立にkNNを求め、k番目距離を `r_z(i,k), r_y(i,k)` とする。元の正解近傍IDに固定した距離ではない。
   - 手法・条件・seedごとに **reference全体から一つだけ** `s=exp(median_ref(log(r_z(i,15)/r_y(i,15))))` を求める。
   - `E(i,k)=log(s*r_y(i,k)/r_z(i,k))`。自然対数、k=5,15、負=圧縮、正=膨張、0=補正後一致。
   - 符号付きE、絶対値|E|、r_z、r_y、sを点ごと保存。withinや橋・各群・queryにも同じsを使用し、再較正しない。
   - **相対局所スケール**を評価する。全体拡大はsで打ち消され、群別の一様拡大は他群との不整合として残る。密度を保ちながら誤近傍を作る場合があるのでRecallと接続指標を併読する。
3. **誤接続**: embedding kNN15の辺が、正解距離でsourceのr_z15の2倍を超え、Aでは角度差>π、Bでは異なる枝なら誤近傍とする。ただし共有分岐点から両端0.6以内の隣接枝は正当とし除外。分母はsourceごとの15近傍。小さな境界入替えや正当な分岐近傍を誤接続と数えない。これは保守的な規則で、全種類のトポロジー誤りを網羅しない。
4. **橋近傍の欠落**: 正解kNN15で少なくとも一端が橋に属する辺のうちembedding kNN15から消えた辺数/対象正解辺数。さらに構造IDが異なる端点接続辺のみの `bridge_cross_edge_loss15` も計算。分母0は欠測。`bridge_near`は対象正解辺を持つsource。橋が切れて島が離れただけでは改善としない。

橋追加の対応比較は同じ内部点IDで行います。within Recallは候補ID集合も一致するため、橋が候補に加わる直接効果と、学習配置の変化を分けて読めます。reference/query平均差は異なる点集合間の記述比較であり、因果分解ではありません。

## 実行・再開・作図

UMAPing直下から既存`.venv`を使用します。

```sh
# 新しいrunを必ず作成し、データ・正解図・入力グラフを完成
.venv/bin/python SyntheticAnalysis/scripts/run.py --phase prepare

# 上で表示されたRUN_DIRを指定（この行だけ実際のrunへ置換）
RUN_DIR=SyntheticAnalysis/runs/<timestamp>

# 最初に2条件×1 seed、所要時間を測定。必須図・報告まで生成
.venv/bin/python SyntheticAnalysis/scripts/run.py --run "$RUN_DIR" --phase train --seeds 0

# 既定の時間判定を通った場合だけ、固定順で残り2 seedsへ拡張
.venv/bin/python SyntheticAnalysis/scripts/run.py --run "$RUN_DIR" --phase train --seeds 1 2

# 評価と全図・報告の再計算（再学習なし）
.venv/bin/python SyntheticAnalysis/scripts/run.py --run "$RUN_DIR" --phase evaluate --seeds 0 1 2

# 保存済み評価から作図・報告のみ（既存のPNG/PDFは再利用）
.venv/bin/python SyntheticAnalysis/scripts/run.py --run "$RUN_DIR" --phase plot

# 上流共有・データ不変・実際の学習グラフ・集計値の整合性監査
.venv/bin/python SyntheticAnalysis/scripts/audit_results.py "$RUN_DIR"

# 数式・self除外・分岐点判定の検証
.venv/bin/python -m pytest SyntheticAnalysis/scripts/test_protocol.py -q
```

開始時刻はrunに保持し、学習には絶対11.5時間のalarmを使用。seed0の2条件と必須図・報告を完成後、実測の2倍の安全係数で残り4条件と30分の報告予算が入るときだけ拡張します。判断と実測値を保存し、結果でseedを選びません。

同じrunで再実行すると完成stage・モデル・埋め込みを再利用します。反発ネットは250 stepsごとのoptimizer/乱数状態、queryは100点ごとの中間値から再開。RetrieverとSpectralは完成stage単位（中断されたstageは同じseedから再実行）。実行lockは生存PIDをチェックし、重複起動を防止します。既存研究成果は変更せず、mainへmergeしません。

## 図

列は正解2D／PCA／通常UMAP／UMAPing Uniform／UMAPing FitGrid、行はreferenceのみ／queryのみ。構造ID、枝ID、進行位置、正解局所密度、符号付き半径誤差、Recallを別図で保存。全手法・両行で色対応を共有し、等アスペクト。各手法の軸範囲はreference/queryの和集合で固定し、Uniform/FitGridは同じ軸範囲。拡大図は正解領域で選んだ同一ID集合を全手法に表示するため、埋め込みが崩れた場合はその崩れを含む範囲になります。PNG/PDFの点群はラスタライズ。主要PNGを実際に確認してlegendと余白を点検します。

現在の結果: [20260918T034857_716429Z/report.md](runs/20260918T034857_716429Z/report.md)。

## 今回のpush対象

ユーザーの指示で保存済みPDF56枚を削除しました。上記runのPNG・全評価結果・設定・データ・最終埋め込み・報告・ソースを公開します。`models/`内の重み・モデルbinaryとoptimizer中間状態、および埋め込みの`*_partial.npz`はローカルに保持します。モデルの再学習は上記コマンド、保存済み結果からの再作図は下記コマンドを使用してください（モデルの再学習・読み込み不要）。

```sh
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "SyntheticAnalysis/scripts")
from plots import all_figures
all_figures(Path("SyntheticAnalysis/runs/20260918T034857_716429Z"))
PY
```
