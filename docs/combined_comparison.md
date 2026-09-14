# 内部結果と外部baselineの統合解析

このCLIは保存済みの数値だけを集計します。学習、モデルのfit/transform、元の評価pipeline、
データ取得・前処理は呼びません。既存の通常UMAPing環境にあるNumPy・pandas・SciPy・matplotlib・PyYAMLを使います。
外部baselineの環境をactivate・再構築・変更する必要はありません。

## 実行

リモートの既存checkoutから以下を一括実行します。出力は毎回新しい時刻付きディレクトリへ保存します。
`--output-dir`はCLI側で新規作成するため、先に作るのはその親ディレクトリだけです。
`tee`のログも時刻付きの新規ファイルに書きます。

```bash
cd /home/suzuki/Learn/UMAPing && bash <<'BASH'
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git fetch origin
git switch analysis/combined-baseline-evaluation
git pull --ff-only origin analysis/combined-baseline-evaluation
EXTERNAL_RUN="$REPO_ROOT/runs_external_baselines/20260914T070151Z_3800762"
test -f "$EXTERNAL_RUN/summary.csv"
test -x "$REPO_ROOT/.venv/bin/python"
mkdir -p "$REPO_ROOT/runs_comparison_analysis"
ANALYSIS_ID="$(date -u '+%Y%m%dT%H%M%SZ')_$$"
OUTPUT_DIR="$REPO_ROOT/runs_comparison_analysis/$ANALYSIS_ID"
LOGFILE="$(mktemp "$REPO_ROOT/runs_comparison_analysis/${ANALYSIS_ID}.log.XXXXXX")"
"$REPO_ROOT/.venv/bin/python" scripts/compare_results.py \
  --internal-runs-root "$REPO_ROOT/runs" \
  --external-run-dir "$EXTERNAL_RUN" \
  --output-dir "$OUTPUT_DIR" \
  --seed 0 --resamples 10000 2>&1 | tee "$LOGFILE"
echo "出力: $OUTPUT_DIR"
echo "レポート: $OUTPUT_DIR/report.md"
echo "主表: $OUTPUT_DIR/tables/main_recall15.csv"
echo "ログ: $LOGFILE"
BASH
```

比較対象の外部runはCLI引数で指定するため、別runでも使用できます。
`python -m umaping.comparison`も同じ引数を受け付けます。
通常は前景実行です。対応検定は比較ごとに進捗を表示し、10,000回の再標本化を小さなバッチに分割します。
データ量とCPUによっては数分以上かかりますが、GPU学習は発生しません。
動作確認用の`--resamples 100`等も受け付けますが、論文用の既定値は10,000です。

## 対象とschema

対象は`coil20`、`coil100`、`pancreas`、`pancreas_batch_corrected`、`fashion_mnist`、
`mnist_oos`、`organoid`、`embryoid_body`の8件です。失敗済み2件はmetadataの除外一覧だけに記録します。

読み込む内部ファイルは各`runs/<dataset>/main/`の以下です。

- `config.yaml`、`metadata.json`
- `metrics/embedding.json`：method別の集約値、k、query件数、診断subsetのID
- `metrics/baseline_comparison.csv`：追加実験の集約値・利用可否
- `metrics/timing.json`：追加baseline段階の時間
- `metrics/advanced_per_query.csv`：method/query_indexごとのRecall、NDCG、配置診断

通常の`metrics/per_query.csv`の`recall_at_k`・`ndcg_at_k`はretrieverの指標です。
これを最終埋め込みのRecall・NDCGと混ぜないため、その列は使いません。

外部runでは`summary.csv`を必須とし、`<dataset>/<baseline>/result.json`と
`advanced_per_query.csv`を発見して照合します。前のrunnerはこのquery別CSVと`embeddings.npz`を保存しています。
JSONの`metrics`とsummaryの同じ列は一致を検査します。外部per-query平均も同じ計算経路の集約値と照合します。
同じファイル内のdataset/methodの重複や、同一測定を表す集約値の競合は黙って選ばずエラーにします。

method名は`ours_full→ours`、`oracle_neighbors→ours_oracle_neighbors`、
`numap_sep_spectralnet→numap`、`param_repulsor→paramrepulsor`に統一します。
旧環境での利用不可行は観測履歴として残し、外部での成功値があれば採用します。
利用不可の数値は欠測にし、主表・順位・勝敗に使いません。CSVの欠測表現は`NaN`、JSONは`null`です。

## 測定段階の区別

元の`analyze-advanced`は当時別途埋め込みを作る実装だったため、`embedding.json`と
`advanced_per_query.csv`は異なる測定段階です。今回の解析ではどちらも再実行しません。
実際にCOIL-100のstandard UMAPで次の差を確認しています。

- 主評価のRecall@15：`0.6773703703703704`
- 追加解析段階のper-query平均：およそ`0.676888888888889`

主表では集約値を保ち、対応検定では追加解析段階の同じqueryの値を使います。
差は`per_query_phase_recall15_mean`、metadata、レポートに明記します。
不足している副指標だけはper-query平均で補い、各列の出典を記録します。
外部結果は同一実行段階ですが、Recall15には次の2つの計算経路があります。

- `neighborhood_recall_at_k`（k=15）：15近傍を直接取得して計算する主評価。
- `recall_at_15`：multi-k評価で最大k（通常30）の近傍を取得し、その先頭15件で計算する値。

元の近傍探索は同距離点の選択順を固定していないため、この2値は一致を保証しません。
同距離点のある合成データで差を再現していますが、実データの差の個別原因は再計算していません。
主表の`recall_at_15`には内部と同じdirect k=15の主評価を優先し、外部の両方の値を
`combined_long.csv`の`recall15_direct`・`recall15_multi_k`に保持します。
対応検定・tail・multi-k図は内部と同じmulti-k経路のper-query値を使います。
summaryとresult JSON、さらに外部per-query平均は、それぞれ同じ計算経路同士で一致を検査し、
本当に矛盾する値はエラーにします。2経路の差はmetadataと日本語レポートにも記録します。

内部の主評価と追加baselineでは時間測定も別実行です。利用できればbaseline/timing段階の時間を優先します。
別測定の元値は`raw_metric_records.json`にすべて保持し、どの値を選んだかを記録します。

## 主比較と順位

主指標は固定referenceに対するNeighborhood Recall@15です。
保存された`neighborhood_recall_at_k`は、保存kが15の場合だけRecall@15として使用します。
平均・中央値はそのmethodの測定済みdatasetだけで計算し、欠測を0にしません。

主表の順位は全候補が測定済みの共通dataset集合上で計算します。
その共通集合が空なら、8件中75%以上で測定された手法群に固定し、その群で共通するdatasetだけを使います。
それでもpanelを作れない場合は順位を欠測とし、無理に平均順位を作りません。
疎な手法も主表から捨てず、測定値・n_datasetsを残します。順位の対象外ならn_rank_datasetsは0です。
このため平均値のn_datasetsと順位のn_rank_datasetsは異なる場合があります。

順位計算時だけ小数12桁に丸め、平均順位を使います。原値は変更しません。
wins/top2は同順位を含めた最小順位が1/2以内の件数です。利用不可を敗北にしません。
副指標も指標ごとに利用可能な固定panelで集計し、そのpanelをmetadataに保存します。

Friedman検定は共通panelに少なくとも3手法・5datasetがある場合だけ探索的に行います。
補正済みpancreasは関連条件としてこの検定の独立dataset数から外します。
小標本・dataset間の依存についてはレポートの制約に明記します。

## 対応検定

method間でquery IDが重複せず、集合・件数が一致することを確認します。
外部との比較は、保存された前処理cacheのSHA-256が内部cacheと一致した場合だけ対応検定を行います。
cacheがない場合も集約表は作れますが、分割同一性が未検証であることを明記し、外部対応検定を省きます。
IDの順序だけが違う場合はIDで並べ替えます。NaNは対応が取れた2値が両方有限なqueryだけを使用します。

oursと、対応値を持つ各methodについてRecall@15・NDCG・local displacementを検定します。
高いほど良い指標はours−baseline、低いほど良い指標はbaseline−oursに向きを揃えます。
平均・中央値の差、平均差の対応bootstrap 95%区間、oursが良いqueryの割合、同点率を保存します。
両側符号反転検定は非ゼロ差が16件以下なら全組合せを列挙し、それ以外は10,000回のMonte Carloです。
Monte Carloのp値は分子・分母に1を加え、0と報告しません。
主仮説の8datasetのp値は1つのHolmファミリー、それ以外は指標別の探索的ファミリーとして補正します。

dataset単位の片側sign testはoursがstandard UMAPに勝つ確率0.5を帰無仮説にします。
同点は除外し、勝数・敗数・同点数・有効nを明示します。query単位の検定と混同しません。
query再標本化のCIは、再学習seed間の変動や患者・細胞・dataset間の依存を評価するものではありません。

## Ablation・裾・周縁配置

`repulsion_oracle_diagnostic`はMonte Carlo斥力、`exact_repulsion_diagnostic`は全referenceによる厳密斥力です。
これらは全queryでの主比較から外し、成功した測定自体はlong表とablation表に保持します。
前者の追加解析CSVはsubset内の連番を保存していたため、`embedding.json`の元query IDへ復元します。
後者は元runnerが使用した保存seed・query件数・`default_rng(seed).choice`の規則から同じsubsetを復元します。
対応するoursの同じsubsetがある場合だけΔを出します。異なる件数の平均を直接引きません。

Retriever gapはoracle neighbors−oursのRecall差の平均絶対値・最大絶対値で定量化します。
反発の寄与ではours/no_repulsionの勝敗と差をそのまま記録し、Recallの改善を前提にしません。
配置幾何の診断は別に比較します。

Recallのworst 5%・1%は低い方、displacementは高い方を使います。対象件数はceil、最低1件です。
periphery_percentileは0〜100なので90/95を閾値にし、accumulationは0〜1なので0.90/0.95を閾値にします。
周縁配置はOOS診断としてラベルを付け、普遍的な品質指標とは主張しません。

## 連続構造

organoid/embryoid_bodyのloaderは検出したgrouping列を最初の保存ラベルにします。
その列が明確な`time`・`day`・`timepoint`・`time_point`で、有限な数値として保存されている場合だけ使います。
`state`や`sample_labels`などを辞書順・文字列の数字抜き出しで順序化しません。
外部の保存座標とcache照合が揃う場合、2Dでのreference 15近傍との平均時刻差と、
最大100,000組のquery-referenceペアにおける時刻差と2D距離のSpearman相関を計算します。
query/referenceペアの乱数はmethod間で揃えます。時刻の単位を推測・変換せず、dataset横断平均もしません。
内部query座標は現行の保存物にはないため、このための再推論は行いません。

## 出力

ルートに`metadata.json`、`raw_metric_records.json`、`combined_long.csv`、`method_availability.csv`、日本語の`report.md`を保存します。
`tables/`には主Recall表（CSV/Markdown）、副指標、standard UMAPに対するgain、対応検定、tail、ablation、periphery、連続構造を保存します。
原ファイルのパス・SHA-256、解析コードのGit commit、日時、seed、順位panel、除外理由をmetadataに残します。

`figures/`はmatplotlibによる300 dpi PNGと編集可能なSVGです。seabornは使いません。
Recall主表、相対改善率、平均順位、複数kのRecall、worst-5% Recall、periphery、速度・品質の図を生成します。
対応値があればdataset別のRecall差分布とdisplacementの経験CDFも作ります。
不足データによって生成できない図は理由をmetadataへ記録します。
図の文字はフォント互換性と論文利用を考慮して英語、README・説明・自動レポートは日本語です。

内部oursの単一query呼出し時間と、外部の一括変換時間÷query件数は測定範囲が異なります。
内部standard UMAPのfit時間もquery変換を含むため、速度倍率を単純に主張しないでください。
図では正の実測時間だけを使い、欠測を0秒にしません。

## 検証

`tests/test_combined_comparison.py`は合成の内部・外部schemaを用いて、読み込み、別名統一、欠測、
重複・矛盾拒否、hash照合、query IDの整列、正確/Monte Carlo検定、bootstrap、Holm、方向性、
tail、subset、連続構造、部分欠測でのレポートと図を検証します。
追加テストではdirect/multi-kの異なる保存値の保持、同距離近傍での差の再現、同じ計算経路内の矛盾拒否も検証します。
統合解析25件と既存のadvanced解析14件、計39テストが通過しています。
リモート用コマンドのbash構文と、CLIの引数表示も確認しました。
実際にGitへ保存済みの内部8datasetはschemaの読み取り確認にのみ使用しました。
ローカルにない外部実測値の結論は作っていません。実データ学習・推論は実行していません。
