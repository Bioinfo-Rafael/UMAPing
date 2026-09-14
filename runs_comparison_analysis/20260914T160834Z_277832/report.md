# UMAPing：保存済み結果の統合比較



解析日時: 2026-09-14T16:09:27.969729+00:00。解析コード: `8f7c31f37269a39565d3655f90a1d1a9bf4260d9`。seed: 0。

学習・推論・前処理は再実行していません。数値は保存済みの測定結果、またはその統計集計です。



## 1. データ範囲・手法の利用可否



| method | n_datasets | n_recall15 |
| --- | --- | --- |
| no_repulsion | 8 | 8 |
| numap | 8 | 8 |
| ours | 8 | 8 |
| ours_oracle_neighbors | 8 | 8 |
| paramrepulsor | 8 | 8 |
| reduced_repulsion_umap | 4 | 4 |
| spectral_only | 8 | 8 |
| standard_umap | 8 | 8 |
| weighted_knn | 4 | 4 |



利用不可・失敗は欠測として扱い、ゼロや敗北に置き換えません。主表は全queryの評価だけです。

一部queryの厳密斥力診断とMonte Carlo斥力診断は、別手法として診断・ablation表に保持しています。

詳細: [利用可否](method_availability.csv)、[全測定値](combined_long.csv)。



## 2. 主結果：Neighborhood Recall@15



高次元と2次元の双方で、各queryから固定reference集合に対する15近傍を比較します。高いほど良い指標です。

主表は保存済みのdirect k=15評価を優先します。外部結果のmulti-k版Recall15は最大k近傍の先頭15件を使う別計算で、同距離近傍の選択により差が生じ得ます。両方の値は全測定値のrecall15_direct・recall15_multi_k列に保持します。

[主表](tables/main_recall15.md)にはdataset別Recall、利用可能dataset上の平均・中央値、固定panelの順位を示します。



| method | mean_recall15 | median_recall15 | mean_rank | n_datasets | n_rank_datasets | wins | top2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_repulsion | 0.3275 | 0.2513 | 3 | 8 | 4 | 0 | 0 |
| numap | 0.03395 | 0.01818 | 9 | 8 | 4 | 0 | 0 |
| ours | 0.3309 | 0.2531 | 1.5 | 8 | 4 | 2 | 4 |
| ours_oracle_neighbors | 0.3311 | 0.2531 | 1.5 | 8 | 4 | 2 | 4 |
| paramrepulsor | 0.194 | 0.1057 | 7 | 8 | 4 | 0 | 0 |
| reduced_repulsion_umap | 0.1876 | 0.1928 | 4.25 | 4 | 4 | 0 | 0 |
| spectral_only | 0.1237 | 0.05649 | 8 | 8 | 4 | 0 | 0 |
| standard_umap | 0.3108 | 0.2241 | 4.75 | 8 | 4 | 0 | 0 |
| weighted_knn | 0.1583 | 0.1583 | 6 | 4 | 4 | 0 | 0 |



順位panel: fashion_mnist, mnist_oos, organoid, embryoid_body。全手法が共通して測定されたデータセット。

値の平均に使うn_datasetsと、同じ手法群・dataset群で順位を平均するn_rank_datasetsは異なる場合があります。



## 3. standard UMAPに対する改善



比較可能な8データセットで、oursは8勝・0敗・0同点でした。

絶対差の平均=0.02013、中央値=0.01695。

相対差の平均=0.1176、中央値=0.1201、最小=0.01679、最大=0.2952。相対差1は100%改善です。

dataset単位の片側正確二項検定: n=8、p=0.003906（同点は除外）。

これはquery単位の有意性ではありません。pancreasの補正あり・なしは関連条件であり、datasetの独立性仮定には制約があります。

| dataset | method | recall_at_15 | standard_umap | absolute_gain | relative_gain |
| --- | --- | --- | --- | --- | --- |
| coil20 | ours | 0.7739 | 0.7611 | 0.01278 | 0.01679 |
| coil100 | ours | 0.6938 | 0.6774 | 0.01644 | 0.02428 |
| pancreas | ours | 0.2365 | 0.2104 | 0.0261 | 0.124 |
| pancreas_batch_corrected | ours | 0.117 | 0.09032 | 0.02666 | 0.2952 |
| fashion_mnist | ours | 0.3029 | 0.286 | 0.01693 | 0.05921 |
| mnist_oos | ours | 0.2697 | 0.2377 | 0.03193 | 0.1343 |
| organoid | ours | 0.163 | 0.146 | 0.01697 | 0.1162 |
| embryoid_body | ours | 0.09059 | 0.07738 | 0.01321 | 0.1708 |



## 4. query単位の対応統計



同じquery IDと由来を確認できる比較だけを使用します。Recall・NDCGはours−baseline、local displacementはbaseline−oursとし、正ならoursが良い方向です。

対応解析・tail・multi-k図のRecall15は、内部・外部とも保存されたmulti-kのper-query値を使います。主表のdirect k=15評価と計算経路が異なるため、主表の差と完全には一致しない場合があります。

対応bootstrapは10000回、両側符号反転検定も原則同数です。非ゼロ差が16件以下なら全符号の正確検定を行います。

CIは平均差のpercentile 95%区間です。主仮説はours vs standard UMAPのRecall@15で、dataset間のp値にHolm補正を適用します。他は指標別の探索的ファミリーとして補正します。



| dataset | n_pairs | ours_mean | baseline_mean | mean_improvement | ci_low | ci_high | fraction_ours_better | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil20 | 360 | 0.7739 | 0.7611 | 0.01278 | 0.002037 | 0.02333 | 0.3417 | 0.02 | 0.02 |
| coil100 | 1800 | 0.6938 | 0.6769 | 0.01693 | 0.009259 | 0.0247 | 0.3728 | 9.999e-05 | 0.0007999 |
| pancreas | 4679 | 0.2365 | 0.2104 | 0.0261 | 0.02206 | 0.03012 | 0.4486 | 9.999e-05 | 0.0007999 |
| pancreas_batch_corrected | 4679 | 0.117 | 0.09032 | 0.02666 | 0.02359 | 0.02971 | 0.4336 | 9.999e-05 | 0.0007999 |
| fashion_mnist | 2000 | 0.3029 | 0.286 | 0.01693 | 0.01157 | 0.0222 | 0.4455 | 9.999e-05 | 0.0007999 |
| mnist_oos | 2000 | 0.2697 | 0.2377 | 0.03193 | 0.02623 | 0.03763 | 0.4895 | 9.999e-05 | 0.0007999 |
| organoid | 766 | 0.163 | 0.146 | 0.01697 | 0.01027 | 0.02385 | 0.4295 | 9.999e-05 | 0.0007999 |
| embryoid_body | 6206 | 0.09059 | 0.07738 | 0.01321 | 0.01097 | 0.01544 | 0.3717 | 9.999e-05 | 0.0007999 |



この測定条件で正の差を支持する結果（CI下限>0かつHolm p<0.05）: coil20, coil100, pancreas, pancreas_batch_corrected, fashion_mnist, mnist_oos, organoid, embryoid_body。

[全対応比較](tables/paired_comparisons.csv)。単一run内のquery再標本化であり、再学習seed間の不確実性や細胞・患者間の依存を推定したCIではありません。



## 5. 裾・大きなquery配置誤差



Recallは低い順の5%・1%、local displacementは高い順の5%・1%を『worst』とします。件数はceilで切り上げ、最低1件です。低Recallをrecall deficitに反転していません。

| dataset | method | mean | median | p10 | p5 | worst_5pct_mean | worst_1pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil20 | ours | 0.7739 | 0.8 | 0.5333 | 0.4667 | 0.3222 | 0.1833 |
| coil20 | standard_umap | 0.7611 | 0.8 | 0.6 | 0.53 | 0.2741 | 0.03333 |
| coil100 | ours | 0.6938 | 0.7333 | 0.4667 | 0.3333 | 0.2133 | 0.08519 |
| coil100 | standard_umap | 0.6769 | 0.7333 | 0.4 | 0.2667 | 0.1274 | 0 |
| pancreas | ours | 0.2365 | 0.2 | 0 | 0 | 0 | 0 |
| pancreas | standard_umap | 0.2104 | 0.2 | 0 | 0 | 0 | 0 |
| pancreas_batch_corrected | ours | 0.117 | 0.06667 | 0 | 0 | 0 | 0 |
| pancreas_batch_corrected | standard_umap | 0.09032 | 0.06667 | 0 | 0 | 0 | 0 |
| fashion_mnist | ours | 0.3029 | 0.2667 | 0.06667 | 0.06667 | 0.008 | 0 |
| fashion_mnist | standard_umap | 0.286 | 0.2667 | 0.06667 | 0.06667 | 0.001333 | 0 |
| mnist_oos | ours | 0.2697 | 0.2667 | 0.06667 | 0 | 0 | 0 |
| mnist_oos | standard_umap | 0.2377 | 0.2 | 0.06667 | 0 | 0 | 0 |
| organoid | ours | 0.163 | 0.1333 | 0 | 0 | 0 | 0 |
| organoid | standard_umap | 0.146 | 0.1333 | 0 | 0 | 0 | 0 |
| embryoid_body | ours | 0.09059 | 0.06667 | 0 | 0 | 0 | 0 |
| embryoid_body | standard_umap | 0.07738 | 0.06667 | 0 | 0 | 0 | 0 |

worst-5% Recallは、対応する8dataset中oursが3件で高く、0件で低い結果でした。

詳細: [tail表](tables/tail_metrics.csv)。これらはquery配置の診断であり、失敗を区切る普遍的な閾値を定義したものではありません。



## 6. Ablation



[dataset別ablation](tables/ablations.csv)、[集約](tables/ablation_summary.csv)。Δはmethod−oursのRecallです。

subset診断は同じqueryのoursと比較できる場合だけΔを出します。Monte Carlo診断とexact診断は混同しません。

Retriever gap（oracle neighbors−ours）の平均絶対値=0.000431、最大絶対値=0.001131、n=8。

Recallでoursがno_repulsionを上回るdataset: coil100, pancreas, pancreas_batch_corrected, fashion_mnist, mnist_oos, organoid, embryoid_body。

no_repulsionがoursを上回るdataset: coil20。

ours−no_repulsionの平均Recall差=0.003427。反発が常にRecallを改善するとは仮定しません。



## 7. OOS periphery・反発診断



periphery_percentileは同ラベルreferenceの中心からの距離に対する百分位です。90/95以上の割合を報告します。

repulsion_accumulation_scoreは0〜1なので、極端値の閾値には0.90/0.95を使います。90/95をそのまま適用しません。

極端な周縁配置の頻度が低い方向を良い方向として扱いますが、普遍的な埋め込み品質指標ではありません。

[periphery表](tables/periphery_metrics.csv)ではours、standard UMAP、no_repulsionなどの実測値を比較できます。



periphery_percentile: 極端値頻度はstandard_umapとの共通8dataset中、oursが7件で低く、1件で高い結果です。

periphery_percentile: 極端値頻度はno_repulsionとの共通8dataset中、oursが2件で低く、5件で高い結果です。

repulsion_accumulation_score: 極端値頻度はstandard_umapとの共通8dataset中、oursが6件で低く、0件で高い結果です。

repulsion_accumulation_score: 極端値頻度はno_repulsionとの共通8dataset中、oursが0件で低く、3件で高い結果です。



## 8. 外部baselineとの比較



parametric_umap: 比較可能な成功値がなく、優劣は判定できません。

numap: 共通8datasetでoursのRecallが高い=8件、低い=0件、平均差=0.297。

paramrepulsor: 共通8datasetでoursのRecallが高い=8件、低い=0件、平均差=0.1369。

外部手法のper-query CSVがあれば対応検定に使います。集約値しかない場合は推論を再実行せず、query単位の値を補いません。

公式OOS-UMAPリポジトリには、このdataset群を忠実に評価できる汎用の学習・fit/transform実行部分が揃っていませんでした。独自再構築による実装上の曖昧さを避けるため、利用不可として保持しています。これは手法自体の失敗を意味しません。



## 9. 速度・品質の関係



正の実測query時間がある手法だけを図に載せます。欠測を0秒として扱いません。

内部oursは単一query呼出しの平均、外部手法は一括transform時間÷query数です。測定環境・Python/依存versionも異なるため、厳密な速度倍率の主張には使えません。

内部standard UMAPのfit_time_secondsは既存実装上fitとquery変換を含みます。外部fit時間と単純な同一条件比較はできません。



## 10. 制約と利用できない比較



測定で支持される事項は上の実測差と条件付きCIに限定します。勝敗が混在する比較、欠測、subset診断を一般的な優越性の主張にまとめません。

主表の集約段階と追加解析段階で再生成された埋め込みが違う場合、保存値を上書きせず両方の出典と差を記録します。

補正済みpancreasはscArchesのquery適応を含む元の表現を再利用しています。厳密なreference-only前処理と同じ条件ではありません。

連続構造はloaderが保存した数値のtime/day列と保存座標が揃う場合だけ評価します。stateやsample_labels等の文字列から順序を推測しません。

[連続構造表](tables/continuous_structure.csv)、[解析メタデータ](metadata.json)、[原測定記録](raw_metric_records.json)。



- coil20/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- coil100/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- coil100/standard_umap: 集約Recall15=0.67737037037、追加解析段階の平均=0.676888888889。主表は集約値、対応検定は追加解析段階の値を使用。

- pancreas/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- pancreas_batch_corrected/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- fashion_mnist/ours/mean_query_latency_seconds: 別実行段階の時間 0.6809891144321301 / 0.08465528387669474。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- fashion_mnist/ours_oracle_neighbors/mean_query_latency_seconds: 別実行段階の時間 0.7750841031540185 / 0.20447841995302588。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- fashion_mnist/no_repulsion/mean_query_latency_seconds: 別実行段階の時間 0.32613500954210756 / 0.031256744207348676。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- fashion_mnist/exact診断: 既存runnerのseedとchoice規則から評価subsetを復元。

- fashion_mnist/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- mnist_oos/ours/mean_query_latency_seconds: 別実行段階の時間 0.14008680943399668 / 0.3566884375717491。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- mnist_oos/ours_oracle_neighbors/mean_query_latency_seconds: 別実行段階の時間 0.20924273075163363 / 0.4021591167668812。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- mnist_oos/no_repulsion/mean_query_latency_seconds: 別実行段階の時間 0.05802798091573641 / 0.1516205161730759。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- mnist_oos/exact診断: 既存runnerのseedとchoice規則から評価subsetを復元。

- mnist_oos/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- organoid/ours/mean_query_latency_seconds: 別実行段階の時間 0.3016973642921732 / 0.35323837182822565。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- organoid/ours_oracle_neighbors/mean_query_latency_seconds: 別実行段階の時間 0.29539677845897744 / 0.34539415482212804。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- organoid/no_repulsion/mean_query_latency_seconds: 別実行段階の時間 0.1458372361629754 / 0.14814032241580496。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- organoid/exact診断: 既存runnerのseedとchoice規則から評価subsetを復元。

- organoid/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- embryoid_body/ours/mean_query_latency_seconds: 別実行段階の時間 0.3476030239373172 / 0.08636873875811571。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- embryoid_body/ours_oracle_neighbors/mean_query_latency_seconds: 別実行段階の時間 0.38411674982049665 / 0.1999910014943837。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- embryoid_body/no_repulsion/mean_query_latency_seconds: 別実行段階の時間 0.1652685813104558 / 0.03189975036250543。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。

- embryoid_body/exact診断: 既存runnerのseedとchoice規則から評価subsetを復元。

- embryoid_body/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。

- coil100/numap: direct k=15 Recall=0.0685555555556、multi-k Recall15=0.0685925925926。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- coil100/numap: direct k=15 Recall=0.0685555555556、multi-k Recall15=0.0685925925926。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- pancreas/numap: direct k=15 Recall=0.0133646790625、multi-k Recall15=0.0133789271212。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- pancreas/numap: direct k=15 Recall=0.0133646790625、multi-k Recall15=0.0133789271212。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- pancreas/paramrepulsor: direct k=15 Recall=0.093809218494、multi-k Recall15=0.0937949704353。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- pancreas/paramrepulsor: direct k=15 Recall=0.093809218494、multi-k Recall15=0.0937949704353。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- pancreas_batch_corrected/paramrepulsor: direct k=15 Recall=0.0408349362399、multi-k Recall15=0.0408206881812。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- pancreas_batch_corrected/paramrepulsor: direct k=15 Recall=0.0408349362399、multi-k Recall15=0.0408206881812。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- embryoid_body/numap: direct k=15 Recall=0.00652057148996、multi-k Recall15=0.00654205607477。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- embryoid_body/numap: direct k=15 Recall=0.00652057148996、multi-k Recall15=0.00654205607477。保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。

- coil20/numap/local_displacement: 対応値なし

- coil20/paramrepulsor/local_displacement: 対応値なし

- coil100/numap/local_displacement: 対応値なし

- coil100/paramrepulsor/local_displacement: 対応値なし

- pancreas/numap/local_displacement: 対応値なし

- pancreas/paramrepulsor/local_displacement: 対応値なし

- pancreas_batch_corrected/numap/local_displacement: 対応値なし

- pancreas_batch_corrected/paramrepulsor/local_displacement: 対応値なし

- fashion_mnist/numap/local_displacement: 対応値なし

- fashion_mnist/paramrepulsor/local_displacement: 対応値なし

- mnist_oos/numap/local_displacement: 対応値なし

- mnist_oos/paramrepulsor/local_displacement: 対応値なし

- organoid/numap/local_displacement: 対応値なし

- organoid/paramrepulsor/local_displacement: 対応値なし

- embryoid_body/numap/local_displacement: 対応値なし

- embryoid_body/paramrepulsor/local_displacement: 対応値なし

- organoid: 保存grouping列='labeling_time'。loaderが順序を保証しないため連続構造解析を省略。

- embryoid_body: 保存grouping列='stage'。loaderが順序を保証しないため連続構造解析を省略。

Friedman検定は実行せず順位要約のみとしました: 共通panelが3手法・5データセット未満、または独立性の制約。
