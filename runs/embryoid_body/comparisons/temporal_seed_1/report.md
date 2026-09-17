# FitGrid最終埋め込み比較



元runは読み取り専用。全手法は同じPreparedDatasetを使用。参照軌道生成は変更していない。

主指標はquery→固定referenceのRecall@15。reference trustworthinessは同一参照の整合性確認のみ。

通常評価のfield.jsonは従来のMC teacher診断。主たる場の評価はexact_field/validation.npzの全点和1000点。

benchmarkのUniformモデルとmainのUniformモデルは別学習であり、混同しない。以下は今回実際に読み込んだモデルの比較。



| method | rmse | cosine | magnitude_error | recall_at_15 | latency_seconds |
| --- | --- | --- | --- | --- | --- |
| uniform_mc | 0.01229 | 0.738 | 0.006131 | 0.1084 | 1.601 |
| fit_grid | 0.009734 | 0.8272 | 0.005499 | 0.1097 | 1.132 |



## 最終埋め込み（全query）

| group | n_queries | recall_at_15 | ndcg | fuzzy_weighted_mse | fuzzy_weighted_bce | local_displacement | repulsion_accumulation_score | density_log_distortion | collapse_rate | global_periphery_percentile | latency_seconds | timepoint_ordinal | horizon | temporal_neighbor_mae | excess_temporal_neighbor_mae | temporal_bracketing_rate | timepoint_label_accuracy | recall_at_15_p05 | local_displacement_p95 | label_accuracy_status | method | status | fit_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 12543 | 0.1084 | 0.1132 | 0.1812 | 0.5724 | 0.3932 | 0.3068 | 0.2799 | 0 | 0.468 | 1.601 | 4.005 | 0.5024 | 1.073 | 0.1381 | 0.3701 | — | 0 | 0.8024 | not_applicable_unseen_timepoints | uniform_mc | success | — |
| all | 12543 | 0.1097 | 0.1155 | 0.1821 | 0.5744 | 0.3963 | 0.3164 | 0.278 | 0 | 0.4696 | 1.132 | 4.005 | 0.5024 | 1.073 | 0.1384 | 0.3801 | — | 0 | 0.8035 | not_applicable_unseen_timepoints | fit_grid | success | — |



## 対応比較

| status | n_pairs | n_dropped | ours_mean | baseline_mean | mean_improvement | median_improvement | ci_low | ci_high | fraction_ours_better | fraction_tied | n_bootstrap | seed | direction | group | fraction_worsened | paired_effect_size_dz | conditional_sign_test_p | test_assumption | inference_scope | conditional_sign_test_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | 12543 | 0 | 0.1097 | 0.1084 | 0.001233 | 0 | 0.0005528 | 0.001945 | 0.1398 | 0.7354 | 10000 | 0 | higher | all | 0.1249 | 0.031 | 0.001241 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.006204 |
| ok | 6241 | 0 | 0.1046 | 0.1038 | 0.0008012 | 0 | -0.0001709 | 0.001795 | 0.1319 | 0.7464 | 10000 | 0 | higher | interpolation | 0.1218 | 0.02039 | 0.1191 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.2383 |
| ok | 6302 | 0 | 0.1147 | 0.113 | 0.001661 | 0 | 0.0006876 | 0.002655 | 0.1476 | 0.7245 | 10000 | 0 | higher | extrapolation | 0.1279 | 0.04126 | 0.003145 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.01258 |
| ok | 6241 | 0 | 0.1046 | 0.1038 | 0.0008012 | 0 | -0.0001709 | 0.001795 | 0.1319 | 0.7464 | 10000 | 0 | higher | timepoint:4-5 | 0.1218 | 0.02039 | 0.1191 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.2383 |
| ok | 6302 | 0 | 0.1147 | 0.113 | 0.001661 | 0 | 0.0006876 | 0.002655 | 0.1476 | 0.7245 | 10000 | 0 | higher | timepoint:8-9 | 0.1279 | 0.04126 | 0.003145 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.01258 |

CIは固定モデル・参照に条件付けたcell bootstrap（10,000回）。生物学的反復やdonor依存を補正していない。

符号検定は独立な非同点queryの正負確率を仮定した参考値。Holm補正を併記し、仮定の成立を断言しない。

効果量は平均Recall差（主）とpaired dz。CIが0を含む場合、改善の証拠は不確定。



## 密度・周辺・collapseの定義

密度歪みは、各空間のqueryの15近傍半径をそのHD真近傍のreference側15近傍半径平均で割り、両空間のlog比の差の絶対値。

global_periphery_percentileは全reference重心からの半径の参照内百分位。未観測時間ラベルでclass条件付き重心を作らない。

collapse_rateは2Dの正規化15近傍半径が0.1未満のquery割合。NDCGは既存距離rank保持指標。

標準/reduced UMAP adapterがquery時間を返さない場合はN/A。fit込み時間をquery latencyに代入しない。

exact_repulsion_diagnosticは配備用手法ではない。oracle_query_indices.jsonの固定subsetのみ。



## 判定

この固定モデル・参照とcell bootstrapの条件では、FitGridの平均Recall改善CIは0より大きい。生物学的一般化は未検証。



## 時間holdout

| timepoint | n_cells | role |
| --- | --- | --- |
| 0-1 | 4575 | reference |
| 2-3 | 7368 | reference |
| 4-5 | 6241 | interpolation |
| 6-7 | 6543 | reference |
| 8-9 | 6302 | extrapolation |

選択規則: T>=5; m=clip(floor(0.6*T),2,T-2); ref=1..m-1,m+1; query=m,m+2..T

時間点分類精度はN/A（queryの時間ラベルはreferenceに存在しない）。別annotationがあればcell-type精度を別列で報告。

temporal-neighbor MAEの単位は年代順の位置差。実時間間隔が不均等でも日数誤差とは解釈しない。

excess MAEはlast observedからの不可避なordinal horizonを差し引く。



| group | n_queries | recall_at_15 | ndcg | fuzzy_weighted_mse | fuzzy_weighted_bce | local_displacement | repulsion_accumulation_score | density_log_distortion | collapse_rate | global_periphery_percentile | latency_seconds | timepoint_ordinal | horizon | temporal_neighbor_mae | excess_temporal_neighbor_mae | temporal_bracketing_rate | timepoint_label_accuracy | recall_at_15_p05 | local_displacement_p95 | label_accuracy_status | method | status | fit_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 12543 | 0.1084 | 0.1132 | 0.1812 | 0.5724 | 0.3932 | 0.3068 | 0.2799 | 0 | 0.468 | 1.601 | 4.005 | 0.5024 | 1.073 | 0.1381 | 0.3701 | — | 0 | 0.8024 | not_applicable_unseen_timepoints | uniform_mc | success | — |
| interpolation | 6241 | 0.1038 | 0.1089 | 0.1884 | 0.5988 | 0.3818 | 0.2959 | 0.2758 | 0 | 0.4892 | 1.629 | 3 | 0 | 1.008 | — | 0.3701 | — | 0 | 0.8055 | not_applicable_unseen_timepoints | uniform_mc | success | — |
| extrapolation | 6302 | 0.113 | 0.1174 | 0.1741 | 0.5462 | 0.4045 | 0.3177 | 0.2839 | 0 | 0.447 | 1.573 | 5 | 1 | 1.138 | 0.1381 | — | — | 0 | 0.7989 | not_applicable_unseen_timepoints | uniform_mc | success | — |
| timepoint:4-5 | 6241 | 0.1038 | 0.1089 | 0.1884 | 0.5988 | 0.3818 | 0.2959 | 0.2758 | 0 | 0.4892 | 1.629 | 3 | 0 | 1.008 | — | 0.3701 | — | 0 | 0.8055 | not_applicable_unseen_timepoints | uniform_mc | success | — |
| timepoint:8-9 | 6302 | 0.113 | 0.1174 | 0.1741 | 0.5462 | 0.4045 | 0.3177 | 0.2839 | 0 | 0.447 | 1.573 | 5 | 1 | 1.138 | 0.1381 | — | — | 0 | 0.7989 | not_applicable_unseen_timepoints | uniform_mc | success | — |
| all | 12543 | 0.1097 | 0.1155 | 0.1821 | 0.5744 | 0.3963 | 0.3164 | 0.278 | 0 | 0.4696 | 1.132 | 4.005 | 0.5024 | 1.073 | 0.1384 | 0.3801 | — | 0 | 0.8035 | not_applicable_unseen_timepoints | fit_grid | success | — |
| interpolation | 6241 | 0.1046 | 0.1109 | 0.1888 | 0.5994 | 0.3776 | 0.2985 | 0.2767 | 0 | 0.4901 | 0.846 | 3 | 0 | 1.008 | — | 0.3801 | — | 0 | 0.7998 | not_applicable_unseen_timepoints | fit_grid | success | — |
| extrapolation | 6302 | 0.1147 | 0.1201 | 0.1755 | 0.5496 | 0.4149 | 0.334 | 0.2793 | 0 | 0.4493 | 1.416 | 5 | 1 | 1.138 | 0.1384 | — | — | 0 | 0.8074 | not_applicable_unseen_timepoints | fit_grid | success | — |
| timepoint:4-5 | 6241 | 0.1046 | 0.1109 | 0.1888 | 0.5994 | 0.3776 | 0.2985 | 0.2767 | 0 | 0.4901 | 0.846 | 3 | 0 | 1.008 | — | 0.3801 | — | 0 | 0.7998 | not_applicable_unseen_timepoints | fit_grid | success | — |
| timepoint:8-9 | 6302 | 0.1147 | 0.1201 | 0.1755 | 0.5496 | 0.4149 | 0.334 | 0.2793 | 0 | 0.4493 | 1.416 | 5 | 1 | 1.138 | 0.1384 | — | — | 0 | 0.8074 | not_applicable_unseen_timepoints | fit_grid | success | — |



外挿時間点が1つだけの場合、horizonに対する単調性は判定不可。



## unavailable / failed

なし。