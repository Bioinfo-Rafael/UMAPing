# FitGrid最終埋め込み比較



元runは読み取り専用。全手法は同じPreparedDatasetを使用。参照軌道生成は変更していない。

主指標はquery→固定referenceのRecall@15。reference trustworthinessは同一参照の整合性確認のみ。

通常評価のfield.jsonは従来のMC teacher診断。主たる場の評価はexact_field/validation.npzの全点和1000点。

benchmarkのUniformモデルとmainのUniformモデルは別学習であり、混同しない。以下は今回実際に読み込んだモデルの比較。



| method | rmse | cosine | magnitude_error | recall_at_15 | latency_seconds |
| --- | --- | --- | --- | --- | --- |
| uniform_mc | 0.01289 | 0.7479 | 0.006382 | 0.1099 | 0.08144 |
| fit_grid | 0.01021 | 0.8221 | 0.005782 | 0.1101 | 0.08159 |



## 最終埋め込み（全query）

| group | n_queries | recall_at_15 | ndcg | fuzzy_weighted_mse | fuzzy_weighted_bce | local_displacement | repulsion_accumulation_score | density_log_distortion | collapse_rate | global_periphery_percentile | latency_seconds | timepoint_ordinal | horizon | temporal_neighbor_mae | excess_temporal_neighbor_mae | temporal_bracketing_rate | timepoint_label_accuracy | recall_at_15_p05 | local_displacement_p95 | label_accuracy_status | method | status | fit_time_seconds | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1.254e+04 | 0.1099 | 0.1151 | 0.1825 | 0.575 | 0.4026 | 0.3181 | 0.2802 | 0 | 0.468 | 0.08144 | 4.005 | 0.5024 | 1.072 | 0.1351 | 0.3793 | — | 0 | 0.8069 | not_applicable_unseen_timepoints | uniform_mc | success | — | — |
| all | 1.254e+04 | 0.1101 | 0.1159 | 0.182 | 0.5739 | 0.395 | 0.3139 | 0.2807 | 0 | 0.4699 | 0.08159 | 4.005 | 0.5024 | 1.073 | 0.1368 | 0.3812 | — | 0 | 0.7975 | not_applicable_unseen_timepoints | fit_grid | success | — | — |
| all | 1.254e+04 | 0.09103 | 0.09279 | 0.1444 | 0.4742 | 0.5598 | 0.4063 | 0.31 | 7.973e-05 | 0.6128 | — | 4.005 | 0.5024 | 1.076 | 0.1431 | 0.451 | — | 0 | 1.101 | not_applicable_unseen_timepoints | standard_umap | success | 9.972 | — |
| all | 1.254e+04 | 0.08853 | 0.09087 | 0.1307 | 0.4384 | 0.3713 | 0.1776 | 0.2775 | 0.0002392 | 0.6096 | — | 4.005 | 0.5024 | 1.095 | 0.1797 | 0.3953 | — | 0 | 0.8026 | not_applicable_unseen_timepoints | reduced_repulsion_umap | success | 8.239 | — |
| all | 1.254e+04 | 0.06536 | 0.06675 | 0.1537 | 0.4841 | 2.896e-06 | 0.2602 | 0.369 | 0 | 0.586 | 0.0002887 | 4.005 | 0.5024 | 1.122 | 0.2327 | 0.4426 | — | 0 | 7.926e-06 | not_applicable_unseen_timepoints | weighted_knn | success | — | — |
| all | 1.254e+04 | 0.1091 | 0.1148 | 0.1779 | 0.5627 | 0.3587 | 0.297 | 0.2796 | 0 | 0.4685 | 1.323 | 4.005 | 0.5024 | 1.078 | 0.1466 | 0.3785 | — | 0 | 0.7347 | not_applicable_unseen_timepoints | fit_grid_oracle_neighbors | success | — | — |
| all | 1.254e+04 | 0.1029 | 0.1071 | 0.1776 | 0.5644 | 0.3435 | 0.2643 | 0.2743 | 0 | 0.4658 | 0.8505 | 4.005 | 0.5024 | 1.082 | 0.155 | 0.3645 | — | 0 | 0.7609 | not_applicable_unseen_timepoints | no_repulsion | success | — | — |
| all | 50 | 0.112 | 0.1169 | 0.1826 | 0.5784 | 0.3902 | 0.3142 | 0.3255 | 0 | 0.4879 | 4.923 | 4.08 | 0.54 | 1.119 | 0.2123 | 0.4348 | — | 0 | 0.7353 | not_applicable_unseen_timepoints | exact_repulsion_diagnostic | success | — | — |



## 対応比較

| status | n_pairs | n_dropped | ours_mean | baseline_mean | mean_improvement | median_improvement | ci_low | ci_high | fraction_ours_better | fraction_tied | n_bootstrap | seed | direction | group | fraction_worsened | paired_effect_size_dz | conditional_sign_test_p | test_assumption | inference_scope | conditional_sign_test_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | 12543 | 0 | 0.1101 | 0.1099 | 0.0001913 | 0 | -0.0005315 | 0.0009142 | 0.1371 | 0.7275 | 10000 | 0 | higher | all | 0.1354 | 0.004699 | 0.7195 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6241 | 0 | 0.1055 | 0.1047 | 0.0008225 | 0 | -0.000203 | 0.001827 | 0.138 | 0.7319 | 10000 | 0 | higher | interpolation | 0.1301 | 0.01984 | 0.2406 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6302 | 0 | 0.1147 | 0.1151 | -0.0004337 | 0 | -0.001375 | 0.0005818 | 0.1363 | 0.7231 | 10000 | 0 | higher | extrapolation | 0.1406 | -0.01085 | 0.5337 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6241 | 0 | 0.1055 | 0.1047 | 0.0008225 | 0 | -0.000203 | 0.001827 | 0.138 | 0.7319 | 10000 | 0 | higher | timepoint:4-5 | 0.1301 | 0.01984 | 0.2406 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6302 | 0 | 0.1147 | 0.1151 | -0.0004337 | 0 | -0.001375 | 0.0005818 | 0.1363 | 0.7231 | 10000 | 0 | higher | timepoint:8-9 | 0.1406 | -0.01085 | 0.5337 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |

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

平均Recall差の95%CIは0を含むため、最終埋め込み改善の証拠は不確定。



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



| group | n_queries | recall_at_15 | ndcg | fuzzy_weighted_mse | fuzzy_weighted_bce | local_displacement | repulsion_accumulation_score | density_log_distortion | collapse_rate | global_periphery_percentile | latency_seconds | timepoint_ordinal | horizon | temporal_neighbor_mae | excess_temporal_neighbor_mae | temporal_bracketing_rate | timepoint_label_accuracy | recall_at_15_p05 | local_displacement_p95 | label_accuracy_status | method | status | fit_time_seconds | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1.254e+04 | 0.1099 | 0.1151 | 0.1825 | 0.575 | 0.4026 | 0.3181 | 0.2802 | 0 | 0.468 | 0.08144 | 4.005 | 0.5024 | 1.072 | 0.1351 | 0.3793 | — | 0 | 0.8069 | not_applicable_unseen_timepoints | uniform_mc | success | — | — |
| interpolation | 6241 | 0.1047 | 0.1098 | 0.1893 | 0.6005 | 0.3881 | 0.3032 | 0.2763 | 0 | 0.4895 | 0.0816 | 3 | 0 | 1.008 | — | 0.3793 | — | 0 | 0.8 | not_applicable_unseen_timepoints | uniform_mc | success | — | — |
| extrapolation | 6302 | 0.1151 | 0.1203 | 0.1758 | 0.5499 | 0.417 | 0.3329 | 0.284 | 0 | 0.4467 | 0.08128 | 5 | 1 | 1.135 | 0.1351 | — | — | 0 | 0.8125 | not_applicable_unseen_timepoints | uniform_mc | success | — | — |
| timepoint:4-5 | 6241 | 0.1047 | 0.1098 | 0.1893 | 0.6005 | 0.3881 | 0.3032 | 0.2763 | 0 | 0.4895 | 0.0816 | 3 | 0 | 1.008 | — | 0.3793 | — | 0 | 0.8 | not_applicable_unseen_timepoints | uniform_mc | success | — | — |
| timepoint:8-9 | 6302 | 0.1151 | 0.1203 | 0.1758 | 0.5499 | 0.417 | 0.3329 | 0.284 | 0 | 0.4467 | 0.08128 | 5 | 1 | 1.135 | 0.1351 | — | — | 0 | 0.8125 | not_applicable_unseen_timepoints | uniform_mc | success | — | — |
| all | 1.254e+04 | 0.1101 | 0.1159 | 0.182 | 0.5739 | 0.395 | 0.3139 | 0.2807 | 0 | 0.4699 | 0.08159 | 4.005 | 0.5024 | 1.073 | 0.1368 | 0.3812 | — | 0 | 0.7975 | not_applicable_unseen_timepoints | fit_grid | success | — | — |
| interpolation | 6241 | 0.1055 | 0.1115 | 0.1888 | 0.5992 | 0.3789 | 0.2971 | 0.2793 | 0 | 0.4907 | 0.08155 | 3 | 0 | 1.008 | — | 0.3812 | — | 0 | 0.7944 | not_applicable_unseen_timepoints | fit_grid | success | — | — |
| extrapolation | 6302 | 0.1147 | 0.1202 | 0.1752 | 0.5488 | 0.4109 | 0.3304 | 0.282 | 0 | 0.4494 | 0.08162 | 5 | 1 | 1.137 | 0.1368 | — | — | 0 | 0.8008 | not_applicable_unseen_timepoints | fit_grid | success | — | — |
| timepoint:4-5 | 6241 | 0.1055 | 0.1115 | 0.1888 | 0.5992 | 0.3789 | 0.2971 | 0.2793 | 0 | 0.4907 | 0.08155 | 3 | 0 | 1.008 | — | 0.3812 | — | 0 | 0.7944 | not_applicable_unseen_timepoints | fit_grid | success | — | — |
| timepoint:8-9 | 6302 | 0.1147 | 0.1202 | 0.1752 | 0.5488 | 0.4109 | 0.3304 | 0.282 | 0 | 0.4494 | 0.08162 | 5 | 1 | 1.137 | 0.1368 | — | — | 0 | 0.8008 | not_applicable_unseen_timepoints | fit_grid | success | — | — |
| all | 1.254e+04 | 0.09103 | 0.09279 | 0.1444 | 0.4742 | 0.5598 | 0.4063 | 0.31 | 7.973e-05 | 0.6128 | — | 4.005 | 0.5024 | 1.076 | 0.1431 | 0.451 | — | 0 | 1.101 | not_applicable_unseen_timepoints | standard_umap | success | 9.972 | — |
| interpolation | 6241 | 0.08923 | 0.09127 | 0.1506 | 0.4914 | 0.5369 | 0.379 | 0.3191 | 0 | 0.571 | — | 3 | 0 | 1.008 | — | 0.451 | — | 0 | 1.097 | not_applicable_unseen_timepoints | standard_umap | success | 9.972 | — |
| extrapolation | 6302 | 0.09282 | 0.0943 | 0.1383 | 0.4572 | 0.5825 | 0.4333 | 0.3009 | 0.0001587 | 0.6541 | — | 5 | 1 | 1.143 | 0.1431 | — | — | 0 | 1.106 | not_applicable_unseen_timepoints | standard_umap | success | 9.972 | — |
| timepoint:4-5 | 6241 | 0.08923 | 0.09127 | 0.1506 | 0.4914 | 0.5369 | 0.379 | 0.3191 | 0 | 0.571 | — | 3 | 0 | 1.008 | — | 0.451 | — | 0 | 1.097 | not_applicable_unseen_timepoints | standard_umap | success | 9.972 | — |
| timepoint:8-9 | 6302 | 0.09282 | 0.0943 | 0.1383 | 0.4572 | 0.5825 | 0.4333 | 0.3009 | 0.0001587 | 0.6541 | — | 5 | 1 | 1.143 | 0.1431 | — | — | 0 | 1.106 | not_applicable_unseen_timepoints | standard_umap | success | 9.972 | — |
| all | 1.254e+04 | 0.08853 | 0.09087 | 0.1307 | 0.4384 | 0.3713 | 0.1776 | 0.2775 | 0.0002392 | 0.6096 | — | 4.005 | 0.5024 | 1.095 | 0.1797 | 0.3953 | — | 0 | 0.8026 | not_applicable_unseen_timepoints | reduced_repulsion_umap | success | 8.239 | — |
| interpolation | 6241 | 0.08922 | 0.09149 | 0.137 | 0.4566 | 0.3777 | 0.1771 | 0.2798 | 0.0001602 | 0.5697 | — | 3 | 0 | 1.009 | — | 0.3953 | — | 0 | 0.8156 | not_applicable_unseen_timepoints | reduced_repulsion_umap | success | 8.239 | — |
| extrapolation | 6302 | 0.08786 | 0.09025 | 0.1244 | 0.4205 | 0.365 | 0.1781 | 0.2752 | 0.0003174 | 0.649 | — | 5 | 1 | 1.18 | 0.1797 | — | — | 0 | 0.789 | not_applicable_unseen_timepoints | reduced_repulsion_umap | success | 8.239 | — |
| timepoint:4-5 | 6241 | 0.08922 | 0.09149 | 0.137 | 0.4566 | 0.3777 | 0.1771 | 0.2798 | 0.0001602 | 0.5697 | — | 3 | 0 | 1.009 | — | 0.3953 | — | 0 | 0.8156 | not_applicable_unseen_timepoints | reduced_repulsion_umap | success | 8.239 | — |
| timepoint:8-9 | 6302 | 0.08786 | 0.09025 | 0.1244 | 0.4205 | 0.365 | 0.1781 | 0.2752 | 0.0003174 | 0.649 | — | 5 | 1 | 1.18 | 0.1797 | — | — | 0 | 0.789 | not_applicable_unseen_timepoints | reduced_repulsion_umap | success | 8.239 | — |
| all | 1.254e+04 | 0.06536 | 0.06675 | 0.1537 | 0.4841 | 2.896e-06 | 0.2602 | 0.369 | 0 | 0.586 | 0.0002887 | 4.005 | 0.5024 | 1.122 | 0.2327 | 0.4426 | — | 0 | 7.926e-06 | not_applicable_unseen_timepoints | weighted_knn | success | — | — |
| interpolation | 6241 | 0.06965 | 0.07128 | 0.1614 | 0.5016 | 2.876e-06 | 0.2594 | 0.3896 | 0 | 0.5443 | 0.0002887 | 3 | 0 | 1.01 | — | 0.4426 | — | 0 | 8.04e-06 | not_applicable_unseen_timepoints | weighted_knn | success | — | — |
| extrapolation | 6302 | 0.06112 | 0.06226 | 0.1461 | 0.4668 | 2.917e-06 | 0.261 | 0.3486 | 0 | 0.6273 | 0.0002887 | 5 | 1 | 1.233 | 0.2327 | — | — | 0 | 7.766e-06 | not_applicable_unseen_timepoints | weighted_knn | success | — | — |
| timepoint:4-5 | 6241 | 0.06965 | 0.07128 | 0.1614 | 0.5016 | 2.876e-06 | 0.2594 | 0.3896 | 0 | 0.5443 | 0.0002887 | 3 | 0 | 1.01 | — | 0.4426 | — | 0 | 8.04e-06 | not_applicable_unseen_timepoints | weighted_knn | success | — | — |
| timepoint:8-9 | 6302 | 0.06112 | 0.06226 | 0.1461 | 0.4668 | 2.917e-06 | 0.261 | 0.3486 | 0 | 0.6273 | 0.0002887 | 5 | 1 | 1.233 | 0.2327 | — | — | 0 | 7.766e-06 | not_applicable_unseen_timepoints | weighted_knn | success | — | — |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | parametric_umap | unavailable | — | umap.parametric_umap.ParametricUMAP requires the optional 'tensorflow' dependency, not installed in this environment (umap.parametric_umap requires Tensorflow >= 2.0). Install with: pip install tensorflow |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | numap_sep_spectralnet | unavailable | — | The official NUMAP/Sep-SpectralNet package is not installed in this environment (No module named 'numap'). Repository: https://github.com/shaham-lab/NUMAP . Install with: pip install numap==0.2.3 (see pyproject.toml's 'external-baselines' extra). |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | param_repulsor | unavailable | — | The official ParamRepulsor package is not installed in this environment (No module named 'parampacmap'). Repository: https://github.com/hyhuang00/ParamRepulsor . Install with: pip install parampacmap==0.1.0 (requires Python <3.12; see pyproject.toml's 'external-baselines' extra). |
| all | 1.254e+04 | 0.1091 | 0.1148 | 0.1779 | 0.5627 | 0.3587 | 0.297 | 0.2796 | 0 | 0.4685 | 1.323 | 4.005 | 0.5024 | 1.078 | 0.1466 | 0.3785 | — | 0 | 0.7347 | not_applicable_unseen_timepoints | fit_grid_oracle_neighbors | success | — | — |
| interpolation | 6241 | 0.1041 | 0.1106 | 0.1851 | 0.5891 | 0.3437 | 0.2818 | 0.2805 | 0 | 0.4884 | 0.8899 | 3 | 0 | 1.008 | — | 0.3785 | — | 0 | 0.7348 | not_applicable_unseen_timepoints | fit_grid_oracle_neighbors | success | — | — |
| extrapolation | 6302 | 0.114 | 0.119 | 0.1707 | 0.5366 | 0.3736 | 0.312 | 0.2788 | 0 | 0.4489 | 1.752 | 5 | 1 | 1.147 | 0.1466 | — | — | 0 | 0.7337 | not_applicable_unseen_timepoints | fit_grid_oracle_neighbors | success | — | — |
| timepoint:4-5 | 6241 | 0.1041 | 0.1106 | 0.1851 | 0.5891 | 0.3437 | 0.2818 | 0.2805 | 0 | 0.4884 | 0.8899 | 3 | 0 | 1.008 | — | 0.3785 | — | 0 | 0.7348 | not_applicable_unseen_timepoints | fit_grid_oracle_neighbors | success | — | — |
| timepoint:8-9 | 6302 | 0.114 | 0.119 | 0.1707 | 0.5366 | 0.3736 | 0.312 | 0.2788 | 0 | 0.4489 | 1.752 | 5 | 1 | 1.147 | 0.1466 | — | — | 0 | 0.7337 | not_applicable_unseen_timepoints | fit_grid_oracle_neighbors | success | — | — |
| all | 1.254e+04 | 0.1029 | 0.1071 | 0.1776 | 0.5644 | 0.3435 | 0.2643 | 0.2743 | 0 | 0.4658 | 0.8505 | 4.005 | 0.5024 | 1.082 | 0.155 | 0.3645 | — | 0 | 0.7609 | not_applicable_unseen_timepoints | no_repulsion | success | — | — |
| interpolation | 6241 | 0.09982 | 0.1046 | 0.1849 | 0.5913 | 0.3403 | 0.2594 | 0.271 | 0 | 0.4878 | 0.8502 | 3 | 0 | 1.008 | — | 0.3645 | — | 0 | 0.7697 | not_applicable_unseen_timepoints | no_repulsion | success | — | — |
| extrapolation | 6302 | 0.106 | 0.1095 | 0.1703 | 0.5378 | 0.3467 | 0.2691 | 0.2775 | 0 | 0.444 | 0.8508 | 5 | 1 | 1.155 | 0.155 | — | — | 0 | 0.7542 | not_applicable_unseen_timepoints | no_repulsion | success | — | — |
| timepoint:4-5 | 6241 | 0.09982 | 0.1046 | 0.1849 | 0.5913 | 0.3403 | 0.2594 | 0.271 | 0 | 0.4878 | 0.8502 | 3 | 0 | 1.008 | — | 0.3645 | — | 0 | 0.7697 | not_applicable_unseen_timepoints | no_repulsion | success | — | — |
| timepoint:8-9 | 6302 | 0.106 | 0.1095 | 0.1703 | 0.5378 | 0.3467 | 0.2691 | 0.2775 | 0 | 0.444 | 0.8508 | 5 | 1 | 1.155 | 0.155 | — | — | 0 | 0.7542 | not_applicable_unseen_timepoints | no_repulsion | success | — | — |
| all | 50 | 0.112 | 0.1169 | 0.1826 | 0.5784 | 0.3902 | 0.3142 | 0.3255 | 0 | 0.4879 | 4.923 | 4.08 | 0.54 | 1.119 | 0.2123 | 0.4348 | — | 0 | 0.7353 | not_applicable_unseen_timepoints | exact_repulsion_diagnostic | success | — | — |
| interpolation | 23 | 0.08116 | 0.08431 | 0.195 | 0.6141 | 0.4155 | 0.3235 | 0.3657 | 0 | 0.5219 | 4.964 | 3 | 0 | 1.009 | — | 0.4348 | — | 0 | 0.7229 | not_applicable_unseen_timepoints | exact_repulsion_diagnostic | success | — | — |
| extrapolation | 27 | 0.1383 | 0.1446 | 0.1721 | 0.5479 | 0.3686 | 0.3063 | 0.2911 | 0 | 0.4589 | 4.889 | 5 | 1 | 1.212 | 0.2123 | — | — | 0 | 0.6948 | not_applicable_unseen_timepoints | exact_repulsion_diagnostic | success | — | — |
| timepoint:4-5 | 23 | 0.08116 | 0.08431 | 0.195 | 0.6141 | 0.4155 | 0.3235 | 0.3657 | 0 | 0.5219 | 4.964 | 3 | 0 | 1.009 | — | 0.4348 | — | 0 | 0.7229 | not_applicable_unseen_timepoints | exact_repulsion_diagnostic | success | — | — |
| timepoint:8-9 | 27 | 0.1383 | 0.1446 | 0.1721 | 0.5479 | 0.3686 | 0.3063 | 0.2911 | 0 | 0.4589 | 4.889 | 5 | 1 | 1.212 | 0.2123 | — | — | 0 | 0.6948 | not_applicable_unseen_timepoints | exact_repulsion_diagnostic | success | — | — |



外挿時間点が1つだけの場合、horizonに対する単調性は判定不可。



## unavailable / failed

| group | n_queries | recall_at_15 | ndcg | fuzzy_weighted_mse | fuzzy_weighted_bce | local_displacement | repulsion_accumulation_score | density_log_distortion | collapse_rate | global_periphery_percentile | latency_seconds | timepoint_ordinal | horizon | temporal_neighbor_mae | excess_temporal_neighbor_mae | temporal_bracketing_rate | timepoint_label_accuracy | recall_at_15_p05 | local_displacement_p95 | label_accuracy_status | method | status | fit_time_seconds | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | parametric_umap | unavailable | — | umap.parametric_umap.ParametricUMAP requires the optional 'tensorflow' dependency, not installed in this environment (umap.parametric_umap requires Tensorflow >= 2.0). Install with: pip install tensorflow |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | numap_sep_spectralnet | unavailable | — | The official NUMAP/Sep-SpectralNet package is not installed in this environment (No module named 'numap'). Repository: https://github.com/shaham-lab/NUMAP . Install with: pip install numap==0.2.3 (see pyproject.toml's 'external-baselines' extra). |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | param_repulsor | unavailable | — | The official ParamRepulsor package is not installed in this environment (No module named 'parampacmap'). Repository: https://github.com/hyhuang00/ParamRepulsor . Install with: pip install parampacmap==0.1.0 (requires Python <3.12; see pyproject.toml's 'external-baselines' extra). |