# FitGrid最終埋め込み比較



元runは読み取り専用。全手法は同じPreparedDatasetを使用。参照軌道生成は変更していない。

主指標はquery→固定referenceのRecall@15。reference trustworthinessは同一参照の整合性確認のみ。

通常評価のfield.jsonは従来のMC teacher診断。主たる場の評価はexact_field/validation.npzの全点和1000点。

benchmarkのUniformモデルとmainのUniformモデルは別学習であり、混同しない。以下は今回実際に読み込んだモデルの比較。



| method | rmse | cosine | magnitude_error | recall_at_15 | latency_seconds |
| --- | --- | --- | --- | --- | --- |
| uniform_mc | 0.01204 | 0.7299 | 0.005964 | 0.09059 | 0.2135 |
| fit_grid | 0.01068 | 0.7782 | 0.005763 | 0.09104 | 0.3134 |



## 最終埋め込み（全query）

| group | n_queries | recall_at_15 | ndcg | fuzzy_weighted_mse | fuzzy_weighted_bce | local_displacement | repulsion_accumulation_score | density_log_distortion | collapse_rate | global_periphery_percentile | latency_seconds | recall_at_15_p05 | local_displacement_p95 | label_accuracy_status | method | status | fit_time_seconds | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 6206 | 0.09059 | 0.09459 | 0.1592 | 0.5061 | 0.3503 | 0.2911 | 0.2657 | 0 | 0.501 | 0.2135 | 0 | 0.7512 | not_requested | uniform_mc | success | — | — |
| all | 6206 | 0.09104 | 0.09503 | 0.1596 | 0.5069 | 0.3624 | 0.3036 | 0.2657 | 0 | 0.5018 | 0.3134 | 0 | 0.7695 | not_requested | fit_grid | success | — | — |
| all | 6206 | 0.07738 | 0.07987 | 0.11 | 0.3724 | 0.496 | 0.3454 | 0.2465 | 0 | 0.4965 | — | 0 | 1.032 | not_requested | standard_umap | success | 23.83 | — |
| all | 6206 | 0.07556 | 0.07781 | 0.103 | 0.3522 | 0.3182 | 0.1486 | 0.2416 | 0 | 0.4947 | — | 0 | 0.6884 | not_requested | reduced_repulsion_umap | success | 18.77 | — |
| all | 6206 | 0.06084 | 0.06164 | 0.118 | 0.3832 | 2.417e-06 | 0.2411 | 0.308 | 0 | 0.4842 | 0.0003931 | 0 | 8.135e-06 | not_requested | weighted_knn | success | — | — |
| all | 6206 | 0.09039 | 0.09463 | 0.1561 | 0.498 | 0.3263 | 0.2853 | 0.2604 | 0 | 0.5015 | 0.3091 | 0 | 0.7044 | not_requested | fit_grid_oracle_neighbors | success | — | — |
| all | 6206 | 0.08673 | 0.09037 | 0.157 | 0.5014 | 0.3122 | 0.2614 | 0.2622 | 0 | 0.5 | 0.1284 | 0 | 0.7138 | not_requested | no_repulsion | success | — | — |
| all | 50 | 0.1013 | 0.1086 | 0.1247 | 0.4063 | 0.418 | 0.354 | 0.1788 | 0 | 0.5683 | 0.7649 | 0 | 0.8797 | not_requested | exact_repulsion_diagnostic | success | — | — |



## 対応比較

| status | n_pairs | n_dropped | ours_mean | baseline_mean | mean_improvement | median_improvement | ci_low | ci_high | fraction_ours_better | fraction_tied | n_bootstrap | seed | direction | group | fraction_worsened | paired_effect_size_dz | conditional_sign_test_p | test_assumption | inference_scope | conditional_sign_test_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | 6206 | 0 | 0.09104 | 0.09059 | 0.0004512 | 0 | -0.0004941 | 0.001407 | 0.1242 | 0.7538 | 10000 | 0 | higher | all | 0.122 | 0.01193 | 0.7395 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.7395 |

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



## unavailable / failed

| group | n_queries | recall_at_15 | ndcg | fuzzy_weighted_mse | fuzzy_weighted_bce | local_displacement | repulsion_accumulation_score | density_log_distortion | collapse_rate | global_periphery_percentile | latency_seconds | recall_at_15_p05 | local_displacement_p95 | label_accuracy_status | method | status | fit_time_seconds | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | parametric_umap | unavailable | — | umap.parametric_umap.ParametricUMAP requires the optional 'tensorflow' dependency, not installed in this environment (umap.parametric_umap requires Tensorflow >= 2.0). Install with: pip install tensorflow |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | numap_sep_spectralnet | unavailable | — | The official NUMAP/Sep-SpectralNet package is not installed in this environment (No module named 'numap'). Repository: https://github.com/shaham-lab/NUMAP . Install with: pip install numap==0.2.3 (see pyproject.toml's 'external-baselines' extra). |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | param_repulsor | unavailable | — | The official ParamRepulsor package is not installed in this environment (No module named 'parampacmap'). Repository: https://github.com/hyhuang00/ParamRepulsor . Install with: pip install parampacmap==0.1.0 (requires Python <3.12; see pyproject.toml's 'external-baselines' extra). |