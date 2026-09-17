# 完了済みseedの最終集計

採用seed: [0, 1]。再学習・推論は実行せず、保存済みの完了結果だけを集計した。

元の3 seed実行の成功とは区別する。元manifestや中断seedの成果物は変更していない。

## seed間の平均・標準偏差

| method | group | recall_at_15_mean | recall_at_15_std | recall_at_15_count | temporal_neighbor_mae_mean | temporal_neighbor_mae_std | temporal_neighbor_mae_count | excess_temporal_neighbor_mae_mean | excess_temporal_neighbor_mae_std | excess_temporal_neighbor_mae_count | temporal_bracketing_rate_mean | temporal_bracketing_rate_std | temporal_bracketing_rate_count | density_log_distortion_mean | density_log_distortion_std | density_log_distortion_count | global_periphery_percentile_mean | global_periphery_percentile_std | global_periphery_percentile_count | collapse_rate_mean | collapse_rate_std | collapse_rate_count | latency_seconds_mean | latency_seconds_std | latency_seconds_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fit_grid | all | 0.1099 | 0.0003195 | 2 | 1.073 | 0.0004848 | 2 | 0.1376 | 0.001077 | 2 | 0.3806 | 0.0007931 | 2 | 0.2793 | 0.001897 | 2 | 0.4698 | 0.0002512 | 2 | 0 | 0 | 2 | 0.607 | 0.7431 | 2 |
| fit_grid | extrapolation | 0.1147 | 1.496e-05 | 2 | 1.138 | 0.001077 | 2 | 0.1376 | 0.001077 | 2 | — | — | 0 | 0.2806 | 0.001923 | 2 | 0.4494 | 0.0001072 | 2 | 0 | 0 | 2 | 0.7489 | 0.9436 | 2 |
| fit_grid | interpolation | 0.105 | 0.0006269 | 2 | 1.008 | 0.0001133 | 2 | — | — | 0 | 0.3806 | 0.0007931 | 2 | 0.278 | 0.00187 | 2 | 0.4904 | 0.0003967 | 2 | 0 | 0 | 2 | 0.4638 | 0.5405 | 2 |
| fit_grid | timepoint:4-5 | 0.105 | 0.0006269 | 2 | 1.008 | 0.0001133 | 2 | — | — | 0 | 0.3806 | 0.0007931 | 2 | 0.278 | 0.00187 | 2 | 0.4904 | 0.0003967 | 2 | 0 | 0 | 2 | 0.4638 | 0.5405 | 2 |
| fit_grid | timepoint:8-9 | 0.1147 | 1.496e-05 | 2 | 1.138 | 0.001077 | 2 | 0.1376 | 0.001077 | 2 | — | — | 0 | 0.2806 | 0.001923 | 2 | 0.4494 | 0.0001072 | 2 | 0 | 0 | 2 | 0.7489 | 0.9436 | 2 |
| uniform_mc | all | 0.1092 | 0.001056 | 2 | 1.073 | 0.001252 | 2 | 0.1366 | 0.002094 | 2 | 0.3747 | 0.006458 | 2 | 0.28 | 0.0002399 | 2 | 0.468 | 4.181e-05 | 2 | 0 | 0 | 2 | 0.8412 | 1.074 | 2 |
| uniform_mc | extrapolation | 0.1141 | 0.001496 | 2 | 1.137 | 0.002094 | 2 | 0.1366 | 0.002094 | 2 | — | — | 0 | 0.284 | 8.17e-05 | 2 | 0.4468 | 0.0001777 | 2 | 0 | 0 | 2 | 0.8271 | 1.055 | 2 |
| uniform_mc | interpolation | 0.1042 | 0.0006118 | 2 | 1.008 | 0.0004003 | 2 | — | — | 0 | 0.3747 | 0.006458 | 2 | 0.276 | 0.0003996 | 2 | 0.4894 | 0.0002634 | 2 | 0 | 0 | 2 | 0.8553 | 1.094 | 2 |
| uniform_mc | timepoint:4-5 | 0.1042 | 0.0006118 | 2 | 1.008 | 0.0004003 | 2 | — | — | 0 | 0.3747 | 0.006458 | 2 | 0.276 | 0.0003996 | 2 | 0.4894 | 0.0002634 | 2 | 0 | 0 | 2 | 0.8553 | 1.094 | 2 |
| uniform_mc | timepoint:8-9 | 0.1141 | 0.001496 | 2 | 1.137 | 0.002094 | 2 | 0.1366 | 0.002094 | 2 | — | — | 0 | 0.284 | 8.17e-05 | 2 | 0.4468 | 0.0001777 | 2 | 0 | 0 | 2 | 0.8271 | 1.055 | 2 |

標準偏差は標本標準偏差（ddof=1）。2 seedでは再現性の根拠は限定的。

## FitGrid−UniformのRecall差

| group | mean | std | count |
| --- | --- | --- | --- |
| all | 0.0007122 | 0.0007366 | 2 |
| extrapolation | 0.0006136 | 0.001481 | 2 |
| interpolation | 0.0008118 | 1.511e-05 | 2 |
| timepoint:4-5 | 0.0008118 | 1.511e-05 | 2 |
| timepoint:8-9 | 0.0006136 | 0.001481 | 2 |

同じqueryをseed間で独立な細胞として水増ししない。seed別のcell bootstrap CIは次表に保持し、CI端点を平均して新しいCIとはしない。

## seed別対応比較

| status | n_pairs | n_dropped | ours_mean | baseline_mean | mean_improvement | median_improvement | ci_low | ci_high | fraction_ours_better | fraction_tied | n_bootstrap | seed | direction | group | fraction_worsened | paired_effect_size_dz | conditional_sign_test_p | test_assumption | inference_scope | conditional_sign_test_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | 12543 | 0 | 0.1101 | 0.1099 | 0.0001913 | 0 | -0.0005315 | 0.0009142 | 0.1371 | 0.7275 | 10000 | 0 | higher | all | 0.1354 | 0.004699 | 0.7195 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6241 | 0 | 0.1055 | 0.1047 | 0.0008225 | 0 | -0.000203 | 0.001827 | 0.138 | 0.7319 | 10000 | 0 | higher | interpolation | 0.1301 | 0.01984 | 0.2406 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6302 | 0 | 0.1147 | 0.1151 | -0.0004337 | 0 | -0.001375 | 0.0005818 | 0.1363 | 0.7231 | 10000 | 0 | higher | extrapolation | 0.1406 | -0.01085 | 0.5337 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6241 | 0 | 0.1055 | 0.1047 | 0.0008225 | 0 | -0.000203 | 0.001827 | 0.138 | 0.7319 | 10000 | 0 | higher | timepoint:4-5 | 0.1301 | 0.01984 | 0.2406 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 6302 | 0 | 0.1147 | 0.1151 | -0.0004337 | 0 | -0.001375 | 0.0005818 | 0.1363 | 0.7231 | 10000 | 0 | higher | timepoint:8-9 | 0.1406 | -0.01085 | 0.5337 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 1 |
| ok | 12543 | 0 | 0.1097 | 0.1084 | 0.001233 | 0 | 0.0005528 | 0.001945 | 0.1398 | 0.7354 | 10000 | 1 | higher | all | 0.1249 | 0.031 | 0.001241 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.006204 |
| ok | 6241 | 0 | 0.1046 | 0.1038 | 0.0008012 | 0 | -0.0001709 | 0.001795 | 0.1319 | 0.7464 | 10000 | 1 | higher | interpolation | 0.1218 | 0.02039 | 0.1191 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.2383 |
| ok | 6302 | 0 | 0.1147 | 0.113 | 0.001661 | 0 | 0.0006876 | 0.002655 | 0.1476 | 0.7245 | 10000 | 1 | higher | extrapolation | 0.1279 | 0.04126 | 0.003145 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.01258 |
| ok | 6241 | 0 | 0.1046 | 0.1038 | 0.0008012 | 0 | -0.0001709 | 0.001795 | 0.1319 | 0.7464 | 10000 | 1 | higher | timepoint:4-5 | 0.1218 | 0.02039 | 0.1191 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.2383 |
| ok | 6302 | 0 | 0.1147 | 0.113 | 0.001661 | 0 | 0.0006876 | 0.002655 | 0.1476 | 0.7245 | 10000 | 1 | higher | timepoint:8-9 | 0.1279 | 0.04126 | 0.003145 | conditional iid cells/non-tied signs; donor/time dependence not modeled | fixed trained models and reference; cell bootstrap, not biological replication | 0.01258 |

CIが0を含む場合、そのseedの改善は不確定。cell間依存や生物学的反復を補正したCIではない。

## 学習時間

| seed | method | build_seconds | train_seconds |
| --- | --- | --- | --- |
| 0 | uniform_mc | 0.003079 | 35.99 |
| 0 | fit_grid | 3.881 | 36.46 |
| 1 | uniform_mc | 0.004473 | 125 |
| 1 | fit_grid | 3.614 | 133.9 |

## 除外seed

| seed | stored_status | disposition | note |
| --- | --- | --- | --- |
| 2 | running | excluded_by_selection | stored running is not a live process check; may be stale after termination |

## 時間split

実時間点: ['0-1', '2-3', '4-5', '6-7', '8-9']

補間: 4-5 / 外挿: ['8-9']
