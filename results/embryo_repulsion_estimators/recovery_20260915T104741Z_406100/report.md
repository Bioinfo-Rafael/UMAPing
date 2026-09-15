# Embryo repulsion teacher比較



dataset: `embryoid_body`、N=24823、入力次元=50、trajectory shape=[201, 24823, 2]。

設定・凍結入力のSHA-256・計算環境はconfig/manifest.json、元設定はconfig/source_config.yaml、今回の共通学習設定はconfig/training.yamlに保存。

checkpoint数=201、seed=0、device=cuda。

RepulsionField・共通学習設定: {"hidden_dim": 256, "n_residual_blocks": 4, "time_embed_dim": 32, "activation": "gelu", "jitter_sigma": 0.1, "teacher_negative_samples": 64, "weight_by_row_mass": false, "lr": 0.001, "weight_decay": 0.0, "steps": 4000, "batch_size": 512, "log_every": 100}

既存kernelの設定: {"a": 1.5769434602697652, "b": 0.8950608778515733, "epsilon": 0.001, "clip": 4.0, "normalization": "1/N, self included"}

## 推定器単体



| method | status | bias | variance | mse | rmse | cosine | magnitude_error | runtime_seconds_per_query | build_seconds | score_seconds_per_query | ess | max_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform_mc | success | 0.005419 | 0.00221 | 0.002252 | 0.04745 | 0.4342 | 0.02642 | 1.002e-05 | 0.006459 | 0 | — | — |
| dual_raw_is | success | 0.00922 | 0.01092 | 0.01116 | 0.1056 | 0.4894 | 0.02666 | 7.89e-05 | 0.03572 | 3.135e-05 | 7.198 | 1096 |
| dual_hub_is | success | 0.008636 | 0.0103 | 0.0105 | 0.1025 | 0.5039 | 0.02597 | 5.977e-05 | 0.02168 | 2.742e-05 | 7.134 | 2442 |
| dual_topl | success | 0.007749 | 0.004261 | 0.004345 | 0.06592 | 0.421 | 0.03501 | 9.158e-05 | 0.01823 | 1.79e-05 | — | — |
| barnes_hut_0.2 | success | 0.004478 | 0 | 5.267e-05 | 0.007258 | 0.9014 | 0.003327 | 0.01039 | 22.53 | 0 | — | — |
| barnes_hut_0.5 | success | 0.004539 | 0 | 5.323e-05 | 0.007296 | 0.902 | 0.003422 | 0.002837 | 13.74 | 0 | — | — |
| barnes_hut_0.8 | success | 0.004673 | 0 | 5.424e-05 | 0.007365 | 0.9034 | 0.003601 | 0.001604 | 23.11 | 0 | — | — |
| fit_grid_64 | success | 0.006484 | 0 | 8.712e-05 | 0.009334 | 0.8749 | 0.005019 | 1.54e-05 | 0.6382 | 0 | — | — |
| fit_grid_128 | success | 0.00511 | 0 | 6.016e-05 | 0.007756 | 0.8981 | 0.00388 | 1.179e-05 | 1.502 | 0 | — | — |
| fit_grid_256 | success | 0.00462 | 0 | 5.365e-05 | 0.007325 | 0.9019 | 0.003451 | 1.187e-05 | 3.768 | 0 | — | — |



biasは有限反復の平均による推定で、真のbiasとは異なる。bias_squared_debiasedは反復分散補正値（負値も保持）。決定論的手法のvariance=0は反復乱数がないことだけを意味する。

## Learned B_phi



| method | status | rmse | cosine | magnitude_error | best_validation_rmse |
| --- | --- | --- | --- | --- | --- |
| uniform_mc | success | 0.0121 | 0.7175 | 0.006353 | 0.01159 |
| dual_raw_is | success | 0.01351 | 0.6861 | 0.006622 | 0.01321 |
| dual_hub_is | success | 0.01378 | 0.6553 | 0.006431 | 0.01335 |
| dual_topl | success | 0.01255 | 0.7163 | 0.006306 | 0.01247 |
| barnes_hut | success | 0.01023 | 0.7983 | 0.005377 | 0.009884 |
| fit_grid | success | 0.01019 | 0.8083 | 0.005458 | 0.009849 |



学習安定性・収束・費用:



| method | train_seconds | rolling_std_mean | late_loss_cv | first_step_half_initial_rmse | seconds_to_half_initial_rmse |
| --- | --- | --- | --- | --- | --- |
| uniform_mc | 43.06 | 0.0001139 | 0.08331 | — | — |
| dual_raw_is | 177.6 | 0.0198 | 3.105 | — | — |
| dual_hub_is | 179.9 | 0.02142 | 2.271 | — | — |
| dual_topl | 271.6 | 0.0002467 | 0.1078 | — | — |
| barnes_hut | 4792 | 3.307e-05 | 0.7208 | 3600 | 4507 |
| fit_grid | 36.48 | 2.998e-05 | 0.8999 | 3800 | 34.68 |



同一モデル初期値、stepごとに再現可能なanchor・連続t・Gaussian jitter、同一optimizer/settingsを使用。exact oracleは評価とteacher variant選択だけに使用。学習targetには使用していない。

完了成果物を再学習せず引き継いだ手法: uniform_mc, dual_raw_is, dual_hub_is, dual_topl。個別のreused_from / resumed_from_stepは学習結果CSVに記録。新しい学習時間はcheckpoint I/Oを除外し、checkpoint_secondsへ別記。

lossは既存どおり座標平均MSE（row-mass有効時は元の重み付き和）。報告のexact MSEはベクトル誤差ノルム二乗の平均であり、2Dでは座標平均MSEの2倍。

## 解釈



1. Uniform-64: RMSE=0.047453、variance=0.00220999。exact場のRMS=0.0199464、RMSE/信号RMS=2.37903（1なら誤差が信号RMSと同規模）。

2. dual_raw_is: variance=0.010923、uniform_mcとの差=0.00871301。RMSE=0.10565。

3. dual_hub_is: variance=0.0102976、dual_raw_isとの差=-0.000625378。RMSE=0.102466。

4. dual_topl: variance=0.00426102、uniform_mcとの差=0.00205103。RMSE=0.0659156。

Top-L−dual_raw_is: variance差=-0.00666198、RMSE差=-0.0397342。名目64-forceは同じだがscore費用も含む秒/queryは9.1576e-05 / 7.89044e-05。

Top-L−dual_hub_is: variance差=-0.0060366、RMSE差=-0.0365508。名目64-forceは同じだがscore費用も含む秒/queryは9.1576e-05 / 5.97667e-05。

5. barnes_hut: 選択=barnes_hut_0.8、RMSE=0.00736472、秒/query=0.00160357。

Uniform比のruntime=159.99倍。構築費用は別途表に記録し、同一64-force費用とは見なさない。

6. fit_grid: 選択=fit_grid_256、RMSE=0.00732489、秒/query=1.18665e-05。

Uniform比のruntime=1.18394倍。構築費用は別途表に記録し、同一64-force費用とは見なさない。

7. 同じvalidation上の最小final RMSEはfit_grid: 0.0101933。単一seedの比較であり一般的優越性は未検証。

8. 学習時間と最終RMSEのPareto候補: fit_grid。次の全pipeline候補は最小RMSEの手法だが、OOS性能改善はこのfield評価だけでは確定しない。



## 制約・異常

BH/gridはcheckpoint間の場の線形補間。厳密oracleは点位置を補間してから非線形forceを計算するため、theta→0でも連続tの時間近似誤差は残る。

BHは近接cellを再帰し、遠方cellでは質量×centroidのclip済みforce。個別点clipの完全再現は葉だけ。GridはCIC・線形FFT・双線形補間による近似で、範囲外queryは明示的な失敗としwrapしない。

theta/gridの選択とネットワーク評価に同じ固定validationを使う探索実験。独立test、複数学習seed、OOS embedding評価は今後の課題。

モデルforward時間は同一architectureのbatch計測であり、query単発latencyとは異なる。teacher初期構築、Dual Q/K cache、hubness構築は別計測。

Low ESS/巨大weightが有限なら主比較に残す。support underflow、NaN、範囲外queryは失敗として記録する。

- dual_raw_is: low ESS / weight explosion: ESS=7.197679954493236, max=1096.196145115137

- dual_hub_is: low ESS / weight explosion: ESS=7.133952887954814, max=2441.641400291404
