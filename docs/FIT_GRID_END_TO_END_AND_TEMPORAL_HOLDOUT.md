# FitGridの最終埋め込み比較と可変時間点holdout

## 今回の到達点（2026-09-15）

`experiments/embryo-repulsion-estimator-benchmark` の最新保存結果 `7274702` を起点に、
`experiments/fit-grid-end-to-end-temporal-holdout` で実装した。mainへのmergeはしない。
既存run・図・checkpoint・設定・rawデータは変更していない。

**実データの最終埋め込み比較とtemporal学習は未実行。** ローカルには保存済みの図・CSV・FitGrid最終checkpointがあるが、
元mainの上流バイナリとraw/PreparedDatasetがない。移植コマンドを実行し、不足ファイルで停止することを確認した。
`runs/embryoid_body/fit_grid` やtemporal runを成功したように作成していない。

[ローカル監査JSON](fit_grid_audit/local_inventory.json)、[観測ラベルCSV](fit_grid_audit/observed_query_timepoints.csv)、
[検証済みcomponent値](fit_grid_audit/verified_component_metrics.csv) に根拠と入力SHA256を保存した。
合成データのテスト結果は実データの実験結果として扱わない。

## 保存済みcomponent結果の確認

元データは `results/embryo_repulsion_estimators/recovery_20260915T104741Z_406100/metrics/` の
`estimator_metrics.csv` と `trained_fields.csv`。checkpoint内のteacherは `fit_grid_256`、grid size 256だった。

| 指標 | Uniform-64 | FitGrid-256 |
| --- | ---: | ---: |
| teacherのexact-field RMSE | 0.047453 | 0.007325 |
| 学習したB_phiのexact-field RMSE | 0.012099 | 0.010193 |
| B_phi cosine | 0.717508 | 0.808279 |
| B_phi magnitude error | 0.006353 | 0.005458 |
| B_phi学習秒（teacher構築を除く） | 43.060742 | 36.481062 |
| teacher構築秒 | 0.370449 | 4.838813 |
| 今回の最終OOS比較 | 未実行 | 未実行 |

これは同じ凍結軌道のcomponent評価であり、最終埋め込み改善や生物学的一般化の証拠ではない。
benchmark内のUniformモデルと `main/checkpoints/repulsion_field.pt` は別学習である。
新しい比較では**実際のmainモデルとFitGridモデル**を共通の1000点・全reference和で再評価する。
通常pipelineの `field.json` は従来のMC teacher診断なので、exact評価の代わりには使わない。

manifest上の参照細胞数は24,823、入力次元50、軌道shapeは `(201, 24823, 2)`。
実バイナリがないためshapeはmanifestからの記録値であり、今回再ロードして検証した値ではない。
RepulsionFieldは2D入出力、hidden 256、residual block 4、time embedding 32、GELU。
設定は4000 steps、batch 512、Adam lr 0.001、jitter 0.1、clip 4、seed 0。

## 実データの時間点監査

`runs/embryoid_body/main/metrics/advanced_per_query.csv` の `ours_full` を読み、queryを重複なく数えた。
**下表は既存easy splitのquery側6206細胞だけ。全raw細胞の群別数ではない。**
元grouping-column名はこのCSVの汎用 `label` 列からは復元できず、未確認。

| 実ラベル（数値窓の年代順） | 既存query細胞数 | 全raw細胞数 |
| --- | ---: | --- |
| 0-1 | 915 | 未確認 |
| 2-3 | 1474 | 未確認 |
| 4-5 | 1248 | 未確認 |
| 6-7 | 1309 | 未確認 |
| 8-9 | 1260 | 未確認 |

観測した一意ラベル数は5。新しい指示に従い、**10点を必須としない**。
ただし実験前にはrawまたはPreparedDataset全体を監査し、列名・全ラベル・全細胞数・年代順・Tを印字する。
監査用には既存PreparedDatasetを使用できるが、temporal前処理を既存easy-split PCA特徴から作り直すことはしない。
学習には注釈付きraw countsが必要である。

ローカルで `Git/`、`Downloads/`、`Documents/` の既存データを探索したが、対象のraw/cachedデータは見つからなかった。
`runs/mock/main` やscanpy同梱の別データは代用していない。自動downloadも行っていない。

## 可変時間点の選択規則

年代順の実ラベルを `τ1,...,τT` とし、T>=5が必要。
`m = min(max(floor(0.6*T), 2), T-2)` とする。

- reference: `τ1,...,τ(m-1),τ(m+1)` の全細胞。
- interpolation: `τm` の全細胞。referenceの `τ(m-1)` と `τ(m+1)` が両側を挟む。
- extrapolation: `τ(m+2),...,τT` の全細胞。
- 最終観測は `τ(m+1)`。外挿horizonは年代順位置差の `1,...,T-m-1`。

数値、`day_0`、`E8.5`、非重複数値窓 `0-1`、ISO日時は数値的に整列する。
ラベル順が曖昧・数値が重複・時間窓が重複する場合は、実ラベルを一度ずつ含む明示的 `--group-order` が必要。
文字列のアルファベット順を年代順として採用しない。時間列候補が複数ある場合も `--group-column` が必要。

全データでも上の5ラベルだけと確認できれば、次のsplitになる。

| 役割 | 実ラベル | horizon |
| --- | --- | --- |
| reference | 0-1, 2-3, 6-7 | — |
| interpolation | 4-5 | 0 |
| extrapolation | 8-9 | 1 |

この5点の場合、外挿horizonは1個なので「誤差がhorizonに対して単調増加するか」は判定不可。
実際に10時間点なら元の例どおりordinal 1,2,3,4,5,7がreferenceとなるが、評価コードにはそれらの固定値を入れていない。
CSV・図・報告には実ラベルを表示し、ordinalとhorizonは別の数値列に保存する。

`temporal_split.json` には `ordered_timepoints`, `interpolation_timepoint`, `last_observed_timepoint`,
`extrapolation_timepoints`, `reference_timepoints`, `query_timepoints`, `cells_per_timepoint`, `selection_rule` と
全cell ID・raw行index・ordinal・列名を保存する。重複cell IDとreference/queryの重なりを拒否する。

## 成果物の再利用と保護

| 成果物 | 既存split FitGrid | temporal FitGrid |
| --- | --- | --- |
| PreparedDataset・split | mainから完全コピー | temporal Uniformから完全コピー |
| reference features・Mu/W graph | 同上 | 同上 |
| Retriever・keys | 同上 | 同上 |
| Spectral・calibration・Y(0) | 同上 | 同上 |
| reference_trajectory.npz | 同上、再生成なし | Uniformで一度だけ生成し完全コピー |
| repulsion_field.pt | 保存済みFitGrid-256に交換 | matched FitGrid-256学習結果に交換 |
| 旧評価値・旧図 | コピーしない | コピーしない |

移植前にbenchmark manifestのconfig・features・trajectory・Retriever・keys・PreparedDatasetのSHA256を検証する。
remote絶対pathはmanifestのfrozen_runを基準に相対化して現在のsourceへ対応付ける。
追加のgraph/Spectral/calibration/Y(0)も現在のsourceからSHA256を取り、コピー前後の不変性を検証する。
これら追加ファイルは元benchmark manifestにhashがないため、過去の状態まで証明できるとは主張しない。
checkpointのteacher選択・hparams・state_dict shape・有限性・force設定・学習設定も検証する。
losses.csvは全stepの並びと有限性を確認し、元の各lossを `metrics/repulsion_training.json` にそのまま保存する。
CSV原本とhashはprovenanceに残す。

既存output・source/outputの祖先/子孫重複を拒否する。検証失敗時には出力runを作らない。
シンボリックリンクは使わず独立コピーし、コピー内容も検証する。
評価時も元runを変更せず、新しいcomparison配下の `uniform_mc_evaluation/`, `fit_grid_evaluation/` に通常suite用コピーを作る。
従って既存main/FitGridに保存済みの図・評価値があっても上書きしない。

## 実行コマンド

### A: 実上流ファイルがあるマシン（元のremote）

```bash
cd /home/suzuki/Learn/UMAPing
git fetch origin
git switch experiments/fit-grid-end-to-end-temporal-holdout
git pull --ff-only

.venv/bin/python -u -m umaping.fit_grid_experiment promote \
  --source runs/embryoid_body/main \
  --benchmark results/embryo_repulsion_estimators/recovery_20260915T104741Z_406100 \
  --output runs/embryoid_body/fit_grid

.venv/bin/python -u -m umaping.fit_grid_experiment compare \
  --uniform runs/embryoid_body/main \
  --fit-grid runs/embryoid_body/fit_grid \
  --output runs/embryoid_body/comparisons/fit_grid_vs_main \
  --standard-suite --device cuda
```

既存outputがある場合は上書きせず停止する。新しい名前を指定する。
`compare` は学習を行わない。`--standard-suite` は通常evaluate・analyze・analyze-advancedを新規コピーで実行する。
baselineは既存adapterを再利用して新たに評価し、保存済みの別実行値を混ぜない。
標準UMAP・reduced repulsion・weighted kNN、任意依存の3手法、FitGrid oracle-neighbor、no-repulsion、exact diagnosticを含む。
exact diagnosticは同一seedの固定query subsetで、IDを保存する。B_phiに依存しないため二モデルで重複実行しない。
依存が未インストールなら理由付きunavailable。失敗はfailedとして残す。追加インストールは行わない。

### B: まず全データを監査

```bash
.venv/bin/python -m umaping.fit_grid_experiment audit \
  --data runs/embryoid_body/main/cache/prepared_dataset.npz

rg --files --hidden --no-ignore data/raw/embryoid_body
```

時間列が一意でなければ、上のエラーが列一覧を表示する。実時間列を `--group-column` に指定する。
rawの実ファイル名はこのローカルにはなく不明。以下の `RAW_FILE` は上で見つかった対象ファイルを設定する。
別データをダウンロードしたり、名前だけ合わせたりしない。

```bash
RAW_FILE='/実在するEmbryoidの注釈付きrawファイル.h5ad'
.venv/bin/python -m umaping.fit_grid_experiment audit --data "$RAW_FILE"

.venv/bin/python -u -m umaping.fit_grid_experiment temporal \
  --data "$RAW_FILE" \
  --config experiments/configs/embryoid_body.yaml \
  --run-root runs/embryoid_body --seeds 0 1 2 --device cuda
```

曖昧な順序では `--group-order '最初の実ラベル' '次の実ラベル' ...` を監査と実験の両方へ指定する。
別のcell-type列があれば `--cell-type-column 実列名` も指定できる。
元のeasy-split configは変更せず、実ラベルと実入力次元を解決した専用configを新run内に保存する。
T<5ならtemporal trainingを開始しない。

## 学習と評価の公平性

HVG/PCAは既存 `_scrna_common.prepare_continuous_scrna_dataset` を再利用してreferenceのみにfitする。
queryのcell単位normalize/log以外の統計をreferenceへ混入させない。HVG名・PCA mean/componentsも保存する。
上流はseed 0で一度だけ学習。Uniform/FitGridのB_phiはseedごとに同一初期stateを複製し、
`TrainingQueries(seed+2000, step)` でanchor/time/jitter列を完全に一致させる。teacher乱数は別stream。
architecture・optimizer・batch・lr・steps・jitter・force・clipは共通。Uniformは64サンプル、FitGridは256格子。
参照軌道を作る関数やg_minusは変更していない。

seed 0で全query・全baseline・通常suiteを実行して数値的に通った後、seed 1,2ではB_phiだけ再学習し、全queryを再評価する。
追加seedでもqueryを間引かない。B_phi以外に依存しないbaselineをseedごとに再学習しない。
`--seeds 0` を明示すれば1seedに限定できるが、その場合seed間一般化は未検証である。
途中失敗はmanifestに記録して残し、既存出力への再実行は拒否する。

主指標はRecall@15。FitGrid−Uniformのcellごとの平均・中央値・改善/同点/悪化率・bootstrap95%CI・paired dzを保存する。
CIは固定参照/モデルに条件付けたcell bootstrap 10,000回。
独立な非同点queryの符号を仮定した符号検定を参考値としてHolm補正付きで記録するが、donor/細胞間依存の成立を断定しない。
CIが0を含む場合は改善を結論しない。複数seedの平均/標準偏差は別表に出す。

時間評価は全query・補間・全外挿・各実ラベル。temporal-neighbor MAEは15近傍referenceのordinal差の平均。
補間のbracketing rateは `τ(m-1)` と `τ(m+1)` の両方を15近傍に含む割合。
外挿excess MAEはMAEから不可避horizonを引く。時間間隔が不均一でもこの誤差を「日数」と解釈しない。
時間点kNN分類精度は未観測classなのでN/A。別cell-type annotationがあれば別列で評価する。

局所密度歪み・global periphery・collapse率・local displacementのtail・NDCGも保存する。
密度歪みは各空間内でreference近傍半径により正規化したquery半径のlog比差。
collapseは正規化2D半径<0.1、peripheryは全reference重心からの半径百分位であり、操作的な診断量。
時間labelが未観測なので、存在しない同label referenceを使うclass-conditional peripheryを主指標にしない。
query時間がadapterから取得できないbaselineはN/Aとし、fit込み時間を混ぜない。

## 出力

- A: `runs/embryoid_body/fit_grid`、`runs/embryoid_body/comparisons/fit_grid_vs_main/`。
- B seed 0: `runs/embryoid_body/temporal_holdout_uniform`、`temporal_holdout_fit_grid`、`comparisons/temporal_holdout/`。
- 追加seed: `temporal_seed_1_uniform` / `temporal_seed_1_fit_grid` 等と `comparisons/temporal_seed_1/` 等。
- 各comparison: `aggregate.csv`, `paired_per_query.csv`, `bootstrap.csv/json`, `accuracy_runtime.csv`, `component_vs_end_to_end.csv`。
- 保存座標: `embeddings/*.npz`（元query index付き）。exact場: `exact_field/validation.npz`, `metrics.csv`, 各モデルの予測。
- temporal: `temporal_horizon.csv`（実ラベル+horizon）、`horizon_monotonicity.json`、`matched_seed_results.csv`, `matched_seed_summary.csv`。
- 基本6図: recall比較、paired差分布、改善割合、密度/周辺、accuracy/latency、component対最終結果の表。
- temporal追加図: 実ラベルをx軸としたrecall/MAE/excess MAE。報告は `report.md`（日本語）。

## 既存mainの参考値と今回の未測定項目

以下は既存 `main/metrics/baseline_comparison.csv` の保存値。今回再実行した比較ではなく、FitGridを加えた対応検定にも使っていない。

| 手法 | 既存Recall@15 | 今回のFitGrid比較 |
| --- | ---: | --- |
| Uniform main | 0.090590 | 未実行 |
| FitGrid | — | 未実行 |
| standard UMAP | 0.077377 | 未実行 |
| reduced repulsion | 0.075561 | 未実行 |
| weighted kNN | 0.060844 | 未実行 |
| oracle neighbors | 0.089569 | 未実行 |
| no repulsion | 0.086733 | 未実行 |
| exact diagnostic（50queryのみ） | 0.101333 | 未実行、全queryとの順位比較不可 |
| Parametric/NUMAP/ParamRepulsor | 元mainではunavailable | 今回未実行 |

時間点 `4-5` / `8-9` のtemporal結果、paired CI、複数seed実結果はすべて未測定。
旧例の6/8/9/10を実ラベルとして捏造しない。

## 不足ファイルと次の作業

現在のローカルにないものは `runs/embryoid_body/main/` の:

```text
cache/prepared_dataset.npz
cache/reference_spectral_embedding.npy
memory/reference_features.npy
memory/reference_trajectory.npz
memory/retriever_keys.npy
memory/graph_directed.npz
memory/graph_symmetric.npz
memory/spectral_calibration.npz
checkpoints/retriever.pt
checkpoints/spectral_encoder.pt
checkpoints/repulsion_field.pt
```

さらに注釈付きrawファイル自体が `data/raw/embryoid_body/` にない。
main configのSHA256はbenchmark manifestと一致した。残りのhash一致はファイル不在で未検証。
従って次は元remoteでAの移植・評価を実行し、全データ監査後にBを実行する。
FitGridはcomponentの有望候補だが、**現段階で最終pipelineの既定teacherを置き換える根拠は未成立**。
最終Recall差のCIと、可変時間holdoutの実測結果を確認してから判断する。

## ローカルで実行した検証

```bash
git switch -c experiments/fit-grid-end-to-end-temporal-holdout origin/experiments/embryo-repulsion-estimator-benchmark
.venv/bin/python -m umaping.fit_grid_experiment promote
.venv/bin/python -m umaping.fit_grid_experiment audit --data runs/embryoid_body/main/cache/prepared_dataset.npz
.venv/bin/python -m pytest tests/test_fit_grid_experiment.py -q
.venv/bin/python -m pytest -q
git diff --check
```

前二つの実験コマンドは上記のファイル不在で停止。実データの成功として扱わない。
テストは合成小データのみ。保護、hash/checkpoint不一致、loss変換、時間点数5/7/10/12、
年代順・曖昧性・ID不重複、時間集計/NA、再現性、query変更でreference PCA/HVGが不変、
3 matched seedsの通し学習・全query比較・図出力を含む。
最終結果: **176 passed / 1 skipped / 41 warnings、57.51秒**。新規テスト17件を含む。
skipはCUDA非搭載によるもの。警告は既存UMAPの固定seed・未導入TensorFlow・Matplotlib非推奨API。
[検証記録](fit_grid_audit/validation.json) を保存した。

## remoteで4段階をまとめて実行するsh

`scripts/run_fit_grid_all.sh` が移植→通常suite付き比較→raw監査→temporal（seed 0,1,2）を順番に実行する。
実行コマンド、開始/完了時刻、失敗段階・終了コードを標準出力/標準エラーに出す。
途中で失敗したら後続段階は実行しない。既存結果の上書きや自動削除はしない。
raw省略時は `data/raw/embryoid_body/` 配下のh5ad/loomが**ちょうど1個**の場合だけ選択する。
rawファイルの存在と出力先の衝突・CUDA利用可否を長時間の比較に入る前に検査する。
時間ラベルの全体監査は比較の後、temporal学習の直前に行う。

```bash
(
  set -e
  cd /home/suzuki/Learn/UMAPing
  git fetch origin
  git switch experiments/fit-grid-end-to-end-temporal-holdout
  git pull --ff-only
  mkdir -p logs
  LOG_FILE="$(mktemp "logs/fit_grid_all_$(date -u +%Y%m%dT%H%M%SZ).log.XXXXXX")"
  nohup bash scripts/run_fit_grid_all.sh > "$LOG_FILE" 2>&1 &
  JOB_PID=$!
  printf '%s\n' "$JOB_PID" > "${LOG_FILE}.pid"
  printf 'PID: %s\nLOG: %s\n' "$JOB_PID" "$LOG_FILE"
  tail -f "$LOG_FILE"
)
```

`Ctrl+C` はログ閲覧を止めるだけで、nohupの実験は継続する。SSH切断後も実行を継続する。
rawが複数ある場合は、`nohup` 行を次のように変更する（対象rawの実パスを指定）。

```bash
nohup bash scripts/run_fit_grid_all.sh \
  --raw-file '/実在する対象raw.h5ad' \
  > "$LOG_FILE" 2>&1 &
```

時間列や順序が曖昧なら `--group-column 実列名` を指定し、順序は
`--group-order '最初の実ラベル' --group-order '次の実ラベル' ...` と一つずつ指定する。
scriptはauditとtemporalの両方に同じ列・順序を渡す。
`--cell-type-column` はtemporalにだけ渡す。

既存splitの比較まで完了済みなら `--start-stage 3`、移植だけ完了なら `--start-stage 2` で
その前の段階を明示的に飛ばせる。`--start-stage 4` でもraw監査は省略しない。
これは失敗した学習のcheckpoint再開ではなく、完了段階を飛ばす指定。
失敗段階の出力が残っていれば保護のため停止するので、別出力名を使う。
`RUN_ROOT` 環境変数で出力root、`PYTHON_BIN` で実行Python（既定 `.venv/bin/python`）を変更できる。

launcher追加時の検証: `bash -n scripts/run_fit_grid_all.sh` と
`.venv/bin/python -m pytest tests/test_fit_grid_launcher.py -q`。
重い実験を起動せず、実行順序・空白入り引数の保持・失敗時停止・temporal前の監査を検証する。
