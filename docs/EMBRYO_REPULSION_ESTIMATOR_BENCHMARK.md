# Embryo：凍結trajectory上のrepulsion teacher比較

## 実行状況（2026-09-15 JST）

実装と合成データでの検証を行いました。**このローカル環境にはEmbryoの凍結成果物がなく、実データのPhase A・Phase Bは未実行です。**
以下の表は未測定を明示するもので、合成テストの数値をEmbryoの実験結果として使っていません。

| 手法 | 実データteacher RMSE / bias / variance | 学習後のexact RMSE | 状態 |
|---|---|---|---|
| Uniform-64 | 未測定 | 未測定 | 凍結成果物待ち |
| Raw Dual IS-64 | 未測定 | 未測定 | 凍結成果物待ち |
| Hub-corrected Dual IS-64 | 未測定 | 未測定 | 凍結成果物待ち |
| Dual Top-32 + tail-32 | 未測定 | 未測定 | 凍結成果物待ち |
| Barnes–Hut | 未測定 | 未測定 | 凍結成果物待ち |
| FIt-style grid | 未測定 | 未測定 | 凍結成果物待ち |

実データの実行後は、runごとの`report.md`が実測値から自動生成されます。この文書や既存の実験出力を自動で上書きしません。

## A. リポジトリ・データの確認

新規ブランチは`experiments/embryo-repulsion-estimator-benchmark`です。
`origin/analysis/combined-baseline-evaluation`の`b7d1c7d`（前回の統合解析成果物を含む）から作成しました。mainへmergeしません。

リポジトリとローカルのPersonalDev・Downloads・Documentsを検索しました。
既存の該当候補は`embryoid_body`（胚様体）で、`src/umaping/data/embryoid_body.py`と
`experiments/configs/embryoid_body.yaml`、`runs/embryoid_body/main/`の完了記録・図・指標・configがあります。
独立した`embryo`という別datasetの実装やrawデータは見つかっていません。
**胚と胚様体は同義とは限らないため、CLIは`--dataset`の明示指定と保存configとの一致を要求します。**
以下の実行例は既存の`embryoid_body`を対象にする場合のものです。別のEmbryo datasetへの暗黙の置換はしません。

既存loaderが記載するデータはMoon et al.のhuman embryoid-body differentiation datasetです。
既存の前処理（referenceでのHVG/PCA、入力50次元）を再利用するため、今回のrunnerは保存済みreference featuresだけを読みます。
loaderのdownload関数、前処理、graph生成、DualEncoder学習、spectral学習を呼びません。
rawの想定ディレクトリ`data/raw/embryoid_body/`はローカルに存在せず、実ファイル名は確認できていません。新規downloadも行っていません。

ローカルで不足しているファイルは、次の絶対パスです。

- `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/embryoid_body/main/memory/reference_trajectory.npz`
- `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/embryoid_body/main/memory/retriever_keys.npy`
- `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/embryoid_body/main/checkpoints/retriever.pt`
- `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/embryoid_body/main/cache/prepared_dataset.npz`

featuresは最後のcacheの代わりに`memory/reference_features.npy`でも読み込めます。これもローカルにはありません。
元configでrow-mass重みを有効にした場合だけ、`memory/graph_symmetric.npz`も必要です。
リモートの既存runを使用するか、同一runからこれらの凍結ファイルをコピーしてください。rawだけあっても、この比較のためにtrajectoryを再生成することはありません。

### 元の処理と保存schema

1. `pipeline.stage_spectral()`がcalibration後のspectral encoderからreference embedding `Y(0)`を作ります。
2. `stage_flow_dynamics()` → `training.flow.build_reference_trajectory()` → `dynamics.simulate_reference_dynamics()`。
3. graphの解析的引力と既存のuniform MC斥力でEuler積分します。成分clipをpairwiseと合算力に適用する現行処理は変更していません。
4. `memory/reference_trajectory.npz`に`times: float64 (T,)`、`positions: float32 (T,N,2)`を保存します。
5. `TorchTrajectoryView`は二つの保存checkpointの**位置**を線形補間して連続時刻の点位置を返します。

保存configは`n_steps=200`、`checkpoint_stride=1`なので、期待されるTは201、正規化時刻は0〜1です。
ただし**実ファイルのshape・N・内容はローカルでは未確認**です。runnerは実ファイルから読み、有限性・単調時刻・features/key件数を検証してmanifestへ記録します。

元の`train_repulsion_field()`は一様anchor、連続一様t、Gaussian jitterを生成し、全referenceから復元抽出した64点のclip済み`g_minus`を平均します。anchor自身も含みます。
元の呼出しでteacher/query_samplerを指定しない場合は乱数消費順とAdam更新を維持します。追加引数はteacher、共有query生成器、評価callbackだけです。

DualEncoderは独立query/key MLP、L2正規化、内積/temperatureのscoreです。
`training/retriever.py`はdirected fuzzy membership Muに従うpositive sampling、in-batch/random negatives、cross-entropyによるDPR/InfoNCE学習を行います。
既存checkpointは`checkpoints/retriever.pt`（`hparams`,`state_dict`）、凍結keyは`memory/retriever_keys.npy`です。
今回のみ、同じcheckpointでreferenceのQをbatch生成し、再計算したKが保存Kと一致することも確認します。再学習は行いません。

## B. 共通条件

| 項目 | 既存embryoid_body configの設定 |
|---|---|
| reference数N | 実ファイル到着後に確認 |
| 入力次元 | 50（実ファイルと照合） |
| RepulsionField | hidden=256、residual blocks=4、time embedding=32、GELU、出力2次元 |
| 初期値 | 単一state_dictを全手法へclone |
| optimizer | Adam、lr=0.001、weight_decay=0 |
| 学習 | 4000 steps、batch=512、jitter sigma=0.1 |
| 行重み | false（指定時は元のrow-mass重みを共通使用） |
| kernel | 既存`g_minus`、`find_ab_params(spread=1,min_dist=.1)`、epsilon=1e-3、成分clip=4 |
| stochastic force budget | 64（Top-Lのみ32+32） |
| validation | 1000固定query、stochastic反復50回、seed=0 |
| parameter gradient clipping | 元学習に存在しないため追加しない。forceの成分clipとは区別 |

実際には指定した保存configのarchitecture・optimizer・jitter・clipを読みます。steps/batchの任意overrideも全手法に共通です。
`config/source_config.yaml`には元configのコピー、`config/training.yaml`には実際の共通学習設定を保存します。
anchor・t・jitterは`SeedSequence([seed,step])`で再現し、teacher固有の乱数から分離します。
Gaussian jitterの分布は従来と同じで、実験間では同じquery列を使用します。過去の既存runそのものの乱数列を再現する実験ではありません。

## C. 六つの推定器

全方式の目標は、anchor/selfを含む全N点の平均です。

$$B(y,t)=\frac1N\sum_{j=1}^N g_-(y,Y_j(t)).$$

`g_minus()`自体とreference trajectoryの生成コードは変更しません。kernelを複製して再実装しません。

### Uniform-64

$$J_m\sim U(1,\ldots,N),\qquad \hat B=\frac1{64}\sum_m g_-(y,Y_{J_m}(t)).$$

### Raw Dual importance sampling

$$s_{ij}=Q_i^TK_j/\tau,\quad p_j=\mathrm{softmax}_j(s_{ij}/T_p),\quad
q_j=(1-\lambda)/N+\lambda p_j,$$

$$J_m\sim q,\qquad \hat B=\frac1{64}\sum_m\frac{g_-(y,Y_{J_m}(t))}{Nq_{J_m}}.$$

主比較は`T_p=1`、`lambda=1`。小さい任意ablationとして`--mixture-ablation`でlambda=0.9を追加できます。
追加ablationのlambdaは`--mixture-lambda`で指定できます。pure版は常に主比較へ残します。
softmaxはfloat64/log-softmaxで計算し、実際にsamplingへ渡した確率で重みを計算します。
確率のunderflowでsupportが消えた場合は失敗とし、黙ったfloor・weight clipping・self除外を行いません。

### Hub-corrected Dual importance sampling

$$h_j=\frac1{K_h}\sum_{i\in\mathrm{TopKQueries}(j)}s_{ij},\quad
\tilde s_{ij}=s_{ij}-\beta h_j,\quad p_j=\mathrm{softmax}_j(\tilde s_{ij}/T_p).$$

`K_h=32`,`beta=1`を既定とし、上記と同じimportance補正を使います。
hの計算はquery/keyの両軸をchunk化してcolumnごとのTop-Kだけを保持します。保存するのはQ、K、hであり、N×N行列は保存しません。
proposal構築でgraph、fuzzy weights、Y(t)、force、spectral encoderは参照しません。

### Dual Top-L + uniform tail

raw scoreのTop-LをC、補集合をRとし、L=32、tail=32です。

$$\hat B=\frac1N\left[\sum_{j\in C}g_-(y,Y_j(t))+
\frac{N-L}{32}\sum_{m=1}^{32}g_-(y,Y_{J_m}(t))\right],\qquad J_m\sim U(R).$$

Top-Lを除いた整数rankから実indexへ写像して、一様復元抽出します。近傍密度による棄却loopはありません。
N≤Lなら全点和になります（小さい単体テストだけの境界条件）。主実験では原空間距離などによるrerankは行いません。

### Barnes–Hut

各保存checkpointに2D四分木を一度構築します。mass・centroid・bounds・childrenを保持します。
queryを含まない遠方cellで`cell_size/distance < theta`なら`mass * g_minus(y,centroid)`とし、近接cellは再帰します。
葉では個別point forceを評価し、全体をNで割ります。

$$\hat B(y,t)=(1-\alpha)\hat B_{\mathrm{tree},lo}(y)+\alpha\hat B_{\mathrm{tree},hi}(y).$$

theta候補は0.2/0.5/0.8です。CPU traversalとCPU/GPU転送もruntimeに含めます。
各queryで実際にkernelへ渡したpoint/centroid件数をinteraction数として記録します。

### FIt-style grid

全trajectoryから共通boundsを決め、各軸に`max(6*jitter_sigma, 5%*span, 1e-3)`のmarginを追加します。
CICでmassを4頂点へ分配し、既存のclip済み`g_minus(delta,0)`からvector kernelを作ります。
`scipy.signal.fftconvolve(...,mode='full')`によるゼロpaddingされた**線形**畳み込みから正しい中心領域を切り出し、Nで正規化します。
空間方向はbilinear、時間方向は二つのcheckpoint fieldの線形補間です。

grid候補は64/128/256です。範囲外queryは件数と失敗を記録し、wrap・黙ったclamp・exact oracleによるtraining fallbackはしません。
これはUMAP kernelに合わせたFIt-SNE-inspired approximationであり、FIt-SNE全体の再実装・同等性の主張ではありません。

## D. Phase A → Phase B

1. 同一query分布の固定validation `(anchor,t,y)` を保存。
2. 既存`exact_all_reference_mean_field()`をquery/referenceの両軸でchunk化して全点和を計算・保存。
3. stochastic teacherを50反復（`--repetitions 100`も可）、BH/gridは各候補を評価。
4. 全反復予測、ESS・weight分布、proposal entropy/top1/min sampled probability、score計算時間、構築時間、interaction数を保存。
5. exact oracleとUniformの有限性を確認。失敗、ESS<6.4、max weight>100を`phase_a_review.json`へ記録してから学習へ進む。
6. 主4方式はpure versionを使用。BH/gridはPhase Aの最小MSEから10%以内の候補で最速のものを選択する。
7. 全6方式のB_phiだけを共通条件で学習。Phase Aで非有限/support喪失等により失敗した方式はskipを明記。有限のlow ESS/巨大weightは隠さず学習比較へ残す。
8. 100stepごとと初期・最終時点に同じexact validationで評価。学習loss、rolling std、後半20%のCV、best/final validation、学習時間、forward時間を保存。初期exact RMSEの半分への初回到達step/時間も共通の収束指標として保存し、未到達はNaNとします。

BH/gridは**fieldの時間補間**、exact oracleは**positionの時間補間後の非線形force**です。
このためthetaを0にしても、連続時刻では時間補間の誤差が残ります。theta→0の全点和への一致テストは保存checkpointで行います。

MSEは`mean(||estimate-exact||²)`、varianceは`mean(||estimate-repetition_mean||²)`です。
biasは有限反復のsample meanとexactの差で、真のbiasではありません。補助列`bias_squared_debiased`ではvariance/(repetitions−1)を引き、負値も保存します。
cosineは両ベクトルが非ゼロのqueryで計算し、有効割合を報告します。relative magnitude errorも小さい真値を除外し、有効割合を記録します。

学習lossは従来の座標平均MSE（row-mass有効時は従来の重み付き式）です。2Dのexact vector MSEとは係数が異なります。
同一validationでteacher variant選択とモデル評価を行う探索実験であり、独立testでの確認ではありません。

## E. リモートでの実行

次は既存の胚様体runを使う場合の実行例です。実データ学習の所要時間は未測定です。
特にBHのPython traversalと全方式4000stepは長くなる可能性があります。Phase Aだけ先に実行し、runtimeと誤差を確認できます。
下記はPhase Aの数値検査を通過後、同じプロセスでPhase Bへ進むコマンドです。

```bash
cd /home/suzuki/Learn/UMAPing && bash <<'BASH'
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git fetch origin
git switch experiments/embryo-repulsion-estimator-benchmark
git pull --ff-only origin experiments/embryo-repulsion-estimator-benchmark
mkdir -p results/embryo_repulsion_estimators
RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')_$$"
OUTPUT_DIR="$REPO_ROOT/results/embryo_repulsion_estimators/$RUN_ID"
LOGFILE="$(mktemp "$REPO_ROOT/results/embryo_repulsion_estimators/${RUN_ID}.log.XXXXXX")"
"$REPO_ROOT/.venv/bin/python" scripts/benchmark_repulsion_estimators.py \
  --frozen-run "$REPO_ROOT/runs/embryoid_body/main" \
  --dataset embryoid_body --output-dir "$OUTPUT_DIR" \
  --stage all --device auto --seed 0 \
  --validation-queries 1000 --repetitions 50 \
  2>&1 | tee "$LOGFILE"
echo "出力: $OUTPUT_DIR"
echo "レポート: $OUTPUT_DIR/report.md"
echo "ログ: $LOGFILE"
BASH
```

Phase Aだけなら`--stage all`を`--stage estimators`へ変更します。
そのrunから後で学習するときは、同じ引数に`--stage train --from-run /absolute/path/to/phase_a_run`を指定し、
**別の新規**`--output-dir`を使います。入力hash・proposal設定・seed等が一致しないPhase Aは受け付けません。
Phase Aの固定validationとexact場をコピーして再利用します。上流artifact生成やexact oracleでのteacher学習はありません。
Q・K・hubnessもPhase Aの保存値をhash照合して再利用し、再計算しません。

保存結果からの図・report再生成は次です（新しい出力先を指定）。

```bash
.venv/bin/python -m umaping.repulsion_estimators.report \
  --run-dir results/embryo_repulsion_estimators/RUN_ID \
  --output-dir results/embryo_repulsion_estimators/NEW_PLOT_RUN_ID
```

## F. 成果物

- `config/`: 元config、共通学習config、入力パス/SHA-256・Git commit・shape・times・seed・環境・完了状態。
- `artifacts/validation.npz`: anchor、t、y、exact field。Q/K・hubness・grid field・共通モデル初期値も保存。
- `estimator_benchmark/`: 全反復予測、batch diagnostics、IS全sample weight。失敗時はpartial予測の未計算部分をNaNで残し、failure JSONを保存。
- `repulsion_training/<method>/`: loss・validation曲線、teacher diagnostics、予測、`hparams/state_dict`付きcheckpoint。
- `metrics/`: estimator CSV/JSON、Phase A review、学習済みfield CSV/JSON。
- `figures/`: 同一軸上の比較図、PNGとSVG。
- `report.md`: 実測値からの日本語report。失敗・未実行を含めて記録。

必須12図は`estimator_rmse_vs_method`、`estimator_variance_vs_method`、`estimator_cosine_vs_method`、
`estimator_runtime_vs_rmse`、`importance_ess`、`importance_weight_distribution`、`training_loss_curves`、
`training_loss_rolling_std`、`exact_validation_rmse`、`exact_validation_cosine`、`exact_validation_magnitude_error`、
`final_runtime_accuracy_pareto`です。追加でestimator magnitude error図を保存します。
利用できない図は`figure_status.json`へ理由を残します。

出力directoryは新規作成のみです。凍結入力hashを実行前後に照合します。大きなcheckpoint・grid・sample weightを含むため生成物はGitで既定除外し、元のrunは変更しません。

## G. 現時点で回答できること

1. Uniform-64のtarget noiseはEmbryoで大きいか：未測定。
2. Raw Dual ISでvarianceが減るか：未測定。不偏性だけでは改善を保証しない。
3. Hubness補正がproposalを改善するか：未測定。
4. 同じ64-force budgetでTop-LはISより良いか：未測定。score計算費用も別途測る。
5. 同等runtimeでのBH精度：未測定。連続時刻の時間近似誤差が含まれる。
6. Grid精度：合成例で解像度改善を検証するが、Embryoの改善は未測定。
7. 最良の学習teacher：未判定。
8. accuracy/runtime tradeoff：未判定。実測後のParetoとfinal exact RMSEで判断する。

**推奨は、現時点ではUniform-64を次の本番pipelineで維持することです。**
このbenchmarkのEmbryo実測が完了してから候補を選び、独立query・複数学習seed・OOS embedding指標でも確認してください。
fieldの改善だけで最終UMAP品質の改善を断定しません。

## H. 検証コマンド

```bash
.venv/bin/python -m pytest tests/test_repulsion_estimators.py tests/test_inference.py tests/test_umap_forces.py tests/test_advanced_analysis.py -q
```

合成fixtureではPhase A→検査→6モデル学習→exact評価→全図生成、およびPhase Aからの分離実行・保存値からの再描画を検証します。
実Embryoへの実行コマンドはローカルで不足artifactを列挙して停止することを確認しました。実データの学習・trajectory生成は実行していません。
CUDAのテストは対応deviceがある場合だけ実行します。CPU-only環境で通過したテストをCUDA検証済みとは表記しません。
上記は**48 passed / 1 skipped（CUDAなし）**でした。既存UMAPのseed/n_jobs警告2件があり、失敗はありません。
合成図のPareto・validation曲線を目視し、凡例・軸目盛も確認しました。
