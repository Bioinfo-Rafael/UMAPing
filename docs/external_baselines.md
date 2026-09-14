# 完了済みrunに対する外部比較

この実行基盤は、UMAPingの学習済みrunから保存済みの前処理結果を読み込み、
未実行だった外部手法だけを比較するものです。UMAPingの再学習、既存baselineの再実行、
データの再取得・再前処理は行いません。READMEへの追記と本ガイドは日本語で管理します。

## 対象

`results/ARCHIVE_INFO.txt`（基点コミット`0eda7ea`）とアーカイブ済み結果を確認しました。
以下の8件だけを、記載順に実行します。

| dataset | 入力run |
|---|---|
| coil20 | `runs/coil20/main` |
| coil100 | `runs/coil100/main` |
| pancreas | `runs/pancreas/main` |
| pancreas_batch_corrected | `runs/pancreas_batch_corrected/main` |
| fashion_mnist | `runs/fashion_mnist/main` |
| mnist_oos | `runs/mnist_oos/main` |
| organoid | `runs/organoid/main` |
| embryoid_body | `runs/embryoid_body/main` |

`twenty_newsgroups`・`hong_ed`は既定一覧から除くだけでなく、個別CLIでも受け付けません。
完了済みrunにも以前の外部手法の`available=False`行があるため、その結果は保持したまま別途比較します。

## 対応手法と公式実装

| CLI名 | CSVのmethod名 | 利用する実装 |
|---|---|---|
| parametric_umap | parametric_umap | `umap-learn==0.5.12`の`ParametricUMAP` |
| numap | numap_sep_spectralnet | `numap==0.2.3`の`NUMAP` |
| paramrepulsor | param_repulsor | `parampacmap==0.1.0`の`ParamPaCMAP` |
| oos_umap | oos_umap | 利用不可を記録。代替アルゴリズムは実行しない |

アダプタは既存の`src/umaping/experiments/baselines.py`を再利用しています。
学習にはreferenceだけを渡し、queryには学習後の`transform`だけを適用します。
Parametric UMAPには保存済みの`spread`も渡します。

NUMAPは既存比較と同じ`use_se=True, use_grease=True, use_residual_connections=True`を使い、
スペクトル次元も埋め込み次元に合わせます。GrEASEの内部処理も公式実装のままです。
参照集合内でのスペクトル学習・固有ベクトル計算があり、大きなデータでは時間とメモリを要します。
`use_grease=True`では公式コードがreference/queryの両方をGrEASEで変換します。
`is_train`は明示しますが、この分岐ではGrEASEを使わない場合と意味が異なります。

ParamRepulsorは`apply_pca=False, apply_scale=None`として、入力表現の再加工を無効にします。
公式既定の`embedding_init="pca"`は低次元埋め込みの初期化であり、入力特徴のPCA再fitとは別です。
反発重みスケジュール等は公式既定値を保ちます。

### OOS-UMAPの忠実性に関する判断（2026-09-14）

[公式リポジトリ](https://github.com/tariqul-islam/OOS_UMAP)の
`5015dc11c92b444530b9237533ea927427712311`について、追跡ファイル一覧とPythonソースを確認しました。
公開ファイルはMNIST・clinical_data・pneumonia_dataの`network_sig.py`、MNISTローダー、図などです。
ネットワークの`forward`はありますが、損失、学習ループ、OOS最適化、汎用`fit/transform`の実行コードはありません。

[2026年論文の第3・4節](https://arxiv.org/html/2606.04451v1)では、CE・MSE・CEMSEによる
学習手順と実験条件を記述しています。公開されたネットワーク定義だけでこれらを動かすには、
学習・グラフ処理をこちらで実装する必要があるため、公式実装を忠実に呼ぶ薄いアダプタにはできません。
したがって、全対象で`available=false, status=unavailable`と根拠を記録します。
他の3手法はそのまま実行します。実行不能なOOS用の環境・外部cloneは作りません。
既存の`reduced_repulsion_umap`をOOS-UMAP公式実装と称することもありません。

確認した一次資料：

- [NUMAP公式](https://github.com/shaham-lab/NUMAP)、[配布版0.2.3](https://pypi.org/project/numap/0.2.3/)
- [ParamRepulsor公式](https://github.com/hyhuang00/ParamRepulsor)、[配布版0.1.0](https://pypi.org/project/parampacmap/0.1.0/)
- [Parametric UMAP公式ガイド](https://umap-learn.readthedocs.io/en/latest/parametric_umap.html)
- [uv CLI](https://docs.astral.sh/uv/reference/cli/)、[PyTorch過去版の公式導入方法](https://pytorch.org/get-started/previous-versions/)

## 入力を一致させる条件

各runの`config.yaml`と`cache/prepared_dataset.npz`を必須入力とします。
`PreparedDataset`のreference/queryの順序・特徴量・ラベルをそのまま使用し、
キャッシュとconfigのSHA-256を結果に保存します。キャッシュがない場合はその比較を失敗として記録します。
GitHubにはPNG等だけを保存していても、元のリモート実行環境にはこのキャッシュが必要です。

`pancreas_batch_corrected`の元表現にはscArchesのquery適応が含まれます。
これは厳密なreference-only前処理ではありません。新しい比較でも同じ表現を再利用し、
この例外を結果メタデータに明記します。今回のrunnerでquery適応をやり直すことはありません。

## 環境と実行

対象はリモートのLinux x86_64環境です。既存`.venv`、conda環境へのインストールやactivateは行いません。
`setup_external_baselines.sh`は、必要ならuv 0.8.22を外部ルートの`bin/`に導入し、
Python 3.11.11と手法別venvを作ります。外部ルートの既定値は`../UMAPing_external_baselines`です。
すでにあるuvを使う場合も、専用venvのPythonを`uv pip --python`で明示します。

NUMAP・ParamRepulsor用のtorchは`2.6.0+cu124`を固定します。
Parametric UMAP用には`tensorflow[and-cuda]==2.18.1`とKeras 3.8.0を入れ、
UMAPingの共通コードが必要とするtorchはCPU版にします。
これにより、同一環境内でTensorFlowとtorchが別のCUDAライブラリを要求する衝突を避けます。
TensorFlowの利用可能デバイスは結果に記録します。CUDAの有無は各公式実装が判定します。

主要依存は`scripts/external_requirements/`に固定し、環境構築後の全依存は
外部ルートの`venvs/<baseline>/requirements.freeze.txt`と、各結果JSONの`packages`に保存します。
環境構築時のインポート確認が成功した環境だけに`.ready`を付けます。
構築に失敗した手法があっても後続の環境を作り、実行時には理由付きで利用不可にします。
全環境が使えない場合も、標準ライブラリのみの親プロセスが32件の利用不可を記録します。
この場合、読めなかった件数・seed・kなどは空欄です。

リモートの既存チェックアウトで、次を一括実行します。

```bash
cd /home/suzuki/Learn/UMAPing && bash <<'BASH'
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git fetch origin
git switch experiments/isolated-external-baselines
git pull --ff-only origin experiments/isolated-external-baselines
EXTERNAL_ROOT="$(dirname "$REPO_ROOT")/UMAPing_external_baselines"
mkdir -p "$EXTERNAL_ROOT"
bash scripts/setup_external_baselines.sh "$EXTERNAL_ROOT"
bash scripts/launch_external_baselines.sh "$EXTERNAL_ROOT"
BASH
```

環境構築は前景で実行し、その後`nohup`で一つの親プロセスを起動します。
比較は逐次実行し、個別のPythonプロセスの失敗・強制終了でも後続へ進みます。
PID・ログファイル・`tail -f`コマンドを表示し、PIDをログと同じディレクトリへ保存します。
SSH切断後も継続します。既存runや外部比較の過去結果への上書きは拒否し、
起動ごとに新しい時刻付きディレクトリを使います。実行中の再起動や環境の再構築は避けてください。

個別実行は`scripts/run_external_baselines.py worker`を専用venvのPythonで呼び出します。
必須引数は`--repo-root --dataset --baseline --run-dir --output-dir`です。
入力runの場所が異なる場合はここで指定できます。`--seed`は省略時に元runのseedを使います。
`--device auto`が既定です。CPU指定ではworker内でCUDAの可視性を無効にします。

## 出力と評価の読み方

出力先は`runs_external_baselines/<UTC時刻_PID>/<dataset>/<baseline>/`です。

| ファイル | 内容 |
|---|---|
| `result.json` | 成功・失敗・利用可否、理由、seed、バージョン、件数、評価値、入力hash、元runメタデータ |
| `baseline_comparison.csv` | 既存比較表とmethod名を合わせた1行の結果 |
| `advanced_per_query.csv` | query_indexごとのRecall@k・NDCG。元queryの順序を維持 |
| `embedding.png` | 既存の描画関数によるreference/queryの図 |
| `embeddings.npz` | 新しく得られたreference/query座標。モデルcheckpointではない |
| `traceback.txt` | Python例外が起きた場合の詳細 |

全比較をまとめた`summary.csv`は時刻付き出力ルート直下に置き、各比較の終了ごとに更新します。
32行は「8データセット×4候補手法」で、OOSは利用不可の行です。
`ALL DONE`は全候補を試した意味で、全候補の成功を意味しません。
成功は`status=success`と`success=true`、依存不足・OOS非対応は`unavailable`、
データ不足・数値異常・worker異常終了は`failed`で区別します。

評価は既存の`evaluate_embedding`と`multi_k_recall_and_ndcg`を呼びます。
主指標のkは元runの`eval.k`を使い、Recall@5/10/15/30とNDCG@30も記録します。
NDCGは高次元の真のtop-30を二値関連度とした、埋め込み近傍ランキングの指標です。
reference trustworthinessは元configの近傍数・全referenceで計算します。
ラベル分類は既存関数の既定15近傍で、これを主指標の`eval.k`と混同しないでください。
元の主ラベル選択を共通moduleへ移しただけなので、旧評価と選び方は同じです。

fit時間とquery変換時間は分離します。NUMAPとParamRepulsorのreference変換時間も別に記録します。
Parametric UMAPのfit時間は`fit_transform(reference)`の時間です。
`mean_query_latency_seconds`は一括query変換時間÷query件数であり、単一点呼出しのレイテンシではありません。
モデル・バージョン・内部前処理・タイミングの条件もJSONで確認してください。

既存CSVに新しい値を挿入したり、既存のPNG/JSONを書き換えたりしません。
`runs_external_baselines/`はgitignore対象です。今回の実装コミットに生成結果、
データ、環境、公式外部リポジトリ、モデルcheckpointは含めません。

## 軽量検証

`tests/test_isolated_external.py`で合成データを使い、キャッシュ再利用、既存runの不変性、
パス衝突とsymlink拒否、任意依存の欠落、NaN、バージョン不一致、失敗後の継続を検証します。
既存の`tests/test_external_baselines.py`も公式APIのmockでreference-only fitを検証します。
本環境では実データ8件の比較を実行していません。

2026-09-14の検証では、新規runner・既存アダプタ・既存追加評価の42テストが成功しました。
環境構築スクリプトはuvをmockした実プロセスで失敗継続とインストール先を確認し、
起動スクリプトは実際の`nohup`でPID・ログ・32件の利用不可サマリを確認しています。
Python構文・標準ライブラリだけでの親CLI import・Bash構文・`git diff --check`も確認しました。
Linux x86_64/Python 3.11向け依存の解決も確認済みですが、CUDAサーバーでの
実際の全環境インストールと公式モデル学習は未実行です。
