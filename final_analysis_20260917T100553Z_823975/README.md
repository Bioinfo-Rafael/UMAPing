# Embryoid body 保存済み成果物の追加解析

[全図の静的ギャラリー](index.html) / [図の索引](99_provenance/figure_catalog.csv) / [スキップ理由](99_provenance/skipped_items.csv)

実行状態：**complete_with_skips**。利用可能だったcomparison：existing, temporal0, temporal1。欠測した比較は未検証です。

## 実行範囲

保存済みCSV・JSON・2D座標・予測値だけをCPUで読み取りました。再学習、モデルロード、推論、UMAP fit/transform、ODE積分、exact場・高次元近傍の再計算はありません。依存パッケージは追加していません。出力は新規ディレクトリで、入力成果物とは分離しています。

既存split、temporal seed 0・1、旧教師benchmark、参照軌道近傍の場評価、空間格子の場評価を区別します。comparison manifestのseedは評価seedです。temporalの学習seedはrun metadataのmatched_seedを照合しています。seed 2は除外しました。

## 保存値・追加集計から直接確認できる結果

### existing split：学習seed not recorded in run metadata

Reference 24,823、query 6,206。各手法のID重複・欠損とstageを検証し、主指標の平均をaggregate.csvと照合しました。

| 方法 | 集団 | n | Recall@15 (%) | NDCG | density log distortion ↓ |
|---|---|---:|---:|---:|---:|
| Standard UMAP | all | 6,206 | 7.738 | 0.0799 | 0.2465 |
| Reduced repulsion UMAP | all | 6,206 | 7.556 | 0.0778 | 0.2416 |
| Weighted kNN | all | 6,206 | 6.084 | 0.0616 | 0.3080 |
| Uniform | all | 6,206 | 9.059 | 0.0946 | 0.2657 |
| FitGrid | all | 6,206 | 9.104 | 0.0950 | 0.2657 |

### temporal holdout：学習seed 0

Reference 18,486、query 12,543。各手法のID重複・欠損とstageを検証し、主指標の平均をaggregate.csvと照合しました。

| 方法 | 集団 | n | Recall@15 (%) | NDCG | density log distortion ↓ |
|---|---|---:|---:|---:|---:|
| Standard UMAP | all | 12,543 | 9.103 | 0.0928 | 0.3100 |
| Standard UMAP | interpolation | 6,241 | 8.923 | 0.0913 | 0.3191 |
| Standard UMAP | extrapolation | 6,302 | 9.282 | 0.0943 | 0.3009 |
| Reduced repulsion UMAP | all | 12,543 | 8.853 | 0.0909 | 0.2775 |
| Reduced repulsion UMAP | interpolation | 6,241 | 8.922 | 0.0915 | 0.2798 |
| Reduced repulsion UMAP | extrapolation | 6,302 | 8.786 | 0.0903 | 0.2752 |
| Weighted kNN | all | 12,543 | 6.536 | 0.0667 | 0.3690 |
| Weighted kNN | interpolation | 6,241 | 6.965 | 0.0713 | 0.3896 |
| Weighted kNN | extrapolation | 6,302 | 6.112 | 0.0623 | 0.3486 |
| Uniform | all | 12,543 | 10.993 | 0.1151 | 0.2802 |
| Uniform | interpolation | 6,241 | 10.466 | 0.1098 | 0.2763 |
| Uniform | extrapolation | 6,302 | 11.514 | 0.1203 | 0.2840 |
| FitGrid | all | 12,543 | 11.012 | 0.1159 | 0.2807 |
| FitGrid | interpolation | 6,241 | 10.549 | 0.1115 | 0.2793 |
| FitGrid | extrapolation | 6,302 | 11.470 | 0.1202 | 0.2820 |

### temporal holdout：学習seed 1

Reference 18,486、query 12,543。各手法のID重複・欠損とstageを検証し、主指標の平均をaggregate.csvと照合しました。

| 方法 | 集団 | n | Recall@15 (%) | NDCG | density log distortion ↓ |
|---|---|---:|---:|---:|---:|
| Uniform | all | 12,543 | 10.843 | 0.1132 | 0.2799 |
| Uniform | interpolation | 6,241 | 10.380 | 0.1089 | 0.2758 |
| Uniform | extrapolation | 6,302 | 11.302 | 0.1174 | 0.2839 |
| FitGrid | all | 12,543 | 10.967 | 0.1155 | 0.2780 |
| FitGrid | interpolation | 6,241 | 10.460 | 0.1109 | 0.2767 |
| FitGrid | extrapolation | 6,302 | 11.468 | 0.1201 | 0.2793 |

### 対応差

Recall差はパーセントポイント（pp）です。相対改善率ではありません。

| split | 学習seed | 集団 | 比較 | Recall差 (pp) | 95% CI (pp) | n |
|---|---:|---|---|---:|---|---:|
| existing split | not recorded in run metadata | all | Uniform − Standard UMAP | +1.321 | [+1.098, +1.540] | 6,206 |
| existing split | not recorded in run metadata | all | FitGrid − Standard UMAP | +1.366 | [+1.154, +1.580] | 6,206 |
| existing split | not recorded in run metadata | all | FitGrid − Uniform | +0.045 | [-0.052, +0.138] | 6,206 |
| temporal holdout | 0 | all | Uniform − Standard UMAP | +1.890 | [+1.720, +2.060] | 12,543 |
| temporal holdout | 0 | interpolation | Uniform − Standard UMAP | +1.544 | [+1.290, +1.789] | 6,241 |
| temporal holdout | 0 | extrapolation | Uniform − Standard UMAP | +2.232 | [+1.977, +2.466] | 6,302 |
| temporal holdout | 0 | all | FitGrid − Standard UMAP | +1.909 | [+1.747, +2.074] | 12,543 |
| temporal holdout | 0 | interpolation | FitGrid − Standard UMAP | +1.626 | [+1.392, +1.858] | 6,241 |
| temporal holdout | 0 | extrapolation | FitGrid − Standard UMAP | +2.189 | [+1.950, +2.413] | 6,302 |
| temporal holdout | 0 | all | FitGrid − Uniform | +0.019 | [-0.054, +0.090] | 12,543 |
| temporal holdout | 0 | interpolation | FitGrid − Uniform | +0.082 | [-0.018, +0.186] | 6,241 |
| temporal holdout | 0 | extrapolation | FitGrid − Uniform | -0.043 | [-0.145, +0.053] | 6,302 |
| temporal holdout | 1 | all | FitGrid − Uniform | +0.123 | [+0.055, +0.193] | 12,543 |
| temporal holdout | 1 | interpolation | FitGrid − Uniform | +0.080 | [-0.014, +0.174] | 6,241 |
| temporal holdout | 1 | extrapolation | FitGrid − Uniform | +0.166 | [+0.063, +0.266] | 6,302 |

### 学習済み場：異なる評価分布

| split | 学習seed | 領域 | 方法 | n | RMSE ↓ | FitGrid/Uniform RMSE |
|---|---:|---|---|---:|---:|---:|
| existing split | not recorded in run metadata | near reference trajectory | Uniform | 1000.0 | 0.0120367 | 0.8872 |
| existing split | not recorded in run metadata | near reference trajectory | FitGrid | 1000.0 | 0.0106792 | 0.8872 |
| temporal holdout | 0 | near reference trajectory | Uniform | 1000.0 | 0.012887 | 0.7926 |
| temporal holdout | 0 | near reference trajectory | FitGrid | 1000.0 | 0.0102141 | 0.7926 |
| temporal holdout | 1 | near reference trajectory | Uniform | 1000.0 | 0.0122928 | 0.7919 |
| temporal holdout | 1 | near reference trajectory | FitGrid | 1000.0 | 0.00973441 | 0.7919 |
| existing split | not recorded in run metadata | spatial grid | Uniform | 300.0 | 0.00941481 | 1.098 |
| existing split | not recorded in run metadata | spatial grid | FitGrid | 300.0 | 0.0103385 | 1.098 |
| temporal holdout | 0 | spatial grid | Uniform | 300.0 | 0.0117369 | 1.679 |
| temporal holdout | 0 | spatial grid | FitGrid | 300.0 | 0.01971 | 1.679 |
| temporal holdout | 1 | spatial grid | Uniform | nan | nan | nan |
| temporal holdout | 1 | spatial grid | FitGrid | nan | nan | nan |

near-referenceは参照軌道近傍1,000点、spatial gridは300点の保存された空間評価です。格子の全体値はmixed tを採用し、固定tの値は別図に載せます。RMSEは2成分の二乗誤差の和を点平均して平方根を取る定義です。tは埋め込み最適化時間であり、生物学的stageではありません。seed 1の空間評価は欠測です。

### 旧教師推定器benchmark

| 設定 | 反復 | query数 | RMSE ↓ | query時間 (µs) | 構築時間 (s) |
|---|---:|---:|---:|---:|---:|
| uniform_mc | 50 | 1000 | 0.047453 | 10.023 | 0.0065 |
| dual_raw_is | 50 | 1000 | 0.10565 | 78.904 | 0.0357 |
| dual_hub_is | 50 | 1000 | 0.102466 | 59.767 | 0.0217 |
| dual_topl | 50 | 1000 | 0.0659156 | 91.576 | 0.0182 |
| barnes_hut_0.2 | 1 | 1000 | 0.00725753 | 10391.466 | 22.5266 |
| barnes_hut_0.5 | 1 | 1000 | 0.00729584 | 2837.127 | 13.7371 |
| barnes_hut_0.8 | 1 | 1000 | 0.00736472 | 1603.566 | 23.1075 |
| fit_grid_64 | 1 | 1000 | 0.00933373 | 15.404 | 0.6382 |
| fit_grid_128 | 1 | 1000 | 0.00775605 | 11.791 | 1.5016 |
| fit_grid_256 | 1 | 1000 | 0.00732489 | 11.867 | 3.7685 |

掲載するtheta・grid sizeは保存された全設定です。旧benchmarkのUniformをmainのUniformと同一視しません。stochasticは反復×query×2、deterministicは1×query×2です。biasは有限反復の標本平均の偏りであり理論的biasではありません。構築時間とquery時間はこの実験内の保存計測値のみです。

保存された教師RMSEはUniform=0.047453、fit_grid_64=0.009334、fit_grid_128=0.007756、fit_grid_256=0.007325です。

旧benchmarkの最終保存stepでのexact validation RMSE：dual_hub_is=0.013775（step 4000）、dual_raw_is=0.013514（step 4000）、uniform_mc=0.012099（step 4000）、dual_topl=0.012553（step 4000）、barnes_hut_0.8=0.010232（step 4000）、fit_grid_256=0.010193（step 4000）。

## その結果から読み取れる解釈

通常UMAPに対する差と、Uniformに対するFitGridの差は別の比較です。正の改善量はRecall/NDCGではours−baseline、densityではbaseline−oursです。良い結果と悪い結果を同じ構成で表示し、CIが0を跨ぐ場合は固定モデルの細胞bootstrapでも差の方向は不確かです。

場の精度が評価領域によって変わる場合、その結果は当該位置・時刻分布に条件づけられています。教師推定器の精度、学習済み場の精度、最終埋め込みのRecallは異なる評価なので、一つの優劣にまとめません。異なる教師へのtraining lossやlossの変動だけからexact精度の順位を決めません。

既存splitとtemporalではreference数とquery集団が異なるため、数値の大小だけで難易度を断定しません。seed別の点は別の学習結果で、同じqueryをseed間で独立細胞としてプールしていません。

## 未検証の仮説と未測定事項

seed 2、未測定baseline、seed 1の通常UMAP・Recall@k・空間格子評価は補完していません。2つの学習seedだけでは一般的な再現性を確定できません。新しい生物学的反復、cell typeの妥当性、将来の複数外挿horizonに対する傾向、他のハードウェアでの速度優位は未検証です。bracketing rateだけで生物学的正しさを結論しません。Recall差の座標図は因果効果を示しません。

## 統計・表示の条件

新規bootstrapは固定seed=20260917から比較ごとに決定的なseedを作り、2,000回、メモリ制限32 MiBのバッチで実行します。対応queryのベクトルを同時再標本化し、temporalでは時間群ごとの件数を固定した層別再標本化を使います。95% CIは平均改善量のpercentile区間です。既存10,000回bootstrapのCIは再利用していません。

CIは固定モデル・固定referenceに対する細胞bootstrapであり、生物学的反復のCIではありません。細胞間・donor間依存はモデル化していません。CSVには元の平均、符号変換前の差、中央値、改善／同点／悪化の割合、反復数・seed・batchを保存しました。

temporalはall／interpolation／extrapolationを表示し、timepoint別名を独立集団として二重計上しません。全stageの色はreferenceとqueryの和集合で固定し、query_indexでラベルを対応付けます。全queryを描画し、PDFの点群はラスタライズしています。Uniform/FitGridのreference座標一致を確認して軸を共有し、通常UMAPは独自の座標系でequal aspect表示しています。

exact repulsionは保存50 queryの同じID集合だけで比較します。temporal内挿／外挿件数はそのCSVのstage_countsに記録します。legacy advancedのMC repulsion oracleとは異なるfull-sum exactです。

## 保存場所と検証

00_overviewは固定した主要図の実体コピーです。01は既存split、02はtemporal、03は旧教師、04は学習済み場、05はablationです。各図のPNG・PDF・CSVは同じstemです。

99_provenance/source_inventory.csvに使用入力のSHA-256と終了時照合、validation_checks.csvに数値照合、figure_catalog.csvに出所・seed・件数・領域、analysis_manifest.jsonに実行条件を保存します。入力の読取りは64 MiB以下の許可形式に制限し、大容量軌道・モデルをハッシュ目的でも読みません。

実行コマンド：`scripts/final_saved_analysis.py --root /home/suzuki/Learn/UMAPing --output /home/suzuki/Learn/UMAPing/final_analysis_20260917T100553Z_823975 --max-seconds 600 --bootstrap 2000 --bootstrap-memory-mb 32 --seed 20260917 --inventory-hint /home/suzuki/Learn/UMAPing/remote_inventory_20260917T083622Z.txt`

## スキップした項目

- temporal_seed_2: Excluded by explicit selection: incomplete, no checkpoint loading or resumption
- temporal_seed_1/standard_umap_and_recall_at_k: Not saved for training seed 1; seed-0 results not substituted
- field/temporal1/uniform_mc/spatial_grid: Saved spatial-grid exact evaluation absent; missing, not imputed
- field/temporal1/fit_grid/spatial_grid: Saved spatial-grid exact evaluation absent; missing, not imputed
