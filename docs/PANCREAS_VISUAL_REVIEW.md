# 保存済みpancreas結果の再作図

`scripts/pancreas_visual_review.py` は既定では保存ファイルだけを読みます。
ユーザーが追加承認した `--regenerate` モードでは、凍結UMAPingのquery再推論・通常UMAPのfit/transform・今回の座標のRecall再評価を行います。
どちらのモードもニューラルネットワーク・前処理の再学習やダウンロードは行いません。

## 座標が未保存の場合の再生成（承認済み）

```bash
SOURCE=/home/suzuki/Learn/UMAPing
CODE=/home/suzuki/Learn/UMAPing-pancreas-review
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$SOURCE/.venv/bin/python" -u "$CODE/scripts/pancreas_visual_review.py" \
  --run "$SOURCE/runs/pancreas_batch_corrected/main" --repo-root "$SOURCE" \
  --regenerate --device cpu --threads 1 --max-seconds 7200
```

- source runは読み取り専用。新しい出力の `tables/regenerated/` に座標・ID・新しいmetrics・実行記録を保存。
- PreparedDatasetをそのまま使い、scVI/scArches、retriever、Spectral、反発ネット、reference trajectoryは再学習・再計算しない。
- 通常UMAPは既存の `run_standard_umap` と同じ設定・seedでreferenceにfit、queryをtransform。ライブラリ版も記録。
- IDは再生成前に固定し、既知の入力行との対応から保存する。以前の座標に行番号を推測で付け足す処理ではない。
- 保存済み旧metricsはID・細胞型照合のみに使い、今回の座標図には再計算したRecall@5/15を使う。旧数値との一致は保証しない。
- 再生成中は30秒ごとのheartbeat、50queryごとの進捗と途中保存。既定2時間の上限を超えたら失敗として停止。
- 成功時は `complete_regenerated_required_plots`。マーカー発現は対応表が無ければ理由付き省略。
- 今回のモードは新しい時刻名で出力するため、前回の部分結果・旧runは上書きしない。
- `--regenerate` と外部embedding/metrics/prepared指定は併用禁止。異なる評価を混ぜないため。

## 実行

```bash
SOURCE=/home/suzuki/Learn/UMAPing
CODE=/home/suzuki/Learn/UMAPing-pancreas-review
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$SOURCE/.venv/bin/python" -u "$CODE/scripts/pancreas_visual_review.py" \
  --run "$SOURCE/runs/pancreas_batch_corrected/main" --repo-root "$SOURCE"
```

出力は `Final_analysis/pancreas_batch_corrected_visual_review_<UTC timestamp>/`。
既存ディレクトリには書き込みません。終了コードは0=必須座標図完了、2=座標不足/対応未検証でmetrics図のみ、1=処理失敗。
0でもマーカー図は対応する発現データがなければ省略します。
図が保存されるたびにPNG/PDFのファイル名を表示します。

## 最初に確認するファイル

- `config.yaml`: pancreasかつbatch_correction=true。
- `metadata.json`: teacherの記録。`repulsion_teacher: uniform_mc` ならUniform MCと表記。
- `metrics/advanced_per_query.csv`: `standard_umap` / `ours_full` のquery_index、label、Recall@5/15。
- `cache/prepared_dataset.npz`: reference/queryの特徴行数、celltype、tech。
- `embeddings/standard_umap.npz` と `embeddings/ours_full.npz`: 以下のID付き保存形式。

別の場所に保存済みの場合は `--prepared`、`--embedding-dir`、または
`--standard-embedding /実在path.npz --umaping-embedding /実在path.npz` で指定できます。
候補が複数あれば自動選択せず停止します。

既存runの旧pipelineはPNGとper-query metricsを保存し、query座標を保存しない版があります。
PNGから正確な座標や細胞IDを復元しません。入力不足時は、入力一覧と不足理由をREADMEへ残し、実データの数値図だけを作成します。

## 座標と対応関係

各embedding NPZには以下が必要です。

| key | 内容 |
|---|---|
| reference | Nr×2 の保存済み最終reference座標 |
| query | Nq×2 の保存済み最終query座標 |
| reference_index | 各reference座標の、PreparedDataset reference内の明示的な行ID |
| query_index | 各query座標の、PreparedDataset query内の明示的な行ID |
| prepared_dataset_sha256 | 対応するPreparedDatasetファイルのSHA256（scalar文字列） |

hashは `<embedding.npz>.json` の同名キーにも対応します。
IDは重複禁止・全splitの完全な集合一致を要求し、ID順でjoinします。座標の行順は自由です。
同じquery IDのcelltypeがmetricsとPreparedDatasetで一致することを確認します。
**IDの無い既存配列に、長さだけを見てarangeのIDを付け足してはいけません。**
元の保存処理・対応情報でIDと入力datasetを証明できない場合、座標図は作成しません。

teacher未記録時は `teacher unrecorded` と表示します。既存実行のteacherを別の記録で確認できた場合のみ、
`--teacher uniform_mc --teacher-evidence '実際に確認した保存記録の場所と内容'` を使えます。
これは探索・教師の変更ではなく名称のprovenanceです。metadataと矛盾する指定は拒否します。

## マーカー発現

対応IDを検証済みのCSVがある場合のみ次を追加します。

```bash
--expression-csv /実在path/markers.csv \
--expression-kind counts \
--expression-description '発現データの出典と対応IDを作った根拠'
```

CSVの列: `split,index,INS,GCG,SST,KRT19,total_counts`。
`split` はreference/query、indexは同じPreparedDatasetに対応する明示的ID。
`<markers.csv>.json` に `prepared_dataset_sha256` を保存して入力を結び付けることが必要です。
countsは全遺伝子のtotal_countsで1万に正規化してlog1p。4遺伝子の和をlibrary sizeに使いません。
既に同じ処理をした値は `--expression-kind log1p_cptt` とし、total_counts列は不要です。
各遺伝子は全手法・reference/queryで0〜全細胞の最大値の共通色域です。未確認の発現や潜在特徴を発現値として扱いません。

## 図・表と検証

1. celltype: 3行（reference/query/overlay）×2手法、固定範囲、完全な図外凡例。
2. tech: 同じ構成。実際のtech名とqueryのtech別点数を出力。
3. query>=100の全celltypeを1型1ページ。左右で表示幅を一致させ、任意の切り取りなし。
4. Recall@5/15は共通0–1、paired差は同じquery IDの値を両座標に描画。
5. 全celltypeの平均Recallドット図と点数。座標不足でもID検証済みmetricsから作成可能。
6. 発現がある場合はINS/GCG/SST/KRT19を1遺伝子1ページ。

使用元CSV、IDで整列した座標/注釈、palette、件数、選択対象、SHA256をtablesに保存。
PNG/PDFと確認用contact sheetを作成し、描画後の凡例境界を検査します。
自動検査と実画像の目視確認は別です。

```bash
python -m unittest discover -s scripts -p test_pancreas_visual_review.py -v
```

テストはシャッフルしたID、重複、集合不一致、hash不一致、label不一致、ID不足、上書き拒否を確認。
全レイアウトのsynthetic fixtureは実データの結果とは別の一時ディレクトリで検証します。
