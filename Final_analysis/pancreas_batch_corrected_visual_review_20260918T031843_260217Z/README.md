# Pancreas batch-corrected visual review

状態: **partial_missing_or_unverified_coordinates**。
入力run: `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main`。
UMAPing teacher: **unrecorded**。根拠: Run metadata does not record teacher; legacy pipeline default is Uniform MC, but not asserted as verified.
queryの測定技術: 未検証（PreparedDatasetの実ラベルをまだ読み取れていません）。config上の予定値を実測値とは扱いません。

## 対応関係の検証
per-query metricsはmethod別のquery_indexで一対一結合し、ID集合・重複・celltypeの一致を検証。座標は明示的なreference_index/query_indexでPreparedDatasetのラベルへ結合。長さが同じという理由だけでは結合しません。
indexはそのPreparedDataset内の行IDです。元の生物学的cell barcodeとは区別します。座標ファイルまたはsidecarに保存されたPreparedDatasetのSHA256も必須として照合します。IDの無い旧形式座標からIDを生成しません。
teacherが未記録なら未確認と表記します。旧pipelineの既定はUniform MCですが、この情報だけで既存runをFitGridまたは検証済みUniformとは表記しません。

## 図一覧
- [05_celltype_recall_dotplot PNG](figures/05_celltype_recall_dotplot.png) / [PDF](figures/05_celltype_recall_dotplot.pdf): 全細胞型の平均Recall@5/15。IDで対応付けた同じquery。

## 観察できる差
座標パネルは未作成のため、配置・重なりの違いは観察できません。

## 数値で裏付けられる差
- Recall@5: 通常UMAP 4.035%、UMAPing 7.980%、paired平均差 +3.945 percentage points（同じquery 4,679点）。
- Recall@15: 通常UMAP 9.032%、UMAPing 11.698%、paired平均差 +2.666 percentage points（同じquery 4,679点）。
平均値は保存済み評価の記述統計です。新しい統計検定や密度指標は追加していません。

## まだ言えないこと
クラスターの位置、回転、コンパクトさだけから生物学的優劣・密度改善は判断できません。拡大図はquery>=100の全細胞型を選び、良い結果の細胞型だけを選択しません。表示幅を左右で一致させても、手法間の座標スケールが生物学的に等しいことを保証しません。
batch-corrected特徴空間での近傍保持と、生物学的なbatch除去・celltype保存は別の評価です。celltypeやtechの混ざり方だけから因果的な改善は主張しません。
教師の比較実験ではないためUniform対FitGridの効果は判断できません。

## マーカー発現
座標入力が不足しており未実行。

## 不足・省略
- {"item": "01/02/03/04/06 coordinate panels", "reason": "Missing saved inputs", "paths": ["/Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main/cache/prepared_dataset.npz", "/Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main/embeddings/standard_umap.npz", "/Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main/embeddings/ours_full.npz"]}

## 入力ファイル
- `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main/config.yaml` (SHA256 `e3a92f021f6a28d41f1f86071f9fc10d6c80b9340b56caefe5b94794f24d112e`)
- `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main/metadata.json` (SHA256 `8f8d109e4a653eaa8c6ba257362d7fa0a458d0254efb96300046fa01326286c3`)
- `/Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main/metrics/advanced_per_query.csv` (SHA256 `83a5f0fa18d0d671da715ab67eec45d9e76c91bcad14bed1485d7d85fef1cdb4`)

全候補一覧: [input_inventory.csv](tables/input_inventory.csv)。

## 再実行コマンド
```bash
python /Users/cls-lab/Git/PersonalDev/UMAPing-pancreas-review/scripts/pancreas_visual_review.py --run /Users/cls-lab/Git/PersonalDev/UMAPing/runs/pancreas_batch_corrected/main --repo-root /Users/cls-lab/Git/PersonalDev/UMAPing-pancreas-review
```
同名出力は拒否します。再実行時は--outputを省略するか、新規ディレクトリを指定してください。

## 画像確認
図外凡例のcanvas内収まりを描画後に検査。PNG/PDFを出力し、tables/qa_contact_sheet_*.pngを作成。実画像を確認した記録は別途保存してください。自動検査を目視確認とは呼びません。

実画像確認済み: このrunの細胞型別Recall図は、全13細胞型・点数・凡例・文字・同値点の重なりを確認しました。全座標図形式は別のsynthetic fixtureで16図を描画して画像を確認しましたが、実pancreasの座標図が完成したことは意味しません。検証記録は [validation.json](tables/validation.json)。

追加提供された保存ファイル一覧はembryoid_bodyのみでpancreasを含まないため、今回の座標対応根拠として使用していません。
