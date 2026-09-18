# Pancreas batch-corrected visual review

状態: **complete_regenerated_required_plots**。
**座標の再生成モードです。** 通常UMAPは保存済みreference特徴でfitしてqueryをtransformし、UMAPingは既存の凍結チェックポイントからqueryのみを再推論する構成です。reference軌道は変更しません。完了段階と成否は `tables/regenerated/regeneration.json` を参照してください。
前処理・scVI/scArches・retriever・Spectral・反発ネットの再学習はありません。Recallは今回の座標に対して再評価し、以前のmetricsを今回の座標図に流用していません。
IDは再計算前にPreparedDatasetの各行へ割り当て、推論ループまたはfit/transformの入力出力対応から保存しました。旧座標の行順を推測して付与したIDではありません。

入力run: `/home/suzuki/Learn/UMAPing/runs/pancreas_batch_corrected/main`。
UMAPing teacher: **unrecorded**。根拠: Run metadata does not record teacher; legacy pipeline default is Uniform MC, but not asserted as verified.
queryの測定技術: celseq2: 2,285 query, smartseq2: 2,394 query

## 対応関係の検証
per-query metricsはmethod別のquery_indexで一対一結合し、ID集合・重複・celltypeの一致を検証。座標は明示的なreference_index/query_indexでPreparedDatasetのラベルへ結合。長さが同じという理由だけでは結合しません。
indexはそのPreparedDataset内の行IDです。元の生物学的cell barcodeとは区別します。座標ファイルまたはsidecarに保存されたPreparedDatasetのSHA256も必須として照合します。IDの無い旧形式座標からIDを生成しません。
teacherが未記録なら未確認と表記します。旧pipelineの既定はUniform MCですが、この情報だけで既存runをFitGridまたは検証済みUniformとは表記しません。

## 図一覧
- [01_overview_by_celltype PNG](figures/01_overview_by_celltype.png) / [PDF](figures/01_overview_by_celltype.pdf): 3行×2列、celltype色。同じ手法の座標範囲は全行固定。
- [02_overview_by_tech PNG](figures/02_overview_by_tech.png) / [PDF](figures/02_overview_by_tech.pdf): 3行×2列、tech色。同じ手法の座標範囲は全行固定。
- [03_01_celltype_zoom_acinar PNG](figures/03_01_celltype_zoom_acinar.png) / [PDF](figures/03_01_celltype_zoom_acinar.pdf): acinar: query>=100の事前規則で対象。両手法の表示幅を統一。
- [03_02_celltype_zoom_activated_stellate PNG](figures/03_02_celltype_zoom_activated_stellate.png) / [PDF](figures/03_02_celltype_zoom_activated_stellate.pdf): activated_stellate: query>=100の事前規則で対象。両手法の表示幅を統一。
- [03_03_celltype_zoom_alpha PNG](figures/03_03_celltype_zoom_alpha.png) / [PDF](figures/03_03_celltype_zoom_alpha.pdf): alpha: query>=100の事前規則で対象。両手法の表示幅を統一。
- [03_04_celltype_zoom_beta PNG](figures/03_04_celltype_zoom_beta.png) / [PDF](figures/03_04_celltype_zoom_beta.pdf): beta: query>=100の事前規則で対象。両手法の表示幅を統一。
- [03_05_celltype_zoom_delta PNG](figures/03_05_celltype_zoom_delta.png) / [PDF](figures/03_05_celltype_zoom_delta.pdf): delta: query>=100の事前規則で対象。両手法の表示幅を統一。
- [03_06_celltype_zoom_ductal PNG](figures/03_06_celltype_zoom_ductal.png) / [PDF](figures/03_06_celltype_zoom_ductal.pdf): ductal: query>=100の事前規則で対象。両手法の表示幅を統一。
- [03_07_celltype_zoom_gamma PNG](figures/03_07_celltype_zoom_gamma.png) / [PDF](figures/03_07_celltype_zoom_gamma.pdf): gamma: query>=100の事前規則で対象。両手法の表示幅を統一。
- [04_query_recall_at_5_and_15 PNG](figures/04_query_recall_at_5_and_15.png) / [PDF](figures/04_query_recall_at_5_and_15.pdf): 保存済みRecall@5/15を共通0–1色域で表示。
- [05_celltype_recall_dotplot PNG](figures/05_celltype_recall_dotplot.png) / [PDF](figures/05_celltype_recall_dotplot.pdf): 全細胞型の平均Recall@5/15。IDで対応付けた同じquery。
- [06_paired_recall_difference_on_both_embeddings PNG](figures/06_paired_recall_difference_on_both_embeddings.png) / [PDF](figures/06_paired_recall_difference_on_both_embeddings.pdf): 同一queryの差を両座標へ表示。共通発散色域。

## 観察できる差
reference/query、celltype、tech、Recallおよび同一queryの差を分けて表示しました。配置・重なりは図から確認できますが、手法間の位置・回転自体を性能差とは解釈しません。

## 数値で裏付けられる差
- Recall@5: 通常UMAP 4.035%、UMAPing 7.980%、paired平均差 +3.945 percentage points（同じquery 4,679点）。
- Recall@15: 通常UMAP 9.032%、UMAPing 11.698%、paired平均差 +2.666 percentage points（同じquery 4,679点）。
平均値は今回再生成した座標の再評価です。旧実行との数値的一致は保証せず、ライブラリ版とseedを記録します。新しい統計検定や密度指標は追加していません。

## まだ言えないこと
クラスターの位置、回転、コンパクトさだけから生物学的優劣・密度改善は判断できません。拡大図はquery>=100の全細胞型を選び、良い結果の細胞型だけを選択しません。表示幅を左右で一致させても、手法間の座標スケールが生物学的に等しいことを保証しません。
batch-corrected特徴空間での近傍保持と、生物学的なbatch除去・celltype保存は別の評価です。celltypeやtechの混ざり方だけから因果的な改善は主張しません。
教師の比較実験ではないためUniform対FitGridの効果は判断できません。

## マーカー発現
省略: 対応IDと発現処理を確認できるマーカー発現表が指定されていない。PCA/scVI特徴量やcelltypeから発現量は復元しない。

## 不足・省略
- {"item": "07 marker expression", "reason": "省略: 対応IDと発現処理を確認できるマーカー発現表が指定されていない。PCA/scVI特徴量やcelltypeから発現量は復元しない。"}

## 入力ファイル
- `/home/suzuki/Learn/UMAPing/runs/pancreas_batch_corrected/main/config.yaml` (SHA256 `e3a92f021f6a28d41f1f86071f9fc10d6c80b9340b56caefe5b94794f24d112e`)
- `/home/suzuki/Learn/UMAPing/runs/pancreas_batch_corrected/main/metadata.json` (SHA256 `8f8d109e4a653eaa8c6ba257362d7fa0a458d0254efb96300046fa01326286c3`)
- `/home/suzuki/Learn/UMAPing/Final_analysis/pancreas_batch_corrected_visual_review_20260918T033600_141054Z/tables/regenerated/regeneration.json` (SHA256 `939eed1d52c0789e6ec64ecfd69e8611b3499fed42fc5ec63206b60ffd5514af`)
- `/home/suzuki/Learn/UMAPing/Final_analysis/pancreas_batch_corrected_visual_review_20260918T033600_141054Z/tables/regenerated/advanced_per_query.csv` (SHA256 `01055b8312f00fc430b77332f2958cff0ff4f5ffd1f1db642cc6b0941059ce92`)
- `/home/suzuki/Learn/UMAPing/runs/pancreas_batch_corrected/main/cache/prepared_dataset.npz` (SHA256 `cf5cadac5b148e94fd414163098366c2b4fe1ba19894710571376dca4be859de`)
- `/home/suzuki/Learn/UMAPing/Final_analysis/pancreas_batch_corrected_visual_review_20260918T033600_141054Z/tables/regenerated/embeddings/standard_umap.npz` (SHA256 `64049c99e48d5f7c3a0c289c014377c6d6c80cb03c50b09497cd89ef0d222faf`)
- `/home/suzuki/Learn/UMAPing/Final_analysis/pancreas_batch_corrected_visual_review_20260918T033600_141054Z/tables/regenerated/embeddings/ours_full.npz` (SHA256 `78f376c683257f6498ec79db0c152a14f3617e9f3854ae199d079500c61494b8`)

全候補一覧: [input_inventory.csv](tables/input_inventory.csv)。

## 再実行コマンド
```bash
python /home/suzuki/Learn/UMAPing-pancreas-review/scripts/pancreas_visual_review.py --run /home/suzuki/Learn/UMAPing/runs/pancreas_batch_corrected/main --repo-root /home/suzuki/Learn/UMAPing --regenerate --device cpu --threads 1 --max-seconds 7200
```
同名出力は拒否します。再実行時は--outputを省略するか、新規ディレクトリを指定してください。

## 画像確認
図外凡例のcanvas内収まりを描画後に検査。PNG/PDFを出力し、tables/qa_contact_sheet_*.pngを作成。実画像を確認した記録は別途保存してください。自動検査を目視確認とは呼びません。

## 保存形式の変更
ユーザー指定により、生成後のPDFを削除しました。図はPNGで保存しています。
