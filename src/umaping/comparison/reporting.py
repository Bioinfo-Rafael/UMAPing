"""日本語レポート。結論は読み込んだ測定値から組み立てる。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fmt(value, digits=4):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(frame):
    if frame.empty:
        return "該当する測定値はありません。"
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def build_report(result, metadata):
    frame, gains = result["frame"], result["gains"]
    summary = result["gain_summary"]
    full = frame[frame.success & frame.scope.eq("full")]
    coverage = full.groupby("method").agg(n_datasets=("dataset", "nunique"), n_recall15=("recall_at_15", "count")).reset_index()
    lines = ["# UMAPing：保存済み結果の統合比較", "",
             f"解析日時: {metadata['timestamp']}。解析コード: `{metadata['git_commit']}`。seed: {metadata['seed']}。",
             "学習・推論・前処理は再実行していません。数値は保存済みの測定結果、またはその統計集計です。", "",
             "## 1. データ範囲・手法の利用可否", "", markdown_table(coverage), "",
             "利用不可・失敗は欠測として扱い、ゼロや敗北に置き換えません。主表は全queryの評価だけです。",
             "一部queryの厳密斥力診断とMonte Carlo斥力診断は、別手法として診断・ablation表に保持しています。",
             "詳細: [利用可否](method_availability.csv)、[全測定値](combined_long.csv)。", "",
             "## 2. 主結果：Neighborhood Recall@15", "",
             "高次元と2次元の双方で、各queryから固定reference集合に対する15近傍を比較します。高いほど良い指標です。",
             "[主表](tables/main_recall15.md)にはdataset別Recall、利用可能dataset上の平均・中央値、固定panelの順位を示します。", "",
             markdown_table(result["main"][[c for c in ("method", "mean_recall15", "median_recall15", "mean_rank", "n_datasets", "n_rank_datasets", "wins", "top2") if c in result["main"]]]), "",
             f"順位panel: {', '.join(result['rank_info']['datasets']) or '構成不可'}。{result['rank_info']['rule']}。",
             "値の平均に使うn_datasetsと、同じ手法群・dataset群で順位を平均するn_rank_datasetsは異なる場合があります。", "",
             "## 3. standard UMAPに対する改善", "",
             f"比較可能な{summary['n_datasets']}データセットで、oursは{summary['wins']}勝・{summary['losses']}敗・{summary['ties']}同点でした。",
             f"絶対差の平均={fmt(summary['mean_absolute_gain'])}、中央値={fmt(summary['median_absolute_gain'])}。",
             f"相対差の平均={fmt(summary['mean_relative_gain'])}、中央値={fmt(summary['median_relative_gain'])}、最小={fmt(summary['min_relative_gain'])}、最大={fmt(summary['max_relative_gain'])}。相対差1は100%改善です。",
             f"dataset単位の片側正確二項検定: n={summary['sign_test_n']}、p={fmt(summary['sign_test_p'])}（同点は除外）。",
             "これはquery単位の有意性ではありません。pancreasの補正あり・なしは関連条件であり、datasetの独立性仮定には制約があります。",
             markdown_table(gains[gains.method.eq("ours")]), "",
             "## 4. query単位の対応統計", "",
             "同じquery IDと由来を確認できる比較だけを使用します。Recall・NDCGはours−baseline、local displacementはbaseline−oursとし、正ならoursが良い方向です。",
             f"対応bootstrapは{metadata['resamples']}回、両側符号反転検定も原則同数です。非ゼロ差が16件以下なら全符号の正確検定を行います。",
             "CIは平均差のpercentile 95%区間です。主仮説はours vs standard UMAPのRecall@15で、dataset間のp値にHolm補正を適用します。他は指標別の探索的ファミリーとして補正します。", ""]
    paired = result["paired"]
    if not paired.empty:
        primary = paired[paired.family.eq("primary_recall15_ours_vs_standard")]
        lines += [markdown_table(primary[[c for c in ("dataset", "n_pairs", "ours_mean", "baseline_mean", "mean_improvement", "ci_low", "ci_high", "fraction_ours_better", "p_value", "p_holm") if c in primary]]), ""]
        if "ci_low" in primary:
            supported = primary[(primary.ci_low > 0) & (primary.p_holm < .05)]
            lines.append("この測定条件で正の差を支持する結果（CI下限>0かつHolm p<0.05）: " + (", ".join(supported.dataset) or "なし") + "。")
    else:
        lines.append("対応値が揃わずquery単位の検定は利用できません。")
    lines += ["[全対応比較](tables/paired_comparisons.csv)。単一run内のquery再標本化であり、再学習seed間の不確実性や細胞・患者間の依存を推定したCIではありません。", "",
              "## 5. 裾・大きなquery配置誤差", "",
              "Recallは低い順の5%・1%、local displacementは高い順の5%・1%を『worst』とします。件数はceilで切り上げ、最低1件です。低Recallをrecall deficitに反転していません。"]
    tails = result["tails"]
    if not tails.empty:
        selected = tails[tails.method.isin(["ours", "standard_umap"]) & tails.metric.eq("recall_at_15")]
        lines += [markdown_table(selected[[c for c in ("dataset", "method", "mean", "median", "p10", "p5", "worst_5pct_mean", "worst_1pct_mean") if c in selected]])]
        pivot = selected.pivot(index="dataset", columns="method", values="worst_5pct_mean")
        if {"ours", "standard_umap"}.issubset(pivot):
            delta = (pivot.ours - pivot.standard_umap).dropna()
            lines.append(f"worst-5% Recallは、対応する{len(delta)}dataset中oursが{int((delta>1e-12).sum())}件で高く、{int((delta < -1e-12).sum())}件で低い結果でした。")
    lines += ["詳細: [tail表](tables/tail_metrics.csv)。これらはquery配置の診断であり、失敗を区切る普遍的な閾値を定義したものではありません。", "",
              "## 6. Ablation", "",
              "[dataset別ablation](tables/ablations.csv)、[集約](tables/ablation_summary.csv)。Δはmethod−oursのRecallです。",
              "subset診断は同じqueryのoursと比較できる場合だけΔを出します。Monte Carlo診断とexact診断は混同しません。"]
    retrieval = result["ablation_summary"].get("retriever", {})
    rep = result["ablation_summary"].get("repulsion", {})
    lines += [f"Retriever gap（oracle neighbors−ours）の平均絶対値={fmt(retrieval.get('mean_absolute_gap'))}、最大絶対値={fmt(retrieval.get('max_absolute_gap'))}、n={retrieval.get('n_datasets', 0)}。",
              "Recallでoursがno_repulsionを上回るdataset: " + (", ".join(rep.get("ours_better", [])) or "なし") + "。",
              "no_repulsionがoursを上回るdataset: " + (", ".join(rep.get("no_repulsion_better", [])) or "なし") + "。",
              f"ours−no_repulsionの平均Recall差={fmt(rep.get('mean_ours_minus_no_repulsion'))}。反発が常にRecallを改善するとは仮定しません。", "",
              "## 7. OOS periphery・反発診断", "",
              "periphery_percentileは同ラベルreferenceの中心からの距離に対する百分位です。90/95以上の割合を報告します。",
              "repulsion_accumulation_scoreは0〜1なので、極端値の閾値には0.90/0.95を使います。90/95をそのまま適用しません。",
              "極端な周縁配置の頻度が低い方向を良い方向として扱いますが、普遍的な埋め込み品質指標ではありません。",
              "[periphery表](tables/periphery_metrics.csv)ではours、standard UMAP、no_repulsionなどの実測値を比較できます。", ""]
    periphery = result["periphery"]
    if not periphery.empty:
        selected = periphery[periphery.method.isin(["ours", "standard_umap", "no_repulsion"])]
        for metric, group in selected.groupby("metric"):
            pivot = group.pivot(index="dataset", columns="method", values="fraction_ge_95")
            for baseline in ("standard_umap", "no_repulsion"):
                if {"ours", baseline}.issubset(pivot):
                    delta = (pivot[baseline] - pivot.ours).dropna()
                    lines.append(f"{metric}: 極端値頻度は{baseline}との共通{len(delta)}dataset中、oursが{int((delta>1e-12).sum())}件で低く、{int((delta < -1e-12).sum())}件で高い結果です。")
    lines += ["", "## 8. 外部baselineとの比較", ""]
    for method in ("parametric_umap", "numap", "paramrepulsor"):
        pivot = full[full.method.isin(["ours", method])].pivot(index="dataset", columns="method", values="recall_at_15")
        if {"ours", method}.issubset(pivot):
            delta = (pivot.ours - pivot[method]).dropna()
            lines.append(f"{method}: 共通{len(delta)}datasetでoursのRecallが高い={int((delta>1e-12).sum())}件、低い={int((delta < -1e-12).sum())}件、平均差={fmt(delta.mean())}。")
        else:
            lines.append(f"{method}: 比較可能な成功値がなく、優劣は判定できません。")
    lines += ["外部手法のper-query CSVがあれば対応検定に使います。集約値しかない場合は推論を再実行せず、query単位の値を補いません。",
              "公式OOS-UMAPリポジトリには、このdataset群を忠実に評価できる汎用の学習・fit/transform実行部分が揃っていませんでした。独自再構築による実装上の曖昧さを避けるため、利用不可として保持しています。これは手法自体の失敗を意味しません。", "",
              "## 9. 速度・品質の関係", "",
              "正の実測query時間がある手法だけを図に載せます。欠測を0秒として扱いません。",
              "内部oursは単一query呼出しの平均、外部手法は一括transform時間÷query数です。測定環境・Python/依存versionも異なるため、厳密な速度倍率の主張には使えません。",
              "内部standard UMAPのfit_time_secondsは既存実装上fitとquery変換を含みます。外部fit時間と単純な同一条件比較はできません。", "",
              "## 10. 制約と利用できない比較", "",
              "測定で支持される事項は上の実測差と条件付きCIに限定します。勝敗が混在する比較、欠測、subset診断を一般的な優越性の主張にまとめません。",
              "主表の集約段階と追加解析段階で再生成された埋め込みが違う場合、保存値を上書きせず両方の出典と差を記録します。",
              "補正済みpancreasはscArchesのquery適応を含む元の表現を再利用しています。厳密なreference-only前処理と同じ条件ではありません。",
              "連続構造はloaderが保存した数値のtime/day列と保存座標が揃う場合だけ評価します。stateやsample_labels等の文字列から順序を推測しません。",
              "[連続構造表](tables/continuous_structure.csv)、[解析メタデータ](metadata.json)、[原測定記録](raw_metric_records.json)。", ""]
    lines += ["- " + str(note) for note in metadata["notes"]]
    friedman = result["friedman"]
    if friedman["status"] == "skipped":
        lines.append("Friedman検定は実行せず順位要約のみとしました: " + friedman["reason"])
    else:
        lines.append(f"探索的Friedman検定: 統計量={fmt(friedman['statistic'])}、p={fmt(friedman['p_value'])}。{friedman['note']}")
    return "\n\n".join(lines) + "\n"
