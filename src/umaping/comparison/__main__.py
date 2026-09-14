"""保存済み結果のみの比較CLI。python -m umaping.comparison"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .loading import load_results, DATASETS, ALIASES
from .analysis import summarize
from .figures import draw_figures
from .reporting import build_report, markdown_table


def safe_output(output, *inputs):
    output = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"出力先が入力と重複しています: {output}")
    if output.exists():
        raise FileExistsError(f"既存の出力先は上書きしません: {output}")
    return output


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, values):
    path.write_text(json.dumps(serializable(values), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_analysis(internal_runs_root, external_run_dir, output_dir, seed=0, resamples=10000):
    if resamples < 1:
        raise ValueError("resamplesは正数が必要です")
    repo = Path(__file__).resolve().parents[3]
    output = safe_output(output_dir, internal_runs_root, external_run_dir, repo / "runs", repo / "runs_external_baselines")
    print("内部・外部の保存schemaと測定値を検証しています。", flush=True)
    data = load_results(internal_runs_root, external_run_dir)
    if not any(r["success"] for r in data.rows.values()):
        raise ValueError("解析可能な成功結果がありません")
    result = summarize(data, seed, resamples)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    (output / "figures").mkdir()
    tables = {"main_recall15": result["main"], "secondary_metrics": result["secondary"],
              "gains_vs_standard_umap": result["gains"], "paired_comparisons": result["paired"],
              "tail_metrics": result["tails"], "ablations": result["ablations"],
              "ablation_summary": result["ablation_aggregate"], "periphery_metrics": result["periphery"],
              "continuous_structure": result["continuous"]}
    paired = result["paired"]
    tables["paired_ours_vs_umap"] = paired[paired.baseline.eq("standard_umap")] if not paired.empty else pd.DataFrame()
    for name, table in tables.items():
        if not len(table.columns):
            table = pd.DataFrame(columns=["dataset", "method", "metric", "status"])
        table.to_csv(output / "tables" / (name + ".csv"), index=False, na_rep="NaN")
    (output / "tables/main_recall15.md").write_text(markdown_table(result["main"]) + "\n", encoding="utf-8")
    result["ranks"].to_csv(output / "tables/rank_panel.csv", na_rep="NaN")
    flat = result["frame"].copy()
    flat["metric_sources"] = flat.metric_sources.map(lambda v: json.dumps(v, ensure_ascii=False))
    flat.to_csv(output / "combined_long.csv", index=False, na_rep="NaN")
    availability = pd.DataFrame(data.availability)
    availability["selected_success"] = [data.rows[d, m]["success"] for d, m in zip(availability.dataset, availability.method)]
    availability.to_csv(output / "method_availability.csv", index=False, na_rep="NaN")
    save_json(output / "raw_metric_records.json", data.raw)
    print("比較図を作成しています。", flush=True)
    figures, skips, multik_panel = draw_figures(result, data, output / "figures")
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True)
    metadata = dict(timestamp=datetime.now(timezone.utc).isoformat(), git_commit=git.stdout.strip() or None,
                    seed=seed, resamples=resamples, included_datasets=list(DATASETS),
                    excluded_datasets=["twenty_newsgroups", "hong_ed"], aliases=ALIASES,
                    internal_runs_root=str(data.internal_root), external_run_dir=str(data.external_root),
                    sources=list(data.sources.values()), rank_panel=result["rank_info"],
                    secondary_rank_panels=result["secondary_rank_info"], multi_k_panel=multik_panel,
                    gain_summary=result["gain_summary"], ablations=result["ablation_summary"],
                    friedman=result["friedman"], figures=figures,
                    notes=data.notes + result["notes"] + skips,
                    validation="保存ファイルを読み取って集計。学習・再推論・元結果への書込みなし。")
    save_json(output / "metadata.json", metadata)
    (output / "report.md").write_text(build_report(result, metadata), encoding="utf-8")
    print(f"解析完了: {output}\nレポート: {output / 'report.md'}\n主表: {output / 'tables/main_recall15.csv'}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--external-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    run_analysis(**vars(args))


if __name__ == "__main__":
    main()
