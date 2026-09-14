"""完了済みrun専用の外部比較。親プロセスは標準ライブラリのみで動作する。"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

COMPLETED_DATASETS = (
    "coil20", "coil100", "pancreas", "pancreas_batch_corrected",
    "fashion_mnist", "mnist_oos", "organoid", "embryoid_body",
)
BASELINES = ("parametric_umap", "numap", "paramrepulsor", "oos_umap")
METHOD_NAMES = {
    "parametric_umap": "parametric_umap", "numap": "numap_sep_spectralnet",
    "paramrepulsor": "param_repulsor", "oos_umap": "oos_umap",
}
OOS_COMMIT = "5015dc11c92b444530b9237533ea927427712311"
OOS_REASON = (
    "公式OOS_UMAPの確認済みコミットにはnetwork_sig.py等のネットワーク定義はあるが、"
    "論文のCE/MSE/CEMSE学習・OOS最適化の実行コードと汎用fit/transform APIがない。"
    "公式手法を独自再実装せず、今回の比較では利用不可とする。"
)
SOURCES = {
    "parametric_umap": {"package": "umap-learn", "version_pin": "0.5.12",
                        "repository": "https://github.com/lmcinnes/umap"},
    "numap": {"package": "numap", "version_pin": "0.2.3",
              "repository": "https://github.com/shaham-lab/NUMAP"},
    "paramrepulsor": {"package": "parampacmap", "version_pin": "0.1.0",
                     "repository": "https://github.com/hyhuang00/ParamRepulsor"},
    "oos_umap": {"repository": "https://github.com/tariqul-islam/OOS_UMAP",
                 "inspected_commit": OOS_COMMIT, "paper": "https://arxiv.org/abs/2606.04451"},
}


def validate_output(repo_root: Path, run_dir: Path, output: Path) -> Path:
    """祖先・子孫・symlinkも確認。既存runsと入力runへの出力を拒否する。"""
    output = output.resolve()
    for protected in ((repo_root / "runs").resolve(), run_dir.resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"出力先が既存runと重複しています: {output}")
    return output


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def base_record(dataset: str, baseline: str) -> dict:
    return dict(dataset=dataset, baseline=baseline, method=METHOD_NAMES[baseline],
                available=False, success=False, status="pending", reason=None,
                seed=None, n_reference=None, n_query=None, k=None,
                fit_time_seconds=None, query_transform_time_seconds=None,
                mean_query_latency_seconds=None, reference_transform_time_seconds=None,
                python_version=platform.python_version(), source=SOURCES[baseline], metrics={})


def run_adapter(baseline, prepared, cfg, device):
    # 既存の薄い公式APIアダプタを再利用。学習はreferenceだけに対して行う。
    from umaping.experiments.baselines import (
        run_numap_baseline, run_param_repulsor_baseline, run_parametric_umap_baseline,
    )
    if baseline == "parametric_umap":
        return run_parametric_umap_baseline(prepared, cfg)
    if baseline == "numap":
        return run_numap_baseline(prepared, cfg, device=device)
    if baseline == "paramrepulsor":
        return run_param_repulsor_baseline(prepared, cfg)
    raise ValueError(baseline)


def evaluate_result(prepared, cfg, outcome, output: Path) -> dict:
    import numpy as np
    import pandas as pd
    from umaping.evaluation.embedding import evaluate_embedding
    from umaping.evaluation.advanced import multi_k_recall_and_ndcg
    from umaping.evaluation.plotting import plot_embedding
    from umaping.evaluation.labels import primary_label

    for name, emb, features in (
        ("reference", outcome.reference_embedding, prepared.reference_features),
        ("query", outcome.query_embedding, prepared.query_features),
    ):
        if emb.shape != (len(features), cfg.umap.embedding_dim) or not np.isfinite(emb).all():
            raise ValueError(f"{name}埋め込みのshapeまたは有限性が不正です")
    ref_labels = primary_label(prepared.reference_labels, cfg.dataset.name)
    query_labels = primary_label(prepared.query_labels, cfg.dataset.name)
    metrics, per_query = evaluate_embedding(
        prepared.query_features, prepared.reference_features,
        outcome.query_embedding, outcome.reference_embedding, k=cfg.eval.k,
        reference_labels=ref_labels, query_labels=query_labels,
        trustworthiness_n_neighbors=cfg.eval.trustworthiness_n_neighbors,
        mean_query_latency_seconds=outcome.mean_query_latency_seconds,
    )
    # 実データでは全kが利用可能。小さいmockでは不可能なkを捏造しない。
    ks = tuple(k for k in (5, 10, 15, 30) if k <= len(prepared.reference_features))
    recall, ndcg = multi_k_recall_and_ndcg(
        prepared.query_features, prepared.reference_features,
        outcome.query_embedding, outcome.reference_embedding, ks=ks,
    )
    values = metrics.to_dict()
    values.update({f"recall_at_{k}": float(v.mean()) for k, v in recall.items()})
    values.update(ndcg=float(ndcg.mean()), ndcg_k=max(ks))
    frame = pd.DataFrame({"query_index": np.arange(len(per_query)),
                          "neighborhood_recall_at_k": per_query, "ndcg": ndcg,
                          **{f"recall_at_{k}": v for k, v in recall.items()}})
    frame.to_csv(output / "advanced_per_query.csv", index=False)
    np.savez_compressed(output / "embeddings.npz", reference=outcome.reference_embedding,
                        query=outcome.query_embedding)
    plot_embedding(outcome.reference_embedding, prepared.reference_labels,
                   output / "embedding.png", title=outcome.name,
                   query_embedding=outcome.query_embedding,
                   query_label_sets=prepared.query_labels)
    return values


def run_one(dataset: str, baseline: str, run_dir: Path, output: Path,
            repo_root: Path, seed: int | None = None, device: str = "auto",
            unavailable_reason: str | None = None) -> dict:
    if dataset not in COMPLETED_DATASETS or baseline not in BASELINES:
        raise ValueError("対象外のdataset/baselineです")
    run_dir, repo_root = run_dir.resolve(), repo_root.resolve()
    output = validate_output(repo_root, run_dir, output)
    output.mkdir(parents=True, exist_ok=False)  # 新規結果も暗黙には上書きしない。
    row = base_record(dataset, baseline)
    row.update(run_dir=str(run_dir), executable=sys.executable, code_commit=git_head(repo_root))
    write_json(output / "result.json", row)
    previous_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        # 公式ライブラリが相対パスで保存するログ/checkpointも新しい出力側へ閉じ込める。
        os.chdir(output)
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import numpy as np
        from umaping.config import Config
        from umaping.data.preprocessing import load_prepared_dataset
        from umaping.utils.seed import set_seed
        from umaping.utils.device import resolve_device
        from umaping.experiments.baselines import BaselineUnavailable

        cfg_path = run_dir / "config.yaml"
        cache_path = run_dir / "cache" / "prepared_dataset.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"既存前処理キャッシュがありません（再作成は行いません）: {cache_path}")
        cfg = Config.load(cfg_path)
        expected = "pancreas" if dataset == "pancreas_batch_corrected" else dataset
        if cfg.dataset.name != expected:
            raise ValueError(f"datasetと入力runが一致しません: {dataset} / {cfg.dataset.name}")
        if seed is not None:
            cfg.seed = seed
        prepared = load_prepared_dataset(cache_path)
        if prepared.input_dim != cfg.dataset.input_dim:
            raise ValueError("キャッシュと保存済みconfigの入力次元が一致しません")
        nr, nq = len(prepared.reference_features), len(prepared.query_features)
        row.update(seed=cfg.seed, k=cfg.eval.k, n_reference=nr, n_query=nq,
                   input_dim=prepared.input_dim, config_sha256=sha256(cfg_path),
                   prepared_dataset_sha256=sha256(cache_path), config=dataclasses.asdict(cfg))
        if nr < max(5, cfg.eval.k) or nq < 1:
            raise ValueError("評価kに対してreference/queryの件数が不足しています")
        if not 1 <= cfg.eval.trustworthiness_n_neighbors < nr / 2:
            raise ValueError("trustworthinessのkはreference件数の半分未満が必要です")
        for features in (prepared.reference_features, prepared.query_features):
            if features.ndim != 2 or features.shape[1] != prepared.input_dim or not np.isfinite(features).all():
                raise ValueError("前処理キャッシュのshapeまたは有限性が不正です")
        if (run_dir / "metadata.json").is_file():
            row["original_run_metadata"] = json.loads((run_dir / "metadata.json").read_text())
        row["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        row["preprocessing"] = {
            "input": "既存PreparedDatasetをそのまま使用。外側の前処理の再fitなし。",
            "internal": ("ParamRepulsor: apply_pca=False, apply_scale=None" if baseline == "paramrepulsor"
                         else "NUMAP: 公式GrEASEによるreference内のスペクトル学習あり" if baseline == "numap"
                         else "追加のPCA/特徴量前処理なし"),
            "batch_corrected_caveat": (
                "scArchesのquery適応を含む既存表現。厳密なreference-only前処理ではない。"
                if cfg.dataset.params.get("batch_correction") else None),
        }
        if baseline == "oos_umap" or unavailable_reason:
            row.update(status="unavailable", reason=unavailable_reason or OOS_REASON)
        else:
            pinned = SOURCES[baseline]
            actual = importlib.metadata.version(pinned["package"])
            row["package_version"] = actual
            if actual != pinned["version_pin"]:
                raise RuntimeError(f"公式packageのバージョン不一致: {actual} != {pinned['version_pin']}")
            set_seed(cfg.seed)
            resolved = resolve_device(device)
            row["device"] = str(resolved)
            row["available"] = True
            write_json(output / "result.json", {**row, "status": "running"})
            outcome = run_adapter(baseline, prepared, cfg, resolved)
            if baseline == "parametric_umap" and "tensorflow" in sys.modules:
                row["tensorflow_visible_devices"] = [
                    d.name for d in sys.modules["tensorflow"].config.get_visible_devices()]
                row["device"] = "tensorflow（visible_devicesを参照）"
            if isinstance(outcome, BaselineUnavailable):
                row.update(status="unavailable", available=False, reason=outcome.reason)
            else:
                row.update(fit_time_seconds=outcome.fit_time_seconds,
                           mean_query_latency_seconds=outcome.mean_query_latency_seconds,
                           query_transform_time_seconds=(outcome.mean_query_latency_seconds * nq
                                                         if outcome.mean_query_latency_seconds is not None else None),
                           adapter_metadata=outcome.extra)
                row["reference_transform_time_seconds"] = outcome.extra.get("reference_transform_time_seconds")
                row["metrics"] = evaluate_result(prepared, cfg, outcome, output)
                row.update(status="success", success=True)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        row.update(status="unavailable", available=False, reason=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc())
    except Exception as exc:
        row.update(status="failed", success=False, reason=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc())
    finally:
        os.chdir(previous_cwd)
        row["total_time_seconds"] = time.perf_counter() - started
        write_json(output / "result.json", row)
        write_summary(output / "baseline_comparison.csv", [row])
    print(f"{dataset}/{baseline}: {row['status']} {row.get('reason') or ''}", flush=True)
    return row


def write_summary(path: Path, rows: list[dict]) -> None:
    flattened = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat.update(row.get("metrics", {}))
        flattened.append(flat)
    required = ["dataset", "method", "available", "success", "status", "seed", "k",
                "n_reference", "n_query", "neighborhood_recall_at_k", "recall_at_5",
                "recall_at_10", "recall_at_15", "recall_at_30", "ndcg",
                "reference_trustworthiness", "label_knn_accuracy", "fit_time_seconds",
                "query_transform_time_seconds", "mean_query_latency_seconds", "reason"]
    fields = required + sorted(set().union(*(r.keys() for r in flattened)) - set(required))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(flattened)
    temporary.replace(path)


def run_suite(repo_root: Path, external_root: Path, output: Path,
              datasets=COMPLETED_DATASETS, baselines=BASELINES, device="auto",
              launch=subprocess.run) -> list[dict]:
    repo_root, external_root = repo_root.resolve(), external_root.resolve()
    if external_root == repo_root or repo_root in external_root.parents:
        raise ValueError("外部環境はリポジトリの外に置いてください")
    if any(d not in COMPLETED_DATASETS for d in datasets) or any(b not in BASELINES for b in baselines):
        raise ValueError("対象外のdataset/baselineです")
    output = validate_output(repo_root, repo_root / "runs", output)
    output.mkdir(parents=True, exist_ok=False)
    script = repo_root / "scripts" / "run_external_baselines.py"
    fallback = next((external_root / "venvs" / b / "bin" / "python" for b in BASELINES[:3]
                     if (external_root / "venvs" / b / ".ready").is_file()
                     and (external_root / "venvs" / b / "bin" / "python").is_file()), None)
    rows = []
    for dataset in datasets:
        for baseline in baselines:
            dest = validate_output(repo_root, repo_root / "runs" / dataset / "main", output / dataset / baseline)
            python = external_root / "venvs" / baseline / "bin" / "python"
            ready = external_root / "venvs" / baseline / ".ready"
            reason = None
            if baseline == "oos_umap":
                reason = OOS_REASON
            elif not python.is_file() or not ready.is_file():
                reason = f"隔離環境の構築未完了。{external_root}/setup_logs/{baseline}.log を確認してください"
            if reason and fallback is None:
                # supervisorの標準ライブラリ環境だけでも失敗を記録できる。
                row = base_record(dataset, baseline)
                row.update(status="unavailable", reason=reason)
                dest.mkdir(parents=True, exist_ok=False)
                write_json(dest / "result.json", row)
                write_summary(dest / "baseline_comparison.csv", [row])
            else:
                if reason:
                    python = fallback
                command = [str(python), str(script), "worker", "--dataset", dataset,
                           "--baseline", baseline, "--run-dir", str(repo_root / "runs" / dataset / "main"),
                           "--output-dir", str(dest), "--repo-root", str(repo_root), "--device", device]
                if reason:
                    command += ["--unavailable-reason", reason]
                print(f"START {dataset}/{baseline}", flush=True)
                try:
                    proc = launch(command, cwd=output, check=False)
                    row = json.loads((dest / "result.json").read_text()) if (dest / "result.json").is_file() else base_record(dataset, baseline)
                    if proc.returncode or row["status"] in ("pending", "running"):
                        row.update(status="failed", success=False,
                                   reason=f"worker終了コード={proc.returncode}; 完了結果がありません。ログを確認してください")
                except Exception as exc:
                    row = base_record(dataset, baseline)
                    row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                dest.mkdir(parents=True, exist_ok=True)
                write_json(dest / "result.json", row)
                write_summary(dest / "baseline_comparison.csv", [row])
            rows.append(row)
            write_summary(output / "summary.csv", rows)
            print(f"END {dataset}/{baseline}: {row['status']}", flush=True)
    print(f"ALL DONE: {output / 'summary.csv'}", flush=True)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--dataset", choices=COMPLETED_DATASETS, required=True)
    worker.add_argument("--baseline", choices=BASELINES, required=True)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--seed", type=int)
    worker.add_argument("--unavailable-reason", help=argparse.SUPPRESS)
    suite = sub.add_parser("suite")
    suite.add_argument("--external-root", type=Path, required=True)
    for p in (worker, suite):
        p.add_argument("--repo-root", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    args["output"] = args.pop("output_dir")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if command == "worker":
        run_one(**args)
    else:
        run_suite(**args)
    return 0
