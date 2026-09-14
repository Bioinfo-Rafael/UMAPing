"""実際の内部・外部保存schemaを正規化し、測定段階と出典を保持する。"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATASETS = ("coil20", "coil100", "pancreas", "pancreas_batch_corrected",
            "fashion_mnist", "mnist_oos", "organoid", "embryoid_body")
ALIASES = {"ours_full": "ours", "oracle_neighbors": "ours_oracle_neighbors",
           "numap_sep_spectralnet": "numap", "param_repulsor": "paramrepulsor"}
# Monte Carlo診断はexact診断に改名してはいけない。
DIAGNOSTICS = {"repulsion_oracle_diagnostic", "exact_repulsion_diagnostic"}
METRICS = ("recall_at_5", "recall_at_10", "recall_at_15", "recall_at_30", "ndcg",
           "reference_trustworthiness", "label_knn_accuracy", "fit_time_seconds",
           "mean_query_latency_seconds", "local_displacement", "fuzzy_weighted_mse",
           "fuzzy_weighted_bce", "repulsion_accumulation_score", "periphery_percentile")
TIMES = {"fit_time_seconds", "mean_query_latency_seconds"}
# 外部runnerはdirect k=15とmax-kのprefixを別々に計算して保存する。
# 同距離近傍の選択は一致を保証しないため、同じ列へ潰さない。
RECORD_METRICS = (*METRICS, "recall15_direct", "recall15_multi_k")


def canonical(name):
    return ALIASES.get(str(name), str(name))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def truth(value):
    if value is True or str(value).strip().lower() in ("true", "1", "1.0"):
        return True
    if value is False or value is None or pd.isna(value) or str(value).strip().lower() in ("false", "0", "0.0", ""):
        return False
    raise ValueError(f"真偽値を解釈できません: {value}")


@dataclass
class Loaded:
    internal_root: Path
    external_root: Path
    rows: dict = field(default_factory=dict)
    queries: dict = field(default_factory=dict)
    raw: list = field(default_factory=list)
    availability: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    external_details: dict = field(default_factory=dict)
    configs: dict = field(default_factory=dict)
    subsets: dict = field(default_factory=dict)

    def track(self, path):
        path = Path(path).resolve()
        if str(path) not in self.sources:
            self.sources[str(path)] = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
        return path

    def json(self, path):
        def unique_object(pairs):
            result = {}
            for key, val in pairs:
                if key in result:
                    raise ValueError(f"JSONキーが重複しています: {path}: {key}")
                result[key] = val
            return result
        return json.loads(self.track(path).read_text(), object_pairs_hook=unique_object)

    def csv(self, path):
        return pd.read_csv(self.track(path), float_precision="round_trip")

    def add(self, dataset, method, record, path, phase, priority=0):
        method = canonical(method)
        row = dict(record)
        metrics = row.pop("metrics", {})
        for name, val in metrics.items():
            if name in row and np.isfinite(finite(row[name])) and np.isfinite(finite(val)):
                if not np.isclose(float(row[name]), float(val), rtol=1e-9, atol=1e-12):
                    raise ValueError(f"result内の値が矛盾しています: {dataset}/{method}/{name}")
            row[name] = val
        self.raw.append(dict(dataset=dataset, method=method, phase=phase, source=str(path), raw=record))
        status = str(row.get("status", ""))
        success = truth(row.get("available", True)) and truth(row.get("success", True))
        if status:
            success = success and status == "success"
        if method == "oos_umap" and success:
            raise ValueError("OOS-UMAPはこの実験では利用不可です。成功行を受け付けません")
        self.availability.append(dict(dataset=dataset, method=method, phase=phase, success=success,
                                      status="success" if success else status or "unavailable",
                                      reason=row.get("reason", ""), source=str(path)))
        k = finite(row.get("k"))
        subset_ids = row.get("query_subset_indices")
        has_subset = isinstance(subset_ids, (list, tuple, np.ndarray)) and len(subset_ids) > 0
        current = dict(dataset=dataset, method=method, success=success,
                       scope="subset" if method in DIAGNOSTICS or has_subset else "full",
                       n_queries=finite(row.get("n_queries", row.get("n_query"))), k=k,
                       phase=phase, priority=priority, source=str(path),
                       metric_sources={}, **{m: finite(row.get(m)) if success else np.nan for m in RECORD_METRICS})
        if success and phase == "external":
            current["recall15_multi_k"] = finite(row.get("recall_at_15"))
        if success and k == 15 and np.isfinite(finite(row.get("neighborhood_recall_at_k"))):
            primary = finite(row["neighborhood_recall_at_k"])
            current["recall15_direct"] = primary
            if np.isfinite(current["recall_at_15"]) and not np.isclose(current["recall_at_15"], primary, rtol=1e-9, atol=1e-12):
                if phase != "external":
                    raise ValueError(f"Recall@15の列同士が矛盾しています: {path}")
                self.notes.append(
                    f"{dataset}/{method}: direct k=15 Recall={primary:.12g}、multi-k Recall15={current['recall_at_15']:.12g}。"
                    "保存実装は15近傍の直接取得と最大k近傍の先頭15件を別々に計算するため、同距離近傍の選択で差が生じ得る。"
                    "主表はdirect、対応・multi-k解析は保存multi-k値を使用。実データで差が生じた個別原因は再計算していない。"
                )
            current["recall_at_15"] = primary
        current["metric_sources"] = {m: str(path) for m in RECORD_METRICS if np.isfinite(current[m])}
        key = dataset, method
        old = self.rows.get(key)
        if old is None or (success and not old["success"]):
            self.rows[key] = current
        elif success and old["success"]:
            for name in (*RECORD_METRICS, "k", "n_queries"):
                x, y = old[name], current[name]
                if np.isfinite(x) and np.isfinite(y) and not np.isclose(x, y, rtol=1e-9, atol=1e-12):
                    if name not in TIMES or (phase == "external" and old["phase"] == "external"):
                        raise ValueError(f"集約結果が矛盾しています: {dataset}/{method}/{name}: {x} vs {y} ({old['source']} / {path})")
                    self.notes.append(f"{dataset}/{method}/{name}: 別実行段階の時間 {x} / {y}。baseline比較・timing段階を優先し、原値はraw_metric_recordsに保存。")
                if np.isfinite(y) and (not np.isfinite(x) or (name in TIMES and priority > old["priority"])):
                    old[name] = y
                    old["metric_sources"][name] = str(path)
            old["priority"] = max(old["priority"], priority)
        return key

    def add_queries(self, dataset, method, group, source, identity_verified=True):
        key = dataset, canonical(method)
        if key in self.queries:
            raise ValueError(f"query結果が重複しています: {key}")
        frame = group.copy()
        if "query_index" not in frame:
            raise ValueError(f"query_indexがありません: {source}")
        idx = pd.to_numeric(frame.query_index, errors="raise")
        if idx.isna().any() or (idx < 0).any() or (idx % 1 != 0).any() or idx.duplicated().any():
            raise ValueError(f"query_indexが不正・重複しています: {key}")
        frame["query_index"] = idx.astype(int)
        if key[1] == "repulsion_oracle_diagnostic":
            mapping = self.subsets.get(key)
            if mapping is None:
                identity_verified = False
            elif np.array_equal(frame.query_index.to_numpy(), np.arange(len(mapping))):
                # 元pipelineはsubset内の連番を保存した。embedding.jsonの元indexへ戻す。
                frame["query_index"] = mapping
                self.notes.append(f"{dataset}/MC斥力診断: 保存されたsubset内連番をquery_subset_indicesで元query IDへ復元。")
            elif set(frame.query_index) != set(mapping):
                raise ValueError(f"診断subsetのquery IDが一致しません: {key}")
        row = self.rows.get(key)
        if row is None or not row["success"]:
            self.notes.append(f"{key}: 成功した集約結果がないためper-query比較から除外。")
            return
        if np.isfinite(row["n_queries"]) and len(frame) != int(row["n_queries"]):
            raise ValueError(f"集約結果とquery件数が違います: {key}")
        external = Path(source).resolve().is_relative_to(self.external_root)
        if external and row["k"] == 15 and "neighborhood_recall_at_k" in frame:
            direct = pd.to_numeric(frame["neighborhood_recall_at_k"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
            expected = row["recall15_direct"]
            if np.isfinite(direct) and np.isfinite(expected) and not np.isclose(direct, expected, rtol=1e-9, atol=1e-12):
                raise ValueError(f"外部の集約値とper-query値が矛盾しています: {key}/neighborhood_recall_at_k")
        for m in METRICS:
            if m in frame:
                frame[m] = pd.to_numeric(frame[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
                avg = frame[m].mean()
                differs = np.isfinite(avg) and np.isfinite(row[m]) and not np.isclose(avg, row[m], rtol=1e-9, atol=1e-12)
                expected = row["recall15_multi_k"] if m == "recall_at_15" and external else row[m]
                conflicts = np.isfinite(avg) and np.isfinite(expected) and not np.isclose(avg, expected, rtol=1e-9, atol=1e-12)
                if conflicts and external:
                    raise ValueError(f"外部の集約値とper-query値が矛盾しています: {key}/{m}")
                if m == "recall_at_15":
                    row["per_query_phase_recall15_mean"] = avg
                    if differs and not external:
                        # analyze-advancedは当時別途埋め込みを生成した段階。primaryを置換しない。
                        self.notes.append(f"{dataset}/{key[1]}: 集約Recall15={row[m]:.12g}、追加解析段階の平均={avg:.12g}。主表は集約値、対応検定は追加解析段階の値を使用。")
                if not np.isfinite(row[m]) and np.isfinite(avg):
                    row[m] = float(avg)
                    row["metric_sources"][m] = str(source) + " (per-query mean)"
        self.queries[key] = dict(frame=frame.set_index("query_index").sort_index(), source=str(source),
                                 identity_verified=identity_verified)

    def frame(self):
        frame = pd.DataFrame(self.rows.values())
        if frame.empty:
            return pd.DataFrame(columns=["dataset", "method", "success", "scope", "n_queries", *METRICS])
        return frame


def unique_methods(frame, path, dataset=None):
    if "method" not in frame:
        raise ValueError(f"method列がありません: {path}")
    keys = [((dataset or r.get("dataset")), canonical(r["method"])) for r in frame.to_dict("records")]
    if len(keys) != len(set(keys)):
        raise ValueError(f"dataset/methodの重複行: {path}")


def load_results(internal_root, external_root):
    data = Loaded(Path(internal_root).resolve(), Path(external_root).resolve())
    summary_path = data.external_root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"外部結果summary.csvがありません: {summary_path}")
    for dataset in DATASETS:
        run = data.internal_root / dataset / "main"
        config = run / "config.yaml"
        if config.is_file():
            data.configs[dataset] = yaml.safe_load(data.track(config).read_text())
        metadata = run / "metadata.json"
        if metadata.is_file():
            data.json(metadata)
        path = run / "metrics/embedding.json"
        if path.is_file():
            records = data.json(path)
            if len({canonical(m) for m in records}) != len(records):
                raise ValueError(f"method別名が重複しています: {path}")
            for method, record in records.items():
                key = data.add(dataset, method, record, path, "internal_evaluate")
                if record.get("query_subset_indices") is not None:
                    data.subsets[key] = np.asarray(record["query_subset_indices"], int)
        path = run / "metrics/baseline_comparison.csv"
        if path.is_file():
            records = data.csv(path)
            unique_methods(records, path, dataset)
            for record in records.to_dict("records"):
                data.add(dataset, record["method"], record, path, "internal_baselines", priority=2)
        timing = run / "metrics/timing.json"
        if timing.is_file():
            for method, values in data.json(timing).items():
                key = dataset, canonical(method)
                if key not in data.rows or not data.rows[key]["success"]:
                    continue
                # 別測定段階のレイテンシをrawで保持した上で、baseline段階の時間を選ぶ。
                data.add(dataset, method, values, timing, "internal_timing", priority=3)
        n_full = data.rows.get((dataset, "ours"), {}).get("n_queries", np.nan)
        exact = (dataset, "exact_repulsion_diagnostic")
        if exact in data.rows and np.isfinite(n_full) and dataset in data.configs:
            n = data.rows[exact]["n_queries"]
            cfg = data.configs[dataset]
            if np.isfinite(n) and "seed" in cfg:
                data.subsets[exact] = np.random.default_rng(cfg["seed"]).choice(int(n_full), size=int(n), replace=False)
                data.notes.append(f"{dataset}/exact診断: 既存runnerのseedとchoice規則から評価subsetを復元。")
        path = run / "metrics/advanced_per_query.csv"
        if path.is_file():
            frame = data.csv(path)
            # 同一ファイル内で異なる別名が同じmethodに潰れることも拒否。
            methods = frame.method.unique()
            if len(set(map(canonical, methods))) != len(methods):
                raise ValueError(f"queryのmethod別名が重複しています: {path}")
            for method, group in frame.groupby("method", sort=False):
                data.add_queries(dataset, method, group, path)

    summary = data.csv(summary_path)
    if "dataset" not in summary:
        raise ValueError("外部summary.csvにdataset列がありません")
    summary = summary[summary.dataset.isin(DATASETS)]
    unique_methods(summary, summary_path)
    for record in summary.to_dict("records"):
        dataset, method = record["dataset"], canonical(record["method"])
        key = data.add(dataset, method, record, summary_path, "external", priority=4)
        folders = []
        for candidate in (data.external_root / dataset).glob("*/result.json"):
            detail = data.json(candidate)
            if detail.get("dataset") != dataset:
                raise ValueError(f"外部resultのdatasetがパスと不一致: {candidate}")
            if canonical(detail.get("method", detail.get("baseline"))) == method:
                folders.append((candidate.parent, detail))
        if len(folders) > 1:
            raise ValueError(f"同一外部methodのresultが複数あります: {key}")
        if not folders:
            data.notes.append(f"{dataset}/{method}: result.jsonなし。summaryの集約値だけを使用。")
            continue
        folder, detail = folders[0]
        expected_success = truth(record.get("success")) and str(record.get("status")) == "success"
        if expected_success != (truth(detail.get("success")) and detail.get("status") == "success"):
            raise ValueError(f"summaryとresultの成功statusが矛盾しています: {key}")
        data.add(dataset, method, detail, folder / "result.json", "external", priority=4)
        data.external_details[key] = dict(folder=folder, record=detail)
        if not data.rows[key]["success"]:
            continue
        config_path = data.internal_root / dataset / "main/config.yaml"
        if config_path.is_file() and detail.get("config_sha256"):
            if digest(config_path) != detail["config_sha256"]:
                raise ValueError(f"外部と内部のconfig hashが不一致: {key}")
        cache = data.internal_root / dataset / "main/cache/prepared_dataset.npz"
        verified = False
        if cache.is_file() and detail.get("prepared_dataset_sha256"):
            data.track(cache)
            if data.sources[str(cache.resolve())]["sha256"] != detail["prepared_dataset_sha256"]:
                raise ValueError(f"外部と内部の前処理cache hashが不一致: {key}")
            verified = True
        detail["identity_verified"] = verified
        perquery = folder / "advanced_per_query.csv"
        if perquery.is_file():
            data.add_queries(dataset, method, data.csv(perquery), perquery, identity_verified=verified)
        else:
            data.notes.append(f"{dataset}/{method}: per-query保存なし。推論を再実行せず集約比較のみ。")
        if not verified:
            data.notes.append(f"{dataset}/{method}: 内部cacheとのhash照合ができないため対応検定は不可。集約値の分割同一性は未検証。")
    # 同じdatasetの全query評価で件数が違う場合は集約でも混ぜない。
    for dataset in DATASETS:
        counts = {r["n_queries"] for (d, _), r in data.rows.items()
                  if d == dataset and r["success"] and r["scope"] == "full" and np.isfinite(r["n_queries"])}
        if len(counts) > 1:
            raise ValueError(f"full queryの件数がmethod間で一致しません: {dataset}: {counts}")
    return data
