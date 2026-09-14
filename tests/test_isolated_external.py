"""実データ/外部学習を使わない隔離比較の検証。"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import types

import numpy as np
import pytest

from umaping.config import Config
from umaping.data.preprocessing import PreparedDataset, save_prepared_dataset
from umaping.experiments import external
from umaping.experiments.baselines import BaselineResult, BaselineUnavailable

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cached_run(tmp_path):
    repo = tmp_path / "repo"
    run = repo / "runs" / "coil20" / "main"
    (run / "cache").mkdir(parents=True)
    cfg = Config.load(REPO / "configs/mock.yaml")
    cfg.dataset.name = "coil20"
    cfg.dataset.input_dim = 4
    cfg.eval.k = 5
    cfg.eval.trustworthiness_n_neighbors = 5
    cfg.save(run / "config.yaml")
    rng = np.random.default_rng(3)
    prepared = PreparedDataset(rng.normal(size=(40, 4)).astype("float32"),
                               rng.normal(size=(8, 4)).astype("float32"),
                               {"object_id": np.arange(40) % 2},
                               {"object_id": np.arange(8) % 2}, 4)
    save_prepared_dataset(run / "cache/prepared_dataset.npz", prepared)
    (run / "metadata.json").write_text('{"git_commit": "original-run-commit"}')
    return repo, run, prepared


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def fake_outcome(p):
    return BaselineResult("parametric_umap", p.reference_features[:, :2].copy(),
                          p.query_features[:, :2].copy(), fit_time_seconds=1.0,
                          mean_query_latency_seconds=0.01)


@pytest.fixture
def pinned_version(monkeypatch):
    monkeypatch.setattr(external.importlib.metadata, "version", lambda _: "0.5.12")


def test_mock_integration_preserves_run_and_uses_existing_metrics(cached_run, tmp_path, monkeypatch, pinned_version):
    from umaping.evaluation.embedding import evaluate_embedding
    repo, run, prepared = cached_run
    # キャッシュ評価にscRNAの生データ処理依存は必要ない。
    monkeypatch.setitem(sys.modules, "scanpy", None)
    monkeypatch.setitem(sys.modules, "anndata", None)
    before = snapshot(run)
    def adapter(name, p, cfg, device):
        np.testing.assert_array_equal(p.reference_features, prepared.reference_features)
        np.testing.assert_array_equal(p.query_features, prepared.query_features)
        assert Path.cwd().is_relative_to(tmp_path / "new-results")
        return fake_outcome(p)
    monkeypatch.setattr(external, "run_adapter", adapter)
    output = tmp_path / "new-results" / "coil20" / "parametric_umap"
    row = external.run_one("coil20", "parametric_umap", run, output, repo, device="cpu")
    assert row["status"] == "success", row.get("reason")
    assert row["n_reference"] == 40 and row["n_query"] == 8 and row["k"] == 5
    assert row["query_transform_time_seconds"] == 0.08
    assert row["original_run_metadata"]["git_commit"] == "original-run-commit"
    assert row["prepared_dataset_sha256"] == external.sha256(run / "cache/prepared_dataset.npz")
    expected, _ = evaluate_embedding(prepared.query_features, prepared.reference_features,
                                    prepared.query_features[:, :2], prepared.reference_features[:, :2],
                                    k=5, trustworthiness_n_neighbors=5)
    assert row["metrics"]["neighborhood_recall_at_k"] == expected.neighborhood_recall_at_k
    assert (output / "embedding.png").stat().st_size > 0
    assert len(list(csv.DictReader((output / "advanced_per_query.csv").open()))) == 8
    assert row["metrics"]["ndcg_k"] == 30
    assert snapshot(run) == before
    assert json.loads((output / "result.json").read_text())["success"]
    with pytest.raises(FileExistsError):
        external.run_one("coil20", "parametric_umap", run, output, repo)


@pytest.mark.parametrize("location", ["runs", "runs/coil20/main/metrics/new", "."])
def test_reject_output_overlapping_old_results(cached_run, location):
    repo, run, _ = cached_run
    with pytest.raises(ValueError):
        external.run_one("coil20", "parametric_umap", run, repo / location, repo)


def test_reject_symlink_to_existing_runs(cached_run, tmp_path):
    repo, run, _ = cached_run
    (tmp_path / "alias").symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError):
        external.validate_output(repo, run, tmp_path / "alias/new")


@pytest.mark.parametrize("dataset", ["twenty_newsgroups", "hong_ed"])
def test_failed_datasets_excluded_and_rejected(cached_run, tmp_path, dataset):
    assert dataset not in external.COMPLETED_DATASETS
    assert len(external.COMPLETED_DATASETS) == 8
    repo, run, _ = cached_run
    with pytest.raises(ValueError):
        external.run_one(dataset, "numap", run, tmp_path / "out", repo)
    with pytest.raises(ValueError):
        external.run_suite(repo, tmp_path / "envs", tmp_path / "out", datasets=[dataset])


def test_missing_cache_is_failure_without_repreprocessing(cached_run, tmp_path):
    repo, run, _ = cached_run
    (run / "cache/prepared_dataset.npz").unlink()
    before = snapshot(run)
    row = external.run_one("coil20", "parametric_umap", run, tmp_path / "out", repo)
    assert row["status"] == "failed"
    assert "キャッシュ" in row["reason"]
    assert snapshot(run) == before


def test_optional_import_failure_and_nonfinite_output(cached_run, tmp_path, monkeypatch, pinned_version):
    repo, run, _ = cached_run
    def absent(*args):
        raise ImportError("optional dependency absent")
    monkeypatch.setattr(external, "run_adapter", absent)
    row = external.run_one("coil20", "parametric_umap", run, tmp_path / "absent", repo)
    assert row["status"] == "unavailable" and not row["available"]
    def broken(name, p, *args):
        outcome = fake_outcome(p)
        outcome.query_embedding[0, 0] = np.nan
        return outcome
    monkeypatch.setattr(external, "run_adapter", broken)
    row = external.run_one("coil20", "parametric_umap", run, tmp_path / "nan", repo)
    assert row["status"] == "failed" and row["available"] and not row["success"]


def test_oos_limitation_recorded_without_calling_adapter(cached_run, tmp_path, monkeypatch):
    repo, run, _ = cached_run
    monkeypatch.setattr(external, "run_adapter", lambda *_: pytest.fail("OOS must not be reimplemented"))
    row = external.run_one("coil20", "oos_umap", run, tmp_path / "out", repo)
    assert row["status"] == "unavailable"
    assert row["source"]["inspected_commit"] == external.OOS_COMMIT
    assert row["n_reference"] == 40


def test_suite_continues_after_worker_crash(tmp_path):
    repo, root, output = tmp_path / "repo", tmp_path / "outside", tmp_path / "results"
    repo.mkdir()
    for name in external.BASELINES[:3]:
        env = root / "venvs" / name
        (env / "bin").mkdir(parents=True)
        (env / "bin/python").touch()
        (env / ".ready").touch()
    invoked = []
    def launch(command, **kwargs):
        name = command[command.index("--baseline") + 1]
        invoked.append(name)
        if name == "parametric_umap":
            return types.SimpleNamespace(returncode=-9)
        dest = Path(command[command.index("--output-dir") + 1])
        dest.mkdir(parents=True)
        row = external.base_record("coil20", name)
        row.update(status="success", available=True, success=True)
        if "--unavailable-reason" in command:
            row.update(status="unavailable", available=False, success=False)
        external.write_json(dest / "result.json", row)
        return types.SimpleNamespace(returncode=0)
    rows = external.run_suite(repo, root, output, datasets=["coil20"], launch=launch)
    assert invoked == list(external.BASELINES)
    assert [r["status"] for r in rows] == ["failed", "success", "success", "unavailable"]
    assert len(list(csv.DictReader((output / "summary.csv").open()))) == 4


def test_suite_missing_environments_records_all_32_jobs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rows = external.run_suite(repo, tmp_path / "outside", tmp_path / "results",
                              launch=lambda *_a, **_kw: pytest.fail("No environment may run"))
    assert len(rows) == 32
    assert all(r["status"] == "unavailable" for r in rows)


def test_parent_cli_imports_with_only_standard_library():
    result = subprocess.run([sys.executable, "-S", str(REPO / "scripts/run_external_baselines.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "suite" in result.stdout


def test_parametric_official_adapter_reference_only(monkeypatch, cached_run):
    from umaping.experiments.baselines import run_parametric_umap_baseline
    repo, run, p = cached_run
    seen = {}
    class FakeParametric:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs
        def fit_transform(self, x):
            np.testing.assert_array_equal(x, p.reference_features)
            return x[:, :2]
        def transform(self, x):
            np.testing.assert_array_equal(x, p.query_features)
            return x[:, :2]
    monkeypatch.setitem(sys.modules, "umap.parametric_umap", types.SimpleNamespace(ParametricUMAP=FakeParametric))
    cfg = Config.load(run / "config.yaml")
    cfg.umap.spread = 1.5
    result = run_parametric_umap_baseline(p, cfg)
    assert isinstance(result, BaselineResult)
    assert seen["kwargs"]["spread"] == 1.5
    assert result.query_embedding.shape == (8, 2)


def test_pinned_package_mismatch_refuses_execution(cached_run, tmp_path, monkeypatch):
    repo, run, _ = cached_run
    monkeypatch.setattr(external.importlib.metadata, "version", lambda _: "99.0")
    monkeypatch.setattr(external, "run_adapter", lambda *_: pytest.fail("wrong version"))
    row = external.run_one("coil20", "numap", run, tmp_path / "out", repo)
    assert row["status"] == "failed" and "バージョン不一致" in row["reason"]


def test_setup_continues_after_install_failure_and_targets_only_isolated_envs(tmp_path):
    """shellのerrexit・環境分離を実プロセスで検証。uvとインポートだけをmockする。"""
    import os
    import shutil
    repo, outside = tmp_path / "repo", tmp_path / "outside"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts/setup_external_baselines.sh", repo / "scripts")
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text("#!/bin/sh\necho mock-commit\n")
    (bin_dir / "git").chmod(0o755)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['MOCK_CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
if args[:2] == ['python', 'find']:
    print({sys.executable!r})
elif args[0] == 'venv':
    target = pathlib.Path(args[-1]) / 'bin'
    target.mkdir(parents=True)
    python = target / 'python'
    python.write_text('#!/bin/sh\\nexit 0\\n')
    python.chmod(0o755)
elif args[:2] == ['pip', 'install'] and '/numap/' in ' '.join(args):
    raise SystemExit(17)
elif args[:2] == ['pip', 'freeze']:
    print('mock==1.0')
''')
    fake_uv.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    original = tmp_path / "original-env"
    original.mkdir()
    (original / "sentinel").write_text("unchanged")
    result = subprocess.run(["bash", str(repo / "scripts/setup_external_baselines.sh"), str(outside)],
                            env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                                 "MOCK_CALLS": str(calls), "VIRTUAL_ENV": str(original)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (outside / "venvs/parametric_umap/.ready").is_file()
    assert not (outside / "venvs/numap/.ready").exists()
    assert (outside / "venvs/paramrepulsor/.ready").is_file()
    assert (original / "sentinel").read_text() == "unchanged"
    for args in map(json.loads, calls.read_text().splitlines()):
        if args[0] == "pip":
            assert Path(args[args.index("--python") + 1]).is_relative_to(outside)


def test_background_launcher_writes_pid_log_and_summary_without_dependencies(tmp_path):
    import shutil
    import time
    repo, outside = tmp_path / "repo", tmp_path / "outside"
    shutil.copytree(REPO / "scripts", repo / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    outside.mkdir()
    (outside / "control-python.txt").write_text(sys.executable + "\n")
    (repo / "runs").mkdir()
    (repo / "runs/sentinel").write_text("original results")
    before = snapshot(repo / "runs")
    result = subprocess.run(["bash", str(repo / "scripts/launch_external_baselines.sh"), str(outside)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "PID:" in result.stdout and "tail -f" in result.stdout
    logs = repo / "runs_external_baselines/logs"
    assert len(list(logs.glob("*.pid"))) == 1
    logfile = next(logs.glob("*.log"))
    deadline = time.monotonic() + 10
    while "ALL DONE" not in logfile.read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "ALL DONE" in logfile.read_text(), logfile.read_text()
    summary = next((repo / "runs_external_baselines").glob("*/summary.csv"))
    assert len(list(csv.DictReader(summary.open()))) == 32
    assert snapshot(repo / "runs") == before
