"""完了済みseedの保存表だけから最終集計。元のmanifestは書き換えない。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .materialize import guard_output, sha
from umaping.comparison.reporting import markdown_table
from umaping.repulsion_estimators.benchmark import write_json


def summarize(run_root, seeds, output):
    root = Path(run_root).resolve()
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('seeds must be unique nonnegative integers')
    sources = []
    results, runtimes, paired = [], [], []
    common_hashes = None
    common_split = None
    expected_groups = None
    source_hashes = {}

    def read(path, csv=False):
        source_hashes[str(path)] = sha(path)
        return pd.read_csv(path) if csv else json.loads(path.read_text())

    # Finish validation before creating any output directory.
    for seed in seeds:
        comparison = root/'comparisons'/('temporal_holdout' if seed == 0 else f'temporal_seed_{seed}')
        source = root/('temporal_holdout_uniform' if seed == 0 else f'temporal_seed_{seed}_uniform')
        fit = root/('temporal_holdout_fit_grid' if seed == 0 else f'temporal_seed_{seed}_fit_grid')
        sources.extend([comparison, source, fit])
        guard_output(output, *sources)
        manifest = read(comparison/'manifest.json')
        if manifest.get('status') != 'complete' or manifest.get('source_hashes_unchanged') is not True:
            raise ValueError(f'seed {seed}: comparison not verified complete')
        hashes = manifest['upstream_hashes']
        if common_hashes is not None and hashes != common_hashes:
            raise ValueError(f'seed {seed}: frozen upstream hashes differ')
        common_hashes = hashes
        split = read(source/'temporal_split.json')
        if common_split is not None and split != common_split:
            raise ValueError(f'seed {seed}: temporal split differs')
        common_split = split
        if read(fit/'temporal_split.json') != split:
            raise ValueError(f'seed {seed}: Uniform/FitGrid split differs')
        initial = []
        for method, run in [('uniform_mc', source), ('fit_grid', fit)]:
            meta = read(run/'metadata.json')
            # comparison manifest.seed is the evaluation seed (normally 0),
            # not the matched B_phi training seed. Read model metadata instead.
            if meta.get('matched_seed') != seed:
                raise ValueError(f'{run}: matched training seed mismatch')
            initial.append(meta['initial_checkpoint_sha256'])
            runtimes.append(dict(seed=seed, method=method,
                build_seconds=meta['teacher_build_seconds'], train_seconds=meta['train_seconds']))
        if initial[0] != initial[1]:
            raise ValueError(f'seed {seed}: initial models differ')
        table = read(comparison/'aggregate.csv', csv=True)
        table = table[table.method.isin(['uniform_mc', 'fit_grid'])].copy()
        if set(table.method) != {'uniform_mc', 'fit_grid'} or not table.status.eq('success').all():
            raise ValueError(f'seed {seed}: missing/failed method')
        if table.duplicated(['method', 'group']).any():
            raise ValueError('duplicate method/group')
        groups = set(zip(table.method, table.group))
        if expected_groups is not None and groups != expected_groups:
            raise ValueError('seed group coverage differs')
        expected_groups = groups
        if set(table[table.method == 'uniform_mc'].group) != set(table[table.method == 'fit_grid'].group):
            raise ValueError('paired group coverage differs')
        if not np.isfinite(table.recall_at_15).all():
            raise ValueError('nonfinite Recall@15')
        table['seed'] = seed
        results.append(table)
        boot = read(comparison/'bootstrap.csv', csv=True)
        if set(boot.group) != set(table.group) or boot.group.duplicated().any():
            raise ValueError('paired statistics group coverage differs')
        boot['seed'] = seed
        paired.append(boot)
        # Require the finished report and cell-level pairing as well.
        for name in ('report.md', 'paired_per_query.csv'):
            source_hashes[str(comparison/name)] = sha(comparison/name)

    excluded = []
    for path in sorted((root/'comparisons').glob('temporal_seed_*/manifest.json')):
        suffix = path.parent.name.removeprefix('temporal_seed_')
        if suffix.isdigit() and int(suffix) not in seeds:
            excluded.append(dict(seed=int(suffix), stored_status=read(path).get('status'),
                disposition='excluded_by_selection',
                note='stored running is not a live process check; may be stale after termination'))
    all_seeds = pd.concat(results, ignore_index=True)
    metrics = [col for col in ('recall_at_15', 'temporal_neighbor_mae',
        'excess_temporal_neighbor_mae', 'temporal_bracketing_rate', 'density_log_distortion',
        'global_periphery_percentile', 'collapse_rate', 'latency_seconds') if col in all_seeds]
    summary = all_seeds.groupby(['method', 'group'])[metrics].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    boot = pd.concat(paired, ignore_index=True)
    differences = boot.groupby('group').mean_improvement.agg(['mean', 'std', 'count']).reset_index()
    for path, digest in source_hashes.items():
        if sha(path) != digest:
            raise ValueError(f'source changed during summarization: {path}')
    output = guard_output(output, *sources)
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', selected_seeds=seeds, excluded_seeds=excluded,
        scope='selected completed seeds only; does not complete the original three-seed execution',
        source_hashes=source_hashes, training_or_inference_executed=False)
    write_json(output/'manifest.json', status)
    try:
        all_seeds.to_csv(output/'matched_seed_results.csv', index=False)
        summary.to_csv(output/'matched_seed_summary.csv', index=False)
        pd.DataFrame(runtimes).to_csv(output/'matched_training_runtime.csv', index=False)
        boot.to_csv(output/'paired_by_seed.csv', index=False)
        differences.to_csv(output/'paired_seed_summary.csv', index=False)
        write_json(output/'temporal_split.json', common_split)
        text = ['# 完了済みseedの最終集計',
            f'採用seed: {seeds}。再学習・推論は実行せず、保存済みの完了結果だけを集計した。',
            '元の3 seed実行の成功とは区別する。元manifestや中断seedの成果物は変更していない。',
            '## seed間の平均・標準偏差', markdown_table(summary),
            '標準偏差は標本標準偏差（ddof=1）。2 seedでは再現性の根拠は限定的。',
            '## FitGrid−UniformのRecall差', markdown_table(differences),
            '同じqueryをseed間で独立な細胞として水増ししない。seed別のcell bootstrap CIは次表に保持し、CI端点を平均して新しいCIとはしない。',
            '## seed別対応比較', markdown_table(boot),
            'CIが0を含む場合、そのseedの改善は不確定。cell間依存や生物学的反復を補正したCIではない。',
            '## 学習時間', markdown_table(pd.DataFrame(runtimes)),
            '## 除外seed', markdown_table(pd.DataFrame(excluded)),
            '## 時間split', f"実時間点: {common_split['ordered_timepoints']}",
            f"補間: {common_split['interpolation_timepoint']} / 外挿: {common_split['extrapolation_timepoints']}"]
        (output/'report.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')
        status['status'] = 'complete'
    except BaseException as exc:
        status.update(status='failed', reason=str(exc))
        raise
    finally:
        write_json(output/'manifest.json', status)
    return output
